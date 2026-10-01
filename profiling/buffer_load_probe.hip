// BUFFER_LOAD_B64 vs global_load_b64 bit-exact + DS_SWIZZLE SWAPX16 vs shfl_xor-16.
// ISA §9 / §16.16 (buffer) and §16.15 (ds_swizzle).
//
//   hipcc --offload-arch=gfx1100 -O3 -o buffer_load_probe buffer_load_probe.hip
//   ./buffer_load_probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define THREADS 256
#define N (THREADS * 64)

__device__ __forceinline__ uint2 ld_global(const uint2* p)
{
    return *p;
}

using buf_rsrc_t = decltype(__builtin_amdgcn_make_buffer_rsrc((void*) nullptr, (short) 0, 0, 0));

__device__ __forceinline__ uint2 ld_buffer(buf_rsrc_t rsrc, const void* base, const void* p)
{
    using v2i = int __attribute__((ext_vector_type(2)));
    int off = (int) ((const char*) p - (const char*) base);
    v2i v = __builtin_amdgcn_raw_buffer_load_b64(rsrc, off, 0, 0);
    uint2 out;
    out.x = (uint32_t) v[0];
    out.y = (uint32_t) v[1];
    return out;
}

__global__ void k_exact(const uint2* src, unsigned long long* mis)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<uint2*>(src), (short) 0, (int) 0xffffffff, 0x31004000);
    uint2 a = ld_global(src + i);
    uint2 b = ld_buffer(rsrc, src, src + i);
    if (a.x != b.x || a.y != b.y) atomicAdd(mis, 1ull);
}

// DS_SWIZZLE SWAPX16: xor_mask=0x10, or=0, and=0x1f → offset 0x401f
__global__ void k_swizzle16(const uint32_t* src, unsigned long long* mis)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t v = src[i];
    uint32_t a = __shfl_xor_sync(0xffffffffffffffffull, v, 16);
    uint32_t b = __builtin_amdgcn_ds_swizzle(v, 0x401f);
    if (a != b) atomicAdd(mis, 1ull);
}

template <int MODE> // 0 global, 1 buffer
__global__ __launch_bounds__(THREADS)
void k_thru(const uint2* src, int iters, unsigned long long* sink)
{
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<uint2*>(src), (short) 0, (int) 0xffffffff, 0x31004000);
    uint2 a = src[threadIdx.x];
    uint2 b = src[(threadIdx.x + 17) & 255];
    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        const uint2* p0 = src + ((threadIdx.x + it * 2) & (N - 1));
        const uint2* p1 = src + ((threadIdx.x + it * 2 + 32) & (N - 1));
        uint2 x, y;
        if constexpr (MODE == 1)
        {
            x = ld_buffer(rsrc, src, p0);
            y = ld_buffer(rsrc, src, p1);
        }
        else
        {
            x = ld_global(p0);
            y = ld_global(p1);
        }
        a.x ^= x.x + y.y;
        a.y += x.y ^ y.x;
        b.x += y.x;
        b.y ^= x.y;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *sink = (unsigned long long) a.x + a.y + b.x + b.y;
}

static void ck(hipError_t e, const char* w)
{
    if (e != hipSuccess)
    {
        fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(1);
    }
}

int main()
{
    std::vector<uint2> h(N);
    for (int i = 0; i < N; ++i)
    {
        h[i].x = 0x9e3779b9u * (uint32_t) (i + 1);
        h[i].y = 0x85ebca6bu * (uint32_t) (i + 7);
    }
    uint2* d = nullptr;
    unsigned long long* mis = nullptr;
    unsigned long long* sink = nullptr;
    ck(hipMalloc(&d, N * sizeof(uint2)), "malloc d");
    ck(hipMalloc(&mis, sizeof(unsigned long long)), "malloc mis");
    ck(hipMalloc(&sink, sizeof(unsigned long long)), "malloc sink");
    ck(hipMemcpy(d, h.data(), N * sizeof(uint2), hipMemcpyHostToDevice), "h2d");
    ck(hipMemset(mis, 0, sizeof(unsigned long long)), "zero mis");

    k_exact<<<N / THREADS, THREADS>>>(d, mis);
    ck(hipDeviceSynchronize(), "exact");
    unsigned long long hmis = 1;
    ck(hipMemcpy(&hmis, mis, sizeof(hmis), hipMemcpyDeviceToHost), "d2h mis");
    printf("buffer_vs_global mismatches %llu\n", (unsigned long long) hmis);

    ck(hipMemset(mis, 0, sizeof(unsigned long long)), "zero mis2");
    k_swizzle16<<<N / THREADS, THREADS>>>((const uint32_t*) d, mis);
    ck(hipDeviceSynchronize(), "swizzle");
    unsigned long long hmis2 = 1;
    ck(hipMemcpy(&hmis2, mis, sizeof(hmis2), hipMemcpyDeviceToHost), "d2h mis2");
    printf("ds_swizzle_xor16 mismatches %llu\n", (unsigned long long) hmis2);

    const int iters = 4096;
    hipEvent_t a, b;
    ck(hipEventCreate(&a), "ea");
    ck(hipEventCreate(&b), "eb");
    auto bench = [&](int mode, const char* name)
    {
        for (int w = 0; w < 5; ++w)
        {
            if (mode == 1) k_thru<1><<<64, THREADS>>>(d, iters, sink);
            else           k_thru<0><<<64, THREADS>>>(d, iters, sink);
        }
        ck(hipDeviceSynchronize(), "warm");
        ck(hipEventRecord(a, 0), "ra");
        for (int r = 0; r < 20; ++r)
        {
            if (mode == 1) k_thru<1><<<64, THREADS>>>(d, iters, sink);
            else           k_thru<0><<<64, THREADS>>>(d, iters, sink);
        }
        ck(hipEventRecord(b, 0), "rb");
        ck(hipEventSynchronize(b), "syncb");
        float ms = 0;
        ck(hipEventElapsedTime(&ms, a, b), "el");
        double bytes = 20.0 * 64.0 * THREADS * iters * 2.0 * 8.0;
        printf("%s %.2f GB/s (%.3f ms)\n", name, bytes / 1e6 / ms, ms);
    };
    bench(0, "global_load");
    bench(1, "buffer_load");
    return (hmis || hmis2) ? 1 : 0;
}
