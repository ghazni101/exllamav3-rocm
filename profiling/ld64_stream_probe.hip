// Bit-exact and bandwidth check for global_load_dwordx2 with slc+dlc.
// Same instruction exl3_ld64_stream emits. Values must match a plain load.
// Bandwidth is only meaningful cold, on a buffer larger than the Infinity Cache.
//
//   hipcc -O3 --offload-arch=gfx1100 ld64_stream_probe.hip -o ld64_stream_probe
//   ./ld64_stream_probe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>

#define CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #cmd, hipGetErrorString(e)); \
        return 1; \
    } \
} while (0)

__device__ __forceinline__ uint2 ld_stream(const uint2* p)
{
    unsigned long long r;
    asm volatile("global_load_dwordx2 %0, %1, off slc dlc" : "=v"(r) : "v"(p));
    uint2 out;
    out.x = (uint32_t) r;
    out.y = (uint32_t) (r >> 32);
    return out;
}

__global__ void copy_kernel(const uint2* in, uint2* out, int n, int stream)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = stream ? ld_stream(in + i) : in[i];
}

// Consume every dword so the load cannot be deleted. One xor per uint2.
__global__ void scan_kernel(const uint2* in, unsigned int* partial, int n, int stream)
{
    unsigned int acc = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        uint2 v = stream ? ld_stream(in + i) : in[i];
        acc ^= v.x;
        acc ^= v.y;
    }
    atomicXor(partial, acc);
}

int main(int argc, char** argv)
{
    const int n = 1 << 20; // 8 MB, exactness
    uint2 *d_in = nullptr, *d_out = nullptr;
    CHECK(hipMalloc(&d_in, (size_t) n * sizeof(uint2)));
    CHECK(hipMalloc(&d_out, (size_t) n * sizeof(uint2)));
    std::vector<uint2> h(n);
    for (int i = 0; i < n; ++i)
    {
        h[i].x = 0x9E3779B9u * (unsigned) i + 1u;
        h[i].y = 0x85EBCA77u * (unsigned) i + 7u;
    }
    CHECK(hipMemcpy(d_in, h.data(), (size_t) n * sizeof(uint2), hipMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    copy_kernel<<<blocks, threads>>>(d_in, d_out, n, 0);
    CHECK(hipDeviceSynchronize());
    std::vector<uint2> plain(n), streamed(n);
    CHECK(hipMemcpy(plain.data(), d_out, (size_t) n * sizeof(uint2), hipMemcpyDeviceToHost));
    CHECK(hipMemset(d_out, 0x5A, (size_t) n * sizeof(uint2)));
    copy_kernel<<<blocks, threads>>>(d_in, d_out, n, 1);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(streamed.data(), d_out, (size_t) n * sizeof(uint2), hipMemcpyDeviceToHost));

    long mismatches = 0;
    for (int i = 0; i < n; ++i)
        if (plain[i].x != streamed[i].x || plain[i].y != streamed[i].y
            || plain[i].x != h[i].x || plain[i].y != h[i].y)
            ++mismatches;
    printf("exact_mismatches %ld / %d\n", mismatches, n);
    if (mismatches) return 2;

    // Optional cold bandwidth: argv[1] is "plain" or "stream". One mode per
    // process. 768 MB is 8x a 96 MB Infinity Cache. A second buffer is written
    // after the fault-in so the timed read does not hit MALL.
    if (argc > 1)
    {
        int mode = (argv[1][0] == 's') ? 1 : 0;
        const size_t bytes = 768ull << 20;
        const int nb = (int) (bytes / sizeof(uint2));
        uint2 *big = nullptr, *flush = nullptr;
        unsigned int* part = nullptr;
        CHECK(hipMalloc(&big, bytes));
        CHECK(hipMalloc(&flush, bytes));
        CHECK(hipMalloc(&part, sizeof(unsigned int)));
        CHECK(hipMemset(big, 1, bytes));          // fault the pages in
        CHECK(hipMemset(flush, 2, bytes));        // evict big from MALL
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemset(part, 0, sizeof(unsigned int)));
        int grid = 96 * 8;
        hipEvent_t a, b;
        CHECK(hipEventCreate(&a));
        CHECK(hipEventCreate(&b));
        CHECK(hipEventRecord(a));
        scan_kernel<<<grid, threads>>>(big, part, nb, mode);
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, a, b));
        unsigned int acc = 0;
        CHECK(hipMemcpy(&acc, part, sizeof(acc), hipMemcpyDeviceToHost));
        double gbps = (bytes / 1e9) / (ms / 1e3);
        printf("cold %s  %.3f ms  %.1f GB/s  xor %u\n",
               mode ? "stream" : "plain ", ms, gbps, acc);
    }
    printf("probe_ok\n");
    return 0;
}
