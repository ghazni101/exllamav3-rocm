// 2x256 LDS LUT vs v_mul_lo_u32 for the mul1 codebook hash.
//
// 16-bit trellis window w:
//   w * 0x83DCD12D == T0[w & 255] + T1[w >> 8]
// with T0[i] = i * C, T1[i] = (i << 8) * C. Bit-identical on 0..65535.
//
// Why this is not the 24-bit split (closed, MODE 6, -27%): that replaced one
// VALU mul with three VALU ops. This moves two of the three onto the LDS pipe
// (ds_load_b32 x2 + v_add), which can dual-issue against v_dot4.
//
//   hipcc --offload-arch=gfx1100 -O3 -mcumode -o hash_lut_probe hash_lut_probe.hip
//   ./hash_lut_probe [iters] [reps] [grid]
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define THREADS 256
#define T0 0
#define T1 256

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c)
{
    return __builtin_amdgcn_sudot4(true, a, false, b, c, false);
}

__device__ __forceinline__ uint32_t hash_mul(uint32_t w)
{
    return w * 0x83DCD12Du;
}

__device__ __forceinline__ uint32_t hash_lut(const uint32_t* tbl, uint32_t w)
{
    return tbl[T0 + (w & 255u)] + tbl[T1 + (w >> 8)];
}

__global__ void k_exact(unsigned long long* mismatches)
{
    __shared__ uint32_t tbl[512];
    if (threadIdx.x < 256)
    {
        tbl[T0 + threadIdx.x] = (uint32_t) threadIdx.x * 0x83DCD12Du;
        tbl[T1 + threadIdx.x] = ((uint32_t) threadIdx.x << 8) * 0x83DCD12Du;
    }
    __syncthreads();
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= 65536u) return;
    uint32_t ref = hash_mul(w);
    uint32_t got = hash_lut(tbl, w);
    if (got != ref) atomicAdd(mismatches, 1ull);
}

// MODE 0 mul+dp4a, 1 lut+dp4a, 2 mul only, 3 lut only
template <int ILP, int MODE>
__global__ __launch_bounds__(THREADS)
void k_mix(const uint32_t* __restrict__ src, int iters, int* __restrict__ sink)
{
    __shared__ uint32_t tbl[512];
    if (threadIdx.x < 256)
    {
        tbl[T0 + threadIdx.x] = (uint32_t) threadIdx.x * 0x83DCD12Du;
        tbl[T1 + threadIdx.x] = ((uint32_t) threadIdx.x << 8) * 0x83DCD12Du;
    }
    __syncthreads();

    uint32_t a[ILP];
    int acc[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; ++i)
    {
        a[i] = src[(threadIdx.x + i) & 255] | 1u;
        acc[i] = 0;
    }
    uint32_t b = src[(threadIdx.x + 64) & 255] | 1u;

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        #pragma unroll
        for (int i = 0; i < ILP; ++i)
        {
            uint32_t w = (a[i] + (uint32_t) it) & 0xffffu;
            uint32_t h = (MODE & 1) ? hash_lut(tbl, w) : hash_mul(w);
            if (MODE < 2) acc[i] = dp4a_us(h, b, acc[i]);
            else a[i] = h;
        }
    }

    int s = 0;
    #pragma unroll
    for (int i = 0; i < ILP; ++i) s += acc[i] ^ (int) a[i];
    if (s == 0x7abcdef) sink[threadIdx.x] = s;
}

template <int ILP, int MODE>
double run_kernel(const uint32_t* d_src, int* d_sink, int grid, int iters, int reps)
{
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    for (int i = 0; i < 3; ++i) k_mix<ILP, MODE><<<grid, THREADS>>>(d_src, iters, d_sink);
    hipDeviceSynchronize();
    hipEventRecord(e0);
    for (int i = 0; i < reps; ++i) k_mix<ILP, MODE><<<grid, THREADS>>>(d_src, iters, d_sink);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return ms;
}

int main(int argc, char** argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 1024;
    int reps  = argc > 2 ? atoi(argv[2]) : 20;
    int grid  = argc > 3 ? atoi(argv[3]) : 384;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("device: %s  CUs=%d  clock=%.3f GHz  warpSize=%d  grid=%d threads=%d iters=%d reps=%d\n",
           prop.name, prop.multiProcessorCount, prop.clockRate * 1e-6, prop.warpSize,
           grid, THREADS, iters, reps);

    unsigned long long host_mis = 0;
    uint32_t t0[256], t1[256];
    for (uint32_t i = 0; i < 256; ++i)
    {
        t0[i] = i * 0x83DCD12Du;
        t1[i] = (i << 8) * 0x83DCD12Du;
    }
    for (uint32_t w = 0; w < 65536u; ++w)
        if (t0[w & 255] + t1[w >> 8] != w * 0x83DCD12Du) ++host_mis;
    printf("lut host exactness (0..65535): %llu mismatches\n", host_mis);

    unsigned long long* d_mis; unsigned long long h_mis = 0;
    hipMalloc(&d_mis, sizeof(unsigned long long));
    hipMemset(d_mis, 0, sizeof(unsigned long long));
    k_exact<<<256, 256>>>(d_mis);
    hipMemcpy(&h_mis, d_mis, sizeof(unsigned long long), hipMemcpyDeviceToHost);
    printf("lut gpu exactness (0..65535 vs w*0x83DCD12D): %llu mismatches\n",
           (unsigned long long) h_mis);
    hipFree(d_mis);

    std::vector<uint32_t> h(256);
    for (int i = 0; i < 256; ++i) h[i] = 0x01020304u * (uint32_t) (i + 1);
    uint32_t* d_src; int* d_sink;
    hipMalloc(&d_src, sizeof(uint32_t) * 256);
    hipMalloc(&d_sink, sizeof(int) * THREADS);
    hipMemcpy(d_src, h.data(), sizeof(uint32_t) * 256, hipMemcpyHostToDevice);

    const double warps = (double) grid * THREADS / prop.warpSize;
    printf("%-8s %-4s %9s %14s\n", "mode", "ILP", "ms", "G warp-hash/s");
    #define CASE(ILP, MODE, NAME)                                                          \
    {                                                                                      \
        double ms = run_kernel<ILP, MODE>(d_src, d_sink, grid, iters, reps);               \
        double nd = warps * (double) ILP * iters * reps;                                   \
        printf("%-8s %-4d %9.3f %14.1f\n", NAME, ILP, ms, nd / (ms * 1e6));                \
    }
    CASE(1, 0, "mul+dot"); CASE(2, 0, "mul+dot"); CASE(4, 0, "mul+dot"); CASE(8, 0, "mul+dot"); CASE(16, 0, "mul+dot");
    CASE(1, 1, "lut+dot"); CASE(2, 1, "lut+dot"); CASE(4, 1, "lut+dot"); CASE(8, 1, "lut+dot"); CASE(16, 1, "lut+dot");
    CASE(4, 2, "mul");     CASE(8, 2, "mul");     CASE(16, 2, "mul");
    CASE(4, 3, "lut");     CASE(8, 3, "lut");     CASE(16, 3, "lut");

    hipFree(d_src);
    hipFree(d_sink);
    return h_mis == 0 && host_mis == 0 ? 0 : 2;
}
