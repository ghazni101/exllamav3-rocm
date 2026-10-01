// DPP16 ROW_XMASK vs __shfl_xor_sync: exactness + throughput on gfx1100.
//
// ISA §7.7.1 / §15.3.8: dpp_ctrl 0x160+mask, mask in {1,2,4,8}:
//   lane n reads lane (n & ~0xf) | ((n & 0xf) ^ mask)
// which is __shfl_xor_sync(v, mask) within each 16-lane group (xor-16 cannot).
//
//   hipcc --offload-arch=gfx1100 -O3 -o dpp_xmask_probe dpp_xmask_probe.hip
//   ./dpp_xmask_probe
//
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define THREADS 256

__device__ __forceinline__ uint32_t dpp_xmask(uint32_t v, int mask)
{
    switch (mask)
    {
        case 1: return __builtin_amdgcn_update_dpp(v, v, 0x161, 0xf, 0xf, false);
        case 2: return __builtin_amdgcn_update_dpp(v, v, 0x162, 0xf, 0xf, false);
        case 4: return __builtin_amdgcn_update_dpp(v, v, 0x164, 0xf, 0xf, false);
        case 8: return __builtin_amdgcn_update_dpp(v, v, 0x168, 0xf, 0xf, false);
        default: return v;
    }
}

__global__ void k_exact(const uint32_t* src, unsigned long long* mis, int mask)
{
    uint32_t v = src[threadIdx.x + blockIdx.x * blockDim.x];
    uint32_t a = __shfl_xor_sync(0xffffffffffffffffull, v, mask);
    uint32_t b = dpp_xmask(v, mask);
    if (a != b) atomicAdd(mis, 1ull);
}

// MODE 0: shfl_xor 1 + add   (what the wide-unit tail emits today)
// MODE 1: dpp xmask 1 + add
// MODE 2: shfl_xor 1 + add, interleaved with ds_load_b128 (inner-loop contention)
// MODE 3: dpp xmask 1 + add, interleaved with ds_load_b128
template <int MODE>
__global__ __launch_bounds__(THREADS)
void k_thru(const uint32_t* src, int iters, int* sink)
{
    __shared__ uint32_t sh[THREADS * 4];
    for (int i = threadIdx.x; i < THREADS * 4; i += THREADS)
        sh[i] = src[i & 255] | 1u;
    __syncthreads();

    uint32_t a0 = src[threadIdx.x & 255] | 1u;
    uint32_t a1 = src[(threadIdx.x + 8) & 255] | 1u;
    uint32_t a2 = src[(threadIdx.x + 16) & 255] | 1u;
    uint32_t a3 = src[(threadIdx.x + 24) & 255] | 1u;
    uint32_t lds_acc = 0;
    uint32_t* sb = sh + (threadIdx.x & 127) * 2;

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        if (MODE >= 2)
        {
            uint4 v = *(const uint4*) (sb + ((it & 3) << 2));
            lds_acc += v.x ^ v.y ^ v.z ^ v.w;
        }
        if (MODE == 0 || MODE == 2)
        {
            a0 += __shfl_xor_sync(0xffffffffffffffffull, a0, 1);
            a1 += __shfl_xor_sync(0xffffffffffffffffull, a1, 1);
            a2 += __shfl_xor_sync(0xffffffffffffffffull, a2, 1);
            a3 += __shfl_xor_sync(0xffffffffffffffffull, a3, 1);
        }
        else
        {
            a0 += dpp_xmask(a0, 1);
            a1 += dpp_xmask(a1, 1);
            a2 += dpp_xmask(a2, 1);
            a3 += dpp_xmask(a3, 1);
        }
    }
    int s = (int) (a0 ^ a1 ^ a2 ^ a3 ^ lds_acc);
    if (s == 0x7abcdef) sink[threadIdx.x] = s;
}

template <int MODE>
double run(const uint32_t* d_src, int* d_sink, int grid, int iters, int reps)
{
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    for (int i = 0; i < 3; ++i) k_thru<MODE><<<grid, THREADS>>>(d_src, iters, d_sink);
    hipDeviceSynchronize();
    hipEventRecord(e0);
    for (int i = 0; i < reps; ++i) k_thru<MODE><<<grid, THREADS>>>(d_src, iters, d_sink);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return ms;
}

int main()
{
    const int grid = 384, iters = 1024, reps = 20;
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("device: %s  CUs=%d  clock=%.3f GHz  warpSize=%d\n",
           prop.name, prop.multiProcessorCount, prop.clockRate * 1e-6, prop.warpSize);

    const int n = THREADS * 32;
    std::vector<uint32_t> h(n);
    for (int i = 0; i < n; ++i) h[i] = 0x9E3779B9u * (uint32_t) (i + 1) + 7u;
    uint32_t* d_src; int* d_sink;
    hipMalloc(&d_src, sizeof(uint32_t) * n);
    hipMalloc(&d_sink, sizeof(int) * THREADS);
    hipMemcpy(d_src, h.data(), sizeof(uint32_t) * n, hipMemcpyHostToDevice);

    unsigned long long* d_mis; unsigned long long h_mis;
    hipMalloc(&d_mis, sizeof(unsigned long long));
    int masks[4] = {1, 2, 4, 8};
    for (int mi = 0; mi < 4; ++mi)
    {
        int m = masks[mi];
        hipMemset(d_mis, 0, sizeof(unsigned long long));
        k_exact<<<32, THREADS>>>(d_src, d_mis, m);
        hipMemcpy(&h_mis, d_mis, sizeof(unsigned long long), hipMemcpyDeviceToHost);
        printf("exact xor-%d vs shfl_xor: %llu mismatches\n", m, h_mis);
    }

    const double warps = (double) grid * THREADS / prop.warpSize;
    const double nops = warps * 4.0 * iters * reps;
    printf("%-8s %9s %12s\n", "mode", "ms", "G xor-add/s");
    double ms0 = run<0>(d_src, d_sink, grid, iters, reps);
    double ms1 = run<1>(d_src, d_sink, grid, iters, reps);
    double ms2 = run<2>(d_src, d_sink, grid, iters, reps);
    double ms3 = run<3>(d_src, d_sink, grid, iters, reps);
    printf("%-8s %9.3f %12.1f\n", "shfl",     ms0, nops / (ms0 * 1e6));
    printf("%-8s %9.3f %12.1f\n", "dpp",      ms1, nops / (ms1 * 1e6));
    printf("%-8s %9.3f %12.1f\n", "shfl+lds", ms2, nops / (ms2 * 1e6));
    printf("%-8s %9.3f %12.1f\n", "dpp+lds",  ms3, nops / (ms3 * 1e6));
    hipFree(d_src); hipFree(d_sink); hipFree(d_mis);
    return 0;
}
