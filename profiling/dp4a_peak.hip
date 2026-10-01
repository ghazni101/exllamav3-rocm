// ---------------------------------------------------------------------------------------------
// A1: V_DOT4 (dp4a) issue-throughput probe for gfx1100.
//
// Question: is the EXL3 int8 GEMV family limited by dp4a issue rate? The probe measures the
// ceiling three ways so the answer does not depend on which instruction mix the compiler ends up
// emitting:
//
// Throughput is reported as WARP instructions/s (what the scheduler issues), not per-lane
// operations: the GEMV issues one v_dot4 per warp per 4 trellis bytes, so the issue-rate question
// is a warp-level one. Lane-level rates are 32x the wave32 numbers.
//
//   MODE 0  pure dp4a, ILP independent accumulator chains          -> dp4a issue ceiling
//   MODE 1  one IMAD (x * 0x83DCD12D) per dp4a                     -> 32-bit codebook mix
//   MODE 2  per 8 dp4a: one LDS.128 + 8 IMAD (operands from the load, so nothing hoists)
//   MODE 3  pure v_fma_f32 chains   -> calibrates the issue ceiling itself (this part's measured
//   MODE 4  pure v_add_nc_u32 chains   issue rate is what the "% of issue" column is relative to)
//   MODE 5  16-bit window * 0x83DCD12D (v_mul_lo_u32) + dp4a       -> GEMV-true mix
//   MODE 6  16-bit window via 24-bit split + dp4a                  -> ISA §V_MUL_LO_U32 note
//   MODE 7  16-bit window * 0x83DCD12D only                        -> mul_lo throughput
//   MODE 8  16-bit window 24-bit split only                        -> 2x u24 + lshl_add
//
// The 24-bit split is bit-identical for w in [0, 65535]:
//   w * 0x83DCD12D == mul_u32_u24(w, 0xDCD12D) + (mul_u32_u24(w, 0x83) << 24)
// ISA: V_MUL_U32_U24 is "as efficient as basic single-precision opcodes"; V_MUL_LO_U32
// says to prefer it for small magnitudes. gfx1100 2026-09-23: C-level umul24
// recombines to v_mul_lo_u32; forced-asm MODE 6 is 43 G vs MODE 5 64 G at ILP 8.
// Three full-rate ops lose. Leave the GEMV hash as v_mul_lo_u32.
//
// ILP is the number of independent chains per thread (1/2/4/8); wave32 and wave64 are separate
// builds (`-mwavefrontsize64`; wave32 is the default for gfx11 and the flag is rejected there).
//
// Build (inside a ROCm container):
//   hipcc --offload-arch=gfx1100 -O3 -o dp4a_peak_w32 dp4a_peak.hip
//   hipcc --offload-arch=gfx1100 -O3 -mwavefrontsize64 -o dp4a_peak_w64 dp4a_peak.hip
// Check the instruction is present:
//   llvm-objdump -d --mcpu=gfx1100 dp4a_peak_w32 | grep -c v_dot
//
// Usage: ./dp4a_peak_w32 [iters] [reps] [grid]      (defaults 1024 20 384)
// ---------------------------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define THREADS 256
#define SHWORDS (THREADS * 32)

// Exactly the intrinsic the GEMV uses for dp4a.u32.s32 (exllamav3_ext/hip_compat_hip.cuh:136):
// __builtin_amdgcn_sudot4(neg_a, a, neg_b, b, c, clamp), here with an unsigned first operand.
__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c)
{
    return __builtin_amdgcn_sudot4(true, a, false, b, c, false);
}

// 16-bit w * 0x83DCD12D via 24-bit muls (ISA V_MUL_U32_U24 + V_LSHL_ADD_U32).
// HIP __umul24 / (hi<<24)+lo is bit-identical but LLVM recombines it back into
// v_mul_lo_u32 (the exactness kernel collapsed to s_endpgm). Force the opcodes.
__device__ __forceinline__ uint32_t hash24_mul1(uint32_t w)
{
    uint32_t lo, hi, out;
    asm volatile("v_mul_u32_u24 %0, %1, 0x00dcd12d" : "=v"(lo) : "v"(w));
    asm volatile("v_mul_u32_u24 %0, %1, 0x83" : "=v"(hi) : "v"(w));
    asm volatile("v_lshl_add_u32 %0, %1, 24, %2" : "=v"(out) : "v"(hi), "v"(lo));
    return out;
}

template <int ILP, int MODE>
__global__ __launch_bounds__(THREADS)
void k_dp4a(const uint32_t* __restrict__ src, int iters, int* __restrict__ sink)
{
    __shared__ uint32_t sh[SHWORDS];
    for (int i = threadIdx.x; i < SHWORDS; i += THREADS) sh[i] = src[i & 255] | 1u;
    __syncthreads();

    uint32_t a[ILP];
    int acc[ILP];
    float facc[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; ++i)
    {
        a[i] = src[(threadIdx.x + i) & 255] | 1u;
        acc[i] = 0;
        facc[i] = 0.5f;
    }
    uint32_t b = src[(threadIdx.x + 64) & 255] | 1u;
    const uint32_t mul = 0x83DCD12Du;
    int lds_acc = 0;
    uint32_t* sb = sh + (threadIdx.x & 127) * 4;
    float fa = (float) (b & 0xffff) * 1e-9f, fb = 1.0000001f;              // MODE 3 operands

    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        if (MODE == 2)
        {
            // One uint4 per 8 dp4a: the wide unit's uint4 splat fetch. Operands come from the load
            // (and the offset rotates), so neither the load nor the multiplications can hoist.
            uint4 v = *(const uint4*) (sb + ((it & 3) << 2));
            uint32_t w0 = v.x * mul, w1 = v.y * mul, w2 = v.z * mul, w3 = v.w * mul;
            #pragma unroll
            for (int i = 0; i < ILP; i += 4)
            {
                acc[i] = dp4a_us(MODE == 2 ? w0 : a[i], b, acc[i]);
                if (ILP > i + 1) acc[i + 1] = dp4a_us(MODE == 2 ? w1 : a[i], b, acc[i + 1]);
                if (ILP > i + 2) acc[i + 2] = dp4a_us(MODE == 2 ? w2 : a[i], b, acc[i + 2]);
                if (ILP > i + 3) acc[i + 3] = dp4a_us(MODE == 2 ? w3 : a[i], b, acc[i + 3]);
            }
            lds_acc += (int) (v.x ^ v.y ^ v.z ^ v.w);
        }
        else if (MODE == 3)
        {
            #pragma unroll
            for (int i = 0; i < ILP; ++i) facc[i] = fmaf(facc[i], fa, fb);
        }
        else if (MODE == 4)
        {
            #pragma unroll
            for (int i = 0; i < ILP; ++i) a[i] += b;      // v_add_nc_u32 chain
        }
        else if (MODE == 5 || MODE == 6)
        {
            #pragma unroll
            for (int i = 0; i < ILP; ++i)
            {
                uint32_t w = (a[i] + (uint32_t) it) & 0xffffu;
                uint32_t h = (MODE == 6) ? hash24_mul1(w) : (w * mul);
                acc[i] = dp4a_us(h, b, acc[i]);
            }
        }
        else if (MODE == 7 || MODE == 8)
        {
            #pragma unroll
            for (int i = 0; i < ILP; ++i)
            {
                uint32_t w = (a[i] + (uint32_t) it) & 0xffffu;
                a[i] = (MODE == 8) ? hash24_mul1(w) : (w * mul);
            }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < ILP; ++i)
            {
                if (MODE == 1) a[i] *= mul;
                acc[i] = dp4a_us(a[i], b, acc[i]);
            }
        }
    }

    int s = lds_acc;
    #pragma unroll
    for (int i = 0; i < ILP; ++i) { s += acc[i]; s ^= (int) facc[i] ^ (int) a[i]; }
    if (s == 0x7abcdef) sink[threadIdx.x] = s;
}

// Variant with the wide unit's cross-lane dependency: two chains, one fed by a shuffle
template <int ILP>
__global__ __launch_bounds__(THREADS)
void k_dp4a_shfl(const uint32_t* __restrict__ src, int iters, int* __restrict__ sink)
{
    uint32_t a0 = src[threadIdx.x & 255] | 1u, a1 = src[(threadIdx.x + 8) & 255] | 1u;
    uint32_t b = src[(threadIdx.x + 64) & 255] | 1u;
    int acc0 = 0, acc1 = 0;
    const uint32_t mul = 0x83DCD12Du;
    const int W = warpSize;
    int lane = threadIdx.x & (W - 1);
    int src_lane = (lane & (W / 2)) | ((lane + W / 2 - 1) & (W / 2 - 1));
    const unsigned long long mask = __builtin_amdgcn_read_exec();
    #pragma unroll 1
    for (int it = 0; it < iters; ++it)
    {
        uint32_t p = __shfl_sync(mask, a1, src_lane);
        #pragma unroll
        for (int i = 0; i < ILP; ++i)
        {
            a0 *= mul;
            acc0 = dp4a_us(a0, b, acc0);
            acc1 = dp4a_us(p, b, acc1);
        }
    }
    if (acc0 + acc1 == 0x7abcdef) sink[threadIdx.x] = acc0;
}

__global__ void k_hash24_exact(unsigned long long* mismatches)
{
    uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= 65536u) return;
    uint32_t ref = w * 0x83DCD12Du;
    uint32_t got = hash24_mul1(w);
    if (got != ref) atomicAdd(mismatches, 1ull);
}

template <int ILP, int MODE>
double run_kernel(const uint32_t* d_src, int* d_sink, int grid, int iters, int reps)
{
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    for (int i = 0; i < 3; ++i) k_dp4a<ILP, MODE><<<grid, THREADS>>>(d_src, iters, d_sink);
    hipDeviceSynchronize();
    hipEventRecord(e0);
    for (int i = 0; i < reps; ++i) k_dp4a<ILP, MODE><<<grid, THREADS>>>(d_src, iters, d_sink);
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    hipEventDestroy(e0); hipEventDestroy(e1);
    return ms;
}

template <int ILP>
double run_shfl(const uint32_t* d_src, int* d_sink, int grid, int iters, int reps)
{
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    for (int i = 0; i < 3; ++i) k_dp4a_shfl<ILP><<<grid, THREADS>>>(d_src, iters, d_sink);
    hipDeviceSynchronize();
    hipEventRecord(e0);
    for (int i = 0; i < reps; ++i) k_dp4a_shfl<ILP><<<grid, THREADS>>>(d_src, iters, d_sink);
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
    int ncu = prop.multiProcessorCount;
    double clk = prop.clockRate * 1e3;          // Hz
    printf("device: %s  CUs=%d  clock=%.3f GHz  warpSize=%d  grid=%d threads=%d iters=%d reps=%d\n",
           prop.name, ncu, clk / 1e9, prop.warpSize, grid, THREADS, iters, reps);

    std::vector<uint32_t> h(256);
    for (int i = 0; i < 256; ++i) h[i] = 0x01020304u * (uint32_t) (i + 1);
    uint32_t* d_src; int* d_sink;
    hipMalloc(&d_src, sizeof(uint32_t) * 256);
    hipMalloc(&d_sink, sizeof(int) * THREADS);
    hipMemcpy(d_src, h.data(), sizeof(uint32_t) * 256, hipMemcpyHostToDevice);

    unsigned long long host_mis = 0;
    for (uint32_t w = 0; w < 65536u; ++w)
    {
        uint32_t lo = w * 0x00DCD12Du;
        uint32_t hi = w * 0x83u;
        uint32_t got = (hi << 24) + lo;
        if (got != w * 0x83DCD12Du) ++host_mis;
    }
    printf("hash24 host exactness (0..65535): %llu mismatches\n", host_mis);

    unsigned long long* d_mis; unsigned long long h_mis = 0;
    hipMalloc(&d_mis, sizeof(unsigned long long));
    hipMemset(d_mis, 0, sizeof(unsigned long long));
    k_hash24_exact<<<256, 256>>>(d_mis);
    hipMemcpy(&h_mis, d_mis, sizeof(unsigned long long), hipMemcpyDeviceToHost);
    printf("hash24 exactness (0..65535 vs w*0x83DCD12D): %llu mismatches\n",
           (unsigned long long) h_mis);
    hipFree(d_mis);

    const double warps = (double) grid * THREADS / prop.warpSize;
    const double issue = ncu * 4.0 * clk;      // 4 SIMD/CU x clock = warp-instructions/s ceiling
    printf("%-5s %-4s %-10s %9s %12s %12s %10s\n",
           "mode", "ILP", "config", "ms", "G warp-dp4a/s", "G warp-inst/s", "% of issue");
    #define CASE(ILP, MODE)                                                                 \
    {                                                                                       \
        double ms = run_kernel<ILP, MODE>(d_src, d_sink, grid, iters, reps);                \
        double nd = warps * (double) ILP * iters * reps;                                     \
        double inst_per = (MODE == 0 ? 1.0 : (MODE == 1 ? 2.0 :                               \
                      (MODE == 2 ? 1.0 + 1.0 / ILP + 0.25 :                                    \
                      (MODE == 5 ? 3.0 : (MODE == 6 ? 5.0 :                                    \
                      (MODE == 7 ? 2.0 : (MODE == 8 ? 4.0 : 1.0)))))));                        \
        double ni = nd * inst_per;                                                             \
        const char* tag = (MODE == 7 || MODE == 8) ? "mul" : "dp4a";                           \
        printf("%-5d %-4d %-10s %9.3f %12.1f %12.1f %9.1f%%\n", MODE, ILP, tag, ms,          \
               nd / (ms * 1e6), ni / (ms * 1e6), 100.0 * ni / (ms * 1e-3) / issue);         \
    }
    CASE(1, 0); CASE(2, 0); CASE(4, 0); CASE(8, 0);
    CASE(1, 1); CASE(2, 1); CASE(4, 1); CASE(8, 1);
    CASE(4, 2); CASE(8, 2);
    CASE(2, 3); CASE(4, 3); CASE(8, 3); CASE(16, 3);
    CASE(4, 4); CASE(8, 4); CASE(16, 0); CASE(16, 1);
    CASE(1, 5); CASE(2, 5); CASE(4, 5); CASE(8, 5);
    CASE(1, 6); CASE(2, 6); CASE(4, 6); CASE(8, 6);
    CASE(4, 7); CASE(8, 7); CASE(4, 8); CASE(8, 8);
    #define CASES(ILP)                                                                      \
    {                                                                                       \
        double ms = run_shfl<ILP>(d_src, d_sink, grid, iters, reps);                        \
        double nd = warps * (double) ILP * iters * reps;                                    \
        printf("%-5s %-4d %-10s %9.3f %12.1f %12.1f %9.1f%%\n", "shfl", ILP, "dp4a", ms,   \
               nd / (ms * 1e6) / 2.0, nd / (ms * 1e6) / 2.0, 100.0 * nd / (ms * 1e-3) / issue); \
    }
    CASES(2); CASES(4); CASES(8);
    printf("issue ceiling (4 SIMD/CU x %d CU x %.3f GHz): %.1f G warp-inst/s  "
           "[wave-%d; multiply lane rates by %d]\n", ncu, clk / 1e9, issue / 1e9,
           prop.warpSize, prop.warpSize);

[Showing lines 1-300 of 303. Use :301 to continue]