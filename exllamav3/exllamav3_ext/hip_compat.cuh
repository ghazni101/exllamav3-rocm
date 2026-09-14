#pragma once

// HIP/ROCm compatibility shims for CUDA constructs that hipify does not translate.
// Included from ptx.cuh and util.cuh; also standalone from headers that use warp
// intrinsics. The qualifier fallbacks below let host-only TUs (.cpp under gcc)
// parse this header; under nvcc/hipcc the real definitions already exist.

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif
#if !defined(__device__)
#define __device__
#endif
#if !defined(__host__)
#define __host__
#endif
#if !defined(__forceinline__)
#define __forceinline__ inline
#endif
#if !defined(__global__)
#define __global__
#endif
#endif

// Tensor core fragment types shared with ptx.cuh. Defined here (outside the
// USE_ROCM guard) so this header can be included standalone from any file that
// needs the warp-intrinsic shims below.

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
    __device__ const T& operator[](int i) const { return elems[i]; }
};

#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

#define EXL3_FULL_WARP_MASK 0xffffffffffffffffULL

// Widen the mask argument of the *_sync warp builtins. The macro name is not
// re-expanded inside its own replacement list, so these forward to the real
// HIP builtins.

#define __shfl_sync(mask, ...)        __shfl_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_up_sync(mask, ...)     __shfl_up_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_down_sync(mask, ...)   __shfl_down_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_xor_sync(mask, ...)    __shfl_xor_sync((unsigned long long)(mask), __VA_ARGS__)
#define __ballot_sync(mask, ...)      __ballot_sync((unsigned long long)(mask), __VA_ARGS__)
#define __all_sync(mask, ...)         __all_sync((unsigned long long)(mask), __VA_ARGS__)
#define __any_sync(mask, ...)         __any_sync((unsigned long long)(mask), __VA_ARGS__)
#define EXL3_SYNCWARP_0()             __syncwarp()
#define EXL3_SYNCWARP_1(m)            __syncwarp((unsigned long long)(m))
#define EXL3_SYNCWARP_PICK(_0, _1, NAME, ...) NAME
#define __syncwarp(...)               EXL3_SYNCWARP_PICK(_, ##__VA_ARGS__, EXL3_SYNCWARP_1, EXL3_SYNCWARP_0)(__VA_ARGS__)
#define __reduce_add_sync(mask, ...)  __reduce_add_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_min_sync(mask, ...)  __reduce_min_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_max_sync(mask, ...)  __reduce_max_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_or_sync(mask, ...)   __reduce_or_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_and_sync(mask, ...)  __reduce_and_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_xor_sync(mask, ...)  __reduce_xor_sync((unsigned long long)(mask), __VA_ARGS__)

// __grid_constant__ is CUDA-only; a plain const kernel argument is the portable form.

#ifndef __grid_constant__
#define __grid_constant__
#endif

// hipify does not map the carveout enum; HIP takes a plain percentage (0-100).

#ifndef cudaSharedmemCarveoutMaxShared
#define cudaSharedmemCarveoutMaxShared 100
#endif

#if defined(__HIPCC__)

// HIP provides __hmax/__hmin for __half but not the __half2 forms.

__device__ __forceinline__ __half2 __hmax2(__half2 a, __half2 b)
{
    return __halves2half2(__hmax(__low2half(a), __low2half(b)),
                          __hmax(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ __half2 __hmin2(__half2 a, __half2 b)
{
    return __halves2half2(__hmin(__low2half(a), __low2half(b)),
                          __hmin(__high2half(a), __high2half(b)));
}

// hipify does not map the _rn/_rz-suffixed bf16 conversions; HIP's __float2bfloat16
// is already round-to-nearest, and truncation is the _rz form.

__device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(float f)
{
    return __float2bfloat16(f);
}

__device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rz(float f)
{
    return __ushort_as_bfloat16((unsigned short) (__float_as_uint(f) >> 16));
}

__device__ __forceinline__ void __nanosleep(uint32_t ns)
{
    // s_sleep takes a constant cycle count (max 127); ~1ns/cycle at ~1GHz.
    // Use 127-cycle chunks to minimize iteration count and oversleep ratio.
    int iters = (int)(ns / 127) + 1;
    while (iters-- > 0) __builtin_amdgcn_s_sleep(127);
}

// __dp4a: 4-way byte dot product. gfx11+ has v_sudot4 (dot8-insts) which handles
// all signed/unsigned combinations natively. Fall back to scalar expansion on
// older targets that lack dot8-insts.

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

__device__ __forceinline__ int __dp4a(int a, int b, int c)
{
    return __builtin_amdgcn_sudot4(false, a, false, b, c, false);
}

__device__ __forceinline__ int __dp4a(unsigned int a, int b, int c)
{
    return __builtin_amdgcn_sudot4(true, a, false, b, c, false);
}

__device__ __forceinline__ int __dp4a(int a, unsigned int b, int c)
{
    return __builtin_amdgcn_sudot4(false, a, true, b, c, false);
}

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c)
{
    return (unsigned int)__builtin_amdgcn_sudot4(true, a, true, b, (int)c, false);
}

#else

__device__ __forceinline__ int __dp4a(int a, int b, int c)
{
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += (int)(int8_t) ((a >> (8 * i)) & 0xff) * (int)(int8_t) ((b >> (8 * i)) & 0xff);
    return r;
}

__device__ __forceinline__ int __dp4a(unsigned int a, int b, int c)
{
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += (int) ((a >> (8 * i)) & 0xff) * (int)(int8_t) ((b >> (8 * i)) & 0xff);
    return r;
}

__device__ __forceinline__ int __dp4a(int a, unsigned int b, int c)
{
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += (int)(int8_t) ((a >> (8 * i)) & 0xff) * (int) ((b >> (8 * i)) & 0xff);
    return r;
}

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c)
{
    unsigned int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += ((a >> (8 * i)) & 0xff) * ((b >> (8 * i)) & 0xff);
    return r;
}

#endif

// L2-coherent loads (ld.global.cg). On AMD a plain load is L2-coherent; the hint only
// affects L1, so these degrade to normal loads.

__device__ __forceinline__ int __ldcg(const int* p) { return *p; }
__device__ __forceinline__ uint32_t __ldcg(const uint32_t* p) { return *p; }
__device__ __forceinline__ uint2 __ldcg(const uint2* p) { return *p; }
__device__ __forceinline__ uint4 __ldcg(const uint4* p) { return *p; }
__device__ __forceinline__ float4 __ldcg(const float4* p) { return *p; }

// Streaming/volatile global accesses. On AMD, __builtin_nontemporal_* generates
// loads/stores with .slc (streaming) cache hints. For 128-bit ops, use a single
// flat_load_dwordx4 / flat_store_dwordx4 via reinterpret_cast to a vector type
// instead of four separate 32-bit ops.

__device__ __forceinline__ uint32_t __ldcs(const uint32_t* p)
{
    return __builtin_nontemporal_load(p);
}

__device__ __forceinline__ uint64_t __ldcs(const uint64_t* p)
{
    return __builtin_nontemporal_load(p);
}

__device__ __forceinline__ void __stwt(uint32_t* p, uint32_t v)
{
    __builtin_nontemporal_store(v, p);
}

__device__ __forceinline__ void __stwt(uint64_t* p, uint64_t v)
{
    __builtin_nontemporal_store(v, p);
}

__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{
    __builtin_nontemporal_store(v, p);
}

__device__ __forceinline__ void stg_wt_u128(uint4* p, const uint4 v)
{
    // Single 128-bit nontemporal store via vector reinterpret.
    typedef int __attribute__((ext_vector_type(4))) int4_v;
    *reinterpret_cast<int4_v*>(p) = *reinterpret_cast<const int4_v*>(&v);
}

__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{
    return __builtin_nontemporal_load(p);
}

__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
    // Single 128-bit load. On AMD a plain 128-bit load is more efficient than
    // four 32-bit nontemporal loads and the cache hint matters less with the
    // unified L2 hierarchy.
    return *p;
}

// System-scope acquire/release for the TP collectives (ll.cuh, barrier_inner.cuh).
// GCN atomics are system-coherent; the ordering comes from the atomics themselves.

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{
    return __atomic_load_n((const uint32_t*) p, __ATOMIC_ACQUIRE);
}

__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{
    return __atomic_load_n((const uint64_t*) p, __ATOMIC_ACQUIRE);
}

__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

// Device-scope acquire/release used by the inter-block barriers in ptx.cuh.

__device__ __forceinline__ int ldg_acquire_gpu_i32(const int* p)
{
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

__device__ __forceinline__ void red_relaxed_gpu_add_i32(int* p, int v)
{
    __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

// Global real-time clock in nanoseconds (matching %globaltimer). CDNA has
// s_memrealtime (ns); RDNA has the 100 MHz steady counter instead.

__device__ __forceinline__ uint64_t globaltimer_ns()
{
#if __AMDGCN_WAVEFRONT_SIZE == 64
    return (uint64_t) __builtin_amdgcn_s_memrealtime();
#else
    return (uint64_t) __builtin_readsteadycounter() * 10;
#endif
}

// Bitfield extract / funnel shift used by the trellis decode paths.

__device__ __forceinline__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t mask = (length >= 64) ? ~0ULL : ((1ULL << length) - 1ULL);
    return static_cast<uint32_t>((value >> offset) & mask);
}

#define FSHF_IMM(dst, lo, hi, imm) dst = __funnelshift_r((lo), (hi), (imm))
#define BFE16_IMM(dst, src, imm) dst = ((src) >> (imm)) & 0xffff

// __fns: Find nth set bit in mask starting from position prev+1.
// CUDA-only intrinsic; emulate with ctz + bit-clear loop for ROCm.
// Returns the bit position (0-indexed) or -1 if not found.

__device__ __forceinline__ int __fns(unsigned int mask, int prev, int n)
{
    if (prev >= 0)
    {
        if (prev >= 31) return -1;
        mask &= ~((1u << (prev + 1)) - 1);  // Clear bits 0..prev
    }
    for (int i = 1; i < n && mask; ++i)
        mask &= mask - 1;  // Clear (n-1) lowest set bits
    return mask ? __builtin_ctz(mask) : -1;
}

// lop3.b32: 3-input bitwise LUT. The immediate selects the output bit for each
// combination of (a, b, c); with a compile-time imm this folds to a few logic ops.

__device__ __forceinline__ uint32_t lop3_u32(uint32_t a, uint32_t b, uint32_t c, uint32_t imm)
{
    uint32_t r = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        if (imm & (1u << i))
            r |= ((i & 4) ? a : ~a) & ((i & 2) ? b : ~b) & ((i & 1) ? c : ~c);
    return r;
}

// cp.async emulation: synchronous 16-byte copies. cp_async_fence/wait become
// no-ops; correctness relies on the __syncthreads() the kernels already issue
// before consuming staged data.
//
// True async global-to-LDS copies on gfx11 require inline asm (global_load_lds
// or buffer_load_lds with s_waitcnt vmcnt tracking). The __builtin_amdgcn_global_load_lds
// builtin is gated on vmem-to-lds-load-insts (gfx9/CDNA only). Future optimization:
// implement via inline asm when RDNA3 hardware is available for testing.
//
// The load and store are split into separate statements to give the compiler
// more scheduling freedom — the global load is non-blocking on AMD, so the
// compiler can overlap it with preceding computation.

__device__ __forceinline__ void cp_async_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    if (pred)
    {
        uint4 data = *((const uint4*) glob_ptr);
        *((uint4*) smem_ptr) = data;
    }
}

__device__ __forceinline__ void cp_async(void* smem_ptr, const void* glob_ptr)
{
    uint4 data = *((const uint4*) glob_ptr);
    *((uint4*) smem_ptr) = data;
}

__device__ __forceinline__ void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{
    uint4 data = *((const uint4*) glob_ptr);
    *((uint4*) smem_ptr) = data;
}

__device__ __forceinline__ void cp_async_fence() {}

template <int n>
__device__ __forceinline__ void cp_async_wait() {}

// ldmatrix.x4 emulation: lane l receives row l/4 of each of the four 8x8 b16
// matrices, 4 bytes at column (l%4)*2. The row base addresses are supplied by
// lanes 0..31 (lanes 8i+r hold matrix i row r) and fetched via shuffles.

__device__ __forceinline__ void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    int lane = threadIdx.x & 31;
    uintptr_t base = (uintptr_t) smem_ptr;
    int row = lane >> 2;
    int col = (lane & 3) * 4;
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // Lane 8*i + row holds the row pointer for matrix i
        uintptr_t row_ptr = (uintptr_t) __shfl_sync(
            EXL3_FULL_WARP_MASK, (unsigned long long) base, 8 * i + row);
        a[i] = *((const uint32_t*) (row_ptr + col));
    }
}

// mma.m16n8k16 emulation via shuffles + FMA. Fragment layouts follow the PTX doc
// (and the ldmatrix.x4 order TL, BL, TR, BR):
//   A (row-major 16x16): lane l = (g, t) = (l/4, l%4); a[j] is a half2 holding
//     A[g + 8*(j%2)][2t + 8*(j/2)] and A[g + 8*(j%2)][2t + 8*(j/2) + 1].
//   B (col-major 16x8):  b[i] is a half2 holding B[2t + 8*i][g] and B[2t + 8*i + 1][g].
//   C/D (16x8):          c[i] holds D[g + 8*(i/2)][2t + (i%2)].
// Each lane gathers its two A rows and two B columns (16 half2 shuffles total),
// then runs 16 FMAs per output element.

__device__ __forceinline__ half2 shfl_h2(half2 v, int src)
{
    uint32_t u = __shfl_sync(EXL3_FULL_WARP_MASK, *(uint32_t*) &v, src);
    return *(half2*) &u;
}

__device__ __forceinline__ void mma_m16n8k16_gather
(
    const FragA& frag_a,
    const FragB& frag_b,
    half2 (&a_rows)[2][8],
    half2 (&b_cols)[2][8]
)
{
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;

    // A row m lives in lanes 4*(m%8)..4*(m%8)+3. For row g: k pairs 0..7 come from
    // frag_a[0], k pairs 8..15 from frag_a[2]; for row g+8: frag_a[1] / frag_a[3].
    // Lane 4g+i holds k = 2i, 2i+1.
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        a_rows[0][i]     = shfl_h2(frag_a[0], 4 * g + i);
        a_rows[0][4 + i] = shfl_h2(frag_a[2], 4 * g + i);
        a_rows[1][i]     = shfl_h2(frag_a[1], 4 * g + i);
        a_rows[1][4 + i] = shfl_h2(frag_a[3], 4 * g + i);
    }

    // B column n lives in lanes 4n..4n+3: lane 4n+i holds k = 2i, 2i+1 in b[0]
    // and k = 2i+8, 2i+9 in b[1]. Columns 2t and 2t+1 are at lane bases 8t, 8t+4.
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        b_cols[0][i]     = shfl_h2(frag_b[0], 8 * t + i);
        b_cols[0][4 + i] = shfl_h2(frag_b[1], 8 * t + i);
        b_cols[1][i]     = shfl_h2(frag_b[0], 8 * t + 4 + i);
        b_cols[1][4 + i] = shfl_h2(frag_b[1], 8 * t + 4 + i);
    }
}

__device__ __forceinline__ void mma_m16n8k16_f32_emu
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC& frag_c
)
{
    half2 a_rows[2][8], b_cols[2][8];
    mma_m16n8k16_gather(frag_a, frag_b, a_rows, b_cols);

    float* c = reinterpret_cast<float*>(&frag_c);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        c[0] += __half2float(__low2half(a_rows[0][i]))  * __half2float(__low2half(b_cols[0][i]));
        c[0] += __half2float(__high2half(a_rows[0][i])) * __half2float(__high2half(b_cols[0][i]));
        c[1] += __half2float(__low2half(a_rows[0][i]))  * __half2float(__low2half(b_cols[1][i]));
        c[1] += __half2float(__high2half(a_rows[0][i])) * __half2float(__high2half(b_cols[1][i]));
        c[2] += __half2float(__low2half(a_rows[1][i]))  * __half2float(__low2half(b_cols[0][i]));
        c[2] += __half2float(__high2half(a_rows[1][i])) * __half2float(__high2half(b_cols[0][i]));
        c[3] += __half2float(__low2half(a_rows[1][i]))  * __half2float(__low2half(b_cols[1][i]));
        c[3] += __half2float(__high2half(a_rows[1][i])) * __half2float(__high2half(b_cols[1][i]));
    }
}

__device__ __forceinline__ void mma_m16n8k16_f16_emu
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC_h& frag_c
)
{
    half2 a_rows[2][8], b_cols[2][8];
    mma_m16n8k16_gather(frag_a, frag_b, a_rows, b_cols);

    // Accumulate each output element in a half2 (even/odd k in the two lanes),
    // then fold. c[0] = {D[g][2t], D[g][2t+1]}, c[1] = {D[g+8][2t], D[g+8][2t+1]}.
    half2* c = reinterpret_cast<half2*>(&frag_c);
    half2 acc00 = __float2half2_rn(0.0f), acc01 = acc00, acc10 = acc00, acc11 = acc00;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        acc00 = __hfma2(a_rows[0][i], b_cols[0][i], acc00);
        acc01 = __hfma2(a_rows[0][i], b_cols[1][i], acc01);
        acc10 = __hfma2(a_rows[1][i], b_cols[0][i], acc10);
        acc11 = __hfma2(a_rows[1][i], b_cols[1][i], acc11);
    }
    c[0] = __hadd2(c[0], __halves2half2(
        __hadd(__low2half(acc00), __high2half(acc00)),
        __hadd(__low2half(acc01), __high2half(acc01))));
    c[1] = __hadd2(c[1], __halves2half2(
        __hadd(__low2half(acc10), __high2half(acc10)),
        __hadd(__low2half(acc11), __high2half(acc11))));
}

// ============================================================================
// WMMA hardware acceleration for gfx11 (RDNA3, wave32)
//
// The gfx11 WMMA instruction performs a 16x16x16 fp16×fp16→fp32 matrix multiply
// in a single hardware instruction, replacing the ~150-250 scalar FMA ops in the
// emulated mma_m16n8k16. One WMMA covers a 16x16 output tile, replacing TWO
// 16x8 PTX MMA calls.
//
// WMMA register layout (verified against AMD matrix_calculator.py):
//   A: lane l holds A[l%16][0..15] (full row, 8 half2). Lanes l and l+16 identical.
//   B: lane l holds B[0..15][l%16] (full column, 8 half2). Lanes l and l+16 identical.
//   D: lane l holds D[2r + l/16][l%16] for r=0..7 (8 float).
//      Lanes 0-15: even rows (0,2,...,14). Lanes 16-31: odd rows (1,3,...,15).
//
// PTX layout (for reference):
//   A: lane l=(g,t)=(l/4,l%4); a[j] holds A[g+8*(j%2)][2t+8*(j/2)..+1]
//   B: lane l=(g,t)=(l/4,l%4); b[i] holds B[2t+8*i][g] and B[2t+8*i+1][g]
//   C: lane l=(g,t); c[i] holds D[g+8*(i/2)][2t+(i%2)]
// ============================================================================

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

using WmmaA16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaB16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaC32 = float __attribute__((ext_vector_type(8)));

// Convert PTX FragA to WMMA A layout.
// WMMA lane l needs A[l%16][0..15] (full row). 8 half2 shuffles, no replication.
// PTX lane (g,t) with g=l/4, t=l%4 holds:
//   a[0]=A[g][2t,2t+1], a[1]=A[g+8][2t,2t+1],
//   a[2]=A[g][2t+8,2t+9], a[3]=A[g+8][2t+8,2t+9]
// For WMMA row r, k=2j..2j+1:
//   j<4 (k<8):  source = a[r<8?0:1] from lane (r%8)*4 + j
//   j>=4 (k>=8): source = a[r<8?2:3] from lane (r%8)*4 + (j-4)
__device__ __forceinline__ WmmaA16 ptx_to_wmma_a(const FragA& frag_a)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;

    const half2* a = reinterpret_cast<const half2*>(&frag_a);
    WmmaA16 result;
    half2* r = reinterpret_cast<half2*>(&result);

    // elem = (row>=8?1:0) + (j>=4?2:0) → {0,1,2,3}
    // Switch ensures compile-time constant VGPR for __shfl_sync.
    #define SHFL_A(dst, src_lane, e) \
        switch (e) { \
            case 0: dst = shfl_h2(a[0], src_lane); break; \
            case 1: dst = shfl_h2(a[1], src_lane); break; \
            case 2: dst = shfl_h2(a[2], src_lane); break; \
            case 3: dst = shfl_h2(a[3], src_lane); break; \
        }

    int base = (row % 8) * 4;
    int e_lo = (row >= 8) ? 1 : 0;  // a[0] or a[1] for k<8
    int e_hi = (row >= 8) ? 3 : 2;  // a[2] or a[3] for k>=8

    SHFL_A(r[0], base + 0, e_lo)
    SHFL_A(r[1], base + 1, e_lo)
    SHFL_A(r[2], base + 2, e_lo)
    SHFL_A(r[3], base + 3, e_lo)
    SHFL_A(r[4], base + 0, e_hi)
    SHFL_A(r[5], base + 1, e_hi)
    SHFL_A(r[6], base + 2, e_hi)
    SHFL_A(r[7], base + 3, e_hi)
    #undef SHFL_A

    return result;
}

// Convert two adjacent PTX FragB (each 16x8, together 16x16) to WMMA B layout.
// WMMA lane l needs B[0..15][l%16] (full column). 16 half2 shuffles + 8 selects.
// PTX lane (g,t) with g=l/4, t=l%4 holds:
//   b[0]=B[2t][g], B[2t+1][g]; b[1]=B[2t+8][g], B[2t+9][g]
// For WMMA column c, k=2j..2j+1:
//   j<4 (k<8):  shuffle b[0] from lane (c%8)*4 + j → gives B[2j][c], B[2j+1][c]
//   j>=4 (k>=8): shuffle b[1] from lane (c%8)*4 + (j-4)
// If c<8, source from frag_b0; if c>=8, source from frag_b1.
// Must shuffle from BOTH frag_b0 and frag_b1 since the source lane's own col
// may differ from the calling lane's col (source lane's b0/b1 selection differs).
__device__ __forceinline__ WmmaB16 ptx_to_wmma_b(const FragB& frag_b0, const FragB& frag_b1)
{
    int lane = threadIdx.x & 31;
    int col = lane % 16;

    const half2* b0 = reinterpret_cast<const half2*>(&frag_b0);
    const half2* b1 = reinterpret_cast<const half2*>(&frag_b1);

    WmmaB16 result;
    half2* r = reinterpret_cast<half2*>(&result);

    // Switch ensures compile-time constant VGPR for __shfl_sync.
    #define SHFL_B0(dst, src, be) \
        switch (be) { \
            case 0: dst = shfl_h2(b0[0], src); break; \
            case 1: dst = shfl_h2(b0[1], src); break; \
        }
    #define SHFL_B1(dst, src, be) \
        switch (be) { \
            case 0: dst = shfl_h2(b1[0], src); break; \
            case 1: dst = shfl_h2(b1[1], src); break; \
        }

    int base = (col % 8) * 4;

    // j=0..3: k<8, b_elem=0. j=4..7: k>=8, b_elem=1.
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int be = j < 4 ? 0 : 1;
        int src = base + (j % 4);
        half2 v0, v1;
        SHFL_B0(v0, src, be)
        SHFL_B1(v1, src, be)
        r[j] = (col < 8) ? v0 : v1;
    }
    #undef SHFL_B0
    #undef SHFL_B1

    return result;
}

// Convert WMMA C (8 float, column-major) back to two PTX FragC (each 4 float, 16x8).
// WMMA D: lane l holds D[2r + l/16][l%16] for r=0..7.
//   Lanes 0-15: even rows (0,2,...,14). Lanes 16-31: odd rows (1,3,...,15).
// PTX C: lane (g,t); c[i] holds D[g+8*(i/2)][2t+(i%2)]
//   d0[0]=D[g][2t], d0[1]=D[g][2t+1], d0[2]=D[g+8][2t], d0[3]=D[g+8][2t+1]
//   d1[0]=D[g][2t+8], d1[1]=D[g][2t+9], d1[2]=D[g+8][2t+8], d1[3]=D[g+8][2t+9]
// D[i][j] is at WMMA lane (i%2)*16 + j, register i/2.
__device__ __forceinline__ void wmma_to_ptx_c(const WmmaC32& wmma_c, FragC& frag_c0, FragC& frag_c1)
{
    int lane = threadIdx.x & 31;
    int g = lane / 4;
    int t = lane % 4;
    const float* c = reinterpret_cast<const float*>(&wmma_c);
    float* d0 = reinterpret_cast<float*>(&frag_c0);
    float* d1 = reinterpret_cast<float*>(&frag_c1);

    // D[i][j] at WMMA lane (i%2)*16 + j, register i/2.
    // d0[0] = D[g][2t]:     WMMA lane (g%2)*16 + 2t,   reg g/2
    // d0[1] = D[g][2t+1]:   WMMA lane (g%2)*16 + 2t+1, reg g/2
    // d0[2] = D[g+8][2t]:   WMMA lane (g%2)*16 + 2t,   reg g/2 + 4
    // d0[3] = D[g+8][2t+1]: WMMA lane (g%2)*16 + 2t+1, reg g/2 + 4
    // d1[0..3] same with +8 lane offset (columns 8-15)

    int e0 = g / 2;       // register for row g
    int e1 = g / 2 + 4;   // register for row g+8
    int lane_base = (g % 2) * 16;

    // Switch ensures compile-time constant VGPR for __shfl_sync.
    #define WMMA_SHFL(dst, src, e) \
        switch (e) { \
            case 0: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[0], src); break; \
            case 1: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[1], src); break; \
            case 2: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[2], src); break; \
            case 3: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[3], src); break; \
            case 4: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[4], src); break; \
            case 5: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[5], src); break; \
            case 6: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[6], src); break; \
            case 7: dst = __shfl_sync(EXL3_FULL_WARP_MASK, c[7], src); break; \
        }

    // FragC0: columns 0-7
    WMMA_SHFL(d0[0], lane_base + 2 * t,     e0)
    WMMA_SHFL(d0[1], lane_base + 2 * t + 1, e0)
    WMMA_SHFL(d0[2], lane_base + 2 * t,     e1)
    WMMA_SHFL(d0[3], lane_base + 2 * t + 1, e1)

    // FragC1: columns 8-15
    WMMA_SHFL(d1[0], lane_base + 2 * t + 8, e0)
    WMMA_SHFL(d1[1], lane_base + 2 * t + 9, e0)
    WMMA_SHFL(d1[2], lane_base + 2 * t + 8, e1)
    WMMA_SHFL(d1[3], lane_base + 2 * t + 9, e1)

    #undef WMMA_SHFL
}

// WMMA 16x16x16 fp16→fp32: replaces two emulated mma_m16n8k16_f32 calls.
// Takes one FragA, two adjacent FragB, accumulates into two adjacent FragC.
// Uses zero-init C + post-add to avoid the PTX→WMMA C conversion shuffles.
__device__ __forceinline__ void wmma_m16n16k16_f32
(
    const FragA& frag_a,
    const FragB& frag_b0,
    const FragB& frag_b1,
    FragC& frag_c0,
    FragC& frag_c1
)
{
    WmmaA16 a = ptx_to_wmma_a(frag_a);
    WmmaB16 b = ptx_to_wmma_b(frag_b0, frag_b1);

    // Compute D = A*B with zero accumulator, then add to existing PTX accumulators.
    // This avoids 8 PTX→WMMA C shuffles at the cost of 8 float adds (net win).
    WmmaC32 c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);

    FragC add_c0 = {}, add_c1 = {};
    wmma_to_ptx_c(c, add_c0, add_c1);

    float* d0 = reinterpret_cast<float*>(&frag_c0);
    float* d1 = reinterpret_cast<float*>(&frag_c1);
    float* a0 = reinterpret_cast<float*>(&add_c0);
    float* a1 = reinterpret_cast<float*>(&add_c1);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        d0[i] += a0[i];
        d1[i] += a1[i];
    }
}

// WMMA 16x16x16 fp16→fp16: replaces two emulated mma_m16n8k16_f16 calls.
// Uses f32 WMMA internally for robustness, converts result to f16 for accumulation.
// The f16 WMMA builtin has complex output layout (8 valid at even/odd indices);
// f32 WMMA + conversion is simpler and more numerically robust at negligible cost.
__device__ __forceinline__ void wmma_m16n16k16_f16
(
    const FragA& frag_a,
    const FragB& frag_b0,
    const FragB& frag_b1,
    FragC_h& frag_c0,
    FragC_h& frag_c1
)
{
    WmmaA16 a = ptx_to_wmma_a(frag_a);
    WmmaB16 b = ptx_to_wmma_b(frag_b0, frag_b1);

    WmmaC32 c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);

    FragC add_c0 = {}, add_c1 = {};
    wmma_to_ptx_c(c, add_c0, add_c1);

    // Convert f32 results to f16 and add to existing f16 accumulators
    float* a0 = reinterpret_cast<float*>(&add_c0);
    float* a1 = reinterpret_cast<float*>(&add_c1);
    half2* d0 = reinterpret_cast<half2*>(&frag_c0);
    half2* d1 = reinterpret_cast<half2*>(&frag_c1);
    d0[0] = __hadd2(d0[0], __floats2half2_rn(a0[0], a0[1]));
    d0[1] = __hadd2(d0[1], __floats2half2_rn(a0[2], a0[3]));
    d1[0] = __hadd2(d1[0], __floats2half2_rn(a1[0], a1[1]));
    d1[1] = __hadd2(d1[1], __floats2half2_rn(a1[2], a1[3]));
}

#endif // gfx11 WMMA support

#endif  // __HIPCC__
#endif  // USE_ROCM
