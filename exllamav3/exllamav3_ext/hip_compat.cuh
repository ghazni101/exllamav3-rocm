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
// The WMMA register layout differs from PTX mma.m16n8k16:
//   WMMA A (K-major, 16x16): lane l holds A[l%16][(l/16)*8 .. (l/16)*8+7]
//   WMMA B (K-major, 16x16): lane l holds B[l%16][(l/16)*8 .. (l/16)*8+7]
//   WMMA C (M-major, 16x16): lane l holds D[l%16][(l/16)*8 .. (l/16)*8+7]
//
// On gfx11, A and B require replication: 8 unique half values duplicated to 16.
//
// PTX layout (for reference):
//   A: lane l=(g,t)=(l/4,l%4); a[j] holds A[g+8*(j%2)][2t+8*(j/2)..+1]
//   B: lane l; b[i] holds B[2t+8*i][g] and B[2t+8*i+1][g]
//   C: lane l; c[i] holds D[g+8*(i/2)][2t+(i%2)]
//
// The conversion uses register shuffles — cheap compared to the FMA ops replaced.
// ============================================================================

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)

using WmmaA16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaB16 = _Float16 __attribute__((ext_vector_type(16)));
using WmmaC32 = float __attribute__((ext_vector_type(8)));

// Convert PTX FragA (8 half, 16x16 K=16 tile) to WMMA A layout.
// 4 half2 shuffles + 4 half2 copies for replication.
__device__ __forceinline__ WmmaA16 ptx_to_wmma_a(const FragA& frag_a)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    int col_group = lane / 16;
    int g = row % 8;
    int elem = col_group * 2 + (row >= 8 ? 1 : 0);

    const half2* a = reinterpret_cast<const half2*>(&frag_a);
    WmmaA16 result;
    half2* r = reinterpret_cast<half2*>(&result);

    // Must shuffle a specific VGPR (a[0..3]) from the source lane, not the source
    // lane's a[elem] which may differ. Switch ensures compile-time constant VGPR.
    #define SHFL_A(dst, src, e) \
        switch (e) { \
            case 0: dst = shfl_h2(a[0], src); break; \
            case 1: dst = shfl_h2(a[1], src); break; \
            case 2: dst = shfl_h2(a[2], src); break; \
            case 3: dst = shfl_h2(a[3], src); break; \
        }

    SHFL_A(r[0], g * 4 + 0, elem)
    SHFL_A(r[1], g * 4 + 1, elem)
    SHFL_A(r[2], g * 4 + 2, elem)
    SHFL_A(r[3], g * 4 + 3, elem)
    #undef SHFL_A

    // Replicate for gfx11 (2x for wave32)
    r[4] = r[0]; r[5] = r[1]; r[6] = r[2]; r[7] = r[3];
    return result;
}

// Convert two adjacent PTX FragB (each 16x8, together 16x16) to WMMA B layout.
// PTX B is N-major (each lane holds 1 column, 4 rows); WMMA B is K-major (each lane
// holds 1 row, 8 columns). This transpose requires 16 shuffles + 16 selects.
// Still far cheaper than the 128-256 scalar FMA ops in the emulated MMA.
__device__ __forceinline__ WmmaB16 ptx_to_wmma_b(const FragB& frag_b0, const FragB& frag_b1)
{
    int lane = threadIdx.x & 31;
    int row = lane % 16;       // k index
    int col_group = lane / 16;  // 0 → cols 0-7 (frag_b0), 1 → cols 8-15 (frag_b1)

    const half2* b0 = reinterpret_cast<const half2*>(&frag_b0);
    const half2* b1 = reinterpret_cast<const half2*>(&frag_b1);

    int k = row;
    int half_k = (k < 8) ? (k / 2) : ((k - 8) / 2);
    int b_elem = (k < 8) ? 0 : 1;
    bool take_low = (k % 2 == 0);

    WmmaB16 result;
    half* r = reinterpret_cast<half*>(&result);

    // Must shuffle specific VGPRs (b[0] or b[1]) from the source lane, not the
    // source lane's b[b_elem] which may differ. Switch ensures correct VGPR.
    // Also must shuffle from both frag_b0 and frag_b1 since source lanes span
    // both col_groups.
    #define SHFL_B(dst, bptr, src, be) \
        switch (be) { \
            case 0: dst = shfl_h2(bptr[0], src); break; \
            case 1: dst = shfl_h2(bptr[1], src); break; \
        }

    #pragma unroll
    for (int n = 0; n < 8; ++n)
    {
        half2 v0, v1;
        SHFL_B(v0, b0, n * 4 + half_k, b_elem)
        SHFL_B(v1, b1, n * 4 + half_k, b_elem)
        half2 v = (col_group == 0) ? v0 : v1;
        r[n] = take_low ? __low2half(v) : __high2half(v);
    }
    #undef SHFL_B

    // Replicate for gfx11 (2x for wave32)
    #pragma unroll
    for (int n = 0; n < 8; ++n)
        r[8 + n] = r[n];
    return result;
}


// Convert WMMA C (8 float, 16x16 M-major) back to two PTX FragC (each 4 float, 16x8).
// 8 float shuffles total (4 per FragC). Uses switch to ensure compile-time constant
// VGPR selection for __shfl_sync — runtime indexing would read the wrong VGPR.
__device__ __forceinline__ void wmma_to_ptx_c(const WmmaC32& wmma_c, FragC& frag_c0, FragC& frag_c1)
{
    int lane = threadIdx.x & 31;
    int g = lane / 4;
    int t = lane % 4;
    const float* c = reinterpret_cast<const float*>(&wmma_c);
    float* d0 = reinterpret_cast<float*>(&frag_c0);
    float* d1 = reinterpret_cast<float*>(&frag_c1);

    // PTX C1 (cols 0-7): d0 = {D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1]}
    // WMMA C: lane l holds D[l%16][(l/16)*8 + 0..7]
    //   col_group 0 (l<16): D[l][0..7]    → source lanes g and g+8
    //   col_group 1 (l≥16): D[l-16][8..15] → source lanes g+16 and g+24

    int e0 = 2 * t;     // even column index within the 8-element row
    int e1 = 2 * t + 1; // odd column index

    // Shuffle float at element e0/e1 from the appropriate WMMA lane
    // Switch ensures the compiler selects the correct VGPR for __shfl_sync
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

    // FragC0: columns 0-7 (WMMA col_group 0, lanes 0-15)
    WMMA_SHFL(d0[0], g, e0)
    WMMA_SHFL(d0[1], g, e1)
    WMMA_SHFL(d0[2], g + 8, e0)
    WMMA_SHFL(d0[3], g + 8, e1)

    // FragC1: columns 8-15 (WMMA col_group 1, lanes 16-31)
    WMMA_SHFL(d1[0], g + 16, e0)
    WMMA_SHFL(d1[1], g + 16, e1)
    WMMA_SHFL(d1[2], g + 24, e0)
    WMMA_SHFL(d1[3], g + 24, e1)

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
