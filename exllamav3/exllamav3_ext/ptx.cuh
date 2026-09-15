#pragma once

#include "hip_compat.cuh"

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda/atomic>
#endif

// Tensor core fragments (Vec<> is defined in hip_compat.cuh)

#if !defined(USE_ROCM)
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;
#endif
#if !defined(USE_ROCM)

// m8n8k4 tensor core matmul (emulated on Ampere and later), don't use
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k4-with-f16-floating-point-type

__device__ inline void ptx_mma_m8n8k4
(
    const Vec<half2, 2>& frag_a,
    const Vec<half2, 2>& frag_b,
    Vec<float, 8>& frag_c
)
{
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3]),"=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])

        :  "r"(a[0]), "r"(a[1]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]), "f"(d[6]), "f"(d[7])
    );
}

#endif  // !USE_ROCM

// m16n8k16 tensor core matmul
//
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type

// FP16 @ FP16 + FP32 -> FP32
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC& frag_c
)
{
#if defined(USE_ROCM)
    mma_m16n8k16_f32_emu(frag_a, frag_b, frag_c);
#else
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    float* c = reinterpret_cast<float*>(&frag_c);
    const float* d = reinterpret_cast<const float*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"

        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3])
    );
#endif
}

// FP16 @ FP16 + FP16 -> FP16
__device__ inline void ptx_mma_m16n8k16
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC_h& frag_c
)
{
#if defined(USE_ROCM)
    mma_m16n8k16_f16_emu(frag_a, frag_b, frag_c);
#else
    const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
    const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
    uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
    const uint32_t* d = reinterpret_cast<const uint32_t*>(&frag_c);

    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"

        : "=r"(c[0]), "=r"(c[1])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
           "r"(b[0]), "r"(b[1]),
           "r"(d[0]), "r"(d[1])
    );
#endif
}

// Paired 16x16 MMA: combines two adjacent 16x8 tiles into one call.
// On gfx11 (RDNA3), uses hardware WMMA (single instruction, 16x16x16) when built
// with -DEXL3_WMMA (EXL3_WMMA=1 at build time). Default off: the WMMA path has a
// known NaN bug on the M>=3 GEMM shapes (see rocm-optimize history); the emulated
// mma_m16n8k16 path is the verified fallback.

// FP16 @ FP16 + FP32 -> FP32 (paired)
__device__ inline void ptx_mma_m16n16k16
(
    const FragA& frag_a,
    const FragB& frag_b0,
    const FragB& frag_b1,
    FragC& frag_c0,
    FragC& frag_c1
)
{
#if defined(USE_ROCM) && defined(EXL3_WMMA) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__))
    wmma_m16n16k16_f32(frag_a, frag_b0, frag_b1, frag_c0, frag_c1);
#else
    ptx_mma_m16n8k16(frag_a, frag_b0, frag_c0);
    ptx_mma_m16n8k16(frag_a, frag_b1, frag_c1);
#endif
}

// FP16 @ FP16 + FP16 -> FP16 (paired)
__device__ inline void ptx_mma_m16n16k16
(
    const FragA& frag_a,
    const FragB& frag_b0,
    const FragB& frag_b1,
    FragC_h& frag_c0,
    FragC_h& frag_c1
)
{
#if defined(USE_ROCM) && defined(EXL3_WMMA) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__))
    wmma_m16n16k16_f16(frag_a, frag_b0, frag_b1, frag_c0, frag_c1);
#else
    ptx_mma_m16n8k16(frag_a, frag_b0, frag_c0);
    ptx_mma_m16n8k16(frag_a, frag_b1, frag_c1);
#endif
}

// Global barrier
__device__ inline void barrier_acquire
(
    int* lock,
    int stage
)
{
    if (threadIdx.x == 0)
    {
#if defined(USE_ROCM)
        int state = -1;
        do
        {
            state = ldg_acquire_gpu_i32(lock);
        }
        while (state != stage);
#else
        volatile int state = -1;
        do
        {
            asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        }
        while (state != stage);
#endif
    }
    __syncthreads();
}

__device__ inline void barrier_release
(
    int* lock,
    int val,
    bool reset
)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (reset)
        {
#if defined(USE_ROCM)
            __atomic_store_n(lock, 0, __ATOMIC_RELEASE);
#else
            asm volatile ("fence.acq_rel.gpu;\n");
            *lock = 0;
#endif
            return;
        }
#if defined(USE_ROCM)
        // Release-ordered atomic ensures all prior writes (output data) are
        // visible before the lock value advances. Replaces the old __threadfence()
        // + relaxed-add pair, which could reorder the store after the atomic.
        __atomic_fetch_add(lock, val, __ATOMIC_RELEASE);
#else
        asm volatile ("fence.acq_rel.gpu;\n");
        asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
#endif
    }
}

#if !defined(USE_ROCM)

// Load global to shared memory, predicated. Seems to produce incorrect code when compiling for Blackwell, but
// `if (...) cp_async(...)` compiles to a predicated instruction anyway

__device__ inline void cp_async_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

// Load global to shared memory

__device__ inline void cp_async(void* smem_ptr, const void* glob_ptr)
{
    const int bytes = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

// Load global to shared memory with cache hint to evict data from L2 ASAP

__device__ inline void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = 16;
    asm volatile
    (
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(bytes)
    );
}

// Async copy fence, commit all pending async copies

__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most n async groups are still pending.

template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// Load 16x16 matrix fragment from shared memory, directly in tensor core layout

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile
    (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}

__device__ inline uint32_t mul_lo_u32(uint32_t x, uint32_t y)
{
    uint32_t w;
    asm volatile
    (
        "mul.lo.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
}

__device__ inline uint32_t mul_hi_u32(uint32_t x, uint32_t y)
{
    uint32_t w;
    asm volatile
    (
        "mul.hi.u32 %0, %1, %2;"
        : "=r"(w)
        :  "r"(x), "r"(y)
    );
    return w;
}

// Memory ops

__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{
    asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(p), "r"(v));
}

__device__ __forceinline__ void stg_wt_u128(uint4* p, const uint4 v)
{
    asm volatile ("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                  :: "l"(p),
                     "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
    uint4 v;
    asm volatile ("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                  : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];"
                 : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{
    uint64_t v;
    asm volatile("ld.global.acquire.sys.u64 %0, [%1];"
                 : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{
    asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{
    asm volatile("st.global.release.sys.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Global time in nanoseconds

__device__ __forceinline__ uint64_t globaltimer_ns()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Bitfield stuff

static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t result64;
    asm ("bfe.u64 %0, %1, %2, %3;"
         : "=l"(result64)
         : "l"(value), "r"(offset), "r"(length));
    return static_cast<uint32_t>(result64);
}

#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

#else  // USE_ROCM

__device__ inline uint32_t mul_lo_u32(uint32_t x, uint32_t y)
{
    return x * y;
}

__device__ inline uint32_t mul_hi_u32(uint32_t x, uint32_t y)
{
    return __umulhi(x, y);
}

#endif  // !USE_ROCM

// Inter-block barrier

__device__ inline void group_barrier
(
    int group_id,
    int group_size,
    int* barrier_counters_sense  // length 2*max(group_id). odd positions are flipped after sync (sense)
)
{
    __syncthreads();

    if (threadIdx.x == 0)
    {
#if defined(USE_ROCM)
        int* counter_p = &barrier_counters_sense[group_id * 2];
        int* sense_p = &barrier_counters_sense[group_id * 2 + 1];

        int old_sense = __hip_atomic_load(sense_p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        int old = __hip_atomic_fetch_add(counter_p, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);

        if (old == group_size - 1)
        {
            __hip_atomic_store(counter_p, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(sense_p, 1 - old_sense, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        else
        {
            while (__hip_atomic_load(sense_p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == old_sense) __nanosleep(32);
        }
#else
        cuda::atomic_ref<int, cuda::thread_scope_device> counter(barrier_counters_sense[group_id * 2]);
        cuda::atomic_ref<int, cuda::thread_scope_device> sense(barrier_counters_sense[group_id * 2 + 1]);

        int old_sense = sense.load(cuda::memory_order_relaxed);
        int old = counter.fetch_add(1, cuda::memory_order_acq_rel);

        if (old == group_size - 1)
        {
            counter.store(0, cuda::memory_order_relaxed);
            sense.store(1 - old_sense, cuda::memory_order_release);
        }
        else
        {
            while (sense.load(cuda::memory_order_acquire) == old_sense) __nanosleep(32);
        }
#endif
    }

    __syncthreads();
}
