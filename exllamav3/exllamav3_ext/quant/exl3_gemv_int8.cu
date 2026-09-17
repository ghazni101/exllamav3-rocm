#include <cuda_fp16.h>
#include "exl3_gemv_int8.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "comp_units/exl3_gemv_int8_instances.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"
#include "hadamard_inner.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <mutex>
#include <ATen/ATen.h>


// Mode 0: disabled; 1: int8 + error-feedback residual pass (~15-16 bit effective activation
// precision, KL at parity with fp16 or better); 2: plain int8 (cheaper, ~0.9% output RMS deviation).
static int _exl3_gemv_int8_mode = 0;
bool _exl3_gemv_int8_mode_chk = false;

static int exl3_gemv_int8_mode()
{
    if (_exl3_gemv_int8_mode_chk) return _exl3_gemv_int8_mode;
    const char* e = getenv("EXL3_INT8_GEMV");
    _exl3_gemv_int8_mode = e ? atoi(e) : 2;
    return _exl3_gemv_int8_mode;
}

bool exl3_gemv_int8_enabled()
{
    return exl3_gemv_int8_mode() != 0;
}

// Kill switch for the multi-matrix/sliced path only (EXL3_INT8_MSQ=0), for A/B verification
// against the cooperative mgemm kernel on identical inputs
bool exl3_gemv_int8_msq_enabled()
{
    static const int on = [] { const char* e = getenv("EXL3_INT8_MSQ"); return e ? atoi(e) : 1; }();
    return on != 0;
}

// Highest K the int8 path accepts; above it the regular kernel wins. The fp16 pipeline must be
// compute/latency-limited for the reduced per-weight work to matter, and where that ends is
// per-arch. Ampere is DRAM-bound from K = 6 up (3090: int8 -29/-22/-9/-6% at K=2/3/4/5, then
// -11..-26% on wide shapes at K=6); Ada is marginal at K=6 (4090: -0..+7%, residual mode loses)
// and keeps the conservative gate. Hopper's fp16 kernel is per-SM INT-throughput-bound at K = 6
// (H200, issue #242: +26/+57% per call, +16% e2e), and Blackwell measures the same way (5090:
// +7..+19% at K=6 across shapes, fp16 kernel at only ~65-78% of DRAM peak; K=7/8 are flat).
// EXL3_INT8_GEMV_MAX_K overrides the per-arch default for testing on unmeasured parts (kernel
// instances exist up to K = 8; at m == 1, K = 7..8 fall through to the cooperative kernel)
int exl3_gemv_int8_max_k(int device)
{
    static const int env_max_k = [] { const char* e = getenv("EXL3_INT8_GEMV_MAX_K"); return e ? atoi(e) : 0; }();
    if (env_max_k) return MIN(env_max_k, 8);
    int cc = DevCtx::instance().get_cc(device);
    return (cc == CC_HOPPER || cc == CC_BLACKWELL) ? 6 : 5;
}

struct GemvInt8Workspace
{
    int* ws = nullptr;
    size_t ws_ints = 0;
};

static GemvInt8Workspace gemv_ws[MAX_DEVICES];
static std::set<void*> gemv_attr_set[MAX_DEVICES];
static std::map<std::pair<void*, size_t>, int> gemv_occ_cache[MAX_DEVICES];

typedef void (*gemv_int8_coop_fn)
    (const half*, const uint16_t*, void*, int, int, int, int*, const half*, half*, const half*);

static void* select_gemv_int8_kernel(int K, bool c_fp32, bool residual)
{
    switch (K)
    {
        case 1: return exl3_gemv_int8_coop_sel_k1(c_fp32, residual);
        case 2: return exl3_gemv_int8_coop_sel_k2(c_fp32, residual);
        case 3: return exl3_gemv_int8_coop_sel_k3(c_fp32, residual);
        case 4: return exl3_gemv_int8_coop_sel_k4(c_fp32, residual);
        case 5: return exl3_gemv_int8_coop_sel_k5(c_fp32, residual);
        case 6: return exl3_gemv_int8_coop_sel_k6(c_fp32, residual);
        case 7: return exl3_gemv_int8_coop_sel_k7(c_fp32, residual);
        case 8: return exl3_gemv_int8_coop_sel_k8(c_fp32, residual);
    }
    return nullptr;
}

static void* select_gemv_int8_sq_kernel(int K, int M, bool c_fp32, bool residual)
{
    switch (K)
    {
        case 1: return exl3_gemv_int8_sq_sel_k1(M, c_fp32, residual);
        case 2: return exl3_gemv_int8_sq_sel_k2(M, c_fp32, residual);
        case 3: return exl3_gemv_int8_sq_sel_k3(M, c_fp32, residual);
        case 4: return exl3_gemv_int8_sq_sel_k4(M, c_fp32, residual);
        case 5: return exl3_gemv_int8_sq_sel_k5(M, c_fp32, residual);
        case 6: return exl3_gemv_int8_sq_sel_k6(M, c_fp32, residual);
    }
    return nullptr;
}

static void* select_gemv_int8_msq_kernel(int K, bool c_fp32, bool residual)
{
    switch (K)
    {
        case 1: return exl3_gemv_int8_msq_sel_k1(c_fp32, residual);
        case 2: return exl3_gemv_int8_msq_sel_k2(c_fp32, residual);
        case 3: return exl3_gemv_int8_msq_sel_k3(c_fp32, residual);
        case 4: return exl3_gemv_int8_msq_sel_k4(c_fp32, residual);
        case 5: return exl3_gemv_int8_msq_sel_k5(c_fp32, residual);
        case 6: return exl3_gemv_int8_msq_sel_k6(c_fp32, residual);
        case 7: return exl3_gemv_int8_msq_sel_k7(c_fp32, residual);
        case 8: return exl3_gemv_int8_msq_sel_k8(c_fp32, residual);
    }
    return nullptr;
}

static bool dbg_gemv_enabled()
{
    static const bool on = [] { const char* e = getenv("EXL3_DBG_GEMV"); return e && atoi(e); }();
    return on;
}

// Fixed-size per-device workspace shared by the sq and coop paths, allocated once and never
// reallocated: the pointer is baked as a kernel argument into captured CUDA graphs, so growing the
// buffer would leave every previously captured graph with a dangling workspace pointer (and let a
// reallocation clobber the self-resetting completion counters at the start of the buffer). Zeroed
// at allocation so the counters begin at zero. Callers must reject work that exceeds the fixed size
// (returns nullptr) and fall through to a non-workspace path.
#define GEMV_INT8_WS_INTS (WORKSPACE_SIZE / sizeof(int))    // 16 MB
static int* gemv_int8_get_ws(int device, size_t ws_ints)
{
    if (ws_ints > GEMV_INT8_WS_INTS) return nullptr;
    GemvInt8Workspace& ws = gemv_ws[device];
    if (!ws.ws)
    {
        cuda_check(cudaMalloc(&ws.ws, GEMV_INT8_WS_INTS * sizeof(int)));
        cuda_check(cudaMemset(ws.ws, 0, GEMV_INT8_WS_INTS * sizeof(int)));
        ws.ws_ints = GEMV_INT8_WS_INTS;
    }
    return ws.ws;
}

// m == 1 fast path: per-slice-scale kernel, regular launch. Returns false to fall through to the
// cooperative kernel (and from there to the regular fp16 kernel).
static bool exl3_gemv_int8_sq
(
    const half* A_ptr, const uint16_t* B_ptr, void* C_ptr,
    int size_m, int size_k, int size_n, int K, bool c_fp32, bool residual,
    const half* suh_ptr, half* A_had_ptr, const half* svh_ptr,
    int device, int num_sms, cudaStream_t stream, Graph* graph
)
{
    if (size_m > 4) return false;
    int M = size_m > 2 ? 4 : size_m;
    void* fn = select_gemv_int8_sq_kernel(K, M, c_fp32, residual);
    if (!fn)
    {
        if (dbg_gemv_enabled())
            fprintf(stderr, "[exl3_sq] no kernel: m=%d k=%d n=%d K=%d c_fp32=%d res=%d\n",
                    size_m, size_k, size_n, K, (int) c_fp32, (int) residual);
        return false;
    }

    int rows_max = gemv_int8_sq_rows_max(M, residual);

    // EXL3_SQ_ROWS_PER pins the slice height (multiple of 8, >= SQ_MINROWS). On RDNA3 the
    // single-wave rule's rows_per = rows_max starves occupancy on wide matrices (lm_head):
    // swept on gfx1101, 64 beats auto by ~19% decode e2e (32/48/96/128/256 all slower).
    static const int rows_per_env = []
    {
        const char* e = getenv("EXL3_SQ_ROWS_PER");
        return e ? atoi(e) : 0;
    }();
#if defined(USE_ROCM)
    int rows_per_arg = MAX(((rows_per_env > 0 ? rows_per_env : 64) + 7) & ~7, SQ_MINROWS);
#else
    int rows_per_arg = rows_per_env > 0 ? MAX((rows_per_env + 7) & ~7, SQ_MINROWS) : 0;
#endif

    // Mirror of the kernel's work decomposition (single-wave rule with a half-wave floor)
    auto decomp = [&] (int grid_, int& ksplit, int& rows_per)
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int r = CEIL_DIVIDE(rows_total * nb256, grid_);
        rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
        rows_per = MAX(rows_per, SQ_MINROWS);
        rows_per = MIN(rows_per, rows_max);
        rows_per = MIN(rows_per, (rows_total + 7) & ~7);
        if (rows_per_arg > 0)
            rows_per = MIN(rows_per_arg, MIN(rows_max, (rows_total + 7) & ~7));
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
    };
    auto smem_for = [&] (int rows_per) -> size_t
    {
        size_t stage = gemv_int8_stage_smem(K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * M * (residual ? 2 : 1)
               + stage + (size_t) 2 * M * 128 * 4;
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max));
#if !defined(USE_ROCM)
        // Match the tensor-core kernels' shared-memory carveout: these kernels interleave with
        // them (hundreds of launches per decoded token), and a smaller carveout would make the GPU
        // drain and reconfigure the SMs on every transition - measured at ~4 us per launch in
        // graph replay. No configurable LDS carveout exists on RDNA; the attribute returns
        // hipErrorInvalidValue there.
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
#endif
        gemv_attr_set[device].insert(fn);
        cuda_check(cudaPeekAtLastError());
    }

    int ksplit, rows_per;
    decomp(6 * num_sms, ksplit, rows_per);
    size_t smem_guess = smem_for(rows_per);
    int maxb;
    auto occ_key = std::make_pair(fn, smem_guess);
    auto occ_it = gemv_occ_cache[device].find(occ_key);
    if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
    else
    {
        maxb = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess);
        gemv_occ_cache[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    decomp(grid, ksplit, rows_per);
    size_t smem = smem_for(rows_per);
    if (ksplit > SQ_KSPLIT_CAP)
    {
        if (dbg_gemv_enabled())
            fprintf(stderr, "[exl3_sq] ksplit cap: ksplit=%d k=%d n=%d K=%d\n", ksplit, size_k, size_n, K);
        return false;
    }
    if (size_n / 256 > SQ_COUNTERS_CAP)
    {
        if (dbg_gemv_enabled())
            fprintf(stderr, "[exl3_sq] counters cap: n=%d\n", size_n);
        return false;
    }

    int pstride = size_n * (residual ? 2 : 1);
    int* ws_ptr = gemv_int8_get_ws(device, SQ_WS_RESERVED + (size_t) ksplit * M * pstride);
    if (!ws_ptr)
    {
        if (dbg_gemv_enabled())
            fprintf(stderr, "[exl3_sq] ws: k=%d n=%d K=%d\n", size_k, size_n, K);
        return false;
    }

    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr,
        (void*) &rows_per_arg
    };

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), kernelArgs, smem, stream);
    if (err != cudaSuccess)
    {
        // Nothing was captured: the caller's fallback kernel records its own parameter sites
        cudaGetLastError();
        return false;
    }
    if (graph)
    {
        graph->record_param(fn, GP_gemm_A, 0);
        graph->record_param(fn, GP_gemm_B_trellis, 1);
        graph->record_param(fn, GP_gemm_C, 2);
        graph->record_param(fn, GP_gemm_B_suh, 7);
        graph->record_param(fn, GP_gemm_A_had, 8);
        graph->record_param(fn, GP_gemm_B_svh, 9);
        graph->record_param(fn, GP_end, 0);
    }
    return true;
}

// m == 1 multi-matrix/sliced fast path: per-slice-scale kernel covering a whole mgemm call in one
// regular launch (see exl3_gemv_int8_msq_kernel). Takes the mgemm entry's cooked pointer arguments;
// the kernel signature matches exl3_mgemm_kernel so graph parameter recording is identical.
// Returns false to fall through to the cooperative mgemm kernel. Unlike the single-matrix gate,
// every K is accepted: the alternative here is the cooperative mgemm kernel, which loses to this
// path even at K = 7-8.
bool exl3_gemv_int8_msq
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    int size_m,
    int size_k,
    int size_n,                     // max slice/matrix width
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list_ptr,
    void** c_list_ptr,
    const int* n_stride_list_ptr,
    const int* had_src_list_ptr,
    int num_had_src,
    int K,
    bool c_fp32,
    int device,
    int num_sms,
    cudaStream_t stream,
    Graph* graph
)
{
    if (size_m < 1 || bszm_out < 1) return false;
    bool residual = exl3_gemv_int8_mode() == 1;
    void* fn = select_gemv_int8_msq_kernel(K, c_fp32, residual);
    if (!fn)
    {
        if (dbg_gemv_enabled())
            fprintf(stderr, "[exl3_msq] no kernel: m=%d k=%d n=%d K=%d c_fp32=%d\n",
                    size_m, size_k, size_n, K, (int) c_fp32);
        return false;
    }

    int rows_total = size_k / 16;
    int nb256_max = CEIL_DIVIDE(size_n, 256);
    int rows_max = gemv_int8_sq_rows_max(1, residual);

    // Mirror of the kernel's work decomposition (sq's single-wave rule over the max width);
    // EXL3_SQ_ROWS_PER pins the slice height (default 64 on RDNA3 — swept on gfx1101)
    static const int rows_per_env = []
    {
        const char* e = getenv("EXL3_SQ_ROWS_PER");
        return e ? atoi(e) : 0;
    }();
#if defined(USE_ROCM)
    int rows_per_arg = MAX(((rows_per_env > 0 ? rows_per_env : 64) + 7) & ~7, SQ_MINROWS);
#else
    int rows_per_arg = rows_per_env > 0 ? MAX((rows_per_env + 7) & ~7, SQ_MINROWS) : 0;
#endif
    auto decomp = [&] (int grid_, int& ksplit, int& rows_per)
    {
        int r = CEIL_DIVIDE(rows_total * nb256_max, grid_);
        rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
        rows_per = MAX(rows_per, SQ_MINROWS);
        rows_per = MIN(rows_per, rows_max);
        rows_per = MIN(rows_per, (rows_total + 7) & ~7);
        if (rows_per_arg > 0)
            rows_per = MIN(rows_per_arg, MIN(rows_max, (rows_total + 7) & ~7));
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
    };
    auto smem_for = [&] (int rows_per) -> size_t
    {
        size_t stage = gemv_int8_stage_smem(K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * (residual ? 2 : 1)
               + stage + (size_t) 2 * 128 * 4;
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max));
#if !defined(USE_ROCM)
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
#endif
        gemv_attr_set[device].insert(fn);
        cuda_check(cudaPeekAtLastError());
    }

    int ksplit, rows_per;
    decomp(6 * num_sms, ksplit, rows_per);
    size_t smem_guess = smem_for(rows_per);
    int maxb;
    auto occ_key = std::make_pair(fn, smem_guess);
    auto occ_it = gemv_occ_cache[device].find(occ_key);
    if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
    else
    {
        maxb = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess);
        gemv_occ_cache[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    decomp(grid, ksplit, rows_per);
    size_t smem = smem_for(rows_per);

    // Workspace: counters share the sq prefix [0..SQ_COUNTERS_CAP); qsums/partials live beyond
    // SQ_WS_RESERVED like the coop kernels. If it doesn't fit, grow rows_per (fewer slices ->
    // less workspace) up to rows_max, then pin the kernel to the same slice height.
    int num_jj = bszm_out * size_m;
    if (num_jj * nb256_max > SQ_COUNTERS_CAP)
    {
        if (dbg_gemv_enabled())
            fprintf(stderr, "[exl3_msq] counters cap: jj=%d nb256=%d m=%d n=%d\n",
                    num_jj, nb256_max, size_m, size_n);
        return false;
    }
    int pstride = nb256_max * 256 * (residual ? 2 : 1);
    int* ws_ptr = nullptr;
    while (true)
    {
        size_t ws_ints = SQ_WS_RESERVED
                       + (size_t) num_jj * ksplit * 4
                       + (size_t) num_jj * ksplit * pstride;
        ws_ptr = gemv_int8_get_ws(device, ws_ints);
        if (ws_ptr) break;
        // ksplit == 1 is the workspace floor (growing rows_per past rows_total can't shrink
        // the allocation further); without this guard a wide-n matrix (lm_head) with a narrow
        // k spins here forever
        if (rows_per >= rows_max || ksplit <= 1)
        {
            if (dbg_gemv_enabled())
                fprintf(stderr, "[exl3_msq] ws: jj=%d ksplit=%d rows_per=%d\n", num_jj, ksplit, rows_per);
            return false;
        }
        rows_per = MIN(rows_per * 2, MIN(rows_max, (rows_total + 7) & ~7));
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
        smem = smem_for(rows_per);
    }
    rows_per_arg = rows_per;    // pin the kernel to the (possibly grown) slice height
    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr_ptr,
        (void*) &indices_ptr,
        (void*) &weights_ptr,
        (void*) &bszm_in,
        (void*) &bszm_out,
        (void*) &min_index,
        (void*) &max_index,
        (void*) &num_tokens,
        (void*) &size_n_list_ptr,
        (void*) &c_list_ptr,
        (void*) &n_stride_list_ptr,
        (void*) &had_src_list_ptr,
        (void*) &num_had_src,
        (void*) &rows_per_arg
    };

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), kernelArgs, smem, stream);
    if (err != cudaSuccess)
    {
        // Nothing was captured: the caller's fallback kernel records its own parameter sites
        cudaGetLastError();
        return false;
    }
    if (graph)
    {
        graph->record_param(fn, GP_mgemm_A, 0);
        graph->record_param(fn, GP_mgemm_C, 2);
        graph->record_param(fn, GP_mgemm_indices, 10);
        graph->record_param(fn, GP_mgemm_weights, 11);
        graph->record_param(fn, GP_end, 0);
    }
    return true;
}

// Single-matrix m > sq-cap routing through the msq kernel (a one-entry bundle). The alternative
// for these calls is the grid-starved cooperative gemm kernel — batch > 4 decode steps and the
// sub-threshold prefill shapes — so the workspace/counters fit checks in the launcher are the
// only gate that matters. The launcher dereferences the pointer arrays on device; B/suh/svh are
// invariant for a loaded module, so the three-entry array is built once per matrix and cached.
// Eager only: graphed callers (bsz == 1 decode, m == 1) never reach this branch.
static bool exl3_gemv_int8_msq_single
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int num_sms,
    cudaStream_t stream
)
{
    static const bool disabled = []
    {
        // Default off: routed shapes measured at parity with the cooperative kernel (K = 1
        // msq ~820 us vs coop ~1000 us per call at m = 8) and m >= 13 batches regressed at a
        // wide cap, so the perf gate keeps it opt-in until the n % 256 single-matrix gate gap
        // (the actual batch-decode coop bulk) is closed. Verified correct for m up to 144:
        // test_msq_single_diag.py, all K, <= 0.9% rel_rms vs the cooperative reference.
        const char* e = getenv("EXL3_INT8_MSQ_SINGLE");
        return !(e && atoi(e));
    }();
    static const int max_m = []
    {
        const char* e = getenv("EXL3_INT8_MSQ_SINGLE_MAX_M");
        return e ? atoi(e) : 144;
    }();
    if (disabled || size_m > max_m) return false;

    // The kernel writes one hadamard-transformed input slab of m * k halves
    if ((int64_t) A_had.numel() < (int64_t) size_m * size_k) return false;

    static std::mutex mtx;
    static std::map<uintptr_t, at::Tensor> args_cache[MAX_DEVICES];

    at::Tensor args;
    uintptr_t B_ptr = (uintptr_t) B.data_ptr();
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto& cache = args_cache[device];
        auto it = cache.find(B_ptr);
        if (it == cache.end())
        {
            int64_t host[3] =
            {
                (int64_t) B_ptr,
                (int64_t) (uintptr_t) suh.data_ptr(),
                (int64_t) (uintptr_t) svh.data_ptr()
            };
            args = at::empty({3}, B.options().dtype(at::kLong));
            args.copy_(at::from_blob(host, {3}, at::kLong), /*non_blocking*/ false);
            it = cache.emplace(B_ptr, args).first;
        }
        args = it->second;
    }

    uintptr_t* base = (uintptr_t*) args.data_ptr();
    return exl3_gemv_int8_msq(
        (const half*) A.data_ptr(),
        (const uintptr_t*) (base + 0),
        C.data_ptr(),
        size_m, size_k, size_n,
        (const uintptr_t*) (base + 1),
        (half*) A_had.data_ptr(),
        (const uintptr_t*) (base + 2),
        nullptr,               // indices
        nullptr,               // weights
        1,                     // bszm_in
        1,                     // bszm_out
        -1, -1,                // min/max index
        1,                     // num_tokens
        nullptr,               // size_n_list
        nullptr,               // c_list
        nullptr,               // n_stride_list
        nullptr,               // had_src_list
        0,                     // num_had_src
        K, c_fp32, device, num_sms, stream, nullptr);
}

bool exl3_gemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    cudaStream_t stream,
    Graph* graph
)
{
    if (!suh.has_value() || !A_had.has_value() || !svh.has_value()) return false;

    // Diagnostics for the single-matrix fast path: every fall-through to the cooperative
    // kernel is grid-starved on RDNA, so EXL3_DBG_GEMV=1 logs why each call missed the
    // sq/coop-int8 paths (zero cost when unset)
    static const bool dbg_gemv = dbg_gemv_enabled();
    auto dbg = [&] (const char* why)
    {
        if (dbg_gemv)
            fprintf(stderr, "[exl3_gemv_int8] miss (%s): m=%d k=%d n=%d K=%d c_fp32=%d\n",
                    why, (int) (A.numel() / A.size(-1)), A.size(-1), B.size(1) * 16, B.size(2) / 16,
                    (int) (C.dtype() == at::kFloat));
    };

    int K = B.size(2) / 16;
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int size_m = A.numel() / size_k;
    if (size_n % 256) { dbg("n %% 256"); return false; }
    if (size_k % 128) { dbg("k %% 128"); return false; }

    int device;
    cudaGetDevice(&device);
    if (K < 1 || K > exl3_gemv_int8_max_k(device)) { dbg("K range"); return false; }
    int num_sms = DevCtx::instance().get_num_sms(device);
    bool c_fp32 = C.dtype() == at::kFloat;
    bool residual = exl3_gemv_int8_mode() == 1;

    // Per-slice-scale kernel: m <= 4 in plain int8 mode (rows share the decoded weights and the B
    // stream). Falls through to the cooperative kernel on a constraint miss; batched rows beyond
    // the gate try the msq kernel as a one-entry bundle (the cooperative gemm alternative is
    // grid-starved on RDNA at every m), then decline.
    if (size_m <= (residual ? 1 : 4) && exl3_gemv_int8_sq(
        (const half*) A.data_ptr(), (const uint16_t*) B.data_ptr(), C.data_ptr(),
        size_m, size_k, size_n, K, c_fp32, residual,
        (const half*) suh->data_ptr(), (half*) A_had->data_ptr(), (const half*) svh->data_ptr(),
        device, num_sms, stream, graph))
        return true;
    if (size_m > 1)
    {
        if (!graph && exl3_gemv_int8_msq_single(
            A, B, C, suh.value(), A_had.value(), svh.value(),
            size_m, size_k, size_n, K, c_fp32, device, num_sms, stream))
            return true;
        dbg("m>1 after sq");
        return false;
    }

    void* fn = select_gemv_int8_kernel(K, c_fp32, residual);
    if (!fn) { dbg("no coop int8 kernel"); return false; }

    // Mirror the kernel's work decomposition for the shared memory size; grid = max co-resident
    // blocks (natural register allocation measures faster than forcing higher occupancy)
    auto smem_for_grid = [&] (int grid_) -> size_t
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int smem_rows_max = residual ? 384 : 768;
        int ksplit = CEIL_DIVIDE(4 * grid_, nb256);
        ksplit = MAX(ksplit, CEIL_DIVIDE(rows_total, smem_rows_max));
        ksplit = MIN(ksplit, rows_total);
        int rows_per = CEIL_DIVIDE(rows_total, ksplit);
        size_t stage = gemv_int8_stage_smem(K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return MAX((size_t) rows_per * 16 * 4 * (residual ? 2 : 1) + stage, (size_t) 8 * 128 * 4);
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        // Upper bound over all shapes: smem_rows_max * 64 B
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 768 * 16 * 4 + GEMV_STAGE_MAX_BYTES);
#if !defined(USE_ROCM)
        // Match the tensor-core kernels' shared-memory carveout: these kernels interleave with
        // them (hundreds of launches per decoded token), and a smaller carveout would make the GPU
        // drain and reconfigure the SMs on every transition - measured at ~4 us per launch in
        // graph replay. No configurable LDS carveout exists on RDNA; the attribute returns
        // hipErrorInvalidValue there.
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
#endif
        gemv_attr_set[device].insert(fn);
        cuda_check(cudaPeekAtLastError());
    }

    size_t smem_guess = smem_for_grid(6 * num_sms);
    int maxb;
    auto occ_key = std::make_pair(fn, smem_guess);
    auto occ_it = gemv_occ_cache[device].find(occ_key);
    if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
    else
    {
        maxb = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess);
#if defined(USE_ROCM)
        // See exl3_gemv.cu: occupancy overestimates co-residency on RDNA; one fewer per SM.
        if (maxb > 1) maxb -= 1;
#endif
        gemv_occ_cache[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    size_t smem = smem_for_grid(grid);

    // Coop region beyond the sq-reserved prefix: [2n accs][4m qsums][grid partial maxes]
    size_t ws_ints = SQ_WS_RESERVED + (size_t) 2 * size_n + 4 * size_m + 1024;
    int* ws_base = gemv_int8_get_ws(device, ws_ints);
    if (!ws_base) { dbg("ws"); return false; }
    int* ws_ptr = ws_base + SQ_WS_RESERVED;

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = C.data_ptr();
    const half* suh_ptr = (const half*) suh->data_ptr();
    half* A_had_ptr = (half*) A_had->data_ptr();   // scratch; used through a raw half* like the regular kernel
    const half* svh_ptr = (const half*) svh->data_ptr();

    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
            graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
            graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
            graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    cudaError_t err = cudaLaunchCooperativeKernel(fn, grid, NUM_THREADS, kernelArgs, smem, stream);
    if (err != cudaSuccess)
    {
        // e.g. cooperative launch unsupported or co-residency violated: fall back to the regular kernel
        // (which records its own graph parameter sites)
        if (dbg_gemv_enabled())
            fprintf(stderr, "[exl3_coopint8] launch failed: %s k=%d n=%d K=%d\n",
                    cudaGetErrorString(err), size_k, size_n, K);
        cudaGetLastError();
        return false;
    }
    add_graph_args((void*) fn);
    return true;
}
