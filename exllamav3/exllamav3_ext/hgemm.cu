#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, size_k,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS
    if (hgemm_f16acc_try(a, w, c)) return;

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

#if defined(USE_ROCM)
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <mutex>

// rocBLAS (via cublasGemmEx) picks a poor kernel for the fp16-in/fp32-out GEMMs of the
// reconstruct (prefill) path on RDNA3: 11.8 ms average for the 2k-token prefill's
// projections vs 3.4 ms through the hipBLASLt backend torch.mm routes to. torch can't
// express that dtype combination (mm requires equal dtypes), so call hipBLASLt here,
// with the heuristic-selected algo cached per shape.
//
// Row-major a[M,K] fp16 @ b[K,N] fp16 -> c[M,N] fp32 is computed as the column-major
// equivalent D[N,M] = B_cm[N,K] x A_cm[K,M] (both OP_N): a row-major matrix reads as its
// own col-major transpose, so the layout interpretation absorbs the transposes.

namespace lt_f32 {
    struct Key
    {
        int m, n, k;
        bool operator<(const Key& o) const
        {
            return m != o.m ? m < o.m : (n != o.n ? n < o.n : k < o.k);
        }
    };
    static hipblasLtHandle_t handle = nullptr;
    static void* ws = nullptr;
    constexpr size_t WS_SIZE = 128u * 1024 * 1024;
    static std::map<Key, hipblasLtMatmulAlgo_t> algos;
    static int sweeps = 0;
    static std::mutex mtx;
}

bool hgemm_lt_f32out(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream)
{
    const int m = a.size(0);
    const int k = a.size(1);
    const int n = b.size(1);
    if ((n | k) % 8) return false;

    std::lock_guard<std::mutex> lock(lt_f32::mtx);
    if (!lt_f32::handle)
    {
        if (hipblasLtCreate(&lt_f32::handle) != HIPBLAS_STATUS_SUCCESS) return false;
        if (cudaMalloc(&lt_f32::ws, lt_f32::WS_SIZE) != cudaSuccess)
        {
            lt_f32::ws = nullptr;
        }
    }

    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    auto cleanup = [&]()
    {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (la) hipblasLtMatrixLayoutDestroy(la);
        if (lb) hipblasLtMatrixLayoutDestroy(lb);
        if (ld) hipblasLtMatrixLayoutDestroy(ld);
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    };

    hipblasOperation_t nop = HIPBLAS_OP_N;
    bool ok = true;
    ok &= hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS;
    ok &= hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &nop, sizeof(nop)) == HIPBLAS_STATUS_SUCCESS;
    ok &= hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &nop, sizeof(nop)) == HIPBLAS_STATUS_SUCCESS;
    ok &= hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, n, k, n) == HIPBLAS_STATUS_SUCCESS;   // B buffer as cm [n,k]
    ok &= hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, k, m, k) == HIPBLAS_STATUS_SUCCESS;   // A buffer as cm [k,m]
    ok &= hipblasLtMatrixLayoutCreate(&ld, HIP_R_32F, n, m, n) == HIPBLAS_STATUS_SUCCESS;   // C as cm [n,m]
    if (!ok)
    {
        cleanup();
        return false;
    }

    // Algo: timed sweep over the heuristic's candidates on first encounter per shape, then
    // cached. The gfx1101 heuristic's top prediction is frequently several times slower
    // than another candidate for the same shape, so predictions alone can't be trusted;
    // one timed pass costs a few hundred ms per shape and amortizes over the run.
    // The sweep only pays off at chunk-aligned (large) m, where prompt chunking produces a
    // handful of recurring shapes; small-m and over-budget cases take the heuristic's top
    // pick so arbitrary prompt remainders can't trigger unbounded sweeps.
    lt_f32::Key key{m, n, k};
    float alpha = 1.0f, beta = 0.0f;
    auto it = lt_f32::algos.find(key);
    if (it == lt_f32::algos.end())
    {
        hipblasLtMatmulPreferenceCreate(&pref);
        size_t ws_size = lt_f32::WS_SIZE;
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));
        hipblasLtMatmulHeuristicResult_t results[8];
        int found = 0;
        hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(
            lt_f32::handle, op, la, lb, ld, ld, pref, 8, results, &found);
        if (hs != HIPBLAS_STATUS_SUCCESS || found == 0)
        {
            cleanup();
            return false;
        }
        bool sweep = m >= 256 && lt_f32::sweeps < 32;
        if (sweep) lt_f32::sweeps++;

        const void* a_ptr = a.data_ptr();
        const void* b_ptr = b.data_ptr();
        void* c_ptr = c.data_ptr();
        hipblasLtMatmulAlgo_t best = results[0].algo;
        if (sweep)
        {
            hipEvent_t ev0, ev1;
            cudaEventCreate(&ev0);
            cudaEventCreate(&ev1);
            float best_ms = 0.0f;
            for (int i = 0; i < found; ++i)
            {
                // warmup (tensile kernel compile), then one timed launch
                hipblasLtMatmul(lt_f32::handle, op, &alpha, b_ptr, la, a_ptr, lb, &beta,
                                c_ptr, ld, c_ptr, ld, &results[i].algo, lt_f32::ws, lt_f32::WS_SIZE, stream);
                cudaEventRecord(ev0, stream);
                hipblasLtMatmul(lt_f32::handle, op, &alpha, b_ptr, la, a_ptr, lb, &beta,
                                c_ptr, ld, c_ptr, ld, &results[i].algo, lt_f32::ws, lt_f32::WS_SIZE, stream);
                cudaEventRecord(ev1, stream);
                cudaEventSynchronize(ev1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                if (best_ms == 0.0f || ms < best_ms)
                {
                    best_ms = ms;
                    best = results[i].algo;
                }
            }
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }
        cleanup();
        it = lt_f32::algos.emplace(key, best).first;
        return true;
    }

    hipblasStatus_t hs = hipblasLtMatmul(
        lt_f32::handle,
        op,
        &alpha,
        b.data_ptr(), la,
        a.data_ptr(), lb,
        &beta,
        c.data_ptr(), ld,
        c.data_ptr(), ld,
        &it->second,
        lt_f32::ws,
        lt_f32::ws ? lt_f32::WS_SIZE : 0,
        stream
    );
    cleanup();
    if (hs != HIPBLAS_STATUS_SUCCESS)
    {
        lt_f32::algos.erase(it);
        return false;
    }
    return true;
}
#endif
