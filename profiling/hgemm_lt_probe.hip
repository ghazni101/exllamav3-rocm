// Standalone hipBLASLt probe for the reconstruct-path GEMM, isolating the call that faulted inside
// the extension (hgemm_lt_impl) and answering the question that decides the prefill fix:
// does hipBLASLt have a fast fp16-input / fp32-output (HSS) kernel, or is the fast path only
// available with fp16 output (HHS, which the incumbent hipBLAS Tensile config also has)?
//
//   hipcc --offload-arch=gfx1100 -O3 -o /out/hgemm_lt_hiptest /prof/hgemm_lt_probe.hip -lhipblaslt
//
// Build/run through profiling/run_hgemm_lt_hip.sh. Every hipblasLt status is printed, so a
// wrong layout/order shows up as a status code or a mismatch, never as a silent wrong number.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    printf("HIP FAIL %s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)
#define LTCK(x) do { hipblasStatus_t s = (x); if (s != HIPBLAS_STATUS_SUCCESS) { \
    printf("LT FAIL %s: status=%d\n", #x, (int) s); exit(2); } } while (0)

int main(int argc, char** argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 2048;
    int K = argc > 2 ? atoi(argv[2]) : 5120;
    int N = argc > 3 ? atoi(argv[3]) : 17408;
    int reps = argc > 4 ? atoi(argv[4]) : 20;
    bool big = M > 64;
    printf("device count check\n");
    int dev = 0;
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, dev));
    printf("gpu=%s  M=%d K=%d N=%d reps=%d\n", prop.name, M, K, N, reps);

    half *a = nullptr, *b0 = nullptr;
    std::vector<half*> bp;
    float* cref = nullptr;
    CK(hipMalloc(&a, (size_t) M * K * 2));
    for (int i = 0; i < 3; ++i) { half* p = nullptr; CK(hipMalloc(&p, (size_t) K * N * 2)); bp.push_back(p); }
    b0 = bp[0];
    CK(hipMalloc(&cref, (size_t) M * N * 4));

    // Fill: the small (verify) case gets deterministic host values; the big (timing) case only
    // needs valid memory, so it is filled on the device and never touches multi-hundred-MB host
    // vectors (those dominated wall time in the first two attempts).
    if (!big)
    {
        std::vector<half> ha(M * K), hb(K * N);
        // NOTE: cast the modulo to a signed type before subtracting, else size_t wraps and the
        // cast to half is inf for every element below the midpoint.
        for (size_t i = 0; i < ha.size(); ++i)
            ha[i] = (half) (0.02f * (float) ((int64_t) ((i * 2654435761u) % 1000) - 500) / 500.0f);
        for (size_t i = 0; i < hb.size(); ++i)
            hb[i] = (half) (0.02f * (float) ((int64_t) ((i * 40503u) % 1000) - 500) / 500.0f);
        CK(hipMemcpy(a, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
        for (int i = 0; i < 3; ++i) CK(hipMemcpy(bp[i], hb.data(), hb.size() * 2, hipMemcpyHostToDevice));
    }
    else
    {
        CK(hipMemset(a, 0x34, (size_t) M * K * 2));       // 0x3434 as half ~ 0.2036, finite
        for (int i = 0; i < 3; ++i) CK(hipMemset(bp[i], 0x34, (size_t) K * N * 2));
    }
    printf("buffers filled\n"); fflush(stdout);

    hipblasLtHandle_t lt;
    LTCK(hipblasLtCreate(&lt));
    printf("handle ok\n");

    for (int arm = 0; arm < 2; ++arm)
    {
        bool out_fp32 = arm == 1;
        printf("\n=== D %s ===\n", out_fp32 ? "fp32 (HSS)" : "fp16 (HHS)");
        hipblasLtMatmulDesc_t desc;
        LTCK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        // Mirror the incumbent cublasGemmEx call exactly (hgemm.cu:hgemm_gemmex_impl): column-major
        // views with OP_N/OP_N, m'=size_n, n'=size_m, k'=size_k, A=b (lda=size_n), B=a (ldb=size_k),
        // C=c (ldc=c_stride_m). Declaring the layouts as row-major instead would read b at stride k
        // rather than n, which is what faulted the in-extension attempt.
        hipblasOperation_t op_n = HIPBLAS_OP_N;
        LTCK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
        LTCK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

        hipblasLtMatrixLayout_t la, lb, lc;
        LTCK(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, N, K, N));      // A = b (N, K), lda = N
        LTCK(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, M, K));      // B = a (K, M), ldb = K
        LTCK(hipblasLtMatrixLayoutCreate(&lc, out_fp32 ? HIP_R_32F : HIP_R_16F, N, M, N));

        hipblasLtMatmulPreference_t pref;
        LTCK(hipblasLtMatmulPreferenceCreate(&pref));
        size_t ws_max = 16u << 20;
        LTCK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &ws_max, sizeof(ws_max)));
        hipblasLtMatmulHeuristicResult_t heur;
        memset(&heur, 0, sizeof(heur));
        int found = 0;
        hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &found);
        printf("heuristic status=%d found=%d workspaceSize=%zu\n", (int) hs, found, heur.workspaceSize);
        if (hs != HIPBLAS_STATUS_SUCCESS || found < 1) { printf("no algo, skipping\n"); continue; }
        printf("algo0: id=%u\n", (unsigned) heur.algo.data[0]);

        void* ws = nullptr;
        size_t ws_size = 0;
        if (heur.workspaceSize > 0)
        {
            CK(hipMalloc(&ws, heur.workspaceSize));
            ws_size = heur.workspaceSize;
            printf("allocated workspace %zu bytes\n", ws_size);
        }
        void* d = nullptr;
        CK(hipMalloc(&d, (size_t) M * N * (out_fp32 ? 4 : 2)));
        float alpha = 1.0f, beta = 0.0f;

        hipblasStatus_t s = hipblasLtMatmul(lt, desc, &alpha, b0, la, a, lb, &beta, d, lc, d, lc,
                                            &heur.algo, ws, ws_size, 0);
        CK(hipDeviceSynchronize());
        printf("matmul status=%d\n", (int) s);
        if (s != HIPBLAS_STATUS_SUCCESS) { printf("matmul failed, skipping verify/time\n"); continue; }

        // verify: compare the first small block against a host fp32 reference (small shapes only;
        // the big-shape version of this does thousands of tiny D2H copies and dominates wall time)
        if (!big) {
            int R = big ? 8 : M, Cn = big ? 16 : N, Kk = K;
            std::vector<half> ha_(R * Kk), hb_(Kk * Cn);
            CK(hipMemcpy(ha_.data(), a, ha_.size() * 2, hipMemcpyDeviceToHost));
            std::vector<half> hb_chunk(Kk * Cn);
            for (int k = 0; k < Kk; ++k)
                CK(hipMemcpy(hb_chunk.data() + (size_t) k * Cn, b0 + (size_t) k * N, Cn * 2,
                             hipMemcpyDeviceToHost));
            double worst = 0.0, refmax = 0.0;
            // d is (M, N) row-major: read the first Cn columns of the first R rows row by row
            std::vector<float> got(R * Cn, 0.f);
            for (int r = 0; r < R; ++r)
            {
                if (out_fp32)
                    CK(hipMemcpy(got.data() + (size_t) r * Cn, (float*) d + (size_t) r * N,
                                 Cn * 4, hipMemcpyDeviceToHost));
                else
                {
                    std::vector<half> gh(Cn);
                    CK(hipMemcpy(gh.data(), (half*) d + (size_t) r * N, Cn * 2, hipMemcpyDeviceToHost));
                    for (int c = 0; c < Cn; ++c) got[(size_t) r * Cn + c] = (float) gh[c];
                }
            }
            printf("sample a[0..2]=%.4f %.4f %.4f  b[0..2]=%.4f %.4f %.4f  got[0..2]=%.5f %.5f %.5f\n",
                   (float) ha_[0], (float) ha_[1], (float) ha_[2],
                   (float) hb_chunk[0], (float) hb_chunk[1], (float) hb_chunk[2],
                   got[0], got[1], got[2]);
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < Cn; ++c)
                {
                    double acc = 0;
                    for (int k = 0; k < Kk; ++k) acc += (double)(float) ha_[r * Kk + k] * (float) hb_chunk[(size_t) k * Cn + c];
                    double err = fabs(acc - got[r * Cn + c]);
                    if (err > worst) worst = err;
                    if (fabs(acc) > refmax) refmax = fabs(acc);
                }
            printf("verify: max_abs_err=%.3e  ref_max=%.3e  rel=%.2e\n", worst, refmax, worst / (refmax + 1e-9));
        }

        printf("verify done\n"); fflush(stdout);
        if (big)
        {
            printf("timing start\n"); fflush(stdout);
            for (int i = 0; i < 3; ++i)      // warmup, rotating buffers
                hipblasLtMatmul(lt, desc, &alpha, bp[i], la, a, lb, &beta, d, lc, d, lc, &heur.algo, ws, ws_size, 0);
            CK(hipDeviceSynchronize());
            hipEvent_t e0, e1;
            CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
            CK(hipEventRecord(e0, 0));
            for (int r = 0; r < reps; ++r)
                for (int i = 0; i < 3; ++i)
                    hipblasLtMatmul(lt, desc, &alpha, bp[i], la, a, lb, &beta, d, lc, d, lc, &heur.algo, ws, ws_size, 0);
            CK(hipEventRecord(e1, 0));
            CK(hipEventSynchronize(e1));
            float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
            double per = ms / (double)(reps * 3), tf = 2.0 * M * K * N / (per * 1e-3) / 1e12;
            double gb = (double) K * N * 2 / 1e9;
            printf("time %.4f ms/call  %.1f TFLOP/s  (weight %.2f GB/call -> %.0f GB/s)\n",
                   per, tf, gb, gb / (per * 1e-3));
        }
        CK(hipFree(d));
        if (ws) CK(hipFree(ws));
        LTCK(hipblasLtMatmulPreferenceDestroy(pref));
        LTCK(hipblasLtMatrixLayoutDestroy(la));
        LTCK(hipblasLtMatrixLayoutDestroy(lb));
        LTCK(hipblasLtMatrixLayoutDestroy(lc));
        LTCK(hipblasLtMatmulDescDestroy(desc));
    }
    LTCK(hipblasLtDestroy(lt));
    printf("\ndone\n");
    return 0;
}
