"""D2: per-shape prefill GEMM configuration matrix (measurement half; no ext change needed).

D1 named the dominant in-model prefill kernel: one hipBLAS Tensile config
(Cijk_Ailk_Bljk_HSS_BH_MT64x32x8_...) takes 68% of prefill device time at ~28.5 TFLOP/s, while the
shape probes and clean loops have measured 54-100 TFLOP/s on the same shapes. This script asks
whether a *different reachable configuration* does better, shape by shape, under the same
cold-buffer discipline (B rotated across >= 2 x 100 MB buffers, never the same buffer twice).

Arms:
  torch_fp16      torch.matmul in fp16 (what the ext's cublasGemmEx path is compared against)
  torch_fp16_rpr  + allow_fp16_reduced_precision_reduction
  torch_t         transposed form (C^T = B^T A^T)
  (blas backend reported: hipBLAS vs hipBLASLt, whichever torch selects; TORCH_BLAS_PREFER_HIPBLASLT=1
   switches it, so run this script twice to compare)

Env: EXL3_MODEL (unused), OUT_DIR, D2_M (2048), D2_ITERS (20)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/prof")

OUT_DIR = os.environ.get("OUT_DIR", "/out")
M = int(os.environ.get("D2_M", "2048"))
ITERS = int(os.environ.get("D2_ITERS", "20"))

# The model's prefill-relevant Linear shapes (k, n) at 4.0 bpw + the lm_head
SHAPES = [
    (5120, 17408),   # mlp gate/up
    (17408, 5120),   # mlp down
    (5120, 10240),
    (5120, 6144),
    (6144, 5120),
    (5120, 12288),   # q_proj
    (5120, 248320),  # lm_head (K=5 trellis, but the GEMM shape is what matters)
]
BUF_MB = 128         # per-buffer size for the rotation pool


def blas_backend():
    try:
        return str(torch.backends.cuda.preferred_blas_library())
    except Exception:                                                          # noqa: BLE001
        return "unknown"


def bench(fns, iters=ITERS, warm=3):
    for _ in range(warm):
        for f in fns:
            f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        for f in fns:
            f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (iters * len(fns))


rows = []
print(f"torch {torch.__version__}  blas_backend={blas_backend()}  M={M}", flush=True)
for (k, n) in SHAPES:
    a = (torch.randn((M, k), dtype=torch.half, device="cuda:0") * 0.02)
    # rotation pool: 2 buffers x BUF_MB each (never the same buffer twice in a row)
    per = BUF_MB * 2 ** 20
    nelem = per // 2
    nbuf = max(2, (n * k) // nelem + 1)
    nbuf = min(nbuf, 3)
    bs = [torch.randn((n, k), dtype=torch.half, device="cuda:0") * 0.02 for _ in range(nbuf)]
    pool_mb = nbuf * n * k * 2 / 2 ** 20
    flops = 2.0 * M * k * n

    def mk(idx):
        def fn():
            torch.matmul(a, bs[idx % nbuf].t())
        return fn

    arms = {}
    fns = [mk(i) for i in range(nbuf)]
    ms = bench(fns)
    arms["torch_fp16"] = round(flops / (ms * 1e-3) / 1e12, 1)

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    fns = [mk(i) for i in range(nbuf)]
    ms = bench(fns)
    arms["torch_fp16_rpr"] = round(flops / (ms * 1e-3) / 1e12, 1)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # transposed form: C^T(n,m) = B(n,k) @ A^T(k,m)
    at = a.t().contiguous()
    fns = [lambda i=i: torch.matmul(bs[i % nbuf], at) for i in range(nbuf)]
    ms = bench(fns)
    arms["torch_t"] = round(flops / (ms * 1e-3) / 1e12, 1)

    row = dict(k=k, n=n, pool_mb=round(pool_mb, 1), **arms)
    print(f"ROW {row}", flush=True)
    rows.append(row)
    print(f"  k={k:6d} n={n:6d} pool={pool_mb:7.1f} MB  " +
          "  ".join(f"{kk}={vv:6.1f} TF/s" for kk, vv in arms.items()), flush=True)
    del a, bs, at
    torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, "d2_gemm_matrix.json"), "w") as f:
    json.dump(dict(m=M, backend=blas_backend(), rows=rows), f, indent=1)
print("wrote d2_gemm_matrix.json", flush=True)
