"""Direct probe of the reconstruct-path GEMM: `ext.hgemm` (hipBLAS cublasGemmEx) vs the hipBLASLt
path behind EXL3_HGEMM_LT=1, on the model's own prefill shapes.

Why this exists: D1/D2 showed the in-model prefill GEMM running at 28.5 TFLOP/s in one hipBLAS
Tensile configuration while torch (hipBLASLt) reaches 99.6-107.8 on the same shapes. This probe
tests the extension's own entry point - correctness against a torch reference first, then speed -
so a bad hipBLASLt descriptor or a declining heuristic shows up in seconds instead of in a 40-minute
bench cycle. `hgemm(a, b, c)` computes C[m,n] = A[m,k] @ B[k,n] (b is (k,n) row-major), c fp16 or
fp32; the buffers are rotated so the weights are not Infinity-Cache-resident.

Env: OUT_DIR, PR_M (2048), PR_K/PR_N lists, PR_ITERS (20), EXL3_HGEMM_LT (read by the extension)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/prof")
from exllamav3.ext import exllamav3_ext as ext                            # noqa: E402

OUT_DIR = os.environ.get("OUT_DIR", "/out")
M = int(os.environ.get("PR_M", "2048"))
ITERS = int(os.environ.get("PR_ITERS", "20"))
SHAPES = [(5120, 17408), (17408, 5120), (5120, 6144), (6144, 5120)]
ARM = "lt" if os.environ.get("EXL3_HGEMM_LT", "0") not in ("0", "") else "incumbent"

rows = []
print(f"arm={ARM}  m={M}  torch={torch.__version__}", flush=True)
for (k, n) in SHAPES:
    a = torch.randn((M, k), dtype=torch.half, device="cuda:0") * 0.02
    pool = [torch.randn((k, n), dtype=torch.half, device="cuda:0") * 0.02 for _ in range(3)]
    pool_mb = 3 * k * n * 2 / 2 ** 20
    ref = a @ pool[0]
    out = {}
    for dtype, label in ((torch.half, "c_fp16"), (torch.float, "c_fp32")):
        c = torch.empty((M, n), dtype=dtype, device="cuda:0")
        ext.hgemm(a, pool[0], c)
        torch.cuda.synchronize()
        got = c.float()
        rel = ((got - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-6)).item()
        # rotated timing: never the same b buffer twice in a row
        for b in pool:
            ext.hgemm(a, b, c)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            for b in pool:
                ext.hgemm(a, b, c)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / (ITERS * len(pool))
        tflops = 2.0 * M * k * n / (ms * 1e-3) / 1e12
        out[label] = dict(ms=round(ms, 4), tflops=round(tflops, 1), max_rel=round(rel, 6))
        print(f"  k={k:6d} n={n:6d} pool={pool_mb:6.0f} MB {label}: {ms:8.4f} ms/call  "
              f"{tflops:6.1f} TF/s  max_rel={rel:.2e}", flush=True)
        del c
    rows.append(dict(k=k, n=n, pool_mb=round(pool_mb, 1), **out))
    del a, pool, ref
    torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, f"hgemm_lt_probe_{ARM}.json"), "w") as f:
    json.dump(dict(arm=ARM, m=M, rows=rows), f, indent=1)
print(f"wrote hgemm_lt_probe_{ARM}.json", flush=True)
