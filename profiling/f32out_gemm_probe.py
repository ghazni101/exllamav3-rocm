"""Measure the fp32-output reconstruct GEMM and the fix that needs no library change.

Finding it rests on: the model's q/k/v/gate/up projections are built with out_dtype=torch.float
(exllamav3/architecture/*.py), so their reconstruct GEMMs write fp32. On this stack an fp16-input
GEMM writing fp32 (Tensile "HSS", MT64x32x8 - the kernel D1 named at 68.2% of prefill device time)
runs at ~19 TFLOP/s through hipBLAS *and* hipBLASLt, while the same call writing fp16 (HHS) runs at
92-103 TFLOP/s. Clean-loop, m=2048, rotated weight buffers.

Arms, all through the installed extension (no rebuild):
  fp32_ref   ext.hgemm(a, b, c_fp32)            - the incumbent path
  f16_convert ext.hgemm(a, b, scratch_fp16); c.copy_(scratch)  - the candidate fix
Reports ms/call, TFLOP/s, and the error of the candidate against the incumbent's fp32 output (the
only numeric difference is one rounding to fp16 of the GEMM result).

Env: OUT_DIR, PR_M (2048), PR_ITERS (20), PR_SHAPES "k:n,k:n".
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/prof")
from exllamav3.ext import exllamav3_ext as ext

OUT_DIR = os.environ.get("OUT_DIR", "/out")
M = int(os.environ.get("PR_M", "2048"))
ITERS = int(os.environ.get("PR_ITERS", "20"))
SHAPES = [tuple(int(x) for x in s.split(":")) for s in
          os.environ.get("PR_SHAPES", "5120:17408,17408:5120,5120:6144,6144:5120").split(",")]

rows = []
print(f"m={M} iters={ITERS} shapes={SHAPES}", flush=True)
for (k, n) in SHAPES:
    a = torch.randn((M, k), dtype=torch.half, device="cuda:0") * 0.02
    pool = [torch.randn((k, n), dtype=torch.half, device="cuda:0") * 0.02 for _ in range(3)]
    c32 = torch.empty((M, n), dtype=torch.float, device="cuda:0")
    c16 = torch.empty((M, n), dtype=torch.half, device="cuda:0")
    out = {}

    def run(arm):
        if arm == "fp32_ref":
            for b in pool:
                ext.hgemm(a, b, c32)
        elif arm == "f16_convert":
            for b in pool:
                ext.hgemm(a, b, c16)
                c32.copy_(c16)
        elif arm == "f16_only":
            for b in pool:
                ext.hgemm(a, b, c16)

    # correctness of the candidate against the incumbent's fp32 result, per rotated buffer
    rels = []
    for b in pool:
        ext.hgemm(a, b, c32)
        ref = c32.clone()
        ext.hgemm(a, b, c16)
        c32.copy_(c16)
        denom = ref.abs().max().clamp_min(1e-6)
        rels.append(((c32 - ref).abs().max() / denom).item())
    out["convert_max_rel_vs_fp32"] = round(max(rels), 7)
    ext.hgemm(a, pool[0], c32)

    for arm in ("fp32_ref", "f16_convert", "f16_only"):
        run(arm)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            run(arm)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / (ITERS * len(pool))
        tflops = 2.0 * M * k * n / (ms * 1e-3) / 1e12
        out[arm] = {"ms": round(ms, 4), "tflops": round(tflops, 1)}
        print(f"  k={k:6d} n={n:6d} {arm:11s}: {ms:8.4f} ms/call  {tflops:6.1f} TF/s", flush=True)

    speedup = out["fp32_ref"]["ms"] / out["f16_convert"]["ms"]
    out["speedup"] = round(speedup, 2)
    print(f"  -> convert path {speedup:.2f}x vs incumbent, max_rel={out['convert_max_rel_vs_fp32']:.2e}",
          flush=True)
    rows.append({"k": k, "n": n, **out})
    del a, pool, c32, c16
    torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, "f32out_gemm_probe.json"), "w") as f:
    json.dump({"m": M, "rows": rows}, f, indent=1)
print("wrote f32out_gemm_probe.json", flush=True)
