"""Standalone lm_head (k=5120, n=248320, K=6) GEMV bench.
Single instance; DRAM>>IC so rotation is a no-op (953MB per call anyway).
Env: EXL3_MODEL, SWEEP_SMS csv (0=auto), REPS.
"""
import json, os, sys, time
import torch
sys.path.insert(0, "/prof")
from exllamav3.ext import exllamav3_ext as ext
import gemv_cold as gc

MODEL_DIR = os.environ["EXL3_MODEL"]
SMS = [int(x) for x in os.environ.get("SWEEP_SMS", "0").split(",")]
REPS = int(os.environ.get("REPS", "10"))
model, found = gc.enumerate_linears(MODEL_DIR)
cands = [(K, k, n, lin) for (K, k, n, lin) in found if K == 6 and n == 248320]
assert cands, "no K=6 n=248320 linear"
K, k, n, lin = cands[0]
print(f"lm_head k={k} n={n} K={K} bytes={k*n*K/8/1e6:.0f}MB", flush=True)
lin.load(device="cuda:0")
x = torch.randn((1, k), dtype=torch.half, device="cuda:0") * 0.05
xh = torch.empty_like(x)
c = torch.empty((1, n), dtype=torch.half, device="cuda:0")
inn = lin.inner
res = {}
for sms in SMS:
    codes = []
    def fn():
        codes.append(ext.exl3_gemm(x, inn.trellis, c, inn.suh, xh, inn.svh, -1, False, True, sms))
    t = gc.bench([fn for _ in range(3)], reps=REPS)
    r = gc.rates(k, n, K, 1, t, 1)
    r["ret"] = sorted(set(codes))
    res[str(sms)] = r
    print(f"sms={sms:4d}  t={r['t_ms_per_call']:8.4f} ms/call  {r['gbps']:7.1f} GB/s  ret={r['ret']}", flush=True)
print(json.dumps(res), flush=True)
