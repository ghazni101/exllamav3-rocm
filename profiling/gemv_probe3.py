"""A2: cold-rotation GEMV rates for the sq/int8 kernels, per K and per m.

Matrix: K in {3,4,5,6} x m in {1,2,4}, all at the model's own trellis tensors (no synthetic data),
plus the K=4 n=1024 IC-resident arm as the ALU-only control (a pool that fits in the 96 MB Infinity
Cache cannot show a DRAM limit; it bounds what the kernel does when B is resident).

Cold discipline: each timed call runs on a different same-shape layer instance and the pool is at
least AB_ROT instances deep. Reported GB/s is bytes-of-trellis per second; for a pool smaller than
the IC the row is marked `ic_resident` and its GB/s must not be read as a DRAM rate.

Env:
  EXL3_MODEL  AB_DIR  AB_ROT (default 6)  AB_REPS (default 6)  AB_MS (default "1,2,4")
  AB_SMS      force_num_sms passed to exl3_gemm (default 0 = hardware geometry)
  AB_ROWS_PER EXL3_SQ_ROWS_PER is read once per process inside the extension
  A2_K        comma list of K to measure (default "3,4,5,6")
  A2_IC       "1" adds the K=4 n<=1024 IC-resident control arm
Writes <AB_DIR>/a2_probe3.json and prints one line per (K, m).
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/prof")
from exllamav3.ext import exllamav3_ext as ext
import gemv_cold as gc

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
OUT_DIR = os.environ.get("AB_DIR", "/out")
ROT = int(os.environ.get("AB_ROT", "6"))
REPS = int(os.environ.get("AB_REPS", "6"))
MS = [int(x) for x in os.environ.get("AB_MS", "1,2,4").split(",")]
SMS = int(os.environ.get("AB_SMS", "0"))
KS = [int(x) for x in os.environ.get("A2_K", "3,4,5,6").split(",")]
WANT_IC = os.environ.get("A2_IC", "1") == "1"

torch.manual_seed(0)
_model, found = gc.enumerate_linears(MODEL_DIR)
print(f"model has {len(found)} quantized Linear instances", flush=True)

results = []
arms = []
for K in KS:
    pick = gc.pick_pool(found, K, ROT)
    if pick is None:
        print(f"K={K}: no shape with >= {ROT} instances - skipped", flush=True)
        continue
    k, n, lins, avail = pick
    arms.append((K, k, n, lins[:ROT], avail, "cold"))
if WANT_IC:
    # IC-resident control: the widest small-n shape at K=4
    pick = gc.pick_pool(found, 4, 1, n_max=1024)
    if pick:
        k, n, lins, avail = pick
        arms.append((4, k, n, lins[:max(1, min(ROT, avail))], avail, "ic_control"))

for K, k, n, pool, avail, tag in arms:
    gc.load_pool(pool)
    rep = gc.pool_report(k, n, K, len(pool))
    rep["tag"] = tag
    rep["available"] = avail
    rep["m"] = {}
    print(f"== K={K} k={k} n={n} [{tag}] pool={len(pool)}/{avail} "
          f"{rep['bytes_mb']} MB/inst, {rep['pool_mb']} MB pool"
          f"{' (IC-RESIDENT)' if rep['ic_resident'] else ''} ==", flush=True)
    for m in MS:
        xs = [torch.randn((m, k), dtype=torch.half, device="cuda:0") * 0.05 for _ in pool]
        xhs = [torch.empty_like(x) for x in xs]
        cs = [torch.empty((m, n), dtype=torch.half, device="cuda:0") for _ in pool]
        codes = []
        fns = []
        for x, xh, c, lin in zip(xs, xhs, cs, pool):
            inn = lin.inner
            def fn(x=x, xh=xh, c=c, inn=inn, codes=codes):
                codes.append(ext.exl3_gemm(x, inn.trellis, c, inn.suh, xh, inn.svh,
                                           -1, False, True, SMS))
            fns.append(fn)
        t = gc.bench(fns, reps=REPS)
        fns[0]()
        torch.cuda.synchronize()
        c1 = cs[0].clone()
        fns[0]()
        torch.cuda.synchronize()
        det = torch.equal(c1, cs[0])
        r = gc.rates(k, n, K, m, t, len(pool))
        r["det"] = bool(det)
        r["ret_codes"] = sorted(set(codes))
        rep["m"][str(m)] = r
        print(f"   m={m} t={r['t_ms_per_call']:8.4f} ms/call  {r['gbps']:7.1f} GB/s  "
              f"{r['g_dp4a_s']:7.1f} G warp-dp4a/s  {r['g_weights_s']:7.1f} G weights/s  "
              f"det={det} ret={sorted(set(codes))}", flush=True)
        del xs, xhs, cs, c1
    results.append(rep)
    del pool

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "a2_probe3.json"), "w") as f:
    json.dump(dict(rot=ROT, reps=REPS, force_num_sms=SMS, arms=results), f, indent=1)
print("wrote a2_probe3.json", flush=True)
