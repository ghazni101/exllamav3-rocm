"""A3: launch-geometry sweep for the sq/int8 GEMV - slice height (EXL3_SQ_ROWS_PER, read once per
process into a static) x grid multiplier (force_num_sms through exl3_gemm's 10th argument, which the
sq path now honours; 0 = hardware geometry).

One rows_per config per process; the grid multiplier is swept in-process because it is a per-call
argument. Same cold-rotation pool discipline as A2 (AB_ROT same-shape instances, IC-resident pools
flagged).

Env:
  EXL3_MODEL  AB_DIR  AB_ROT (6)  AB_REPS (6)  EXL3_SQ_ROWS_PER (default: the extension's own)
  SWEEP_K     comma list of K (default "3,4")
  SWEEP_M     comma list of m (default "1")
  SWEEP_N     optional exact output width to pin the shape
  SWEEP_SMS   comma list of force_num_sms (default "0,48,96,192")
  SWEEP_TAG   label for the output file
Writes <AB_DIR>/a3_sweep_<tag>_rows<rows_per>.json
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
KS = [int(x) for x in os.environ.get("SWEEP_K", "3,4").split(",")]
MS = [int(x) for x in os.environ.get("SWEEP_M", "1").split(",")]
SMS = [int(x) for x in os.environ.get("SWEEP_SMS", "0,48,96,192").split(",")]
WANT_N = int(os.environ.get("SWEEP_N", "0")) or None
ROWS_PER = os.environ.get("EXL3_SQ_ROWS_PER", "auto")
TAG = os.environ.get("SWEEP_TAG", "a3")

torch.manual_seed(0)
_model, found = gc.enumerate_linears(MODEL_DIR)

out = dict(rows_per=ROWS_PER, rot=ROT, reps=REPS, cases=[])
for K in KS:
    pick = gc.pick_pool(found, K, ROT, prefer_n=WANT_N)
    if pick is None:
        print(f"K={K}: no pool (n={WANT_N})", flush=True)
        continue
    k, n, lins, avail = pick
    pool = lins[:ROT]
    gc.load_pool(pool)
    rep = gc.pool_report(k, n, K, len(pool))
    rep["available"] = avail
    rep["m"] = {}
    print(f"== K={K} k={k} n={n} pool={len(pool)}/{avail} {rep['pool_mb']} MB"
          f"{' IC-RESIDENT' if rep['ic_resident'] else ''} rows_per={ROWS_PER} ==", flush=True)
    for m in MS:
        xs = [torch.randn((m, k), dtype=torch.half, device="cuda:0") * 0.05 for _ in pool]
        xhs = [torch.empty_like(x) for x in xs]
        cs = [torch.empty((m, n), dtype=torch.half, device="cuda:0") for _ in pool]
        row = {}
        for sms in SMS:
            fns = []
            codes = []
            for x, xh, c, lin in zip(xs, xhs, cs, pool):
                inn = lin.inner
                def fn(x=x, xh=xh, c=c, inn=inn, codes=codes, sms=sms):
                    codes.append(ext.exl3_gemm(x, inn.trellis, c, inn.suh, xh, inn.svh,
                                               -1, False, True, sms))
                fns.append(fn)
            t = gc.bench(fns, reps=REPS)
            r = gc.rates(k, n, K, m, t, len(pool))
            r["ret_codes"] = sorted(set(codes))
            row[str(sms)] = r
            print(f"   m={m} force_sms={sms:4d}  t={r['t_ms_per_call']:8.4f} ms/call  "
                  f"{r['gbps']:7.1f} GB/s  {r['g_dp4a_s']:7.2f} G warp-dp4a/s  "
                  f"ret={sorted(set(codes))}", flush=True)
        rep["m"][str(m)] = row
        del xs, xhs, cs
    out["cases"].append(rep)
    del pool

os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, f"{TAG}_rows{ROWS_PER}.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path, flush=True)
