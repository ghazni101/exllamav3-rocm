"""A3 follow-up: what grid does the sq path actually launch, and why does force_num_sms=0 differ?

The A3 sweep showed force_num_sms=0 and 48 giving different times on identical code paths
(force_num_sms=0 means "use DevCtx::get_num_sms"), so this probe first prints the device-level
numbers the launcher uses, then sweeps a fine grid multiplier with per-call min/mean/max so a
single slow arm cannot hide behind an average.

Env: EXL3_MODEL AB_DIR AB_K (default "3,4") AB_M (default 1) AB_SHAPE ("k,n", default "5120,17408")
     AB_SMS (default "0,8,16,24,32,48,64,96,128,192") AB_REPS (default 6)
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
KS = [int(x) for x in os.environ.get("AB_K", "3,4").split(",")]
M = int(os.environ.get("AB_M", "1"))
SHAPE = tuple(int(x) for x in os.environ.get("AB_SHAPE", "5120,17408").split(","))
SMS = [int(x) for x in os.environ.get("AB_SMS", "0,8,16,24,32,48,64,96,128,192").split(",")]
REPS = int(os.environ.get("AB_REPS", "6"))

prop = torch.cuda.get_device_properties(0)
print(f"torch sees: {prop.name}  multiprocessor_count={prop.multi_processor_count}  "
      f"warp={prop.warp_size}", flush=True)
try:
    print(f"ext.g_get_num_sms(0) = {ext.g_get_num_sms(0)}", flush=True)
except Exception as e:                                   # noqa: BLE001
    print("g_get_num_sms unavailable:", e, flush=True)

torch.manual_seed(0)
_model, found = gc.enumerate_linears(MODEL_DIR)
out = dict(shape=SHAPE, m=M, sms=SMS, cases=[])

for K in KS:
    pick = gc.pick_pool(found, K, 6, prefer_n=SHAPE[1] if SHAPE[1] else None)
    if pick is None:
        print(f"K={K}: no pool for n={SHAPE[1]}", flush=True)
        continue
    k, n, lins, avail = pick
    pool = lins[:6]
    gc.load_pool(pool)
    xs = [torch.randn((M, k), dtype=torch.half, device="cuda:0") * 0.05 for _ in pool]
    xhs = [torch.empty_like(x) for x in xs]
    cs = [torch.empty((M, n), dtype=torch.half, device="cuda:0") for _ in pool]
    rec = dict(K=K, k=k, n=n, avail=avail, by_mb=round(gc.trellis_bytes(k, n, K) / 1e6, 2), arms={})
    print(f"== K={K} k={k} n={n} avail={avail} {rec['by_mb']} MB/inst ==", flush=True)
    for sms in SMS:
        fns = []
        for x, xh, c, lin in zip(xs, xhs, cs, pool):
            inn = lin.inner
            def fn(x=x, xh=xh, c=c, inn=inn, sms=sms):
                return ext.exl3_gemm(x, inn.trellis, c, inn.suh, xh, inn.svh, -1, False, True, sms)
            fns.append(fn)
        # per-call event timing so min/mean/max are visible (36 calls: 2 warmup passes + 6 timed)
        for _ in range(2):
            for f in fns:
                f()
        torch.cuda.synchronize()
        times = []
        for _ in range(REPS):
            for f in fns:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                f()
                e.record()
                times.append((s, e))
        torch.cuda.synchronize()
        ms = sorted(s.elapsed_time(e) for s, e in times)
        mean = sum(ms) / len(ms)
        by = gc.trellis_bytes(k, n, K)
        rec["arms"][str(sms)] = dict(
            min_ms=round(ms[0], 4), med_ms=round(ms[len(ms) // 2], 4), mean_ms=round(mean, 4),
            max_ms=round(ms[-1], 4),
            gbps_mean=round(by / (mean * 1e6), 1), gbps_med=round(by / (ms[len(ms) // 2] * 1e6), 1))
        print(f"   sms={sms:4d}  min={ms[0]:7.4f} med={ms[len(ms) // 2]:7.4f} mean={mean:7.4f} "
              f"max={ms[-1]:7.4f} ms  -> {by / (ms[len(ms) // 2] * 1e6):7.1f} GB/s (median)", flush=True)
    out["cases"].append(rec)
    del xs, xhs, cs

with open(os.path.join(OUT_DIR, "a3_gridprobe.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote a3_gridprobe.json", flush=True)
