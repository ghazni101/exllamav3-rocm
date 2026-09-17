"""A3 grid/slice sweep, ONE configuration per process.

Rationale: the earlier multi-arm sweep (all force_num_sms values in one process) produced a
systematic first-arm penalty (K=3: 188 GB/s at sms=0 vs 416 at sms=48, tight min/max within each
arm, i.e. not noise) even though force_num_sms=0 resolves to the same num_sms=48
(`ext.g_get_num_sms(0)` = 48). One configuration per process removes arm-order effects entirely,
which is also how the rows_per knob has to be swept (it is read into a static).

Env:
  EXL3_MODEL AB_DIR AB_ROT AB_REPS
  AB_K, AB_M, AB_SHAPE ("k,n" or "auto" for the widest shape with >= rot instances)
  AB_SMS      force_num_sms for this process (0 = the extension's own grid)
  EXL3_SQ_ROWS_PER / EXL3_SQ_STAGE_SMEM  the extension-side knobs under test
  AB_TAG      label for the result file; results are appended to <AB_DIR>/a3_procs.json
Prints one JSON line with the median-of-reps per-call time and derived rates.
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
ROT = int(os.environ.get("AB_ROT", "20"))
REPS = int(os.environ.get("AB_REPS", "6"))
K = int(os.environ.get("AB_K", "4"))
M = int(os.environ.get("AB_M", "1"))
SHAPE = os.environ.get("AB_SHAPE", "auto")
SMS = int(os.environ.get("AB_SMS", "0"))
TAG = os.environ.get("AB_TAG", "x")
ROWS_PER = os.environ.get("EXL3_SQ_ROWS_PER", "auto")
STAGE = os.environ.get("EXL3_SQ_STAGE_SMEM", "auto")

torch.manual_seed(0)
_model, found = gc.enumerate_linears(MODEL_DIR)

if SHAPE == "auto":
    pick = gc.pick_pool(found, K, ROT)
else:
    kk, nn = (int(x) for x in SHAPE.split(","))
    pick = gc.pick_pool(found, K, ROT, prefer_n=nn)
if pick is None:
    print(json.dumps(dict(tag=TAG, K=K, m=M, sms=SMS, skipped="no pool")))
    sys.exit(0)
k, n, lins, avail = pick
pool = lins[:ROT]
gc.load_pool(pool)
by = gc.trellis_bytes(k, n, K)

xs = [torch.randn((M, k), dtype=torch.half, device="cuda:0") * 0.05 for _ in pool]
xhs = [torch.empty_like(x) for x in xs]
cs = [torch.empty((M, n), dtype=torch.half, device="cuda:0") for _ in pool]
codes = []
fns = []
for x, xh, c, lin in zip(xs, xhs, cs, pool):
    inn = lin.inner
    def fn(x=x, xh=xh, c=c, inn=inn, codes=codes):
        codes.append(ext.exl3_gemm(x, inn.trellis, c, inn.suh, xh, inn.svh, -1, False, True, SMS))
    fns.append(fn)

# Cold protocol: no warmup pass - the first pass over a pool that is many times the 96 MB Infinity
# Cache is the cold measurement (the earlier 6-instance pools were only ~2x the IC, so a pass
# re-read much of itself and the reported GB/s depended on what had run before). Pass 2 is
# reported separately as the warm contrast; each call is timed with its own event pair.
def one_pass():
    ev = []
    for f in fns:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        f()
        e.record()
        ev.append((s, e))
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in ev)

p1 = one_pass()                      # cold: every tensor untouched by this process so far
p2 = one_pass()                      # warm: up to IC-size worth of the pool is now resident
ms = p1
med = ms[len(ms) // 2]
warm_med = p2[len(p2) // 2]
rec = dict(tag=TAG, K=K, m=M, k=k, n=n, sms=SMS, rows_per=ROWS_PER, stage=STAGE, avail=avail,
           pool_mb=round(by * len(pool) / 1e6, 1), ic_resident=by * len(pool) < gc.IC_BYTES,
           pool_over_ic=round(by * len(pool) / gc.IC_BYTES, 2), ret=sorted(set(codes)),
           cold_min_ms=round(p1[0], 4), cold_med_ms=round(med, 4), cold_max_ms=round(p1[-1], 4),
           warm_med_ms=round(warm_med, 4), gbps=round(by / (med * 1e6), 1),
           gbps_warm=round(by / (warm_med * 1e6), 1))
rec.update({kk: v for kk, v in gc.rates(k, n, K, M, med, len(pool)).items()
            if kk in ("g_dp4a_s", "g_weights_s")})
path = os.path.join(OUT_DIR, "a3_procs.json")
rows = json.load(open(path)) if os.path.exists(path) else []
rows.append(rec)
with open(path, "w") as f:
    json.dump(rows, f, indent=1)
print("JSON " + json.dumps(rec), flush=True)
