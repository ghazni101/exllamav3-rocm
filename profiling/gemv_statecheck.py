"""A2 sanity: is the cold-rotation actually cold, and is the pool distinct?

The first A2 run reported 3.2 TB/s on the K=4 (5120,12288) arm - above this card's DRAM peak, i.e.
cache-resident or not reading B at all. This script decides which, by timing the same kernel call
under three pool regimes on the same tensors:

  pool=1        the same instance every call    -> IC-warm reference (upper bound)
  pool=ROT      distinct same-shape instances   -> what A2/A3 assumed is "cold"
  pool=large    enough distinct tensors to blow the cache many times over -> true cold rate

Plus a duplicate audit of the enumeration (a module reachable through more than one attribute
would put the *same* Linear in the pool several times, which is exactly how a 6-instance rotation
degrades into a 1-instance warm loop).

Env: EXL3_MODEL AB_DIR AB_K (default 4) AB_SHAPE ("k,n", default "5120,12288") AB_REPS
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
K_WANT = int(os.environ.get("AB_K", "4"))
SHAPE = tuple(int(x) for x in os.environ.get("AB_SHAPE", "5120,12288").split(","))
REPS = int(os.environ.get("AB_REPS", "6"))

torch.manual_seed(0)
_model, found = gc.enumerate_linears(MODEL_DIR)
ids = [id(l) for _K, _k, _n, l in found]
keys = [l.key for _K, _k, _n, l in found]
uniq_ids = len(set(ids))
print(f"enumerated {len(found)} Linear hits, {uniq_ids} distinct objects, "
      f"{len(set(keys))} distinct keys", flush=True)
dupes = [k for k, c in __import__("collections").Counter(keys).items() if c > 1]
print(f"keys appearing more than once: {len(dupes)} {dupes[:5]}", flush=True)

pool = [l for (K, k, n, l) in found if K == K_WANT and k == SHAPE[0] and n == SHAPE[1]]
pool = list({id(l): l for l in pool}.values())          # dedupe by object identity
avail = len(pool)
print(f"distinct K={K_WANT} ({SHAPE[0]},{SHAPE[1]}) instances: {avail}", flush=True)
print("first keys:", [l.key for l in pool[:8]], flush=True)

results = []
for tag, sel in (("pool1", pool[:1]),
                 ("pool6", pool[:6]),
                 ("pool12", pool[:12]),
                 ("pool_all", pool)):
    if not sel:
        continue
    gc.load_pool(sel)
    m = 1
    xs = [torch.randn((m, SHAPE[0]), dtype=torch.half, device="cuda:0") * 0.05 for _ in sel]
    xhs = [torch.empty_like(x) for x in xs]
    cs = [torch.empty((m, SHAPE[1]), dtype=torch.half, device="cuda:0") for _ in sel]
    fns = []
    for x, xh, c, lin in zip(xs, xhs, cs, sel):
        inn = lin.inner
        def fn(x=x, xh=xh, c=c, inn=inn):
            ext.exl3_gemm(x, inn.trellis, c, inn.suh, xh, inn.svh, -1, False, True, 0)
        fns.append(fn)
    t = gc.bench(fns, reps=REPS)
    r = gc.rates(SHAPE[0], SHAPE[1], K_WANT, m, t, len(sel))
    r.update(tag=tag, instances=len(sel))
    results.append(r)
    print(f"  {tag:9s} instances={len(sel):3d} pool={len(sel) * gc.trellis_bytes(*SHAPE, K_WANT) / 1e6:7.1f} MB  "
          f"t={r['t_ms_per_call']:8.4f} ms/call  {r['gbps']:8.1f} GB/s", flush=True)
    del xs, xhs, cs, fns

with open(os.path.join(OUT_DIR, "a2_statecheck.json"), "w") as f:
    json.dump(dict(K=K_WANT, shape=SHAPE, enumerated=len(found), distinct_objects=uniq_ids,
                   duplicate_keys=dupes[:20], avail=avail, arms=results), f, indent=1)
print("wrote a2_statecheck.json", flush=True)
