#!/usr/bin/env python3
"""Standalone A/B for the T1 route: single-matrix m > 4 through the msq kernel.

Cold-state timing: in-model decode re-reads a weight tensor after ~13 GB of other traffic,
so a bench loop over ONE tensor measures Infinity-Cache-resident kernels (96 MB IC), not
the production regime. This bench ROTATES over several same-shape layer instances so each
call streams cold weights.

Modes (numeric reference needs its own process: the msq kill switch is read once per
process; a forced-shape coop kernel at m=5 is not a valid reference either - the generator
only reaches coop through the autotuner, and forced shape configs are exactly the M>=3
history documented in exl3_gemm_inner.cuh):
  AB_MODE=msq      (default) force=-1 -> m<=4 sq; m>4 msq (the T1 route). Cold-timed,
                   determinism-checked; saves outputs.
  AB_MODE=coopref  run with EXL3_INT8_MSQ=0 so force=-1 lands on the AUTOTUNED coop kernel
                   (the incumbent the model actually uses); cold-timed; saves outputs.
  AB_MODE=compare  loads both output sets, recomputes the sq reference (chunks of 4 through
                   the production m<=4 kernel), reports tolerances and speedups.

Pass: deterministic msq; maxrel vs sqref < 1e-3 (same per-(row, k-slice) quantization; per-M
instantiations may reassociate the fp32 combine); rel_rms vs autotuned coop < 0.02 (per-slice
vs global activation scales).
"""
import os, sys, json, collections
import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
MODE = os.environ.get("AB_MODE", "msq")   # coopref = autotuned-coop reference (EXL3_INT8_MSQ=0)
SAVENAME = "coop" if MODE == "coopref" else MODE
DIR = os.environ.get("AB_DIR", "/out")
MS = [int(x) for x in os.environ.get("AB_MS", "1,2,4,5,8").split(",")]
REPS = int(os.environ.get("AB_REPS", "6"))
ROT = int(os.environ.get("AB_ROT", "6"))     # rotation pool depth (x weight bytes > cache)

torch.manual_seed(0)
config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)

# Group same-shape Linear instances by (k, n); .inner only exists after load, so shapes drive
# selection. lm_head (n > 32768) and non-256-multiples are filtered.
groups = collections.defaultdict(list)
for m_ in model.modules:
    for attr in ("attn", "mlp"):
        mod = getattr(m_, attr, None)
        if mod is None:
            continue
        for name in dir(mod):
            try:
                lin = getattr(mod, name)
            except Exception:
                continue
            if type(lin).__name__ != "Linear":
                continue
            k, n = lin.in_features, lin.out_features
            if k % 128 or n % 256 or n > 32768:
                continue
            key = getattr(lin, "key", "") or ""
            groups[(k, n)].append(lin)

# Prefer the biggest shape with at least ROT instances
target = None
for (k, n), lins in sorted(groups.items(), key = lambda kv: -kv[0][0] * kv[0][1]):
    if len(lins) >= ROT:
        target = (k, n, lins[:ROT])
        break
assert target, "no shape with enough instances"
k, n, pool = target
for lin in pool:
    lin.load(device = "cuda:0")

# Instance 0 of each K subgroup is the correctness representative
byK = collections.defaultdict(list)
for lin in pool:
    byK[lin.inner.K].append(lin)
print("pool:", {K: [l.key for l in lins] for K, lins in byK.items()}, flush=True)

def bench(fns, reps=REPS):
    for _ in range(2):
        for f in fns: f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        for f in fns: f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (reps * len(fns))

def collect(mode, xs_source = None):
    """mode 'msq' or 'coop': {(K, m): (t_ms_per_call, C, det, x)}. xs_source supplies the
    exact activations used by a previous mode so outputs are same-input comparable."""
    out = {}
    for K in sorted(byK):
        lins = byK[K]
        trellis, suh, svh = lins[0].inner.trellis, lins[0].inner.suh, lins[0].inner.svh
        tb = trellis.numel() * 2
        print(f"== [{mode}] K={K} k={k} n={n} pool={len(lins)} trellis={tb/1e6:.1f} MB ==", flush=True)
        for m in MS:
            if xs_source is not None:
                x0 = xs_source[(K, m)]
                xs = [x0.clone() for _ in lins]
            else:
                xs = [torch.randn((m, k), dtype = torch.half, device = "cuda:0") * 0.05 for _ in lins]
            xhs = [torch.empty_like(x) for x in xs]
            cs = [torch.empty((m, n), dtype = torch.half, device = "cuda:0") for _ in lins]
            fns = []
            for x, xh, c, lin in zip(xs, xhs, cs, lins):
                fns.append(lambda x = x, xh = xh, c = c, lin = lin:
                           ext.exl3_gemm(x, lin.inner.trellis, c, lin.inner.suh, xh,
                                         lin.inner.svh, -1, False, True, 0))
            t = bench(fns)
            rep = fns[0]
            rep(); torch.cuda.synchronize()
            c1 = cs[0].clone()
            rep(); torch.cuda.synchronize()
            det = torch.equal(c1, cs[0])
            out[(K, m)] = (t, cs[0].cpu().clone(), det, xs[0].cpu().clone())
            print(f"   [{mode}] m={m:4d} t={t:7.4f} ms ({m * tb / t / 1e6:6.1f} GB/s cold) det={det}",
                  flush=True)
            del xs, xhs, cs, c1
    return out

if MODE in ("msq", "coop", "coopref"):
    xs_source = None
    if SAVENAME == "coop":
        prev = torch.load(os.path.join(DIR, "ab_msq.pt"), weights_only = False)
        xs_source = {kk: v[3].cuda() for kk, v in prev.items()}
    res = collect(SAVENAME, xs_source)
    torch.save(res, os.path.join(DIR, f"ab_{SAVENAME}.pt"))
    times = {}
    for (K, m), (t, _c, _d, _x) in res.items():
        tb = byK[K][0].inner.trellis.numel() * 2
        times[f"{K}_{m}"] = dict(t_ms = round(t, 4), gbps = round(m * tb / t / 1e6, 1))
    json.dump({"mode": SAVENAME, "cases": times},
              open(os.path.join(DIR, f"ab_{SAVENAME}.json"), "w"), indent = 1)
    print(f"[{SAVENAME}] saved", flush=True)

elif MODE == "compare":
    a = torch.load(os.path.join(DIR, "ab_msq.pt"), weights_only = False)
    b = torch.load(os.path.join(DIR, "ab_coop.pt"), weights_only = False)
    results = []
    for key, (t, c_msq, det, x) in sorted(a.items()):
        K, m = key
        c_msq = c_msq.cuda()
        c_coop = b[key][1].cuda()
        x = x.cuda()
        lin = byK[K][0]
        inn = lin.inner
        xh = torch.empty_like(x)
        c_ref = torch.empty((m, n), dtype = torch.half, device = "cuda:0")
        for i in range(0, m, 4):
            j = min(i + 4, m)
            xh_c = torch.empty((j - i, k), dtype = torch.half, device = "cuda:0")
            ext.exl3_gemm(x[i:j].contiguous(), inn.trellis, c_ref[i:j], inn.suh,
                          xh_c, inn.svh, -1, False, True, 0)
        torch.cuda.synchronize()
        maxrel_sq = ((c_ref.float() - c_msq.float()).abs().max()
                     / c_ref.float().abs().max().clamp_min(1e-6)).item()
        d = (c_msq.float() - c_coop.float()).abs()
        rel_rms = (d.pow(2).mean().sqrt() / c_coop.float().pow(2).mean().sqrt().clamp_min(1e-6)).item()
        ok = det and maxrel_sq < 1e-3 and rel_rms < 0.02
        print(f"   K={K} m={m:4d} maxrel_sq={maxrel_sq:.2e} rel_rms_coop={rel_rms:.5f} det={det} "
              f"t_msq={t:7.4f}ms t_coop={b[key][0]:7.4f}ms "
              f"speedup={b[key][0] / t:5.2f}x {'OK' if ok else 'BAD'}", flush=True)
        results.append(dict(K = K, m = m, maxrel_sq = round(maxrel_sq, 8),
                            rel_rms_coop = round(rel_rms, 6), det = bool(det),
                            t_msq_ms = round(t, 4), t_coop_ms = round(b[key][0], 4), ok = ok))
        del x, xh, c_ref, c_msq, c_coop
    json.dump(results, open(os.path.join(DIR, "ab_compare.json"), "w"), indent = 1)
    bad = [r for r in results if not r["ok"]]
    print(f"[{'FAIL' if bad else 'PASS'}] {len(bad)} bad of {len(results)} cases", flush=True)
    sys.exit(1 if bad else 0)
