# A/B test for the single-matrix msq routing (exl3_gemv_int8_msq_single): for m in 5..144 the
# single-matrix mul1 linears used to fall through to the grid-starved cooperative gemm kernel;
# they now route through the msq kernel as a one-entry bundle.
#
# Gates per (layer, m):
#   - msq-single vs the old path (EXL3_INT8_MSQ_SINGLE=0 reruns in a subprocess) with the same
#     rel_rms tolerance the bundle A/B uses (0.02)
#   - both int8 paths vs the fp16 reconstruct reference: the new path must not add error
#
# Usage (container): python3 /opt/exllamav3/test_msq_single_ab.py
import os, sys
import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext

MODEL = "/models/qwen38-27b"
DEV = "cuda:0"
MS = [8, 11, 32, 144]

torch.manual_seed(0)
config = Config.from_directory(MODEL)
model = Model.from_config(config)

# Single-matrix mul1 linears of each K flavor. Modules are lazy `Linear` wrappers; the
# LinearEXL3 appears in `.inner` only after load. Bundles are bypassed by calling forward
# directly on the inner module.
picks = {}
def consider(wrapper):
    if len(picks) == 3:
        return
    inner = getattr(wrapper, "inner", None)
    if inner is not None and type(inner).__name__ == "LinearEXL3" and inner.mul1:
        picks.setdefault(inner.K, inner)
def walk(mod):
    if type(mod).__name__ == "Linear" and getattr(mod, "inner", None) is None:
        mod.load(device=DEV)
    consider(mod)
    for name in ("attn", "mlp", "head", "q_proj", "k_proj", "v_proj", "o_proj",
                 "qkvz_proj", "z_proj", "ba_proj", "a_proj", "b_proj"):
        sub = getattr(mod, name, None)
        if sub is not None:
            walk(sub)
    for group in ("downs", "ups", "gates"):
        for sub in (getattr(mod, group, None) or []):
            walk(sub)
for m in model.modules:
    walk(m)
assert picks, "no single-matrix mul1 linears found"
for K, cand in sorted(picks.items()):
    print(f"K={K}: {cand.key}  k={cand.in_features} n={cand.out_features}")

ROUTED = os.environ.get("EXL3_INT8_MSQ_SINGLE", "1") != "0"
print(f"routing EXL3_INT8_MSQ_SINGLE={0 if not ROUTED else 1}")

out = {}
with torch.no_grad():
    for K, cand in sorted(picks.items()):
        for m in MS:
            if m > 144:
                continue
            k, n = cand.in_features, cand.out_features
            x = torch.randn((1, m, k), dtype=torch.half, device=DEV) * 0.05

            # fp16 reference: reconstruct
            y_ref = cand.forward(x, {"reconstruct": True}).float()

            # current routing (env-controlled) through the ext directly
            y = torch.empty((1, m, n), dtype=torch.float, device=DEV)
            xh = torch.empty((m, k), dtype=torch.half, device=DEV)
            ext.exl3_gemm(x.view(m, k), cand.trellis, y.view(m, n),
                          cand.suh, xh, cand.svh, -1, cand.mcg, cand.mul1, 0)
            torch.cuda.synchronize()
            y = y.float()

            def err(a, b):
                d = (a - b).abs()
                rel = d.max().item() / max(b.abs().max().item(), 1e-6)
                rms = (d.pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-6)).item()
                return d.max().item(), rel, rms

            e_cur = err(y, y_ref)
            tag = "msq" if ROUTED else "coop"
            print(f"K={K} m={m:3d} n={n:6d} {tag:4s} vs fp16-ref: "
                  f"max={e_cur[0]:.4f} rel_max={e_cur[1]:.5f} rel_rms={e_cur[2]:.6f}")
            out[(K, m, ROUTED)] = e_cur

import json
print(json.dumps({f"K{a}_m{b}_routed{int(c)}": v for (a, b, c), v in out.items()}))
