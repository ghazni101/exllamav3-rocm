# Diagnostic: msq correctness at m>4, K=1/2/3, bundle (mgemm) vs single (gemm + routing).
#   - bundle: MultiLinear of two same-K linears; ext.exl3_mgemm(-1) routes msq,
#     ext.exl3_mgemm(force_shape_idx=2) forces the cooperative mgemm kernel.
#   - single: one linear through ext.exl3_gemm; env EXL3_INT8_MSQ_SINGLE picks msq-single or coop.
# References: fp16 reconstruct per linear; each half of the bundle compared to its own linear.
import os, sys
import torch
from exllamav3 import Config, Model
from exllamav3.modules.multilinear import MultiLinear
from exllamav3.ext import exllamav3_ext as ext

MODEL = "/models/qwen38-27b"
DEV = "cuda:0"
MS = [int(v) for v in os.environ.get("MSQ_DIAG_M", "8").split(",")]

torch.manual_seed(0)
config = Config.from_directory(MODEL)
model = Model.from_config(config)

picks = {}
def consider(wrapper):
    inner = getattr(wrapper, "inner", None)
    if inner is not None and type(inner).__name__ == "LinearEXL3" and inner.mul1:
        picks.setdefault(inner.K, []).append((wrapper, inner))
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
            consider(sub)
for m in model.modules:
    walk(m)
# two DISTINCT linears per K with identical shapes (bundle requirement)
pairs = {}
for K, lst in picks.items():
    for i, a in enumerate(lst):
        for b in lst[i + 1:]:
            if (a[1].in_features, a[1].out_features) == (b[1].in_features, b[1].out_features):
                pairs[K] = (a, b)
                break
        if K in pairs:
            break
picks = pairs
print(f"Ks with matching pairs: {sorted(picks)}")

def rel(a, b):
    d = (a.float() - b.float()).abs()
    return (d.max().item() / max(b.float().abs().max().item(), 1e-6),
            (d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp_min(1e-6)).item())

with torch.no_grad():
  for M in MS:
    for K in sorted(picks):
        (wa, la), (wb, lb) = picks[K]
        k, n = la.in_features, la.out_features
        x = torch.randn((1, M, k), dtype=torch.half, device=DEV) * 0.05

        # --- bundle through mgemm: msq (-1) vs forced coop (2); coop is the reference ---
        bund = MultiLinear(DEV, [wa, wb])
        carrier = torch.empty((2, M, n), dtype=torch.float, device=DEV)
        xh_b = torch.empty((2, M, k), dtype=torch.half, device=DEV)
        def run_bundle(force):
            carrier.zero_()
            ext.exl3_mgemm(x, bund.ptrs_trellis, carrier, bund.ptrs_suh, xh_b, bund.ptrs_svh,
                           None, None, bund.K, force, bund.mcg, bund.mul1, -1, -1, 0, 1,
                           None, None, None, None, 0)
            torch.cuda.synchronize()
            return carrier[0].clone(), carrier[1].clone()
        msq_a, msq_b = run_bundle(-1)
        coop_a, coop_b = run_bundle(2)
        r_x = rel(msq_a, coop_a)
        print(f"K={K} m={M} BUNDLE  msq-vs-coop rel_max={r_x[0]:.5f} rel_rms={r_x[1]:.6f}")

        # --- single through gemm: env-selected routing (msq-single or coop) ---
        y = torch.empty((M, n), dtype=torch.float, device=DEV)
        xh = torch.empty((M, k), dtype=torch.half, device=DEV)
        ROUTED = os.environ.get("EXL3_INT8_MSQ_SINGLE", "1") != "0"
        ext.exl3_gemm(x.view(M, k), la.trellis, y, la.suh, xh, la.svh, -1, la.mcg, la.mul1, 0)
        torch.cuda.synchronize()
        r_s_coop = rel(y, coop_a)
        tag = "msq" if ROUTED else "coop"
        ok = r_s_coop[1] < 0.02
        print(f"K={K} m={M} SINGLE[{tag}]  vs-bundle-coop rel_max={r_s_coop[0]:.5f} "
              f"rel_rms={r_s_coop[1]:.6f} {'OK' if ok else 'FAIL'}")
