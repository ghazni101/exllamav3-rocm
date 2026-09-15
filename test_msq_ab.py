# A/B test: exl3_gemv_int8_msq_kernel vs the cooperative exl3_mgemm_kernel on identical inputs.
# Covers both mgemm modes the msq path takes over at m == 1:
#   - sliced (SlicedMultiLinear: qkvz bundle, attention qkv) via size_n_list/c_ptrs/had_src_list
#   - plain multi-matrix (MultiLinear: MLP gate+up) via B/suh/svh pointer lists
# msq is not bit-exact vs coop (per-slice int8 activation scales vs global) - we compare against
# the coop output with a tolerance, and against each matrix's own exl3_gemm int8 (sq) output,
# which shares the quantization scheme and should agree closely.
#
# Usage (container): python3 /src/test_msq_ab.py
import os, sys
sys.path.insert(0, "/src")
import torch
from exllamav3 import Config, Model
from exllamav3.modules.multilinear import SlicedMultiLinear, MultiLinear
from exllamav3.ext import exllamav3_ext as ext

MODEL = "/models/qwen38-27b"
DEV = "cuda:0"

torch.manual_seed(0)
config = Config.from_directory(MODEL)
model = Model.from_config(config)

# Find a GatedDeltaNet layer (split qkv/z -> SlicedMultiLinear) and an MLP (gate+up MultiLinear)
gdn = None
mlp = None
for m in model.modules:
    if type(m).__name__ != "TransformerBlock":
        continue
    attn = getattr(m, "attn", None)
    if gdn is None and attn is not None and type(attn).__name__ == "GatedDeltaNet" \
            and getattr(attn, "qkv_proj", None) is not None:
        gdn = attn
    if mlp is None and getattr(m, "mlp", None) is not None:
        mlp = m.mlp
    if gdn and mlp:
        break
assert gdn is not None, "no split-projection GatedDeltaNet found"
assert mlp is not None, "no MLP found"
print(f"gdn key: {gdn.key}   mlp key: {mlp.key}")

gdn.load(device = DEV)
mlp.load(device = DEV)

def report(name, a, b):
    d = (a.float() - b.float()).abs()
    rel = d.max().item() / max(b.float().abs().max().item(), 1e-6)
    rms = (d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp_min(1e-6)).item()
    print(f"  {name}: max_abs={d.max().item():.5f} rel_max={rel:.5f} rel_rms={rms:.6f}")
    return rms

failures = []

# ---------------- sliced mode (qkvz bundle), m = 1 and m = 4 ----------------
mq = SlicedMultiLinear(DEV, [gdn.qkv_proj, gdn.z_proj])
print(f"sliced: slices={mq.num_slices} width={mq.width} K={mq.K} mul1={mq.mul1} mcg={mq.mcg} srcs={mq.num_src}")
def sliced_case(m):
    global failures
    print(f"== sliced mgemm (GDN qkvz), m={m} ==")
    hidden = gdn.qkv_proj.in_features
    x = torch.randn((1, m, hidden), dtype = torch.half, device = DEV) * 0.05
    xh = torch.empty((mq.num_src, m, hidden), dtype = torch.half, device = DEV)
    qkv = torch.empty((1, m, gdn.qkv_proj.out_features), dtype = torch.float, device = DEV)
    z = torch.empty((1, m, gdn.z_proj.out_features), dtype = torch.float, device = DEV)
    c_ptrs = mq.c_ptrs([qkv.view(m, -1), z.view(m, -1)])
    carrier = torch.empty((mq.num_slices, m, mq.width), dtype = torch.float, device = DEV)

    def run(force_shape_idx):
        qkv.zero_(); z.zero_()
        ext.exl3_mgemm(
            x, mq.ptrs_trellis, carrier, mq.ptrs_suh, xh, mq.ptrs_svh,
            None, None, mq.K, force_shape_idx, mq.mcg, mq.mul1, -1, -1, 0, 1,
            mq.size_n_list, c_ptrs, mq.n_stride_list, mq.had_src_list, mq.num_src)
        torch.cuda.synchronize()
        return qkv.clone(), z.clone()

    qkv_msq, z_msq = run(-1)          # msq path
    qkv_coop, z_coop = run(2)         # forced coop kernel
    r = report(f"qkv m{m} msq vs coop", qkv_msq, qkv_coop); failures += [r > 0.02]
    r = report(f"z   m{m} msq vs coop", z_msq, z_coop);   failures += [r > 0.02]

    # Per-matrix reference: each source matrix through the single-matrix int8 path
    qkv_ref = gdn.qkv_proj.forward(x, {"reconstruct": False})
    z_ref = gdn.z_proj.forward(x, {"reconstruct": False})
    r = report(f"qkv m{m} msq vs per-matrix int8", qkv_msq, qkv_ref); failures += [r > 0.02]
    r = report(f"z   m{m} msq vs per-matrix int8", z_msq, z_ref);   failures += [r > 0.02]
    return qkv_msq, z_msq, run

qkv_msq4, z_msq4, _ = sliced_case(4)
qkv_msq, z_msq, run_sliced = sliced_case(1)
print("sliced m4 done", flush=True)

# ---------------- plain multi-matrix (MLP gate+up), m = 1 and m = 4 ----------------
print("building mgu", flush=True)
mgu = MultiLinear(DEV, [mlp.gates[0], mlp.ups[0]])
print(f"== plain mgemm (MLP gate+up): mats=2 width={mgu.out_features} K={mgu.K} mul1={mgu.mul1} mcg={mgu.mcg}")
dim = mlp.gates[0].in_features

def plain_case(m):
    global failures
    x2 = torch.randn((1, m, dim), dtype = torch.half, device = DEV) * 0.05
    guh = torch.empty((2, m, dim), dtype = torch.half, device = DEV)
    gu = torch.empty((2, m, mgu.out_features), dtype = torch.half, device = DEV)

    def run(force_shape_idx):
        gu.zero_()
        ext.exl3_mgemm(
            x2, mgu.ptrs_trellis, gu, mgu.ptrs_suh, guh, mgu.ptrs_svh,
            None, None, mgu.K, force_shape_idx, mgu.mcg, mgu.mul1, -1, -1, 0, 1, None, None)
        torch.cuda.synchronize()
        return gu.clone()

    gu_msq = run(-1)
    gu_coop = run(2)
    r = report(f"gate m{m} msq vs coop", gu_msq[0], gu_coop[0]); failures += [r > 0.02]
    r = report(f"up   m{m} msq vs coop", gu_msq[1], gu_coop[1]); failures += [r > 0.02]

    g_ref = mlp.gates[0].forward(x2, {"reconstruct": False})
    u_ref = mlp.ups[0].forward(x2, {"reconstruct": False})
    r = report(f"gate m{m} msq vs per-matrix int8", gu_msq[0], g_ref); failures += [r > 0.02]
    r = report(f"up   m{m} msq vs per-matrix int8", gu_msq[1], u_ref); failures += [r > 0.02]
    return gu_msq, run

gu_msq, run_plain = plain_case(1)
print("plain m1 done", flush=True)
gu_msq4, _ = plain_case(4)
print("plain m4 done", flush=True)

# ---------------- determinism: same call twice ----------------
print("== determinism ==")
qkv2, z2 = run_sliced(-1)
same = torch.equal(qkv_msq, qkv2) and torch.equal(z_msq, z2)
print(f"  sliced repeat identical: {same}")
failures += [not same]
gu2 = run_plain(-1)
same = torch.equal(gu_msq, gu2)
print(f"  plain repeat identical:  {same}")
failures += [not same]

print("FAIL" if any(failures) else "PASS")
sys.exit(1 if any(failures) else 0)
