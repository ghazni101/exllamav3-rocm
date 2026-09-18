"""Isolated probe of _paged_attn_decode_split_kernel on the target Qwen3.8 shape.

Shape: bsz=1, q_len=1, n_q_heads=24, n_kv_heads=4, head_dim=256, seq len ~4224.
Arms sweep num_warps x block_n x num_splits, fp16 caches, one process per arm value
set (fresh python), reporting per-call ms and split-kernel scratch/vgpr from
rocprofv3 --kernel-trace (optional; used by the caller when --trace is set).
"""
import argparse

import torch
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode

torch.manual_seed(0)

BSZ, Q_LEN, NQ, NKV, HD = 1, 1, 24, 4, 256
SEQ = 4100
PAGE = 256
GROUP = NQ // NKV
HD_PAD = 256
N_PAGES = (SEQ + PAGE - 1) // PAGE + 4

ap = argparse.ArgumentParser()
ap.add_argument("--warps", type=int, default=4)
ap.add_argument("--block-n", type=int, default=32)
ap.add_argument("--splits", type=int, default=48)
ap.add_argument("--reps", type=int, default=50)
args = ap.parse_args()

dev = torch.device("cuda:0")
q = torch.randn((BSZ, Q_LEN, NQ, HD), dtype=torch.float16, device=dev) * 0.1
k_cache = torch.randn((BSZ, N_PAGES * PAGE, NKV, HD), dtype=torch.float16, device=dev) * 0.05
v_cache = torch.randn_like(k_cache)
bt = torch.arange(N_PAGES, dtype=torch.int32, device=dev).repeat(BSZ, 1)
sl = torch.full((BSZ,), SEQ, dtype=torch.int32, device=dev)
k_new = torch.randn((BSZ, 1, NKV, HD), dtype=torch.float16, device=dev) * 0.1
v_new = torch.randn_like(k_new)

# Dense reference on the full cache slice for correctness of the paged path
kt = k_cache[0, :SEQ]          # [SEQ, NKV, HD]
vt = v_cache[0, :SEQ]
kt = kt.permute(1, 0, 2)       # [NKV, SEQ, HD]
vt = vt.permute(1, 0, 2)
qs = q[0, 0].view(NKV, GROUP, HD)
scores = torch.einsum("ghd,gsd->ghs", qs.float(), kt.float()) / HD**0.5
ref = torch.einsum("ghs,gsd->ghd", scores.softmax(-1), vt.float())
ref = ref.reshape(1, 1, NQ, HD).to(torch.float16)


def call():
    return paged_attn_triton_decode(
        q=q, k=k_new, v=v_new,
        k_cache=k_cache, v_cache=v_cache,
        block_table=bt, cache_seqlens=sl,
        causal=True, block_n=args.block_n, num_splits=args.splits,
        num_warps=args.warps,
    )


out = call()
torch.cuda.synchronize()
err = (out.float() - ref.float()).abs().max().item()
rel = err / ref.float().abs().max().item()

# Timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(5):
    call()
torch.cuda.synchronize()
start.record()
for _ in range(args.reps):
    call()
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / args.reps
print(f"RESULT warps={args.warps} block_n={args.block_n} splits={args.splits} "
      f"ms_per_call={ms:.4f} max_abs_err={err:.3e} rel_err={rel:.3e}")
