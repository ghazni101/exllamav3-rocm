#!/usr/bin/env python3
"""Stress-repro for the one-time all-zeros paged_attn_triton_prefill failure (2026-10-01 review).

Observed once: 36/58 tests in test_triton_paged_hdpad.py returned rel err 1.0 (output ~ 0)
in a full-suite run; never reproduced in 6 later attempts. Suspect class: transient
launcher/arg-staging race in triton-rocm's kernel launcher (stale or zeroed pinned arg
buffer), which would produce a computed-but-wrong result exactly once.

This hammers prefill + decode with rapidly varying args and values, verifying EVERY call
against an fp32 torch reference, and additionally checks that each output differs from the
previous call's output when inputs changed (catches stale-arg reuse that happens to be
finite). Prints a compact PASS line per phase and full detail on any failure.
"""
import sys, time, random, platform
import torch

sys.path.insert(0, "/opt/exllamav3")
from exllamav3.modules.attention_fn.triton_paged import (
    paged_attn_triton_decode, paged_attn_triton_prefill,
)
from exllamav3.constants import PAGE_SIZE
import triton

device = "cuda:0"

def ref_attn(q, k, v, causal, past, window=None):
    B, Q, H, D = q.shape
    T, KVH = k.shape[1], k.shape[2]
    g = H // KVH
    kk = k.repeat_interleave(g, dim=2).float(); vv = v.repeat_interleave(g, dim=2).float()
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * D ** -0.5
    qpos = (T - Q + torch.arange(Q, device=q.device)).view(Q, 1)
    kpos = torch.arange(T, device=q.device).view(1, T)
    mask = torch.ones((Q, T), dtype=torch.bool, device=q.device)
    if causal: mask &= kpos <= qpos
    if window is not None: mask &= kpos >= qpos - window
    s = s.masked_fill(~mask.view(1, 1, Q, T), -float("inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vv)

def env_info():
    import exllamav3
    print(f"python {sys.version.split()[0]} torch {torch.__version__} hip {torch.version.hip}")
    print(f"triton {triton.__version__} device {torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    print(f"gcn {p.gcnArchName} warpsize {p.warp_size}")

def make_case(rnd, hd_choices):
    B = rnd.choice([1, 2])
    kvh = rnd.choice([1, 2, 4])
    hd = rnd.choice(hd_choices)
    q_len = rnd.choice([1, 8, 17, 130, 300, 513, 700, 900])
    past = rnd.choice([0, 40, 300, 500])
    window = rnd.choice([None, None, 64])
    T_alloc = past + q_len + rnd.randint(1, 5)
    torch.manual_seed(rnd.randint(0, 2**31 - 1))
    pages = -(-T_alloc // PAGE_SIZE)
    kc = torch.randn((B * pages, PAGE_SIZE, kvh, hd), dtype=torch.half, device=device)
    vc = torch.randn_like(kc)
    perm = torch.randperm(B * pages, device=device, dtype=torch.int32)
    bt = perm.view(B, pages)
    sl = torch.full((B,), past, dtype=torch.int32, device=device)
    k = torch.randn((B, q_len, kvh, hd), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    q = torch.randn((B, q_len, kvh * 4 if kvh < 8 else kvh, hd), dtype=torch.half, device=device)
    return B, q_len, past, window, kc, vc, bt, sl, k, v, q

def gather(kc, bt, T):
    B, pages = bt.shape
    flat = kc[bt.long().view(-1)].view(B, pages * PAGE_SIZE, kc.shape[2], kc.shape[3])
    return flat[:, :T]

def run(iters, hd_choices, seed, phase):
    rnd = random.Random(seed)
    prev_out = None
    prev_sig = None
    fails = 0
    t0 = time.time()
    for i in range(iters):
        B, q_len, past, window, kc, vc, bt, sl, k, v, q = make_case(rnd, hd_choices)
        T = past + q_len
        if q_len <= 16:
            out = paged_attn_triton_decode(q, k, v, kc, vc, bt, sl, causal=True,
                                           window_size=(window, 0) if window else None)
        else:
            out = paged_attn_triton_prefill(q, k, v, kc, vc, bt, sl, causal=True,
                                            window_size=(window, 0) if window else None)
        ref = ref_attn(q, gather(kc, bt, T), gather(vc, bt, T), True, past, window)
        rel = (out.float() - ref).abs().max().item() / ref.abs().max().item()
        sig = (tuple(out.shape), float(out.float().abs().sum()))
        if not torch.isfinite(out.float()).all() or rel > 8e-3:
            fails += 1
            print(f"[{phase}] FAIL iter {i}: rel {rel:.4f} finite={bool(torch.isfinite(out.float()).all())} "
                  f"B{B} qlen{q_len} past{past} win{window} hd{q.shape[-1]} kvh{k.shape[2]} "
                  f"outnorm {out.float().norm():.4f} refnorm {ref.norm():.4f}", flush=True)
        # stale-arg detector: same shape + near-identical signature back-to-back with
        # different random inputs would indicate a reused/zeroed arg buffer
        if prev_out is not None and sig[0] == prev_sig[0]:
            d = (out.float() - prev_out).abs().max().item()
            if d < 1e-6:
                fails += 1
                print(f"[{phase}] STALE iter {i}: identical output for different inputs "
                      f"(max diff {d:.2e})", flush=True)
        prev_out, prev_sig = out.float().clone(), sig
        if i % 200 == 0:
            print(f"[{phase}] {i}/{iters} ({time.time()-t0:.0f}s) fails={fails}", flush=True)
    print(f"[{phase}] DONE {iters} iters in {time.time()-t0:.0f}s, fails={fails}", flush=True)
    return fails

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    env_info()
    total = 0
    total += run(iters, [72, 96, 128, 160], 1234, "mixed-hd")     # shapes the tests used
    total += run(iters, [128], 5678, "pow2-only")                 # control dim, was failing too
    total += run(iters, [72, 80, 96, 112, 128, 160, 192, 256], 909, "wide-hd")
    print(f"TOTAL FAILS: {total}")
    sys.exit(1 if total else 0)
