#!/usr/bin/env python3
"""Cross-arm logits comparison for the numcheck captures (`correctness_gate.py numcheck save`).

The in-tree numcheck compares within one binary (bit-reproducibility). A change that alters the
numeric path - e.g. routing the prefill GEMM through a different library - needs the *cross-arm*
KLD instead, which is what the project's numeric rule uses (<= 4e-3 for the accepted mgemm delta,
findings-log 10.3). Logits carry uninitialized channels beyond actual_vocab_size in this model, so
both arms are trimmed to the shortest common width before the softmax.

Usage: numcheck_diff.py <ref.pt> <new.pt>
"""
import os
import sys

import torch

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")


def vocab_width():
    """channels >= actual_vocab_size are uninitialized in returned logits (findings-log 11.4), so
    both arms are trimmed to it - the same trim the gate itself applies."""
    try:
        from exllamav3 import Config, Tokenizer
        return Tokenizer.from_config(Config.from_directory(MODEL_DIR)).actual_vocab_size
    except Exception as e:                                                      # noqa: BLE001
        print(f"(tokenizer unavailable: {e}; trimming to the common width)")
        return None


W = vocab_width()


def trim(t):
    return t[:, :W] if W else t[:, :t.shape[-1]]


def main():
    ref = torch.load(sys.argv[1], weights_only=False)
    new = torch.load(sys.argv[2], weights_only=False)
    rt, rl = ref["tokens"], ref["logits"]
    nt, nl = new["tokens"], new["logits"]
    worst = 0.0
    bad = []
    for n, lg in nl.items():
        if n not in rl:
            print(f"{n}: not in reference, skipped")
            continue
        r = trim(rl[n].float().double())
        x = trim(lg.float().double())
        k = min(r.shape[0], x.shape[0])
        r, x = r[:k], x[:k]
        if not (torch.isfinite(r).all() and torch.isfinite(x).all()):
            print(f"{n}: NON-FINITE LOGITS")
            bad.append(n)
            continue
        lp_r = torch.log_softmax(r, dim=-1)
        lp_x = torch.log_softmax(x, dim=-1)
        p_r = lp_r.exp()
        kld = (p_r * (lp_r - lp_x)).where(p_r > 0, torch.zeros_like(p_r)).sum(-1)
        mk = kld.max().item()
        worst = max(worst, mk)
        div = [i for i in range(k)
               if nt[n][len(nt[n]) - k + i] != rt[n][len(rt[n]) - k + i]]
        flag = "OK" if mk <= 4e-3 else "OVER"
        print(f"{n:16s} max KLD {mk:.3e}  divergent greedy steps {len(div)} {div[:6]}  {flag}")
        if mk > 4e-3:
            bad.append(n)
    print(f"worst KLD {worst:.3e} over {len(nl)} prompts; "
          f"{'PASS' if worst <= 4e-3 else 'FAIL'} (threshold 4e-3); bad: {bad}")


main()
