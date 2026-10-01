#!/usr/bin/env python3
"""Adversarial EXL3 kernel fuzz v2 (chunked fp32 reference; per-dispatch-family row consistency).

Findings from v1 informed the checks:
 - run_alloc dispatches m<=4 -> int8 sq GEMV (rel ~8e-3), m>=5 -> fp16 reconstruct path
   (rel ~6e-4). Both families must be INTERNALLY batch-size independent for a fixed row.
Checks: numeric error vs fp32 reference (chunked matmul), per-row consistency across m
within each dispatch family, determinism, fp32-out variant, fused vs plain reconstruct.
"""
import json, sys
import torch, safetensors
from exllamav3.model.config import Config
from exllamav3.modules.quant.exl3 import LinearEXL3

MODEL = "/models/qwen38-27b"
DEV = "cuda"
torch.manual_seed(1234)

M_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 31, 32, 33, 63, 64, 100, 127, 128,
           143, 144, 145, 200, 300, 511, 512, 1024, 1025, 2048]

failures = []
notes = []

def rel_rms(a, b):
    d = (a.double() - b.double())
    den = b.double().square().mean().clamp_min(1e-12)
    return float((d.square().mean() / den).sqrt())

def max_abs(a, b):
    return float((a.double() - b.double()).abs().max())

def ref_matmul(x, W, chunk=256):
    """fp32 reference, row-chunked to bound memory."""
    m = x.shape[0]
    out = torch.empty((m, W.shape[1]), dtype=torch.float32, device=x.device)
    for i in range(0, m, chunk):
        out[i:i+chunk] = x[i:i+chunk].float() @ W.float()
    return out

def want_names(w):
    return [k.rsplit('.', 1)[1] for k in w]

def load_prefix(prefix, cfg):
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    want = {}
    for n in ["suh", "svh", "trellis", "mul1", "mcg", "bias"]:
        k = f"{prefix}.{n}"
        if k in wm:
            want[k] = wm[k]
    assert "trellis" in want_names(want), f"{prefix}: no trellis"
    shards = sorted(set(want.values()))
    tensors = {}
    for s in shards:
        with safetensors.safe_open(f"{MODEL}/{s}", framework="pt", device=DEV) as f:
            for k in want:
                if wm[k] == s:
                    tensors[k.rsplit('.', 1)[1]] = f.get_tensor(k)
    suh, svh, trellis = tensors["suh"], tensors["svh"], tensors["trellis"]
    return LinearEXL3(
        config=cfg, in_features=suh.numel(), out_features=svh.numel(),
        suh=suh, svh=svh, trellis=trellis,
        mcg=tensors.get("mcg"), mul1=tensors.get("mul1"),
        bias=tensors.get("bias"), key=prefix,
    )

def check(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"  FAIL: {msg}", flush=True)

def main():
    cfg = Config.from_directory(MODEL)
    prefixes = [
        "model.language_model.layers.27.mlp.up_proj",
        "model.language_model.layers.27.self_attn.k_proj",
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.linear_attn.out_proj",
        "lm_head",
    ]
    for prefix in prefixes:
        try:
            mod = load_prefix(prefix, cfg)
        except Exception as e:
            notes.append(f"{prefix}: skipped ({e})")
            continue
        print(f"[{prefix}] in={mod.in_features} out={mod.out_features} K={mod.K} "
              f"mcg={mod.mcg} mul1={mod.mul1}", flush=True)
        W = mod.get_weight_tensor()
        check(bool(torch.isfinite(W.float()).all()), f"{prefix}: W non-finite")
        rows_bank = torch.randn(2048, mod.in_features, device=DEV, dtype=torch.half) * 0.5

        base_row3 = {}
        for m in M_SWEEP:
            x = rows_bank[:m].contiguous()
            y_gemv = mod.bc.run_alloc(x, mod.out_features, False)
            y_gemv32 = mod.bc.run_alloc(x, mod.out_features, True)
            y_rec = mod.reconstruct_hgemm(x, None) if m >= 5 else None
            ref = ref_matmul(x, W)

            check(y_gemv.dtype == torch.half, f"{prefix} m={m}: gemv dtype {y_gemv.dtype}")
            check(y_gemv32.dtype == torch.float32, f"{prefix} m={m}: gemv32 dtype {y_gemv32.dtype}")
            check(bool(torch.isfinite(y_gemv.float()).all()), f"{prefix} m={m}: gemv non-finite")
            if y_rec is not None:
                check(bool(torch.isfinite(y_rec.float()).all()), f"{prefix} m={m}: rec non-finite")
            e_gemv = rel_rms(y_gemv, ref)
            e_32 = rel_rms(y_gemv32, ref)
            line = f"  m={m:5d}  gemv {e_gemv:.4f}  gemv32 {e_32:.4f}"
            check(e_gemv < 0.05, f"{prefix} m={m}: gemv rel {e_gemv:.4f}")
            check(e_32 < 0.05, f"{prefix} m={m}: gemv32 rel {e_32:.4f}")

            # determinism within a family
            y2 = mod.bc.run_alloc(x, mod.out_features, False)
            check(bool((y2 == y_gemv).all()), f"{prefix} m={m}: gemv nondeterministic")

            # row-3 batch-independence within dispatch family (m<=4 int8; m>=5 fp16).
            # The family baseline MUST come from a batched run of that family: a
            # single-row run dispatches to the int8 path, so using it as the fp16
            # baseline measures the int8-vs-fp16 dispatch gap (~0.1 abs on out_proj)
            # and false-fails.
            fam = "int8" if m <= 4 else "fp16"
            if m > 3:
                if fam not in base_row3:
                    base_row3[fam] = y_gemv[3].float().clone()
                else:
                    d = max_abs(y_gemv[3].float(), base_row3[fam])
                    line += f"  row3[{fam}] {d:.5f}"
                    check(d < 0.05, f"{prefix} m={m}: row3 differs within {fam} family: {d:.4f}")

            if y_rec is not None and m >= 5:
                er = rel_rms(y_rec, ref)
                line += f"  rec {er:.4f}"
                check(er < 0.02, f"{prefix} m={m}: rec rel {er:.4f}")
                if "rec_row3" not in base_row3:
                    base_row3["rec_row3"] = y_rec[3].float() if m > 3 else None
                elif m > 3:
                    d2 = max_abs(y_rec[3].float(), base_row3["rec_row3"])
                    check(d2 < 0.05, f"{prefix} m={m}: rec row3 batch-dependent {d2:.4f}")
            print(line, flush=True)
            del ref, y_gemv, y_gemv32, y_rec

        # fused vs plain reconstruct at m=2048
        mod._fused_reconstruct = False
        y_nf = mod.reconstruct_hgemm(rows_bank[:1024].contiguous(), None)
        mod._fused_reconstruct = True
        y_f = mod.reconstruct_hgemm(rows_bank[:1024].contiguous(), None)
        d = max_abs(y_nf, y_f)
        ref = ref_matmul(rows_bank[:1024], W)
        e_nf, e_f = rel_rms(y_nf, ref), rel_rms(y_f, ref)
        check(e_nf < 0.02 and e_f < 0.02,
              f"{prefix}: reconstruct rel nf {e_nf:.4f} f {e_f:.4f}")
        print(f"  fused-vs-plain @1024: max abs {d:.5f}  (nf rel {e_nf:.4f}, f rel {e_f:.4f})", flush=True)

        del W, mod, ref, y_nf, y_f, rows_bank
        torch.cuda.empty_cache()

    print("\n==== SUMMARY ====")
    for n in notes:
        print(n)
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f_ in failures:
            print(" -", f_)
        sys.exit(1)
    print("ALL FUZZ CHECKS PASSED")

if __name__ == "__main__":
    main()
