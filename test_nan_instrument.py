#!/usr/bin/env python3
"""Instrument ext.exl3_mgemm and ext.exl3_gemm to find NaN source in MLP."""
import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.ext import exllamav3_ext as ext

# Save originals
_orig_mgemm = ext.exl3_mgemm
_orig_gemm = ext.exl3_gemm

def check(name, t):
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        print(f"    {name}: NaN={has_nan} Inf={has_inf} shape={t.shape} *** FAIL")
    else:
        print(f"    {name}: OK shape={t.shape} range=[{t.min().item():.3f}, {t.max().item():.3f}]")

def patched_mgemm(A, B, C, suh, A_had, svh, indices, weights, K, *args, **kwargs):
    print(f"  [mgemm] A={A.shape} C={C.shape} K={K}")
    check("  mgemm A", A)
    result = _orig_mgemm(A, B, C, suh, A_had, svh, indices, weights, K, *args, **kwargs)
    torch.cuda.synchronize()
    check("  mgemm C", C)
    return result

def patched_gemm(A, B, C, suh, A_had, svh, *args, **kwargs):
    print(f"  [gemm] A={A.shape} C={C.shape}")
    check("  gemm A", A)
    result = _orig_gemm(A, B, C, suh, A_had, svh, *args, **kwargs)
    torch.cuda.synchronize()
    check("  gemm C", C)
    return result

ext.exl3_mgemm = patched_mgemm
ext.exl3_gemm = patched_gemm

model_path = "/models/exl3/turboderp/Qwen3.8-27B-SC_4.00bpw_H5_V6"
config = Config.from_directory(model_path)
model = Model.from_config(config)
model.load(device="cuda")
tokenizer = Tokenizer.from_config(config)

ids = tokenizer.encode("Hello, how are you today? I am doing well, thanks for asking me about my day.")

for seq_len in [1, 3, 4]:
    print(f"\n=== M={seq_len} ===")
    input_ids = ids[:, :seq_len].to("cuda")
    with torch.no_grad():
        logits = model.forward(input_ids=input_ids, params={})
    torch.cuda.synchronize()
    has_nan = torch.isnan(logits).any().item()
    print(f"  logits: NaN={has_nan}")
