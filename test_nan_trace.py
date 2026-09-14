#!/usr/bin/env python3
"""Instrument ext.exl3_mgemm with tracebacks to find NaN source."""
import traceback
import torch
from exllamav3 import Config, Model, Tokenizer
from exllamav3.ext import exllamav3_ext as ext

_orig_mgemm = ext.exl3_mgemm
_orig_gemm = ext.exl3_gemm
_call_count = 0

def check(name, t):
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        print(f"    {name}: NaN={has_nan} Inf={has_inf} shape={t.shape} *** FAIL")
    else:
        print(f"    {name}: OK shape={t.shape}")

def patched_mgemm(A, B, C, suh, A_had, svh, indices, weights, K, *args, **kwargs):
    global _call_count
    _call_count += 1
    caller = traceback.extract_stack(limit=8)[-2]
    print(f"  [mgemm#{_call_count}] A={A.shape} C={C.shape} K={K} from {caller.filename}:{caller.lineno} {caller.name}")
    check(f"  A", A)
    result = _orig_mgemm(A, B, C, suh, A_had, svh, indices, weights, K, *args, **kwargs)
    torch.cuda.synchronize()
    check(f"  C", C)
    return result

def patched_gemm(A, B, C, suh, A_had, svh, *args, **kwargs):
    global _call_count
    _call_count += 1
    caller = traceback.extract_stack(limit=8)[-2]
    print(f"  [gemm#{_call_count}] A={A.shape} C={C.shape} from {caller.filename}:{caller.lineno} {caller.name}")
    check(f"  A", A)
    result = _orig_gemm(A, B, C, suh, A_had, svh, *args, **kwargs)
    torch.cuda.synchronize()
    check(f"  C", C)
    return result

ext.exl3_mgemm = patched_mgemm
ext.exl3_gemm = patched_gemm

model_path = "/models/exl3/turboderp/Qwen3.8-27B-SC_4.00bpw_H5_V6"
config = Config.from_directory(model_path)
model = Model.from_config(config)
model.load(device="cuda")
tokenizer = Tokenizer.from_config(config)

ids = tokenizer.encode("Hello, how are you today? I am doing well, thanks for asking me about my day.")

for seq_len in [1, 3]:
    print(f"\n=== M={seq_len} ===")
    _call_count = 0
    input_ids = ids[:, :seq_len].to("cuda")
    with torch.no_grad():
        logits = model.forward(input_ids=input_ids, params={})
    torch.cuda.synchronize()
    has_nan = torch.isnan(logits).any().item()
    print(f"  logits: NaN={has_nan}")
