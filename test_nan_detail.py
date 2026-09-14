#!/usr/bin/env python3
"""Detailed NaN diagnostic: check intermediate tensors in layer 0."""
import torch
from exllamav3 import Config, Model, Tokenizer

model_path = "/models/exl3/turboderp/Qwen3.8-27B-SC_4.00bpw_H5_V6"
config = Config.from_directory(model_path)
model = Model.from_config(config)
model.load(device="cuda")
tokenizer = Tokenizer.from_config(config)

ids = tokenizer.encode("Hello, how are you today? I am doing well, thanks for asking me about my day.")

def check_tensor(name, t):
    if not isinstance(t, torch.Tensor):
        print(f"  {name}: not a tensor ({type(t)})")
        return
    if not t.is_floating_point():
        print(f"  {name}: not floating point ({t.dtype})")
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        print(f"  {name}: NaN={has_nan} Inf={has_inf} shape={t.shape} *** FAIL")
    else:
        print(f"  {name}: OK shape={t.shape} range=[{t.min().item():.3f}, {t.max().item():.3f}]")

# Get layer 0 and its submodules by iterating
layer0 = None
modules_by_key = {}
for m in model:
    key = getattr(m, "key", type(m).__name__)
    modules_by_key[key] = m
    if key == "model.language_model.layers.0":
        layer0 = m

if layer0 is None:
    print("ERROR: Could not find layer 0")
    exit(1)

print(f"Layer 0: {type(layer0).__name__}")
print(f"Layer 0 attrs: {[a for a in dir(layer0) if not a.startswith('_')]}")

# List all submodules of layer 0
for m in model:
    key = getattr(m, "key", type(m).__name__)
    if key.startswith("model.language_model.layers.0"):
        print(f"  submodule: {key} ({type(m).__name__})")

for seq_len in [1, 3, 4, 7]:
    print(f"\n=== M={seq_len} ===")
    input_ids = ids[:, :seq_len].to("cuda")

    # Hook all submodules of layer 0
    hooks = {}
    for m in model:
        key = getattr(m, "key", type(m).__name__)
        if key.startswith("model.language_model.layers.0"):
            orig = m.forward
            def make_hook(k, fn):
                def hooked(*a, **kw):
                    out = fn(*a, **kw)
                    check_tensor(k, out)
                    return out
                return hooked
            hooks[key] = (m, orig, make_hook(key, orig))
            m.forward = hooks[key][2]

    with torch.no_grad():
        logits = model.forward(input_ids=input_ids, params={})
    torch.cuda.synchronize()

    # Restore
    for key, (m, orig, _) in hooks.items():
        m.forward = orig

    has_nan = torch.isnan(logits).any().item()
    print(f"  logits: NaN={has_nan}")
    if not has_nan:
        print(f"  logits range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")
