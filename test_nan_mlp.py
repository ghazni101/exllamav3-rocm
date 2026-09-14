#!/usr/bin/env python3
"""Granular MLP NaN diagnostic: hook individual GEMM calls inside MLP."""
import torch
from exllamav3 import Config, Model, Tokenizer

model_path = "/models/exl3/turboderp/Qwen3.8-27B-SC_4.00bpw_H5_V6"
config = Config.from_directory(model_path)
model = Model.from_config(config)
model.load(device="cuda")
tokenizer = Tokenizer.from_config(config)

ids = tokenizer.encode("Hello, how are you today? I am doing well, thanks for asking me about my day.")

def check_tensor(name, t):
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if has_nan or has_inf:
        print(f"    {name}: NaN={has_nan} Inf={has_inf} shape={t.shape} *** FAIL")
    else:
        print(f"    {name}: OK shape={t.shape} range=[{t.min().item():.3f}, {t.max().item():.3f}]")

# Get layer 0 MLP
layer0 = None
for m in model:
    key = getattr(m, "key", "")
    if key == "model.language_model.layers.0":
        layer0 = m
        break

mlp = layer0.mlp
print(f"MLP type: {type(mlp).__name__}")
print(f"MLP attrs: {[a for a in dir(mlp) if not a.startswith('_') and not callable(getattr(mlp, a, None))]}")

# Check if mgemm is used
uses_mgemm = hasattr(mlp, 'inner') and hasattr(mlp.inner, 'gu_ptrs_trellis') and mlp.inner.gu_ptrs_trellis is not None
if hasattr(mlp, 'inner'):
    print(f"MLP inner: {type(mlp.inner).__name__}")
    print(f"  gu_ptrs_trellis: {getattr(mlp.inner, 'gu_ptrs_trellis', 'N/A')}")
else:
    print("No inner attr")

for seq_len in [1, 3, 4]:
    print(f"\n=== M={seq_len} ===")
    input_ids = ids[:, :seq_len].to("cuda")
    
    # Hook MLP submodules
    hooks = {}
    for m in model:
        key = getattr(m, "key", "")
        if "layers.0.mlp" in key:
            orig = m.forward
            def make_hook(k, fn):
                def hooked(*a, **kw):
                    out = fn(*a, **kw)
                    check_tensor(k, out)
                    return out
                return hooked
            hooks[key] = (m, orig)
            m.forward = make_hook(key, orig)
    
    # Also hook the MLP forward to check x and d
    orig_mlp_fwd = mlp.forward
    def mlp_hook(*a, **kw):
        x = a[0] if a else kw.get('x')
        if isinstance(x, torch.Tensor):
            check_tensor("mlp_input_x", x)
        out = orig_mlp_fwd(*a, **kw)
        return out
    mlp.forward = mlp_hook
    
    with torch.no_grad():
        logits = model.forward(input_ids=input_ids, params={})
    torch.cuda.synchronize()
    
    # Restore
    for key, (m, orig) in hooks.items():
        m.forward = orig
    mlp.forward = orig_mlp_fwd
    
    has_nan = torch.isnan(logits).any().item()
    print(f"  logits: NaN={has_nan}")
