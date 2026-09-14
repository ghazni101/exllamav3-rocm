#!/usr/bin/env python3
"""Test: disable MoE coop fused path (mlp.bc=None) to isolate NaN source."""
import torch
from exllamav3 import Config, Model, Tokenizer

model_path = "/models/exl3/turboderp/Qwen3.8-27B-SC_4.00bpw_H5_V6"
config = Config.from_directory(model_path)
model = Model.from_config(config)
model.load(device="cuda")
tokenizer = Tokenizer.from_config(config)

# Disable the fused MoE coop path on all layers
count = 0
for m in model:
    if type(m).__name__ == "BlockSparseMLP" and getattr(m, "bc", None) is not None:
        m.bc = None
        count += 1
print(f"Disabled bc on {count} BlockSparseMLP modules")

ids = tokenizer.encode("Hello, how are you today? I am doing well, thanks for asking me about my day.")
print(f"Total tokens: {ids.shape[1]}")

for seq_len in [1, 3, 4, 7]:
    input_ids = ids[:, :seq_len].to("cuda")
    with torch.no_grad():
        logits = model.forward(input_ids=input_ids, params={})
    torch.cuda.synchronize()
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    if has_nan or has_inf:
        print(f"  M={seq_len}: NaN={has_nan} Inf={has_inf} *** FAIL")
    else:
        print(f"  M={seq_len}: NaN=False, range=[{logits.min().item():.3f}, {logits.max().item():.3f}]  OK")
