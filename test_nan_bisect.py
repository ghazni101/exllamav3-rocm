#!/usr/bin/env python3
"""NaN bisection diagnostic for exllamav3 ROCm.

Hooks every module's forward pass to detect the first layer producing NaN.
Tests 1-token (GEMV path) vs multi-token (GEMM path) to confirm the bug is
sequence-length dependent.

Usage:
    python3 test_nan_bisect.py [--model PATH] [--seq-len N]
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/models/exl3/turboderp/Qwen3.8-27B-SC_4.00bpw_H5_V6",
    )
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    from exllamav3 import Cache, Config, Model, Tokenizer

    config = Config.from_directory(args.model)
    config.max_seq_len = args.max_seq_len
    model = Model.from_config(config)
    print("Loading model...")
    model.load()
    print("Model loaded.")
    tokenizer = Tokenizer.from_config(config)

    # Hook every module's forward to check for NaN
    nan_found = [False]
    first_nan_module = [None]

    def make_hook(name, module):
        original_forward = module.forward

        def hooked_forward(*a, **kw):
            out = original_forward(*a, **kw)
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                has_nan = torch.isnan(out).any().item()
                has_inf = torch.isinf(out).any().item()
                if (has_nan or has_inf) and not nan_found[0]:
                    nan_found[0] = True
                    first_nan_module[0] = name
                    print(
                        f"  [FIRST NaN/Inf] {name}: shape={out.shape}, "
                        f"NaN={has_nan}, Inf={has_inf}, "
                        f"range=[{out.min().item() if not has_nan else 'nan'}, "
                        f"{out.max().item() if not has_nan else 'nan'}]"
                    )
                elif has_nan or has_inf:
                    print(f"  [NaN/Inf] {name}: shape={out.shape}")
            return out

        module.forward = hooked_forward

    # Install hooks on all modules
    for name, module in model.modules.named_modules():
        make_hook(name, module)

    # Test 1: single token (GEMV path)
    print("\n=== Test 1: single token (GEMV path) ===")
    nan_found[0] = False
    first_nan_module[0] = None
    ids = tokenizer.encode("Hello")
    if isinstance(ids, torch.Tensor):
        input_ids = ids.cuda()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    print(f"  input_ids shape: {input_ids.shape}")

    cache1 = Cache(model, max_num_tokens=args.max_seq_len)
    with torch.no_grad():
        logits = model.forward(input_ids, cache=cache1)
    torch.cuda.synchronize()
    has_nan = torch.isnan(logits).any().item()
    print(f"  1 token: NaN={has_nan}", end="")
    if not has_nan:
        print(f", range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")
    else:
        print()
    if first_nan_module[0]:
        print(f"  First NaN at: {first_nan_module[0]}")

    # Test 2: multi-token (GEMM path)
    for seq_len in [2, 4, 7]:
        if seq_len > args.seq_len:
            break
        print(f"\n=== Test 2: {seq_len} tokens (GEMM path) ===")
        nan_found[0] = False
        first_nan_module[0] = None
        text = "Hello, how are you today?"[: seq_len * 3]
        ids = tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            input_ids = ids.cuda()
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        # Trim to exact seq_len
        if input_ids.shape[1] > seq_len:
            input_ids = input_ids[:, :seq_len]
        print(f"  input_ids shape: {input_ids.shape}")

        cache_n = Cache(model, max_num_tokens=args.max_seq_len)
        with torch.no_grad():
            logits = model.forward(input_ids, cache=cache_n)
        torch.cuda.synchronize()
        has_nan = torch.isnan(logits).any().item()
        print(f"  {seq_len} tokens: NaN={has_nan}", end="")
        if not has_nan:
            print(f", range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")
        else:
            print()
        if first_nan_module[0]:
            print(f"  First NaN at: {first_nan_module[0]}")
        else:
            print("  No NaN detected in intermediate activations.")

    print("\nDone.")


if __name__ == "__main__":
    main()
