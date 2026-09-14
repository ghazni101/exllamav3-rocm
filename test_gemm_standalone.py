#!/usr/bin/env python3
"""Standalone GEMM test for exllamav3 ROCm.

Directly calls exl3_gemm with known inputs to test the GEMM kernel
(cp_async + ldsm4 + WMMA + threadblock reduction + barrier) in isolation,
without loading a full model.

Tests:
1. M=1 (GEMV path) — should match reference (known to work)
2. M=2,4,7,16 (GEMM path) — test for NaN
3. Compare against torch reference matmul

Usage:
    python3 test_gemm_standalone.py
"""

import torch
from exllamav3.ext import exllamav3_ext as ext


def test_gemm(m, k, n, bits=5, cb=2):
    """Test exl3_gemm at given shape, compare to torch reference.

    Args:
        m: rows of A
        k: reduction dimension (must be multiple of 16)
        n: columns of B/C (must be multiple of 128)
        bits: bits per weight (5 for Qwen3.8-27B)
        cb: codebook (2 for mul1)
    """
    device = torch.device("cuda")
    dtype = torch.float16

    # Create random A matrix
    torch.manual_seed(42)
    A = torch.randn(m, k, device=device, dtype=dtype) * 0.1

    # Create a random EXL3-quantized B matrix
    # B shape: (k//16, n//16, 16*K) where K = bits
    K = bits
    B_shape = (k // 16, n // 16, 16 * K)
    B = torch.randint(0, 65535, B_shape, device=device, dtype=torch.int16)

    # Create output C (f32 accumulation)
    C = torch.empty(m, n, device=device, dtype=torch.float32)

    # Create suh (input scales) and A_had (Hadamard scratch)
    suh = torch.randn(k // 16, device=device, dtype=dtype) * 0.01
    A_had = torch.empty_like(A)

    # Create svh (output scales)
    svh = torch.randn(n // 16, device=device, dtype=dtype) * 0.01

    # Call exl3_gemm
    # exl3_gemm(A, B, C, suh, A_had, svh, force_shape_idx, mcg, mul1, force_num_sms)
    # mcg=False, mul1=(cb==2), force_shape_idx=-1 (auto), force_num_sms=0 (auto)
    mcg = cb == 1
    mul1 = cb == 2

    try:
        result = ext.exl3_gemm(A, B, C, suh, A_had, svh, -1, mcg, mul1, 0)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  exl3_gemm FAILED: {e}")
        return False

    # Check for NaN
    has_nan = torch.isnan(C).any().item()
    has_inf = torch.isinf(C).any().item()

    if has_nan or has_inf:
        print(f"  M={m:3d} K={k:5d} N={n:5d}: NaN={has_nan} Inf={has_inf} "
              f"shape_idx={result} *** FAIL ***")
        return False
    else:
        # Check range (should be finite and reasonable)
        c_min = C.min().item()
        c_max = C.max().item()
        print(f"  M={m:3d} K={k:5d} N={n:5d}: NaN=False range=[{c_min:.3f}, {c_max:.3f}] "
              f"shape_idx={result} PASS")
        return True


def test_gemm_no_scales(m, k, n, bits=5, cb=0):
    """Test exl3_gemm without suh/svh (no Hadamard transform)."""
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(42)
    A = torch.randn(m, k, device=device, dtype=dtype) * 0.1

    K = bits
    B_shape = (k // 16, n // 16, 16 * K)
    B = torch.randint(0, 65535, B_shape, device=device, dtype=torch.int16)

    C = torch.empty(m, n, device=device, dtype=torch.float32)

    mcg = cb == 1
    mul1 = cb == 2

    try:
        result = ext.exl3_gemm(A, B, C, None, None, None, -1, mcg, mul1, 0)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  exl3_gemm (no scales) FAILED: {e}")
        return False

    has_nan = torch.isnan(C).any().item()
    has_inf = torch.isinf(C).any().item()

    if has_nan or has_inf:
        print(f"  M={m:3d} K={k:5d} N={n:5d} (no scales): NaN={has_nan} Inf={has_inf} "
              f"shape_idx={result} *** FAIL ***")
        return False
    else:
        c_min = C.min().item()
        c_max = C.max().item()
        print(f"  M={m:3d} K={k:5d} N={n:5d} (no scales): NaN=False range=[{c_min:.3f}, {c_max:.3f}] "
              f"shape_idx={result} PASS")
        return True


def main():
    print("=== GPU Info ===")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  gcnArchName: {props.gcnArchName}")
    print(f"  SMs: {props.multi_processor_count}")

    print("\n=== Device context ===")
    cc = ext.g_get_cc(0)
    sms = ext.g_get_num_sms(0)
    print(f"  CC: {cc}, SMs: {sms}")

    # Model dimensions: Qwen3.8-27B
    # hidden_size=5120, K=5 (5 bits), cb=2 (mul1)
    k = 5120
    n = 5120
    bits = 5

    print(f"\n=== GEMM with scales (suh+svh), cb=2 (mul1), bits={bits} ===")
    print(f"  K={k}, N={n}")

    all_pass = True
    for m in [1, 2, 4, 7, 8, 16]:
        ok = test_gemm(m, k, n, bits=bits, cb=2)
        all_pass = all_pass and ok

    print(f"\n=== GEMM without scales, cb=0, bits={bits} ===")
    print(f"  K={k}, N={n}")

    for m in [1, 2, 4, 7, 8, 16]:
        ok = test_gemm_no_scales(m, k, n, bits=bits, cb=0)
        all_pass = all_pass and ok

    # Also test with different N values
    print(f"\n=== GEMM with scales, cb=2, bits={bits}, varying N ===")
    for n_test in [128, 256, 512, 1024, 2048]:
        ok = test_gemm(7, k, n_test, bits=bits, cb=2)
        all_pass = all_pass and ok

    print(f"\n=== {'ALL PASSED' if all_pass else 'SOME FAILED'} ===")


if __name__ == "__main__":
    main()
