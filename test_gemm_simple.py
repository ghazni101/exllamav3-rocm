#!/usr/bin/env python3
"""Simple GEMM test: call exl3_gemm with force_num_sms=1 to avoid cooperative launch."""
import torch
from exllamav3.ext import exllamav3_ext as ext

device = torch.device("cuda")
dtype = torch.float16
k = 5120
n = 5120
bits = 5

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CC: {ext.g_get_cc(0)}, SMs: {ext.g_get_num_sms(0)}")

for m in [1, 2, 3, 4, 7, 8, 16]:
    torch.manual_seed(42)
    A = torch.randn(m, k, device=device, dtype=dtype) * 0.1
    B = torch.randint(0, 32767, (k // 16, n // 16, 16 * bits), device=device, dtype=torch.int16)
    C = torch.empty(m, n, device=device, dtype=torch.float32)
    suh = torch.randn(k, device=device, dtype=dtype) * 0.01
    A_had = torch.empty_like(A)
    svh = torch.randn(n, device=device, dtype=dtype) * 0.01

    for num_sms in [1, 0]:
        C.zero_()
        try:
            result = ext.exl3_gemm(A, B, C, suh, A_had, svh, -1, False, True, num_sms)
            torch.cuda.synchronize()
            has_nan = torch.isnan(C).any().item()
            if has_nan:
                print(f"  M={m:3d} sms={num_sms}: NaN=True *** FAIL")
            else:
                print(f"  M={m:3d} sms={num_sms}: NaN=False range=[{C.min().item():.3f}, {C.max().item():.3f}] OK")
        except Exception as e:
            print(f"  M={m:3d} sms={num_sms}: ERROR: {e}")
