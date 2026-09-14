#!/usr/bin/env python3
"""Standalone mgemm test: call exl3_mgemm directly to isolate from model."""
import torch
from exllamav3.ext import exllamav3_ext as ext

device = torch.device("cuda")
dtype = torch.float16
k = 5120
n = 5120  # gate+up combined width
bits = 5

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CC: {ext.g_get_cc(0)}, SMs: {ext.g_get_num_sms(0)}")

# mgemm needs: A [1, M, K], B [ptr_list], C [2, M, N], suh [ptr_list], A_had [2, M, K], svh [ptr_list]
# For the MLP gate+up fused: bszm_in=1, bszm_out=2

for m in [1, 2, 3, 4, 7, 8]:
    torch.manual_seed(42)
    A = torch.randn(1, m, k, device=device, dtype=dtype) * 0.1
    # B is a pointer table (int64) of 2 entries
    B1 = torch.randint(0, 32767, (k // 16, n // 16, 16 * bits), device=device, dtype=torch.int16)
    B2 = torch.randint(0, 32767, (k // 16, n // 16, 16 * bits), device=device, dtype=torch.int16)
    B_ptrs = torch.tensor([B1.data_ptr(), B2.data_ptr()], device=device, dtype=torch.int64)
    
    C = torch.empty(2, m, n, device=device, dtype=dtype)
    
    suh = torch.randn(k, device=device, dtype=dtype) * 0.01
    suh_ptrs = torch.tensor([suh.data_ptr(), suh.data_ptr()], device=device, dtype=torch.int64)
    A_had = torch.empty(2, m, k, device=device, dtype=dtype)
    
    svh = torch.randn(n, device=device, dtype=dtype) * 0.01
    svh_ptrs = torch.tensor([svh.data_ptr(), svh.data_ptr()], device=device, dtype=torch.int64)
    
    C.zero_()
    try:
        result = ext.exl3_mgemm(
            A, B_ptrs, C, suh_ptrs, A_had, svh_ptrs,
            None, None, bits, -1, False, True, -1, -1, 0, 1, None, None
        )
        torch.cuda.synchronize()
        has_nan = torch.isnan(C).any().item()
        if has_nan:
            print(f"  M={m:3d}: NaN=True *** FAIL (shape_idx={result})")
        else:
            print(f"  M={m:3d}: NaN=False range=[{C.min().item():.3f}, {C.max().item():.3f}] OK (shape_idx={result})")
    except Exception as e:
        print(f"  M={m:3d}: ERROR: {e}")
