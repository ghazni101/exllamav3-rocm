#!/usr/bin/env python3
"""Adversarial hgemm fuzz v2: odd/random shapes for plain hgemm (2-D, any shape);
K%64==0/N%128==0 shapes for hgemm_f16acc (batched); padding canaries; fp64 ref."""
import sys
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.manual_seed(777)
DEV = "cuda"
failures = []

def check(cond, msg):
    if not cond:
        failures.append(msg)
        print(f"FAIL: {msg}", flush=True)

def rel_rms(a, b):
    d = (a.double() - b.double())
    den = b.double().square().mean().clamp_min(1e-12)
    return float((d.square().mean() / den).sqrt())

def run_case(M, N, K, batch=1, use_f16acc=False):
    sa, sb, ldc = M * K + 8, K * N + 8, N + 16
    sc = M * ldc + 2
    a = torch.empty(batch * sa, device=DEV, dtype=torch.half).as_strided((batch, M, K), (sa, K, 1))
    b = torch.empty(batch * sb, device=DEV, dtype=torch.half).as_strided((batch, K, N), (sb, N, 1))
    a.normal_(std=0.25); b.normal_(std=0.25)
    dtype = torch.half
    storage = torch.full((batch * sc + 8,), 123, device=DEV, dtype=dtype)
    c = storage.as_strided((batch, M, N), (sc, ldc, 1))
    mask = torch.zeros_like(storage, dtype=torch.bool)
    mask.as_strided((batch, M, N), (sc, ldc, 1)).fill_(True)

    if use_f16acc:
        ext.hgemm_f16acc(a, b, c)
        ref = (a.double() @ b.double())
    else:
        assert batch == 1
        ext.hgemm(a[0], b[0], c[0])
        ref = (a[0].double() @ b[0].double())
    check(bool((storage[~mask] == 123).all()),
          f"M{M} N{N} K{K} b{batch} f16acc={use_f16acc}: padding clobbered")
    e = rel_rms(c.float(), ref)
    check(e < 0.01, f"M{M} N{N} K{K} b{batch} f16acc={use_f16acc}: rel {e:.4f}")
    return e

print("plain hgemm (2-D, arbitrary shapes):")
for (M, N, K) in [
    (1, 1, 1), (1, 2, 3), (2, 3, 5), (7, 11, 13), (1, 17, 257), (33, 65, 129),
    (128, 128, 128), (127, 129, 255), (256, 512, 1024), (511, 1023, 513),
    (1000, 100, 100), (100, 1000, 100), (100, 100, 1000), (1, 4096, 4096),
    (2048, 1, 4096), (2048, 4096, 1), (17, 257, 65),
]:
    e = run_case(M, N, K)
    print(f"  M{M} N{N} K{K}: rel {e:.4f}", flush=True)

print("hgemm_f16acc (batched, K%64==0, N%128==0):")
for (M, N, K, B) in [
    (1, 128, 64, 1), (17, 128, 128, 1), (511, 256, 512, 1), (127, 1024, 1024, 1),
    (513, 2048, 512, 1), (129, 128, 256, 4), (8, 512, 1024, 3), (1, 2048, 64, 8),
]:
    e = run_case(M, N, K, batch=B, use_f16acc=True)
    print(f"  M{M} N{N} K{K} b{B}: rel {e:.4f}", flush=True)

print("hgemm_recon:")
for (M, N, K) in [(1, 128, 128), (17, 1024, 2048), (1024, 4096, 2048), (129, 511, 257)]:
    a = torch.randn(M, K, device=DEV, dtype=torch.half) * 0.3
    b = torch.randn(K, N, device=DEV, dtype=torch.half) * 0.3
    c = torch.empty(M, N, device=DEV, dtype=torch.half)
    ext.hgemm_recon(a, b, c)
    ref = a.double() @ b.double()
    e = rel_rms(c, ref)
    check(e < 0.01, f"hgemm_recon M{M} N{N} K{K}: rel {e:.4f}")
    print(f"  M{M} N{N} K{K}: rel {e:.4f}", flush=True)

a = torch.randn(64, 512, device=DEV, dtype=torch.half)
b = torch.randn(512, 512, device=DEV, dtype=torch.half)
r1 = torch.empty(64, 512, device=DEV, dtype=torch.float32); ext.hgemm(a, b, r1)
r2 = torch.empty(64, 512, device=DEV, dtype=torch.float32); ext.hgemm(a, b, r2)
check(bool((r1 == r2).all()), "hgemm nondeterministic")
print("determinism: ok")

# f16acc batched determinism (plain hgemm determinism above covers only the 2-D path)
a3 = torch.randn(3, 64, 512, device=DEV, dtype=torch.half) * 0.25
b3 = torch.randn(3, 512, 512, device=DEV, dtype=torch.half) * 0.25
c3a = torch.empty(3, 64, 512, device=DEV, dtype=torch.half)
c3b = torch.empty(3, 64, 512, device=DEV, dtype=torch.half)
ext.hgemm_f16acc(a3, b3, c3a)
ext.hgemm_f16acc(a3, b3, c3b)
check(bool((c3a == c3b).all()), "hgemm_f16acc nondeterministic")
print("f16acc determinism: ok")

# Adversarial values: NaN / Inf / overflow-boundary elements, each poisoning exactly
# one row of one batch element. Policies under test:
#  - NaN/Inf must propagate to that row's output (never silently replaced by finite data)
#  - an 8000-magnitude element (fp16-acc overflow boundary) may saturate to Inf but must
#    not collapse toward zero and must not corrupt any other row
M, N, K, B = 17, 128, 512, 3
for tag, poison in [("nan", float("nan")), ("inf", float("inf")), ("overflow", 8000.0)]:
    av = torch.randn(B, M, K, device=DEV, dtype=torch.half) * 0.25
    bv = torch.randn(B, K, N, device=DEV, dtype=torch.half) * 0.25
    av[0, 0, 0] = poison
    ref = av.double() @ bv.double()
    cv = torch.empty(B, M, N, device=DEV, dtype=torch.half)
    ext.hgemm_f16acc(av, bv, cv)
    # every unpoisoned row (all of batch 1..B-1 plus rows 1..M-1 of batch 0) must be
    # finite and match the reference: poison must not leak across rows
    unpois = cv.clone(); unpois[0, 0] = 0.0
    ref_u = ref.clone(); ref_u[0, 0] = 0.0
    check(bool(torch.isfinite(unpois).all()), f"{tag}: unpoisoned rows non-finite (cross-row corruption)")
    e_u = rel_rms(unpois, ref_u)
    check(e_u < 0.01, f"{tag}: unpoisoned rows rel {e_u:.4f}")
    row = cv[0, 0]
    ref_row_abs = ref[0, 0].abs().max()
    if tag == "overflow":
        # only outputs paired with a large b[0,j] are large; the row's MAX must
        # carry the poison (saturating to Inf is also acceptable for fp16 acc)
        big = bool((~torch.isfinite(row)).any()) or float(row.abs().max()) >= 100.0
        check(big, f"{tag}: poisoned row collapsed (max |out| {row.abs().max():.1f} vs ref {ref_row_abs:.0f})")
    else:
        check(bool((~torch.isfinite(row)).all()), f"{tag}: poisoned row silently finite")
    print(f"  {tag}: unpoisoned rel {e_u:.4f}, poisoned row "
          f"{'non-finite' if not bool(torch.isfinite(row).all()) else 'finite'}", flush=True)

if failures:
    print(f"\n{len(failures)} FAILURES")
    sys.exit(1)
print("ALL HGEMM FUZZ CHECKS PASSED")
