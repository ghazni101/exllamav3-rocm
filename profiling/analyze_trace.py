#!/usr/bin/env python3
"""Analyze a rocprofv3 1_kernel_trace.csv (long format, dispatch rows) from bench_rocm.py.

Phases are located by signature: bench_rocm.py runs load -> warmup(8) -> decode x3 (128 tok)
-> ctx1k prefill+decode -> ctx4k prefill+decode -> prefill 512/2048/4096 -> batch8.
Outputs: phase boundaries, device-busy (interval union) vs wall, per-class shares,
dispatch/token, grid-size census for exl3 kernels.
"""
import csv, sys, collections, re

path = sys.argv[1]
rows = []
with open(path) as f:
    r = csv.reader(f)
    hdr = next(r)
    i_name = hdr.index("Kernel_Name")
    i_start = hdr.index("Start_Timestamp")
    i_end = hdr.index("End_Timestamp")
    i_grid = hdr.index("Grid_Size_X")
    i_wg = hdr.index("Workgroup_Size_X")
    i_vgpr = hdr.index("VGPR_Count")
    i_queue = hdr.index("Queue_Id")
    for row in r:
        try:
            rows.append((int(row[i_start]), int(row[i_end]), row[i_name],
                         int(row[i_grid]), int(row[i_wg]), int(row[i_vgpr]), int(row[i_queue])))
        except (ValueError, IndexError):
            continue
rows.sort()
print(f"dispatches: {len(rows)}")
t0, t1 = rows[0][0], max(r[1] for r in rows)
wall_s = (t1 - t0) / 1e9
print(f"span: {wall_s:.1f}s")

def cls(name):
    if "exl3_gemv_int8" in name:
        m = re.search(r'exl3_gemv_int8_(\w+?)_kernel', name)
        return "gemv_" + (m.group(1) if m else "?")
    if "exl3_gemm_kernel" in name: return "exl3_gemm_coop"
    if "Cijk" in name: return "hipBLAS_" + ("MFMA" if "MI16" in name or "MI" in name else "HSS")
    if "reconstruct" in name: return "reconstruct"
    if "paged_attn" in name: return "paged_attn_" + ("prefill" if "prefill" in name else "decode")
    if "FillFunctor" in name or "fillBuffer" in name: return "fill"
    if "gdn" in name or "recurrent" in name or "chunk_gated" in name or "recompute_w_u" in name or "fused_recurrent" in name: return "GDN"
    if "elementwise" in name: return "elementwise"
    if "reduce" in name: return "reduce"
    if "CatArrayBatched" in name: return "cat"
    if "wmma" in name.lower(): return "wmma"
    return "other"

def classify_all():
    c = {}
    for _, _, n, *_ in rows:
        k = cls(n)
        if k not in c: c[k] = n
    return c
names = classify_all()
for k, n in sorted(names.items()):
    print(f"  class {k:22s} e.g. {n[:80]}")

# Dispatch-rate profile to find phase boundaries: per-second dispatch counts and class mix
import math
sec = collections.defaultdict(lambda: collections.Counter())
for s, e, n, *_ in rows:
    sec[int((s - t0) // 1_000_000_000)][cls(n)] += 1
print("\nper-second class counts (s: total hipblas gemv coop recon):")
prev = None
for s in sorted(sec):
    c = sec[s]
    tag = ""
    if c["hipBLAS_HSS"] + c["hipBLAS_MFMA"] > 20: tag = "PREFILL?"
    elif c["gemv_sq"] + c["gemv_msq"] > 500: tag = "DECODE?"
    elif c["fill"] > 100: tag = "LOAD?"
    print(f"  t={s:4d} n={sum(c.values()):6d} blas={c['hipBLAS_HSS']+c['hipBLAS_MFMA']:4d} gemv={c['gemv_sq']+c['gemv_msq']:5d} coop={c['exl3_gemm_coop']:3d} recon={c['reconstruct']:3d} fill={c['fill']:5d} {tag}")

# grid census for exl3 kernels
grids = collections.Counter()
for s, e, n, g, wg, vgpr, q in rows:
    if "exl3_gemm_kernel" in n or "exl3_gemv_int8" in n:
        grids[(cls(n), g, wg, vgpr)] += 1
print("\nexl3 kernel (class, grid, workgroup, vgpr) -> count:")
for k, v in sorted(grids.items(), key=lambda x: -x[1])[:12]:
    print(f"  {k}: {v}")
