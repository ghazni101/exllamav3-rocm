#!/usr/bin/env python3
"""Phase stats from bench_rocm kernel trace: busy union vs wall per segment,
per-class time shares, dispatches/token, gap structure. Segments are found by
splitting the dispatch timeline at gaps > GAP ms, then labelled by signature."""
import csv, sys, collections, re

path, GAP_MS = sys.argv[1], float(sys.argv[2] if len(sys.argv) > 2 else 50)
rows = []
with open(path) as f:
    r = csv.reader(f)
    hdr = next(r)
    i_name = hdr.index("Kernel_Name"); i_start = hdr.index("Start_Timestamp")
    i_end = hdr.index("End_Timestamp"); i_grid = hdr.index("Grid_Size_X")
    i_wg = hdr.index("Workgroup_Size_X")
    for row in r:
        try:
            rows.append((int(row[i_start]), int(row[i_end]), row[i_name], int(row[i_grid]), int(row[i_wg])))
        except (ValueError, IndexError):
            continue
rows.sort()
t0 = rows[0][0]

def cls(name):
    if "exl3_gemv_int8_msq" in name: return "gemv_msq"
    if "exl3_gemv_int8_sq" in name: return "gemv_sq"
    if "exl3_gemm_kernel" in name: return "coop"
    if "Cijk" in name: return "hipBLAS"
    if "reconstruct" in name: return "reconstruct"
    if "paged_attn" in name: return "paged_attn"
    if "FillFunctor" in name or "fillBuffer" in name: return "fill"
    if "copyBuffer" in name: return "copybuf"
    if "gdn" in name or "recurrent" in name or "chunk_gated" in name or "recompute_w_u" in name or "fused_recurrent" in name or "delta_rule" in name: return "GDN"
    if "elementwise" in name: return "elementwise"
    if "reduce" in name: return "reduce"
    if "CatArray" in name: return "cat"
    return "other"

# segment on gaps
segs, cur = [], [rows[0]]
for r_ in rows[1:]:
    if (r_[0] - cur[-1][1]) > GAP_MS * 1e6:
        segs.append(cur); cur = [r_]
    else:
        cur.append(r_)
segs.append(cur)

def busy_union(seg):
    iv = sorted((s, e) for s, e, *_ in seg)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if ce is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    return (tot + (ce - cs if ce else 0)) / 1e9

def segstats(seg):
    wall = (seg[-1][1] - seg[0][0]) / 1e9
    busy = busy_union(seg)
    per = collections.defaultdict(float)
    cnt = collections.Counter()
    for s, e, n, g, wg in seg:
        per[cls(n)] += (e - s) / 1e9
        cnt[cls(n)] += 1
    return seg[0][0], wall, busy, per, cnt

print(f"{'seg':>3} {'t_start':>8} {'wall_s':>7} {'busy_s':>7} {'busy%':>6} {'n':>7}  top classes (ms, share of busy)")
for i, seg in enumerate(segs):
    st, wall, busy, per, cnt = segstats(seg)
    if wall < 0.4 and sum(cnt.values()) < 50: continue
    tops = sorted(per.items(), key=lambda x: -x[1])[:6]
    tops = " ".join(f"{k}:{v*1e3:.0f}ms({v/max(busy,1e-9)*100:.0f}%)" for k, v in tops)
    print(f"{i:>3} {(st-t0)/1e9:>8.1f} {wall:>7.2f} {busy:>7.2f} {busy/wall*100:>5.0f}% {sum(cnt.values()):>7}  {tops}")
