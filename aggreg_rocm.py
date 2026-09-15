#!/usr/bin/env python3
"""Aggregate rocprofv3 kernel-trace CSVs: per-kernel total/avg/count, ranked.

Usage: python3 aggreg_rocm.py <trace_dir_or_csv> [--top 40] [--min-ms 1]
Handles rocprofv3 CSV variants (kernel-trace dispatch records). Durations are
normalized to microseconds.
"""
import csv, glob, os, sys, collections

def pick(row, *keys):
    for k in keys:
        for rk in row:
            if rk.lower() == k.lower():
                return row[rk]
    return None

def main():
    path = sys.argv[1]
    top = 40
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    totals = collections.defaultdict(float)
    counts = collections.Counter()
    grids = collections.defaultdict(collections.Counter)
    nrows = 0
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                name = pick(row, "Kernel_Name", "KernelName", "Name")
                dur = pick(row, "Duration(ns)", "Duration_Ns", "Duration(nsec)", "Duration")
                if name is None or dur is None:
                    continue
                try:
                    d = float(dur)
                except ValueError:
                    continue
                # normalize: if values look like ns, keep ns internally
                totals[name] += d
                counts[name] += 1
                g = pick(row, "Grid_Size", "GridSize", "grid_size")
                if g:
                    grids[name][g] += 1
                nrows += 1
    if not totals:
        print("no kernel rows found; columns of first file:")
        if files:
            with open(files[0], newline="") as fh:
                print(next(csv.reader(fh)))
        return

    # Scale: if max total < 1e4 assume values are already usec
    scale = 1.0
    max_t = max(totals.values())
    if max_t > 1e7:      # ns -> us
        scale = 1e-3
    total_all = sum(totals.values()) * scale / 1000.0  # ms
    print(f"rows={nrows}  kernels={len(totals)}  total_gpu_time={total_all:.1f} ms\n")
    print(f"{'total_ms':>10} {'%':>6} {'count':>8} {'avg_us':>9}  kernel [grid]")
    for name, t in sorted(totals.items(), key=lambda kv: -kv[1])[:top]:
        ms = t * scale / 1000.0
        pct = 100.0 * t / sum(totals.values())
        c = counts[name]
        avg = t * scale / c
        g = grids[name].most_common(1)[0][0] if grids[name] else "?"
        short = name if len(name) <= 90 else name[:87] + "..."
        print(f"{ms:10.1f} {pct:5.1f}% {c:8d} {avg:9.1f}  {short} [{g}]")

if __name__ == "__main__":
    main()
