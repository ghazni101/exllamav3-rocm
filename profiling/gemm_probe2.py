#!/usr/bin/env python3
"""pp-0 probe: why does the model's prefill deliver 15-29 TFLOP/s when hipBLASLt
does 83.5 on the headline shape? Times BOTH candidate device paths on the model's
exact per-layer (k, n) list derived from the checkpoint headers:
  a) torch.matmul (bf16->fp16 heuristic path)
  b) ext.hgemm_recon (the binding the model's reconstruct_hgemm actually calls)
Reports per-shape TFLOP/s and the layer-count-weighted aggregate.
"""
import os, json, struct, time
import torch

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
MS = [int(x) for x in os.environ.get("PROBE_MS", "512,1700,2048,4096").split(",")]

def linear_inventory(d):
    with open(os.path.join(d, "model.safetensors"), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    shapes = {}
    for k, v in hdr.items():
        if k.endswith("svh") and (".layer" in k or k.startswith("lm_head")):
            base = k[:-4]
            kdim = hdr[base + "suh"][0]
            ndim = v[0]
            shapes.setdefault((kdim, ndim), set()).add(base.split(".layers.")[0] if ".layers." in base else base)
    # layer counts
    counts = {}
    for k, v in hdr.items():
        if k.endswith("svh") and ".layers." in k:
            import re
            m = re.search(r"\.layers\.(\d+)\.", k)
            if m:
                counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
    n_layers = max(counts) + 1 if counts else 0
    full_every = 4
    n_full = sum(1 for i in range(n_layers) if (i + 1) % full_every == 0)
    n_gdn = n_layers - n_full
    inv = {}
    for k, v in hdr.items():
        if k.endswith("svh") and ".layers." in k:
            base = k[:-4]
            kdim, ndim = hdr[base + "suh"][0], v[0]
            inv[(kdim, ndim)] = inv.get((kdim, ndim), 0) + 1
    # per-layer counts -> total counts
    total = {(k, n): c * n_layers for (k, n), c in inv.items()}
    lm = hdr.get("lm_head.svh")
    if lm: total[(hdr["lm_head.suh"][0], lm[0])] = 1
    return total

def bench(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

def main():
    import exllamav3_ext as ext
    d = MODEL_DIR
    total = linear_inventory(d)
    print(f"{len(total)} unique (k,n) shapes, weighted by layer count")
    agg = {"flops": 0.0, "t_matmul": 0.0, "t_hgemm": 0.0}
    for m in MS:
        print(f"\n=== m = {m} ===")
        rows = []
        for (k, n), cnt in sorted(total.items(), key=lambda x: -x[0][0] * x[0][1]):
            a = torch.randn(m, k, dtype=torch.float16, device="cuda")
            b = torch.randn(k, n, dtype=torch.float16, device="cuda")
            c = torch.empty(m, n, dtype=torch.float16, device="cuda")
            t_mm = bench(lambda: torch.matmul(a, b, out=c))
            try:
                t_hg = bench(lambda: ext.hgemm_recon(a, b, c))
            except Exception as e:
                t_hg = None
                hg_err = str(e)[:80]
            fl = 2.0 * m * k * n
            tf_mm = fl / t_mm / 1e12
            tf_hg = fl / t_hg / 1e12 if t_hg else None
            agg["flops"] += fl * cnt; agg["t_matmul"] += t_mm * cnt
            if t_hg: agg["t_hgemm"] += t_hg * cnt
            rows.append((k, n, cnt, tf_mm, tf_hg))
            del a, b, c
        for k, n, cnt, tf_mm, tf_hg in rows:
            print(f"  k={k:6d} n={n:6d} x{cnt:3d}  matmul={tf_mm:7.1f} TF/s  hgemm_recon={'-' if tf_hg is None else f'{tf_hg:7.1f} TF/s'}")
        wmm = agg["flops"] / agg["t_matmul"] / 1e12
        whg = agg["flops"] / agg["t_hgemm"] / 1e12 if agg["t_hgemm"] else None
        print(f"  WEIGHTED aggregate: matmul={wmm:.1f} TF/s  hgemm_recon={'-' if whg is None else f'{whg:.1f} TF/s'}")
        agg = {"flops": 0.0, "t_matmul": 0.0, "t_hgemm": 0.0}

if __name__ == "__main__":
    main()
