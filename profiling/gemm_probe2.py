#!/usr/bin/env python3
"""pp-0 probe: why does the model's prefill deliver 15-29 TFLOP/s when hipBLASLt
does 83.5 on the headline shape? Times both candidate device paths on the model's
exact per-layer (k, n) inventory derived from the checkpoint headers:
  a) torch.matmul (the heuristic path)
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
    inv = {}
    for k, v in hdr.items():
        # serve-visible linears only: model layers (mtp.* is dropped by conversion, exclude it);
        # the loop iterates every layer, so counts are already totals
        if k.endswith(".svh") and ".layers." in k and "mtp" not in k and (k[:-3] + "suh") in hdr:
            base = k[:-3]
            kdim = hdr[base + "suh"]["shape"][0]
            ndim = v["shape"][0]
            inv[(kdim, ndim)] = inv.get((kdim, ndim), 0) + 1
    if "lm_head.svh" in hdr:
        inv[(hdr["lm_head.suh"]["shape"][0], hdr["lm_head.svh"]["shape"][0])] = 1
    return inv

def bench(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

def main():
    import exllamav3_ext as ext
    total = linear_inventory(MODEL_DIR)
    print(f"{len(total)} unique (k,n) shapes, weighted by layer count")
    for m in MS:
        print(f"\n=== m = {m} ===")
        agg = {"flops": 0.0, "t_mm": 0.0, "t_hg": 0.0}
        for (k, n), cnt in sorted(total.items(), key=lambda x: -x[0][0] * x[0][1]):
            a = torch.randn(m, k, dtype=torch.float16, device="cuda")
            b = torch.randn(k, n, dtype=torch.float16, device="cuda")
            c = torch.empty(m, n, dtype=torch.float16, device="cuda")
            t_mm = bench(lambda: torch.matmul(a, b, out=c))
            t_hg = None
            try:
                t_hg = bench(lambda: ext.hgemm_recon(a, b, c))
            except Exception as e:
                print(f"  (hgemm_recon({m},{k},{n}) unavailable: {str(e)[:70]})")
            fl = 2.0 * m * k * n
            agg["flops"] += fl * cnt; agg["t_mm"] += t_mm * cnt
            if t_hg: agg["t_hg"] += t_hg * cnt
            tf_mm = fl / t_mm / 1e12
            tf_hg = fl / t_hg / 1e12 if t_hg else None
            print(f"  k={k:6d} n={n:6d} x{cnt:3d}  matmul={tf_mm:7.1f} TF/s  "
                  f"hgemm_recon={'-' if tf_hg is None else f'{tf_hg:7.1f} TF/s'}")
            del a, b, c
        wmm = agg["flops"] / agg["t_mm"] / 1e12
        whg = agg["flops"] / agg["t_hg"] / 1e12 if agg["t_hg"] else None
        print(f"  WEIGHTED aggregate: matmul={wmm:.1f} TF/s  "
              f"hgemm_recon={'-' if whg is None else f'{whg:.1f} TF/s'}")

if __name__ == "__main__":
    main()
