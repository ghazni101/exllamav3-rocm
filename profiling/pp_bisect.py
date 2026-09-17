#!/usr/bin/env python3
"""P1 bisect: name the mechanism behind in-model prefill GEMMs running at ~30 TF/s
while ext.hgemm_recon does 90-100 TF/s on identical shapes in a clean loop.

Variant matrix over the model's exact weighted (k, n) inventory:
  reuse      all buffers preallocated (the probe condition; ~90-100 TF/s expected)
  fresh_b    weight tensor torch.empty'd per call (the model's pp-1.2 churn site)
  fresh_c    output fresh per call
  fresh_all  both
  b_fill     fresh b, filled by a memory-bound kernel before the GEMM (simulates the
             dequant write that precedes every model GEMM; dequant traffic ~= fill)
  model      fresh b + fill + fresh c per call, outputs kept alive per iteration like
             the model's y (freed after)
Whichever cell falls to ~30 TF/s names the fix. torch caching-allocator behavior is
part of the measurement (allocs are real alloc/free pairs through the cache).
"""
import os, json, struct, time
import torch

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
M = int(os.environ.get("PP_M", "2048"))
REPS = int(os.environ.get("PP_REPS", "10"))

def linear_inventory(d):
    with open(os.path.join(d, "model.safetensors"), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    inv = {}
    for k, v in hdr.items():
        if k.endswith(".svh") and ".layers." in k and "mtp" not in k and (k[:-3] + "suh") in hdr:
            base = k[:-3]
            kdim = hdr[base + "suh"]["shape"][0]
            ndim = v["shape"][0]
            inv[(kdim, ndim)] = inv.get((kdim, ndim), 0) + 1
    if "lm_head.svh" in hdr:
        inv[(hdr["lm_head.suh"]["shape"][0], hdr["lm_head.svh"]["shape"][0])] = 1
    return inv

def main():
    import exllamav3_ext as ext
    inv = linear_inventory(MODEL_DIR)
    shapes = sorted(inv.items(), key = lambda x: -x[0][0] * x[0][1])
    flops_total = sum(2.0 * M * k * n * cnt for (k, n), cnt in shapes)
    print(f"{len(shapes)} shapes, {flops_total/1e12:.1f} GFLOP per pass at m={M}")

    # Preallocated pool for the reuse config
    pool = []
    for (k, n), cnt in shapes:
        for _ in range(min(cnt, 3)):    # distinct buffers per weight class
            pool.append((
                torch.randn(M, k, dtype = torch.half, device = "cuda") * 0.05,
                torch.randn(k, n, dtype = torch.half, device = "cuda") * 0.05,
                torch.empty(M, n, dtype = torch.half, device = "cuda"),
            ))
    # flat list with (k, n) expanded per layer, paired with round-robin pool entries
    calls = []
    pi = 0
    for (k, n), cnt in shapes:
        for _ in range(cnt):
            calls.append(((k, n), pool[pi % len(pool)]))
            pi += 1

    def run(cfg):
        for (_kk, _nn), (a, b, c) in calls:
            M_, k_, n_ = a.shape[0], a.shape[1], b.shape[1]
            if cfg == "reuse":
                ext.hgemm_recon(a, b, c)
            elif cfg == "fresh_b":
                b2 = torch.empty(k_, n_, dtype = torch.half, device = "cuda")
                ext.hgemm_recon(a, b2, c)
            elif cfg == "fresh_c":
                c2 = torch.empty(M_, n_, dtype = torch.half, device = "cuda")
                ext.hgemm_recon(a, b, c2)
            elif cfg == "fresh_all":
                b2 = torch.empty(k_, n_, dtype = torch.half, device = "cuda")
                c2 = torch.empty(M_, n_, dtype = torch.half, device = "cuda")
                ext.hgemm_recon(a, b2, c2)
            elif cfg == "b_fill":
                b2 = torch.empty(k_, n_, dtype = torch.half, device = "cuda")
                b2.fill_(0.5)
                ext.hgemm_recon(a, b2, c)
            elif cfg == "model":
                b2 = torch.empty(k_, n_, dtype = torch.half, device = "cuda")
                b2.fill_(0.5)
                c2 = torch.empty(M_, n_, dtype = torch.half, device = "cuda")
                ext.hgemm_recon(a, b2, c2)
        torch.cuda.synchronize()

    out = {}
    for cfg in ["reuse", "fresh_b", "fresh_c", "fresh_all", "b_fill", "model"]:
        run(cfg)   # warm (allocator + lib caches)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(cfg)
        dt = time.perf_counter() - t0
        tf = flops_total / dt / 1e12
        out[cfg] = round(tf, 1)
        print(f"  {cfg:9s}: {tf:6.1f} TF/s", flush=True)
    print(json.dumps(out))

main()
