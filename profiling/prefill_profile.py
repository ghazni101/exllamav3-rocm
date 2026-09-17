"""D1: prefill attribution - per-op shapes, device time and TFLOP/s for one cold 2048-token prefill.

Why this exists: three mutually inconsistent GEMM rates have been reported for this model
(28.7 TFLOP/s in-model, 54-60 in the shape probe, 83.5-100 in the clean loop). Nothing was
decidable until the profiler named the dominant kernel and its per-shape rate.

Method: one nonced (cold) 2048-token prefill with torch.profiler over CPU+CUDA, then
  (a) key_averages(group_by_input_shape=True) -> per-(op, shape) CPU and CUDA time, from which the
      GEMM shapes (aten::mm/bmm/addmm) get TFLOP/s = 2*m*k*n / device_time;
  (b) key_averages() filtered to CUDA kernel events -> the kernel-level share (reconstruct_*,
      hipBLAS Cijk_*, elementwise, ...) and the host gap fraction.
Chunked prefill means the prefill may be several chunks; the table is per chunk, not per token.

Env: EXL3_MODEL EXL3_CACHE_TOKENS PROMPT_TOKENS (default 2048) OUT_DIR (default /out)
"""
import json
import os
import random
import string
import sys
import time

import torch

sys.path.insert(0, "/prof")
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer           # noqa: E402
from exllamav3.generator.sampler import GreedySampler                           # noqa: E402

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
N = int(os.environ.get("PROMPT_TOKENS", "2048"))
OUT_DIR = os.environ.get("OUT_DIR", "/out")

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=CACHE_TOKENS)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
base = tok.encode("The history of computing machinery begins in the nineteenth century with", add_bos=True)
chunk = tok.encode("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor " * 4,
                   add_bos=False)
ids = base
while ids.shape[1] < N:
    ids = torch.cat([ids, chunk], dim=1)
nonce = tok.encode("".join(random.choices(string.ascii_letters, k=24)), add_bos=False)
prompt = torch.cat([nonce, ids[:, :N]], dim=1)

# warm up the kernels and the autotuner on a different nonce, then measure the cold prefill
gen.enqueue(Job(input_ids=prompt, max_new_tokens=2, sampler=GreedySampler()))
while True:
    done = any(r.get("eos") for r in gen.iterate())
    if done:
        break
nonce2 = tok.encode("".join(random.choices(string.ascii_letters, k=24)), add_bos=False)
prompt2 = torch.cat([nonce2, ids[:, :N]], dim=1)

from torch.profiler import ProfilerActivity, profile                                # noqa: E402

job = Job(input_ids=prompt2, max_new_tokens=2, sampler=GreedySampler())
gen.enqueue(job)
t0 = time.time()
res = None
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    while True:
        hit = [r for r in gen.iterate() if r.get("eos")]
        if hit:
            res = hit[0]
            break
wall = time.time() - t0
prefill_ms = round(res["time_prefill"] * 1e3, 3) if res else None
print(f"prompt_tokens={int(prompt2.shape[1])} wall={wall:.3f}s prefill_ms={prefill_ms} "
      f"prefill_tps={round(prompt2.shape[1] / res['time_prefill'], 1) if res else None}", flush=True)

# (a) per-op-shape view: GEMM shapes -> TFLOP/s
rows = []
for ev in prof.key_averages(group_by_input_shape=True):
    cuda_us = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
    if cuda_us <= 0:
        continue
    shapes = ""
    try:
        shapes = str(ev.input_shapes)
    except Exception:                                                              # noqa: BLE001
        pass
    flops = 0
    if ev.key in ("aten::mm", "aten::bmm", "aten::addmm", "aten::matmul") and shapes:
        try:
            dims = [int(x) for x in __import__("re").findall(r"\d+", shapes)]
            if len(dims) >= 3:
                m, k, n = dims[-3], dims[-2], dims[-1]
                flops = 2 * m * k * n
        except Exception:                                                          # noqa: BLE001
            flops = 0
    rows.append(dict(key=ev.key, count=ev.count, cuda_ms=round(cuda_us / 1e3, 3),
                     shapes=shapes if len(shapes) < 160 else shapes[:157] + "...",
                     tflops=round(flops / cuda_us * 1e-6, 1) if flops else None))
rows.sort(key=lambda r: -r["cuda_ms"])
print("\n== top ops by device time (grouped by input shape) ==")
for r in rows[:28]:
    tf = f"{r['tflops']:7.1f} TF/s" if r["tflops"] else " " * 12
    print(f"  {r['cuda_ms']:9.3f} ms  n={r['count']:5d}  {tf}  {r['key'][:34]:34s} {r['shapes']}")

# (b) kernel-level share
krows = []
for ev in prof.key_averages():
    cuda_us = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
    if cuda_us <= 0 or ev.count == 0:
        continue
    krows.append((cuda_us / 1e3, ev.count, ev.key))
krows.sort(reverse=True)
total = sum(r[0] for r in krows)
print(f"\n== device time by entry (total {total:.1f} ms) ==")
for ms, cnt, key in krows[:22]:
    print(f"  {ms:9.3f} ms {100*ms/total:5.1f}%  n={cnt:6d}  {key[:96]}")

# (c) host attribution: the same events ranked by self CPU time. The prefill wall exceeds the
# device total by ~2.2 s on this model, and only a CPU-ranked table says where it goes (the op
# table above drops CPU-only ops such as aten::empty/view and the python-side dispatch around
# reconstruct_hgemm).
cpu_rows = []
for ev in prof.key_averages():
    cpu_us = getattr(ev, "self_cpu_time_total", 0) or 0
    if cpu_us <= 0 or ev.count == 0:
        continue
    cpu_rows.append((cpu_us / 1e3, ev.count, ev.key))
cpu_rows.sort(reverse=True)
total_cpu = sum(r[0] for r in cpu_rows)
print(f"\n== host time by entry (total self CPU {total_cpu:.1f} ms) ==")
for ms, cnt, key in cpu_rows[:20]:
    print(f"  {ms:9.3f} ms  n={cnt:6d}  {key[:96]}")

with open(os.path.join(OUT_DIR, "d1_prefill_profile.json"), "w") as f:
    json.dump(dict(prompt_tokens=int(prompt2.shape[1]), wall_s=round(wall, 3),
                   prefill_ms=prefill_ms, total_cuda_ms=round(total, 3),
                   ops=rows, kernels=[dict(ms=round(a, 3), count=b, key=c) for a, b, c in krows[:40]],
                   total_cpu_ms=round(total_cpu, 3),
                   cpu=[dict(ms=round(a, 3), count=b, key=c) for a, b, c in cpu_rows[:40]]),
              f, indent=1)
print("wrote d1_prefill_profile.json", flush=True)
