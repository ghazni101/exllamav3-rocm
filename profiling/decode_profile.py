"""B3.0/E1: pure-decode device attribution with torch.profiler (no rocprofv3, no shim).

A b1 decode window is profiled in isolation - the session-2 CLI trace mixed warmup, prefill,
long-context and batch phases in one file, which is why its cluster attribution could not be
mapped to a phase with confidence. This script measures exactly one thing: N decode steps.

Outputs per-kernel device totals (share of the window) and per-op shape-grouped device time, plus
the host-gap fraction (window wall time minus the device-busy union).

Env: EXL3_MODEL EXL3_CACHE_TOKENS DECODE_N (192) CTX (0 = short prompt) OUT_DIR TAG
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
from torch.profiler import ProfilerActivity, profile                            # noqa: E402

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
N = int(os.environ.get("DECODE_N", "192"))
CTX = int(os.environ.get("CTX", "0"))
TAG = os.environ.get("TAG", "decode")
OUT_DIR = os.environ.get("OUT_DIR", "/out")


def nonce(n=24):
    return "".join(random.choices(string.ascii_letters, k=n))


config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=CACHE_TOKENS)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
base = tok.encode("The history of computing machinery begins in the nineteenth century with", add_bos=True)
filler = tok.encode("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor " * 4,
                    add_bos=False)
ids = base
while ids.shape[1] < max(CTX, 8):
    ids = torch.cat([ids, filler], dim=1)
prompt = torch.cat([tok.encode(nonce(), add_bos=False), ids[:, :max(CTX, 1)]], dim=1)


def run(ids_, n):
    job = Job(input_ids=ids_, max_new_tokens=n, sampler=GreedySampler())
    gen.enqueue(job)
    while True:
        for r in gen.iterate():
            if r.get("eos"):
                return r


run(prompt, 8)                                                  # warmup / autotune / graph capture
r = run(prompt, N)                                              # unprofiled reference timing
ref_tps = r["new_tokens"] / r["time_generate"]
print(f"reference: {ref_tps:.2f} tok/s over {r['new_tokens']} tokens "
      f"({r['time_generate'] / r['new_tokens'] * 1e3:.2f} ms/token)", flush=True)

# Profiled window: a second run on a fresh nonce so no prefix-cache hit.
# Drain prefill (and the iterate that emits the first streaming token) BEFORE
# starting the profiler so hipBLAS/reconstruct/exl3_gemm are not mixed into
# the decode attribution. CTX=4096 prefill was 40% of the previous capture.
prompt2 = torch.cat([tok.encode(nonce(), add_bos=False), ids[:, :max(CTX, 1)]], dim=1)
job = Job(input_ids=prompt2, max_new_tokens=N, sampler=GreedySampler())
gen.enqueue(job)
prefill_iters = 0
while True:
    rs = gen.iterate()
    prefill_iters += 1
    if any(r_.get("eos") for r_ in rs):
        raise RuntimeError("job finished during prefill drain")
    if any(r_.get("stage") == "streaming" for r_ in rs):
        break
print(f"prefill drain iters={prefill_iters} (first streaming token excluded)", flush=True)
t0 = time.time()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    while True:
        if any(r_.get("eos") for r_ in gen.iterate()):
            break
wall = time.time() - t0
print(f"profiled wall={wall:.3f}s", flush=True)

krows = []
for ev in prof.key_averages():
    dev = getattr(ev, "self_device_time_total", None)
    if dev is None:
        dev = getattr(ev, "self_cuda_time_total", 0)
    if dev <= 0 or ev.count == 0:
        continue
    krows.append((dev / 1e3, ev.count, ev.key))
krows.sort(reverse=True)
total = sum(x[0] for x in krows)
print(f"\n== device time by entry, total {total:.1f} ms over {wall:.2f}s wall ==")
for ms, cnt, key in krows[:26]:
    print(f"  {ms:9.3f} ms {100*ms/total:5.1f}%  n={cnt:7d}  avg={ms*1e3/cnt:8.1f} us  {key[:84]}")
print(f"  device busy union >= {total:.1f} ms = {100*total/(wall*1e3):.0f}% of wall; "
      f"host gap <= {100*(1 - total/(wall*1e3)):.0f}%")

with open(os.path.join(OUT_DIR, f"{TAG}_profile.json"), "w") as f:
    json.dump(dict(ctx=CTX, decode_n=N, ref_tps=round(ref_tps, 2), wall_s=round(wall, 3),
                   total_dev_ms=round(total, 3),
                   kernels=[dict(ms=round(a, 3), count=b, key=c) for a, b, c in krows[:60]]),
              f, indent=1)
print(f"wrote {TAG}_profile.json", flush=True)
