"""Attribute the prefill host gap by call stack.

The shipped profile showed the prefill wall exceeding device time by ~2.2 s, spending 1374 ms in
hipMemcpyWithStream (102 calls), 418 ms in hipDeviceSynchronize (68) and 245+89 ms in 64
hipStreamCreateWithFlags/Destroy pairs, with python-side ops accounting for only ~30 ms. Ranking is
not attribution: this runs the same nonced 2048-token prefill with torch.profiler(with_stack=True)
and prints the stack of the top host entries, so the caller is named.

Env: EXL3_MODEL EXL3_CACHE_TOKENS PROMPT_TOKENS OUT_DIR
"""
import json
import os
import random
import string
import sys

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
prompt = torch.cat([tok.encode("".join(random.choices(string.ascii_letters, k=16)), add_bos=False),
                    ids[:, :N - 16]], dim=1)

# warmup so autotune/JIT is outside the profile
job = Job(input_ids=base, max_new_tokens=8, sampler=GreedySampler())
gen.enqueue(job)
for r in gen.iterate():
    if r.get("eos"):
        break

job = Job(input_ids=prompt, max_new_tokens=2, sampler=GreedySampler())
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA],
                            with_stack=True) as prof:
    gen.enqueue(job)
    for r in gen.iterate():
        if r.get("eos"):
            break

WANT = ("hipMemcpyWithStream", "hipDeviceSynchronize", "hipStreamCreateWithFlags",
        "hipStreamDestroy", "hipMemcpyAsync", "hipMemcpy")
stacks = {}
counts = {}
for ev in prof.events():
    if ev.key not in WANT:
        continue
    counts[ev.key] = counts.get(ev.key, 0) + 1
    if ev.key in stacks:
        continue
    st = list(getattr(ev, "stack", []) or [])
    frames = []
    for f in st:
        nm = getattr(f, "name", None) or "?"
        fn = getattr(f, "filename", "") or ""
        ln = getattr(f, "lineno", 0) or 0
        if "torch" in fn and "profiler" in fn:
            continue
        frames.append(f"{nm} ({os.path.basename(fn)}:{ln})")
    stacks[ev.key] = frames[-6:]

print("\n== host entries: count and python stack (innermost last) ==")
for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"\n--- {k}: {c} calls")
    for fr in stacks.get(k, [])[-6:]:
        print(f"      {fr}")

with open(os.path.join(OUT_DIR, "host_gap_stacks.json"), "w") as f:
    json.dump({"counts": counts, "stacks": stacks}, f, indent=1)
print("\nwrote host_gap_stacks.json", flush=True)
