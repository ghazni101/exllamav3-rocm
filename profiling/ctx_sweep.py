"""E1: long-context decode scaling (quality-neutral levers only).

Decode b1 at a growing context, each level on a fresh nonce so the prefix cache cannot serve it:
tok/s, and the fp16 KV read per token for the budget check
(16 full-attn layers x 2 x kv_heads x head_dim x 2 B per token).

Env: EXL3_MODEL EXL3_CACHE_TOKENS GEN_TOKENS (192) CTX_LIST ("1024,4096,8192,16384,32768") OUT_DIR
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
GEN = int(os.environ.get("GEN_TOKENS", "192"))
# 32768 needs a cache > prompt+generation: with EXL3_CACHE_TOKENS=32768 a 32 k prompt plus 192
# generated tokens cannot be allocated, and the generator spins on a job that can never complete
# (observed as a hang). Keep the sweep below the cache or raise the cache with it.
CTX = [int(x) for x in os.environ.get("CTX_LIST", "1024,4096,8192,16384").split(",")]
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


def run(ids, n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler())
    gen.enqueue(job)
    while True:
        for r in gen.iterate():
            if r.get("eos"):
                return r


run(base, 8)                                                    # warmup / autotune

out = []
for ctx in CTX:
    ids = base
    while ids.shape[1] < ctx:
        ids = torch.cat([ids, filler], dim=1)
    p = torch.cat([tok.encode(nonce(), add_bos=False), ids[:, :ctx]], dim=1)
    r = run(p, GEN)
    tps = r["new_tokens"] / r["time_generate"]
    pp = p.shape[1] / r["time_prefill"]
    out.append(dict(ctx=ctx, decode_tps=round(tps, 2), prefill_tps=round(pp, 1),
                    ttft_s=round(r["time_prefill"], 3), gen_tokens=r["new_tokens"]))
    print(json.dumps(out[-1]), flush=True)   # per-level line: partial results survive a hang

with open(os.path.join(OUT_DIR, "e1_ctx_sweep.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote e1_ctx_sweep.json", flush=True)
