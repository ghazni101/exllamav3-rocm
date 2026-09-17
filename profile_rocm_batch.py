#!/usr/bin/env python3
"""Batched-decode trace target for T3: enqueue B concurrent jobs, decode N tokens, exit.
Under a rocprofv3 --kernel-trace this yields per-launch sq/msq/coop counts and durations
per batch size, to answer whether the generator actually issues m=B GEMV calls and
whether the m<=4 sq instantiations amortize (T3, plan §9)."""
import os
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
B = int(os.environ.get("TRACE_BATCH", "4"))
WARMUP_TOKENS = int(os.environ.get("EXL3_WARMUP_TOKENS", "16"))
DECODE_TOKENS = int(os.environ.get("EXL3_DECODE_TOKENS", "24"))

def main():
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model=model, cache=cache, tokenizer=tok)

    ids = tok.encode("The history of computing machinery begins in the nineteenth century with", add_bos=True)
    job = Job(input_ids=ids, max_new_tokens=WARMUP_TOKENS, sampler=GreedySampler())
    gen.enqueue(job)
    for r in gen.iterate():
        if r.get("eos"):
            break

    # B identical-but-distinct prompts, enqueued together -> one batched m=B decode round
    jobs = []
    for i in range(B):
        j = Job(input_ids=ids + i, max_new_tokens=DECODE_TOKENS, sampler=GreedySampler())
        gen.enqueue(j)
        jobs.append(j)
    done = 0
    while done < B:
        for r in gen.iterate():
            if r.get("eos"):
                done += 1
    print(f"[profile] batch-{B} decode {DECODE_TOKENS} tokens done")

main()
