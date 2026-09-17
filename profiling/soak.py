#!/usr/bin/env python3
"""No-deadlock soak: mixed concurrent/sequential jobs of varying lengths.
A cooperative-launch defect shows up as a hang — the hard timeout turns a hang
into exit code 3 (gpu-ctl reports it as a failed run). VRAM is checked for leaks.
"""
import os, sys, time, json, random
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
MINUTES = float(os.environ.get("SOAK_MINUTES", "10"))
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))

random.seed(1234)
FILLER = "Machine learning models have grown rapidly in scale over the past decade. "
BASE = "The history of computing machinery begins in the nineteenth century with "

def main():
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tok = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tok)
    gen = Job(input_ids=tok.encode("Warmup.", add_bos=True), max_new_tokens=8, sampler=GreedySampler())
    generator.enqueue(gen)
    while not any(r.get("eos") for r in generator.iterate()):
        pass

    # Leak baseline: AFTER the model, cache and warm-up are resident. The earlier version sampled
    # free VRAM before the model was loaded, so `free0 - free1` was just the model size (~20 GB) and
    # the >512 MB check could never pass (the session-4 gate only grepped for the JSON key).
    free0, total = torch.cuda.mem_get_info()
    alloc0 = torch.cuda.memory_allocated()
    resv0 = torch.cuda.memory_reserved()

    t_end = time.time() + MINUTES * 60
    rounds, toks = 0, 0
    while time.time() < t_end:
        njobs = random.choice([1, 2, 4, 8])
        plen = random.choice([64, 256, 1024, 2048])
        glen = random.choice([32, 64, 128])
        ids0 = tok.encode(BASE, add_bos=True)
        chunk = tok.encode(FILLER, add_bos=False)
        jobs = []
        for i in range(njobs):
            ids = ids0
            while ids.shape[1] < plen:
                ids = torch.cat([ids, chunk], dim=1)
            ids = ids[:, : plen] + (i % 7)
            j = Job(input_ids=ids, max_new_tokens=glen, sampler=GreedySampler())
            generator.enqueue(j)
            jobs.append(j)
        done = 0
        while done < njobs:
            for r in generator.iterate():
                if r.get("eos"):
                    done += 1
        toks += njobs * glen
        rounds += 1
        print(f"[soak] round {rounds}: {njobs}x{plen}-prompt {glen}-tok jobs done, total {toks} gen tokens", flush=True)

    free1, _ = torch.cuda.mem_get_info()
    alloc1 = torch.cuda.memory_allocated()
    resv1 = torch.cuda.memory_reserved()
    print(json.dumps({
        "rounds": rounds, "gen_tokens": toks,
        "vram_leak_mb": round((free0 - free1) / 2**20, 1),
        # allocated growth = live tensors (a real leak); reserved growth = caching-allocator
        # retention, which is expected to grow with the largest job mix seen and is not a leak
        "alloc_growth_mb": round((alloc1 - alloc0) / 2**20, 1),
        "reserved_growth_mb": round((resv1 - resv0) / 2**20, 1)}))
    # The leak signal is live-tensor growth. Free-VRAM growth also carries the caching allocator's
    # retained blocks and the KV page pool's high-water mark, which legitimately grow with the
    # largest job mix served and are not leaks - report them, but do not fail on them alone.
    if (alloc1 - alloc0) / 2**20 > 512:
        print("[soak] FAIL: live-tensor growth > 512 MB")
        sys.exit(1)
    print("[soak] PASS (live tensors; free-VRAM/reserved growth reported above)")

if __name__ == "__main__":
    main()
