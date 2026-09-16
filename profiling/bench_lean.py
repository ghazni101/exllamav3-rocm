"""Lean A/B bench: decode b1, prefill 2k (cold), batch-4 and batch-8 aggregate."""
import os, json, time, random, string
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))

def run(generator, ids, n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler())
    generator.enqueue(job)
    while True:
        for r in generator.iterate():
            if r.get("eos"):
                return r

def run_batch(generator, ids_list, n):
    for ids in ids_list:
        generator.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()))
    done = 0
    t0 = time.time()
    while done < len(ids_list):
        for r in generator.iterate():
            if r.get("eos"):
                done += 1
    return len(ids_list) * n / (time.time() - t0)

def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n//2-1] + xs[n//2]) / 2

def main():
    t0 = time.time()
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tok = Tokenizer.from_config(config)
    gen = Generator(model=model, cache=cache, tokenizer=tok)
    load_s = time.time() - t0
    base = tok.encode("The history of computing machinery begins in the nineteenth century with", add_bos=True)
    out = {"load_s": round(load_s, 1)}

    run(gen, base, 8)   # warmup / autotune / JIT

    ds = []
    for _ in range(3):
        r = run(gen, base, 192)
        ds.append(r["new_tokens"] / r["time_generate"])
    out["decode_b1_tps"] = round(med(ds), 2)

    # cold prefill 2k (unique nonce defeats prefix cache)
    chunk = tok.encode("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor " * 4, add_bos=False)
    ids = base
    while ids.shape[1] < 2048:
        ids = torch.cat([ids, chunk], dim=1)
    nonce = tok.encode("".join(random.choices(string.ascii_letters, k=24)), add_bos=False)
    p = torch.cat([nonce, ids[:, :2048]], dim=1)
    r = run(gen, p, 4)
    out["prefill_2k_tps"] = round(p.shape[1] / r["time_prefill"], 1)

    out["batch4_agg_tps"] = round(run_batch(gen, [base + i for i in range(4)], 128), 2)
    out["batch8_agg_tps"] = round(run_batch(gen, [base + i for i in range(8)], 96), 2)
    print(json.dumps(out), flush=True)

main()
