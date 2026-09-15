import os, time
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

# Deterministic, bounded-length generation for profiling runs.
# Keeps total kernel count small enough that rocprofv3 trace files stay manageable.

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
WARMUP_TOKENS = int(os.environ.get("EXL3_WARMUP_TOKENS", "16"))
DECODE_TOKENS = int(os.environ.get("EXL3_DECODE_TOKENS", "96"))
PREFILL_TOKENS = int(os.environ.get("EXL3_PREFILL_TOKENS", "0"))


def main():
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    ids = tokenizer.encode(
        "The history of computing machinery begins in the nineteenth century with",
        add_bos=True)

    # Warmup (allocator, autotuner, JIT paths) — profiled too, but easy to slice off
    job = Job(input_ids=ids, max_new_tokens=WARMUP_TOKENS, sampler=GreedySampler())
    generator.enqueue(job)
    while True:
        if any(r.get("eos") for r in generator.iterate()):
            break

    # Steady-state decode — the region of interest
    t0 = time.time()
    job = Job(input_ids=ids, max_new_tokens=DECODE_TOKENS, sampler=GreedySampler())
    generator.enqueue(job)
    while True:
        if any(r.get("eos") for r in generator.iterate()):
            break
    dt = time.time() - t0
    print(f"[profile] decode {DECODE_TOKENS} tokens in {dt:.3f}s "
          f"= {DECODE_TOKENS/dt:.2f} tok/s", flush=True)

    if PREFILL_TOKENS > 0:
        chunk = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 64,
                                 add_bos=False)
        pids = ids
        while pids.shape[1] < PREFILL_TOKENS:
            pids = torch.cat([pids, chunk], dim=1)
        pids = pids[:, :PREFILL_TOKENS]
        t0 = time.time()
        job = Job(input_ids=pids, max_new_tokens=1, sampler=GreedySampler())
        generator.enqueue(job)
        while True:
            if any(r.get("eos") for r in generator.iterate()):
                break
        dt = time.time() - t0
        print(f"[profile] prefill {PREFILL_TOKENS} tokens in {dt:.3f}s "
              f"= {PREFILL_TOKENS/dt:.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
