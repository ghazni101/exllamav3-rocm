import os, time, json
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

# Fallback hotspot profile via torch.profiler (in-process roctracer), used when
# external rocprofv3 injection crashes on the extension (documented in
# docs/rocm-perf-baseline.md). Captures per-kernel device time + launch counts
# and host-side ops for a handful of steady-state decode steps.

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
WARMUP_TOKENS = 16
PROFILE_TOKENS = int(os.environ.get("EXL3_PROFILE_TOKENS", "6"))


def gen_n(generator, ids, n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler())
    generator.enqueue(job)
    while True:
        if any(r.get("eos") for r in generator.iterate()):
            return


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

    gen_n(generator, ids, WARMUP_TOKENS)  # JIT + allocator warmup
    gen_n(generator, ids, 4)              # extra settle

    # Wall-clock reference for the same number of steps
    t0 = time.time()
    gen_n(generator, ids, PROFILE_TOKENS)
    wall = time.time() - t0
    print(f"[ref] {PROFILE_TOKENS} tokens in {wall:.3f}s = "
          f"{PROFILE_TOKENS/wall:.2f} tok/s", flush=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        t0 = time.time()
        gen_n(generator, ids, PROFILE_TOKENS)
        prof_wall = time.time() - t0

    print(f"[profiler] wall {prof_wall:.3f}s\n", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=45),
          flush=True)
    prof.export_chrome_trace("/tmp/decode_trace.json")

    ka = prof.key_averages()
    rows = []
    for e in ka:
        dt_us = getattr(e, "device_time_total", 0) or 0
        if dt_us <= 0:
            continue
        rows.append({"name": e.key[:120], "device_ms": dt_us / 1000,
                     "count": e.count})
    rows.sort(key=lambda r: -r["device_ms"])
    total_dev = sum(r["device_ms"] for r in rows)
    summary = {
        "wall_s": prof_wall,
        "profiled_tokens": PROFILE_TOKENS,
        "device_total_ms": total_dev,
        "host_gap_ms": max(0.0, prof_wall * 1000 - total_dev),
        "host_gap_pct": 100.0 * max(0.0, prof_wall * 1000 - total_dev) /
                        (prof_wall * 1000),
        "top": rows[:40],
    }
    with open("/tmp/hotspots.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "top"},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
