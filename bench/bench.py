import os, time, json
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
GEN_TOKENS = int(os.environ.get("EXL3_GEN_TOKENS", "128"))


def vram_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**30


def run_gen(generator, ids, max_new_tokens):
    job = Job(input_ids=ids, max_new_tokens=max_new_tokens, sampler=GreedySampler())
    generator.enqueue(job)
    while True:
        for r in generator.iterate():
            if r.get("eos"):
                return r


def main():
    t0 = time.time()
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    print(f"[load] {time.time()-t0:.1f}s  vram={vram_gb():.2f} GB", flush=True)

    base_ids = tokenizer.encode(
        "The history of computing machinery begins in the nineteenth century with",
        add_bos=True)
    chunk = tokenizer.encode(
        "Machine learning models have grown rapidly in scale over the past decade. " * 40,
        add_bos=False)
    prompts = {"short": base_ids}
    for name, n in (("ctx1k", 1024), ("ctx4k", 4096)):
        ids = base_ids
        while ids.shape[1] < n:
            ids = torch.cat([ids, chunk], dim=1)
        prompts[name] = ids[:, :n]

    results = {}

    # Warmup (allocator + JIT paths)
    run_gen(generator, prompts["short"], 8)

    # Decode TPS, short context, best of 3
    gens = []
    for _ in range(3):
        r = run_gen(generator, prompts["short"], GEN_TOKENS)
        gens.append(r["new_tokens"] / r["time_generate"])
    results["decode_short_tps"] = gens

    # Prefill + decode at longer context
    for name in ("ctx1k", "ctx4k"):
        r = run_gen(generator, prompts[name], GEN_TOKENS)
        pp = prompts[name].shape[1] / r["time_prefill"]
        dg = r["new_tokens"] / r["time_generate"]
        results[f"prefill_{name}_tps"] = pp
        results[f"decode_{name}_tps"] = dg

    # Prefill scaling
    for pp_len in (512, 2048, 4096):
        ids = prompts["short"]
        while ids.shape[1] < pp_len:
            ids = torch.cat([ids, chunk], dim=1)
        ids = ids[:, :pp_len]
        r = run_gen(generator, ids, 2)
        results[f"prefill_{pp_len}_tps"] = pp_len / r["time_prefill"]

    # Batch aggregate: 8 concurrent short jobs
    for i in range(8):
        generator.enqueue(Job(input_ids=prompts["short"] + i,
                              max_new_tokens=GEN_TOKENS, sampler=GreedySampler()))
    t0 = time.time()
    done = 0
    while done < 8:
        for r in generator.iterate():
            if r.get("eos"):
                done += 1
    wall = time.time() - t0
    results["decode_batch8_tps"] = 8 * GEN_TOKENS / wall

    print(json.dumps(results, indent=2), flush=True)
    with open("/tmp/bench_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
