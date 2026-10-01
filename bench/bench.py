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


# Each benchmark prompt is built from its own filler text: the prefix cache keeps
# completed pages content-hashed, so two prompts sharing a leading run of pages
# would let the second skip most of its prefill and the tps number would measure
# cache hits, not the prefill kernels (2026-10-01 review). The eos result reports
# cached_tokens; every prefill metric below asserts it is ~0 so a future prompt
# overlap fails loudly instead of silently inflating.
FILLERS = {
    "ctx1k": "The development of printed books changed how knowledge spread across early modern Europe. ",
    "ctx4k": "Ocean currents transport heat around the planet and shape regional climates in complex ways. ",
    512:    "Advances in materials science have repeatedly redefined what engineering can build. ",
    2048:   "The evolution of musical notation reflects centuries of changing performance practice. ",
    4096:   "Urban planning decisions made in one decade constrain transportation choices for generations. ",
}
BATCH_PROMPTS = [
    "Summarize the causes of the industrial revolution in three paragraphs.",
    "Describe how a modern GPU executes thousands of threads. Be specific about warps.",
    "Write a short story about a lighthouse keeper who discovers a strange signal.",
    "List the planets and one distinguishing fact about each.",
    "Explain how public-key cryptography works to a curious teenager.",
    "Compare and contrast the Roman Republic with the Roman Empire.",
    "Write a Python one-liner that reverses the words in a sentence, then explain it.",
    "What are the primary greenhouse gases and their main sources?",
]


def build_prompt(tokenizer, filler, n):
    ids = tokenizer.encode(
        "The history of computing machinery begins in the nineteenth century with",
        add_bos=True)
    chunk = tokenizer.encode(filler * 64, add_bos=False)
    while ids.shape[1] < n:
        ids = torch.cat([ids, chunk], dim=1)
    return ids[:, :n]


def check_uncached(r, tag, results):
    cached = int(r.get("cached_tokens", 0))
    results[f"cached_tokens_{tag}"] = cached
    if cached > 8:
        raise RuntimeError(
            f"bench {tag}: prefill hit {cached} cached tokens - prompt overlap with an "
            f"earlier run invalidates the prefill timing; give this prompt its own filler")


def main():
    t0 = time.time()
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    print(f"[load] {time.time()-t0:.1f}s  vram={vram_gb():.2f} GB", flush=True)

    prompts = {"short": tokenizer.encode(
        "The history of computing machinery begins in the nineteenth century with",
        add_bos=True)}
    for name, n in (("ctx1k", 1024), ("ctx4k", 4096)):
        prompts[name] = build_prompt(tokenizer, FILLERS[name], n)

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
        check_uncached(r, name, results)
        pp = prompts[name].shape[1] / r["time_prefill"]
        dg = r["new_tokens"] / r["time_generate"]
        results[f"prefill_{name}_tps"] = pp
        results[f"decode_{name}_tps"] = dg

    # Prefill scaling (each length has its own filler: no prompt is a prefix of another)
    for pp_len in (512, 2048, 4096):
        ids = build_prompt(tokenizer, FILLERS[pp_len], pp_len)
        r = run_gen(generator, ids, 2)
        check_uncached(r, f"scale{pp_len}", results)
        results[f"prefill_{pp_len}_tps"] = pp_len / r["time_prefill"]

    # Batch aggregate: 8 concurrent jobs, distinct real prompts (token-id shifting
    # produces garbage inputs and can early-EOS, overcounting throughput)
    done_results = []
    for text in BATCH_PROMPTS:
        generator.enqueue(Job(input_ids=tokenizer.encode(text, add_bos=True),
                              max_new_tokens=GEN_TOKENS, sampler=GreedySampler()))
    t0 = time.time()
    done = 0
    while done < len(BATCH_PROMPTS):
        for r in generator.iterate():
            if r.get("eos"):
                done += 1
                done_results.append(r)
    wall = time.time() - t0
    total_new = sum(r["new_tokens"] for r in done_results)
    results["decode_batch8_tps"] = total_new / wall
    results["decode_batch8_tokens"] = total_new

    print(json.dumps(results, indent=2), flush=True)
    with open("/tmp/bench_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
