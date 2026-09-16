# Golden-output parity harness for the ROCm perf work (docs/rocm-perf-baseline.md, gate #1).
#
# Greedy-decodes a fixed prompt set through the generator and saves token IDs + text to JSON,
# or compares a saved baseline against the current build. Greedy decode is deterministic for a
# fixed kernel configuration, so any token divergence means a numeric/semantic regression.
# A change that intentionally alters numerics (e.g. enabling WMMA) must first be A/B-verified
# at kernel level (see test_msq_ab.py / tests); this harness then catches end-to-end drift.
#
# Usage (container):
#   python3 /opt/exllamav3/golden_rocm.py --save /out/golden_baseline.json
#   python3 /opt/exllamav3/golden_rocm.py --check /out/golden_baseline.json
import argparse, json, os, sys, time
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
GEN_TOKENS = int(os.environ.get("EXL3_GEN_TOKENS", "96"))

REPEAT = "Machine learning models have grown rapidly in scale over the past decade. "

PROMPTS = [
    "The history of computing machinery begins in the nineteenth century with",
    "Write a Python function that merges two sorted lists without using sort():",
    "Explain the difference between TCP and UDP to a network engineer:",
    "Derive the quadratic formula step by step:",
    "Le chat aime dormir sur le clavier parce que",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    REPEAT * 24 + "Summarize the passage above in one sentence:",          # ~1.3k ctx
    REPEAT * 44 + "What is the one word that appears most often in the text above?",  # ~2.4k ctx
]


def run_job(generator, ids, max_new_tokens):
    job = Job(input_ids=ids, max_new_tokens=max_new_tokens, sampler=GreedySampler())
    generator.enqueue(job)
    # Per-result token_ids is that step's slice, so accumulate across results
    chunks = []
    while True:
        for r in generator.iterate():
            t = r.get("token_ids")
            if t is not None:
                chunks.append(t[0].tolist())
            if r.get("eos"):
                return [i for c in chunks for i in c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--check", type=str, default=None)
    args = ap.parse_args()
    if not args.save and not args.check:
        print("usage: golden_rocm.py --save OUT.json | --check REF.json", file=sys.stderr)
        sys.exit(2)

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    entries = []
    for i, prompt in enumerate(PROMPTS):
        ids = tokenizer.encode(prompt, add_bos=True)
        t0 = time.time()
        out_list = run_job(generator, ids, GEN_TOKENS)
        dt = time.time() - t0
        entries.append({
            "prompt": prompt,
            "prompt_tokens": int(ids.shape[1]),
            "gen_tokens": len(out_list),
            "token_ids": out_list,
            "text": tokenizer.decode(torch.tensor([out_list], dtype=torch.long)),
        })
        print(f"[golden {i+1}/{len(PROMPTS)}] {ids.shape[1]}+{len(out_list)} tok "
              f"in {dt:.1f}s", flush=True)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(entries, f, indent=1)
        print(f"saved {len(entries)} golden outputs to {args.save}")
        return

    with open(args.check) as f:
        ref = json.load(f)
    ok = True
    if len(ref) != len(entries):
        print(f"FAIL: prompt count {len(entries)} != baseline {len(ref)}")
        sys.exit(1)
    for i, (r, e) in enumerate(zip(ref, entries)):
        if r["prompt"] != e["prompt"]:
            print(f"FAIL: prompt {i} mismatch")
            ok = False
            continue
        n = min(len(r["token_ids"]), len(e["token_ids"]))
        first_diff = next((j for j in range(n)
                           if r["token_ids"][j] != e["token_ids"][j]), None)
        if first_diff is not None or len(r["token_ids"]) != len(e["token_ids"]):
            j = first_diff if first_diff is not None else n
            print(f"FAIL: prompt {i} diverges at generated token {j}: "
                  f"baseline {r['token_ids'][j:j+6]} vs now {e['token_ids'][j:j+6]}")
            ok = False
        else:
            print(f"ok: prompt {i} ({e['prompt_tokens']}+{e['gen_tokens']} tok) identical")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
