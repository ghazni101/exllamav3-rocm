#!/usr/bin/env python3
"""Correctness gate: golden-token parity + batch/sequential equivalence.

Modes:
  baseline <out.json>   run the fixed prompt set on the CURRENT build, save token ids
  compare  <baseline>   re-run and require identical greedy token ids (hard gate);
                        exits 1 on any mismatch
  batch     <baseline>  run 4 mid-length prompts sequentially (m<=4 sq GEMV path) and
                        then concurrently (m>1 msq/batched path); sequences must match

Greedy decoding is deterministic: any numeric change (grid size, reduction order,
tiling) shows up as divergent token ids long before it is visible as loss.
"""
import os, sys, json, time
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
CACHE_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
GEN = int(os.environ.get("GATE_GEN_TOKENS", "256"))

FILLER = "Machine learning models have grown rapidly in scale over the past decade. "
BASE = "The history of computing machinery begins in the nineteenth century with "

def build_prompts(tok):
    def long_prompt(n):
        ids = tok.encode(BASE, add_bos=True)
        chunk = tok.encode(FILLER, add_bos=False)
        while ids.shape[1] < n:
            ids = torch.cat([ids, chunk], dim=1)
        return ids[:, :n]
    texts = [
        ("short_general", "Write a detailed essay about the history of computing."),
        ("numeric", "What is 17 * 23? Explain step by step, then give the final number."),
        ("code", "Write a Python function that merges two sorted lists without using sort()."),
        ("unicode", "Explain the meaning of these emojis: 🚀 🌍 ⚗️ 🤖, in French."),
        ("reasoning", "Alice has 3 brothers and 2 sisters. How many sisters does Alice's brother have? Reason carefully."),
    ]
    out = [(n, tok.encode(t, add_bos=True)) for n, t in texts]
    out.append(("longctx_4k", long_prompt(4160)))
    shared = long_prompt(1536)
    out.append(("prefix_A", torch.cat([shared, tok.encode(" At the end of this passage, write the word BLUE.", add_bos=False)], dim=1)))
    out.append(("prefix_B", torch.cat([shared, tok.encode(" At the end of this passage, write the word RED.", add_bos=False)], dim=1)))
    return out

def main():
    mode, path = sys.argv[1], sys.argv[2]
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    model.load()
    tok = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tok)

    def run_jobs(jobs_spec, concurrent):
        """jobs_spec: list[(name, ids, n)]. Returns {name: [seq token ids incl prompt]}."""
        jobs = []
        t_start = time.time()
        if concurrent:
            for nm, ids, n in jobs_spec:
                j = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler())
                generator.enqueue(j)
                jobs.append((nm, j))
            done = 0
            while done < len(jobs_spec):
                for r in generator.iterate():
                    if r.get("eos"):
                        done += 1
        else:
            for nm, ids, n in jobs_spec:
                j = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler())
                generator.enqueue(j)
                jobs.append((nm, j))
                done = 0
                while done < 1:
                    for r in generator.iterate():
                        if r.get("eos"):
                            done += 1
        out = {}
        for nm, j in jobs:
            ids = j.sequences[0].sequence_ids.torch().flatten().tolist()
            out[nm] = [int(x) for x in ids]
        print(f"[gate] {len(jobs_spec)} jobs ({'concurrent' if concurrent else 'sequential'}) "
              f"in {time.time()-t_start:.1f}s", flush=True)
        return out

    t0 = time.time()
    if mode in ("baseline", "compare"):
        jobs_spec = [(n, ids, GEN) for n, ids in build_prompts(tok)]
        res = run_jobs(jobs_spec, concurrent=False)
        results = {n: v for n, v in res.items()}
        if mode == "baseline":
            json.dump({"meta": {"model": MODEL_DIR, "gen": GEN, "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                       "tokens": results}, open(path, "w"))
            print(f"[gate] baseline written to {path} ({time.time()-t0:.0f}s)")
        else:
            ref = json.load(open(path))["tokens"]
            bad = []
            for n, a in ref.items():
                b = results.get(n, [])
                if a != b:
                    div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    print(f"[gate] MISMATCH {n}: diverges at generated-token {div} "
                          f"(ref len {len(a)}, got len {len(b)})")
                    bad.append(n)
            if bad:
                print(f"[gate] FAIL: {len(bad)}/{len(ref)} prompts diverged: {bad}")
                sys.exit(1)
            print(f"[gate] PASS: all {len(ref)} prompts token-identical ({time.time()-t0:.0f}s)")

    elif mode == "batch":
        names = ["batch_a", "batch_b", "batch_c", "batch_d"]
        ptexts = [
            "Summarize the causes of the industrial revolution in three paragraphs.",
            "Describe how a modern GPU executes thousands of threads. Be specific about warps.",
            "Write a short story about a lighthouse keeper who discovers a strange signal.",
            "List the planets and one distinguishing fact about each.",
        ]
        pids = [tok.encode(t, add_bos=True) for t in ptexts]
        spec = list(zip(names, pids, [128] * 4))
        seq = run_jobs(spec, concurrent=False)
        conc = run_jobs(spec, concurrent=True)
        bad = [n for n in names if seq[n] != conc[n]]
        if bad:
            print(f"[gate] BATCH MISMATCH: {bad}")
        else:
            print(f"[gate] PASS: batch-vs-sequential token-identical ({time.time()-t0:.0f}s)")
        if path != "-":
            json.dump({"sequential": seq, "concurrent": conc}, open(path, "w"))
        sys.exit(0 if not bad else 1)

if __name__ == "__main__":
    main()
