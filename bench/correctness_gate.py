#!/usr/bin/env python3
"""Correctness gate: golden-token parity + batch/sequential equivalence + logits KLD.

Modes:
  baseline <out.json>   run the fixed prompt set on the CURRENT build, save token ids
  compare  <baseline>   re-run and require identical greedy token ids (hard gate);
                        exits 1 on any mismatch
  batch     <baseline>  run 4 mid-length prompts sequentially (m<=4 sq GEMV path) and
                        then concurrently (m>1 msq/batched path); sequences must match
  numcheck  save|compare <path>
                        capture the first NUMCHECK_STEPS generated-step logits (full
                        vocab) per prompt. save: store to path (torch.save). compare:
                        KLD(ref || new) per prompt must be < NUMCHECK_KLD (1e-3) - the
                        gate for changes that intentionally alter numerics; token
                        divergence vs the reference run is reported per prompt. Capture
                        the reference on the incumbent numerics in the same image.

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
NUMCHECK_STEPS = int(os.environ.get("NUMCHECK_STEPS", "8"))
NUMCHECK_KLD = float(os.environ.get("NUMCHECK_KLD", "1e-3"))

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

    def run_numcheck():
        """Prompt set with return_logits: the eos result carries (1, n_steps, vocab) logits.
        A short warmup generation first, so per-shape JIT/autotune is paid outside any
        graph-capture window (a capture-time autotune aborts with 'stream is capturing')."""
        wj = Job(input_ids=tok.encode("Warmup prompt for autotune.", add_bos=True),
                 max_new_tokens=8, sampler=GreedySampler())
        generator.enqueue(wj)
        for r in generator.iterate():
            if r.get("eos"):
                break
        vocab = tok.actual_vocab_size
        # Only the short prompts: the T1 route changes prefill numerics for m in 5..144
        # exactly there. The long/reconstruct prompts keep bit-identical numerics and are
        # covered end-to-end by the golden-token compare; running them here would drag the
        # chunked-prefill graph capture (and its autotune-during-capture fragility) into the
        # KLD gate.
        # NUMCHECK_LONG=1 is the numeric-change variant of this gate: a change that intentionally
        # alters reconstruct-path numerics (e.g. EXL3_HGEMM_F16OUT, which rounds the fp32-output
        # prefill GEMM result to fp16) is invisible to the short prompts, which never reach
        # reconstruct_hgemm (AUTO_RECONSTRUCT_THRESHOLD = 144). Including the long prompts measures
        # the KLD the change actually causes on the path it touches.
        sel = os.environ.get("NUMCHECK_LONG") == "1"
        prompts = [p for p in build_prompts(tok)
                   if sel or not p[0].startswith(("longctx", "prefix"))]
        tokens, logits = {}, {}
        t0_ = time.time()
        for name, ids in prompts:
            j = Job(input_ids=ids, max_new_tokens=NUMCHECK_STEPS, sampler=GreedySampler(),
                    return_logits=True)
            generator.enqueue(j)
            out_logits = None
            for r in generator.iterate():
                if r.get("logits") is not None:
                    out_logits = r["logits"]
                if r.get("eos"):
                    break
            if out_logits is None:
                # prefix-cache-hit prompts emit no held logits; their parity is covered by
                # the golden-token compare (phase B)
                print(f"[gate] numcheck: no logits emitted for {name} (prefix hit), skipped")
                continue
            # trim the padded vocab tail: channels beyond actual_vocab_size are uninitialized
            lg = out_logits[0, :NUMCHECK_STEPS, :vocab].float().cpu()
            bad_num = not torch.isfinite(lg).all()
            logits[name] = lg
            if bad_num:
                print(f"[gate] WARNING: non-finite logits in {name} (will fail the gate)")
            seq = j.sequences[0].sequence_ids.torch().flatten().tolist()
            tokens[name] = [int(x) for x in seq]
        print(f"[gate] numcheck: {len(prompts)} prompts x {NUMCHECK_STEPS} steps in {time.time()-t0_:.1f}s", flush=True)
        return tokens, logits

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

    elif mode == "numcheck":
        sub = sys.argv[2]
        path = sys.argv[3]
        tokens, logits = run_numcheck()
        if sub == "save":
            torch.save({"tokens": tokens, "logits": logits}, path)
            print(f"[gate] numcheck reference written to {path}")
        else:
            ref = torch.load(path, weights_only=False)
            rt, rl = ref["tokens"], ref["logits"]
            worst = 0.0
            bad = []
            if len(logits) < 3:
                print("[gate] NUMCHECK FAIL: fewer than 3 prompts emitted logits")
                sys.exit(1)
            for n, lg in logits.items():
                if n not in rl:
                    print(f"[numcheck] {n}: not in reference, skipped")
                    continue
                rlg = rl[n]
                k = min(lg.shape[0], rlg.shape[0])
                if not (torch.isfinite(rlg[:k]).all() and torch.isfinite(lg[:k]).all()):
                    print(f"[numcheck] {n}: NON-FINITE LOGITS")
                    bad.append(n)
                    continue
                logp_ref = torch.log_softmax(rlg[:k].double(), dim=-1)
                logp_new = torch.log_softmax(lg[:k].double(), dim=-1)
                p_ref = logp_ref.exp()
                # KLD(ref || new); p_ref == 0 terms contribute exactly 0 - mask them so
                # log-space underflow cannot produce nan in the product
                kld = (p_ref * (logp_ref - logp_new)).where(p_ref > 0, torch.zeros_like(p_ref)).sum(-1)
                mk = kld.max().item()
                worst = max(worst, mk)
                div = [i for i in range(k) if tokens[n][len(tokens[n]) - k + i] != rt[n][len(rt[n]) - k + i]]
                print(f"[numcheck] {n}: KLD max {mk:.3e}, divergent steps {div if div else 'none'}")
                if mk >= NUMCHECK_KLD:
                    bad.append(n)
            if bad:
                print(f"[gate] NUMCHECK FAIL: KLD >= {NUMCHECK_KLD} on {bad} (worst {worst:.3e})")
                sys.exit(1)
            print(f"[gate] NUMCHECK PASS: all KLD < {NUMCHECK_KLD} (worst {worst:.3e}, {time.time()-t0:.0f}s)")

    elif mode == "batch":
        # 8 mid-length prompts: sequential (m=1, sq GEMV path) vs concurrent (m=8, the
        # msq path for m>4). The m=8 concurrent run is what exercises tg-1b.
        names = ["batch_a", "batch_b", "batch_c", "batch_d", "batch_e", "batch_f", "batch_g", "batch_h"]
        ptexts = [
            "Summarize the causes of the industrial revolution in three paragraphs.",
            "Describe how a modern GPU executes thousands of threads. Be specific about warps.",
            "Write a short story about a lighthouse keeper who discovers a strange signal.",
            "List the planets and one distinguishing fact about each.",
            "Explain how public-key cryptography works to a curious teenager.",
            "Compare and contrast the Roman Republic with the Roman Empire.",
            "Write a Python one-liner that reverses the words in a sentence, then explain it.",
            "What are the primary greenhouse gases and their main sources?",
        ]
        pids = [tok.encode(t, add_bos=True) for t in ptexts]
        spec = list(zip(names, pids, [128] * len(names)))
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
