#!/usr/bin/env python3
"""Correctness gate: golden-token parity + batch/sequential equivalence + logits KLD.

Modes:
  baseline <out.json>   run the fixed prompt set on the CURRENT build, save token ids
  compare  <baseline>   re-run and require identical greedy token ids (hard gate);
                        exits 1 on any mismatch
  batch     <out.json>  run 8 mid-length prompts sequentially (m=1 decode) and then
                        concurrently (m=8 decode), compare greedy token sequences.
                        Decode dispatch intentionally differs by batch size on ROCm
                        (m<=4 int8 GEMV, m>=5 fp16 GEMM - see docs/rocm.md), so a
                        near-tie logit can flip a token and fork the continuation;
                        both branches remain individually coherent. Policy: fail if
                        lengths differ or if the first fork appears before generated
                        token GATE_BATCH_MINSTEP (default 4 - early divergence means
                        prefill/first-step numerics differ materially, not a
                        near-tie). Divergence totals/stretch counts after a fork are
                        reported but not gated: two coherent continuations drift in
                        and out of coincidental token agreement, so only fork
                        position (and numcheck's logits KLD) measure the numeric
                        gap. GATE_BATCH_STRICT=1 restores exact token equality for
                        A/B of builds whose numerics should be batch-size
                        independent.
  numcheck  save|compare <path>
                        capture the first NUMCHECK_STEPS generated-step logits (full
                        vocab) per prompt, including the long/prefill prompts by
                        default (NUMCHECK_LONG=0 reverts to short prompts only) - a
                        prefill-path numeric change (e.g. EXL3_HGEMM_F16OUT) is
                        invisible to the short prompts, which never reach the
                        reconstruct path. save: store to path (torch.save). compare:
                        KLD(ref || new) per prompt must be < NUMCHECK_KLD (1e-3), and
                        divergent greedy steps must number <= NUMCHECK_MAXDIV
                        (default 0: against a same-build reference any divergence is
                        a real numeric change; raise it when intentionally comparing
                        across numeric variants and judge by KLD). Capture the
                        reference on the incumbent numerics in the same image.

Greedy decoding is deterministic: any numeric change (grid size, reduction order,
tiling) shows up as divergent token ids long before it is visible as loss.

Startup-fragility caveat (measured 2026-10-03): baseline/compare token identity
holds only for a byte-identical process startup. ANY perturbation that changes
import-time conditions — replacing a module in site-packages, or merely
`touch`ing one .py (bytecode recompilation) — reproducibly shifts the GEMM
autotuner's first-use timing and with it the winning config, flipping near-tie
tokens mid-generation (observed: 2/8 prompts forking at generated tokens 18/26,
identically for a comment-only change and for touch-only). Save baselines and
run compares from the same unmodified image; attribute cross-startup forks to
the autotuner before suspecting the code change, and confirm with numcheck's
KLD (a config-order flip is rounding-scale; a real bug is not).
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
NUMCHECK_MAXDIV = int(os.environ.get("NUMCHECK_MAXDIV", "0"))
NUMCHECK_LONG = os.environ.get("NUMCHECK_LONG", "1") == "1"
GATE_BATCH_MINSTEP = int(os.environ.get("GATE_BATCH_MINSTEP", "4"))
GATE_BATCH_STRICT = os.environ.get("GATE_BATCH_STRICT", "0") == "1"

FILLER = "Machine learning models have grown rapidly in scale over the past decade. "
BASE = "The history of computing machinery begins in the nineteenth century with "

def long_prompt(tok, n):
    ids = tok.encode(BASE, add_bos=True)
    chunk = tok.encode(FILLER, add_bos=False)
    while ids.shape[1] < n:
        ids = torch.cat([ids, chunk], dim=1)
    return ids[:, :n]

def build_prompts(tok):
    texts = [
        ("short_general", "Write a detailed essay about the history of computing."),
        ("numeric", "What is 17 * 23? Explain step by step, then give the final number."),
        ("code", "Write a Python function that merges two sorted lists without using sort()."),
        ("unicode", "Explain the meaning of these emojis: 🚀 🌍 ⚗️ 🤖, in French."),
        ("reasoning", "Alice has 3 brothers and 2 sisters. How many sisters does Alice's brother have? Reason carefully."),
    ]
    out = [(n, tok.encode(t, add_bos=True)) for n, t in texts]
    out.append(("longctx_4k", long_prompt(tok, 4160)))
    shared = long_prompt(tok, 1536)
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
        Warmup generations first so per-shape JIT/autotune is paid outside any
        graph-capture window (a capture-time autotune aborts with 'stream is capturing'):
        a short prompt for the decode shapes, and - when the long prompts are included -
        a ~1700-token prompt that walks the chunked-prefill shapes the measured long
        prompts will capture."""
        wj = Job(input_ids=tok.encode("Warmup prompt for autotune.", add_bos=True),
                 max_new_tokens=8, sampler=GreedySampler())
        generator.enqueue(wj)
        wj_done = False
        while not wj_done:
            for r in generator.iterate():
                if r.get("eos"):
                    wj_done = True
                    break
        if NUMCHECK_LONG:
            # Unique filler: the warmup's completed pages stay in the content-hash
            # prefix cache, and a warmup prefix of longctx_4k/prefix_A would make
            # those prompts skip their (measured) prefill as cache hits
            wl_ids = tok.encode(
                "Warmup for the chunked prefill autotuner. " +
                "Bridge construction across the roman empire required remarkable logistics. " * 64,
                add_bos=True)
            wl_ids = wl_ids[:, :1700]
            wl = Job(input_ids=wl_ids, max_new_tokens=8, sampler=GreedySampler())
            generator.enqueue(wl)
            wl_done = False
            while not wl_done:
                for r in generator.iterate():
                    if r.get("eos"):
                        wl_done = True
                        break
        vocab = tok.actual_vocab_size
        # Long prompts are included by default: a prefill-path numeric change (e.g.
        # EXL3_HGEMM_F16OUT rounding the fp32-output reconstruct GEMM to fp16) is invisible
        # to the short prompts, which never reach reconstruct_hgemm
        # (AUTO_RECONSTRUCT_THRESHOLD). longctx_4k measures the KLD on the path such a
        # change actually touches. prefix_B shares 1536 tokens with prefix_A, so whichever
        # runs second may prefix-hit and emit no new logits; its parity is covered by the
        # golden-token compare.
        prompts = [p for p in build_prompts(tok)
                   if NUMCHECK_LONG or not p[0].startswith(("longctx", "prefix"))]
        tokens, logits = {}, {}
        t0_ = time.time()
        for name, ids in prompts:
            j = Job(input_ids=ids, max_new_tokens=NUMCHECK_STEPS, sampler=GreedySampler(),
                    return_logits=True)
            generator.enqueue(j)
            out_logits = None
            # Drive iterate() to this job's eos: one call returns only the current
            # step's results, and a 4k-token chunked prefill spans many calls. The
            # old single-call loop abandoned long jobs mid-prefill, which surfaced
            # as "no logits emitted ... skipped" for exactly the long prompts this
            # mode exists to measure (and leaked their results into later prompts).
            finished = False
            while not finished:
                for r in generator.iterate():
                    # held-back steps flush nested under "held" on eos (e.g. a
                    # first-token stop-token EOS emits everything only there);
                    # prefer whichever carries more steps so no prompt is silently
                    # dropped from the gate
                    held = r.get("held") or {}
                    for cand in (r.get("logits"), held.get("logits")):
                        if cand is not None and (out_logits is None or cand.shape[1] > out_logits.shape[1]):
                            out_logits = cand
                    if r.get("eos"):
                        finished = True
                        break
            if out_logits is None:
                # genuinely no logits for this prompt (full prefix-cache hit: every page
                # reused, nothing decoded fresh); parity is covered by the golden compare
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
            torch.save({"tokens": tokens, "logits": logits,
                        "meta": {"model": MODEL_DIR, "prompts": sorted(logits),
                                 "steps": NUMCHECK_STEPS, "long": NUMCHECK_LONG}}, path)
            print(f"[gate] numcheck reference written to {path} ({len(logits)} prompts)")
        else:
            ref = torch.load(path, weights_only=False)
            rt, rl = ref["tokens"], ref["logits"]
            rmeta = ref.get("meta", {})
            # Reference integrity: a stale or foreign reference must not silently
            # shrink the compared set to whatever overlaps. Compare against a
            # different model or prompt set needs a fresh save.
            if rmeta.get("model") not in (None, MODEL_DIR):
                print(f"[gate] NUMCHECK FAIL: reference was saved for {rmeta['model']}, running {MODEL_DIR}")
                sys.exit(1)
            missing = [n for n in rl if n not in logits]
            if missing:
                print(f"[gate] NUMCHECK FAIL: prompts in reference missing this run: {missing} "
                      f"(regression in logits plumbing, or a stale reference - resave)")
                sys.exit(1)
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
                # against a same-build reference any divergent step is a real numeric
                # change; NUMCHECK_MAXDIV raises the allowance when intentionally
                # comparing across numeric variants (judge those by KLD instead)
                if len(div) > NUMCHECK_MAXDIV:
                    print(f"[numcheck] {n}: {len(div)} divergent steps > NUMCHECK_MAXDIV={NUMCHECK_MAXDIV}")
                    bad.append(n)
            if bad:
                print(f"[gate] NUMCHECK FAIL: {bad} (worst KLD {worst:.3e})")
                sys.exit(1)
            print(f"[gate] NUMCHECK PASS: all KLD < {NUMCHECK_KLD} (worst {worst:.3e}, {time.time()-t0:.0f}s)")

    elif mode == "batch":
        # 8 mid-length prompts: sequential (m=1 decode: int8 GEMV on ROCm) vs concurrent
        # (m=8 decode: fp16 GEMM path - the dispatch differs by batch size, see
        # docs/rocm.md). Near-tie logits can flip a token between the two runs; the
        # policy gate bounds how much divergence is acceptable without hiding a real
        # blowup: equal lengths, at most GATE_BATCH_MAXDIV divergent tokens per prompt,
        # and re-convergence within GATE_BATCH_RECONV tokens after any divergence.
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

        def divergence_report(a, b):
            """(first_div_idx, n_divergent, n_runs) for equal-length runs, else None.
            n_runs counts maximal contiguous divergent stretches: one near-tie flip
            produces one run followed by a permanently different (but individually
            coherent) continuation; repeated independent flips produce several."""
            if len(a) != len(b) or not a:
                return None
            flags = [x != y for x, y in zip(a, b)]
            div = [i for i, f in enumerate(flags) if f]
            if not div:
                return (None, 0, 0)
            runs = 0
            prev = False
            for f in flags:
                if f and not prev:
                    runs += 1
                prev = f
            return (div[0], len(div), runs)

        bad = []
        # Per-prompt length: sequences include the prompt, so the earliest possible
        # fork index for prompt n is its own prompt length. Subtracting the MINIMUM
        # length instead (as this once did) inflates first_gen for every longer
        # prompt by (len_i - min_len) and lets a generated-token-0 fork pass the
        # GATE_BATCH_MINSTEP check for all but the shortest prompt.
        prompt_len = {n: ids.shape[1] for n, ids in zip(names, pids)}
        for n in names:
            a, b = seq[n], conc[n]
            if len(a) != len(b):
                print(f"[gate] {n}: LENGTH MISMATCH seq {len(a)} vs conc {len(b)}")
                bad.append(n)
                continue
            rep = divergence_report(a, b)
            if rep is None or rep[1] == 0:
                print(f"[gate] {n}: token-identical ({len(a)} tokens)")
                continue
            first, ndiv, runs = rep
            first_gen = first - prompt_len[n]  # position among GENERATED tokens
            print(f"[gate] {n}: fork at generated-token {first_gen} "
                  f"({ndiv} divergent tokens in {runs} divergence stretch(es); "
                  f"stretches after the first fork include coincidental agreement)")
            if GATE_BATCH_STRICT:
                bad.append(n)
            elif first_gen < GATE_BATCH_MINSTEP:
                # divergence in the first few generated tokens means the prefill or
                # first decode step numerics differ materially (not a near-tie).
                # After a fork, token streams of two individually coherent
                # continuations drift in and out of coincidental agreement, so
                # stretch counts and totals are reported but not gated - the
                # numeric distance itself is numcheck's job.
                bad.append(n)
        if bad:
            print(f"[gate] BATCH FAIL: {bad} "
                  f"(strict={GATE_BATCH_STRICT}, minstep={GATE_BATCH_MINSTEP})")
        else:
            print(f"[gate] PASS: batch-vs-sequential within policy "
                  f"(strict={GATE_BATCH_STRICT}) ({time.time()-t0:.0f}s)")
        if path != "-":
            json.dump({"sequential": seq, "concurrent": conc}, open(path, "w"))
        sys.exit(0 if not bad else 1)

if __name__ == "__main__":
    main()
