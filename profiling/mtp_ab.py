"""One-prompt greedy decode, with or without the in-checkpoint MTP head.

model_init --mtp loads the mtp component of the same directory as the draft.
Prints one JSON line: tok/s over the timed generation, and draft acceptance.
"""
import argparse
import json
import sys

import torch

from exllamav3 import Generator, GreedySampler, Job, model_init

PROMPT = (
    "Explain, in a few paragraphs, how a residual stream in a transformer "
    "carries information from one layer to the next, and why a low-rank "
    "update can still change the next-token distribution."
)


def run(generator, tokenizer, n):
    ids = tokenizer.encode(PROMPT, add_bos=True)
    job = Job(
        input_ids=ids,
        max_new_tokens=n,
        stop_conditions=[],
        sampler=GreedySampler(),
    )
    generator.enqueue(job)
    last = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            last = result
    new_tokens = last["new_tokens"]
    gen_s = last["time_generate"]
    dacc = last.get("accepted_draft_tokens")
    drej = last.get("rejected_draft_tokens")
    text = last.get("full_completion") or last.get("text") or ""
    return {
        "new_tokens": new_tokens,
        "gen_s": gen_s,
        "tok_s": new_tokens / gen_s if gen_s else None,
        "accepted": dacc,
        "rejected": drej,
        "acc_per_round": (dacc / (new_tokens - dacc)) if dacc is not None and new_tokens != dacc else None,
        "preview": text[:180].replace("\n", " "),
    }


def main():
    parser = argparse.ArgumentParser()
    model_init.add_args(
        parser,
        default_cache_size=8192,
        add_draft_model_args=True,
        default_chunk_size=2048,
    )
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=192)
    args = parser.parse_args()
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(
        model=model,
        cache=cache,
        draft_model=draft_model,
        draft_cache=draft_cache,
        tokenizer=tokenizer,
        num_draft_tokens=args.num_draft_tokens,
        max_chunk_size=args.chunk_size,
    )
    run(generator, tokenizer, args.warmup)
    torch.cuda.synchronize()
    timed = run(generator, tokenizer, args.tokens)
    timed["mtp"] = bool(args.mtp)
    timed["draft_tokens"] = args.num_draft_tokens
    print("JSON " + json.dumps(timed), flush=True)


if __name__ == "__main__":
    main()
