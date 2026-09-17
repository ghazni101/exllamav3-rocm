"""Fixed batch-one 4096/256 autoregressive benchmark; saves tokens for A/B parity."""
import json
import os
import statistics
import time
from pathlib import Path

import torch
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler

model_dir = os.environ["EXL3_MODEL"]
output = Path(os.environ["TG_OUTPUT"])
config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok)
assert gen.num_draft_tokens == 0
base = tok.encode("The history of computing machinery begins in the nineteenth century with ", add_bos=True)
filler = tok.encode("Machine learning models have grown rapidly in scale over the past decade. ", add_bos=False)
assert isinstance(base, torch.Tensor) and isinstance(filler, torch.Tensor)
prompt = torch.cat([base, filler.repeat(1, 4096 // filler.shape[1] + 1)], dim=1)[:, :4096]
assert prompt.shape == (1, 4096)


def run(n):
    job = Job(input_ids=prompt, max_new_tokens=n, stop_conditions=[], sampler=GreedySampler(), seed=0)
    gen.enqueue(job)
    first = None
    while True:
        events = gen.iterate()
        torch.cuda.synchronize()
        now = time.perf_counter()
        for event in events:
            if event.get("stage") == "streaming" and first is None:
                first = now
            if event.get("eos"):
                assert event["new_tokens"] == n, event
                tokens = job.sequences[0].sequence_ids.torch().flatten().tolist()[-n:]
                return {"tokens": tokens, "generated": n, "engine_tps": n / event["time_generate"],
                        "wall_tps": (n - 1) / (now - first) if first is not None and now > first else None}


run(8)
rows = []
for index in range(3):
    row = run(256)
    rows.append(row)
    print(json.dumps({"run": index + 1, **{k: v for k, v in row.items() if k != "tokens"}}), flush=True)
assert all(row["tokens"] == rows[0]["tokens"] for row in rows), "Repeated greedy outputs differ"
result = {"model": model_dir, "prompt_tokens": 4096, "generation_tokens": 256,
          "kv": "fp16", "speculative": False, "runs": rows,
          "median_engine_tps": statistics.median(r["engine_tps"] for r in rows)}
reference = os.environ.get("TG_REFERENCE")
if reference:
    ref = json.loads(Path(reference).read_text())
    result["token_parity"] = rows[0]["tokens"] == ref["runs"][0]["tokens"]
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "runs"}), flush=True)
if reference:
    assert result["token_parity"], "Greedy output differs from baseline"
