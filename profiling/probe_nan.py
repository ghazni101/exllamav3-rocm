#!/usr/bin/env python3
"""Model-level NaN bisect: spy every Linear's forward during a two-token generation on a
short prompt; report the FIRST module whose output is non-finite, with its (k, n, K, rows).
Run twice: EXL3_INT8_MSQ=0 (incumbent) and default (T1 route)."""
import os, torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 32768)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok)

events = []
spies = []

def make_spy(mod, key):
    orig = type(mod).forward
    def spy(self, x, params, out_dtype = None):
        rows = x.numel() // x.shape[-1]
        out = orig(self, x, params, out_dtype)
        if rows > 1 and len(events) < 4000:
            fin = torch.isfinite(out).all().item()
            events.append((key, rows, self.in_features, self.out_features, self.inner.K, bool(fin)))
        return out
    spies.append((mod, orig))
    return spy

for m in model.modules:
    for attr in ("attn", "mlp"):
        blk = getattr(m, attr, None)
        if blk is None:
            continue
        for name in dir(blk):
            try:
                lin = getattr(blk, name)
            except Exception:
                continue
            if type(lin).__name__ == "Linear" and getattr(lin, "key", None):
                type(lin).forward = make_spy(lin, lin.key)

lm = [m for m in model.modules if type(m).__name__ == "Linear"]
for lin in lm:
    type(lin).forward = make_spy(lin, lin.key)

ids = tok.encode("Write a detailed essay about the history of computing.", add_bos = True)
j = Job(input_ids = ids, max_new_tokens = 2, sampler = GreedySampler())
gen.enqueue(j)
for r in gen.iterate():
    if r.get("eos"):
        break

for mod, orig in spies:
    type(mod).forward = orig

bad = [e for e in events if not e[5]]
print(f"events={len(events)} non-finite={len(bad)}")
if bad:
    first = bad[0]
    print(f"FIRST NON-FINITE: key={first[0]} rows={first[1]} k={first[2]} n={first[3]} K={first[4]}")
    for e in bad[:8]:
        print(f"  bad: {e[0]} rows={e[1]} k={e[2]} n={e[3]} K={e[4]}")
else:
    print("all linear outputs finite")
print(f"env EXL3_INT8_MSQ={os.environ.get('EXL3_INT8_MSQ', 'unset')}")
