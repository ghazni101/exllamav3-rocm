import os, sys, time
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler.custom import CustomSampler, SS_Argmax

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
NUMCHECK_STEPS = 4

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 8192)
model.load()
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok)

def build_prompts(tok):
    return [
        ("short", tok.encode("Write a detailed essay about the history of computing.", add_bos = True)),
        ("numeric", tok.encode("What is 17 * 23? Explain step by step.", add_bos = True)),
    ]

# exact numcheck pattern
def run_numcheck():
    class CaptureSampler(CustomSampler):
        def __init__(self, store):
            super().__init__([SS_Argmax()])
            self.store = store
            self.step = 0
        def forward(self, logits, *args, **kwargs):
            if self.step < NUMCHECK_STEPS:
                self.store.append(logits[0, -1].float().cpu().clone())
            self.step += 1
            return super().forward(logits, *args, **kwargs)

    prompts = build_prompts(tok)
    tokens, logits = {}, {}
    for name, ids in prompts:
        store = []
        sampler = CaptureSampler(store)
        j = Job(input_ids = ids, max_new_tokens = NUMCHECK_STEPS, sampler = sampler)
        gen.enqueue(j)
        broke = False
        for r in gen.iterate():
            if r.get("eos"):
                broke = True
                break
        logits[name] = torch.stack(store) if store else None
        seq = j.sequences[0].sequence_ids.torch().flatten().tolist()
        tokens[name] = [int(x) for x in seq]
        print(f"[{name}] broke={broke} store={len(store)} gen_len={len(seq)}", flush = True)

run_numcheck()
