import os, sys, collections
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768)
model.load()
tokenizer = Tokenizer.from_config(config)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
ids = tokenizer.encode("The history of computing machinery begins in the nineteenth century with", add_bos=True)

def gen_n(n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler())
    generator.enqueue(job)
    while True:
        if any(r.get("eos") for r in generator.iterate()):
            return

gen_n(16)  # warmup
gen_n(4)
# census window: capture stderr externally; steady-state decode
gen_n(8)
print("CENSUS_DONE", flush=True)
