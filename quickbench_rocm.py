import os, time
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

config = Config.from_directory(os.environ["EXL3_MODEL"])
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768)
t0 = time.time()
model.load()
tokenizer = Tokenizer.from_config(config)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=16)
print(f"[load {time.time()-t0:.0f}s] vram={(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**30:.2f} GB", flush=True)

ids = tokenizer.encode("The history of computing machinery begins in the nineteenth century with", add_bos=True)

def run_batch(bsz, n, label, warm=8):
    jobs = [Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()) for _ in range(bsz)]
    for j in jobs: generator.enqueue(j)
    steps = 0
    t0 = time.perf_counter()
    while True:
        rs = generator.iterate()
        steps += 1
        if all(r.get("eos") for r in rs):
            break
    wall = time.perf_counter() - t0
    print(f"{label}: {n} steps x {bsz} streams in {wall:.2f}s -> aggregate {bsz*n/wall:.1f} t/s ({wall/n*1000:.1f} ms/step)", flush=True)

# warm
run_batch(1, 16, "warm")
run_batch(1, 64, "decode_b1")
run_batch(4, 64, "decode_b4")
run_batch(8, 64, "decode_b8")
run_batch(16, 64, "decode_b16")

# prefill: fresh prompts at 512 tokens (force full prefill via unique filler)
import random
words = open("/usr/share/dict/words").read().split() if os.path.exists("/usr/share/dict/words") else None
base = "The history of computing machinery begins in the nineteenth century with"
filler = "Machine learning models have grown rapidly in scale over the past decade. "
for plen in (128, 512):
    ts = []
    for trial in range(3):
        prompt = base + f" doc {trial}:" + filler * ((plen // 13) + trial)
        pids = tokenizer.encode(prompt, add_bos=True)
        job = Job(input_ids=pids, max_new_tokens=1, sampler=GreedySampler())
        generator.enqueue(job)
        t0 = time.perf_counter()
        while True:
            if any(r.get("eos") for r in generator.iterate()):
                break
        ts.append((time.perf_counter() - t0))
    best = min(ts)
    n = pids.shape[1]
    print(f"prefill_{plen}: {n} tokens best {best*1000:.0f} ms -> {n/best:.0f} t/s", flush=True)
