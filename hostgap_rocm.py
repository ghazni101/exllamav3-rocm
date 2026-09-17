import os, time
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

config = Config.from_directory(os.environ["EXL3_MODEL"])
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

gen_n(16); gen_n(4)

# per-token: wall vs stream-time (CUDA events bracket each iterate round)
N = 24
job = Job(input_ids=ids, max_new_tokens=N, sampler=GreedySampler())
generator.enqueue(job)
stream = torch.cuda.current_stream()
evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]
t0 = time.perf_counter()
i = 0
while True:
    evs[i][0].record(stream)
    results = generator.iterate()
    evs[i][1].record(stream)
    if any(r.get("eos") for r in results):
        break
    i += 1
torch.cuda.synchronize(stream)
wall = time.perf_counter() - t0
times = [e0.elapsed_time(e1) for e0, e1 in evs[:i+1]]
stream_s = sum(times) / 1000
print(f"tokens={i+1} wall={wall:.3f}s stream_time={stream_s:.3f}s "
      f"gap={(wall-stream_s):.3f}s ({(wall-stream_s)/wall*100:.1f}%) "
      f"decode_tps={(i+1)/wall:.2f} stream_tps={(i+1)/stream_s:.2f}")
print("per-token stream ms:", [f"{t:.1f}" for t in times])
