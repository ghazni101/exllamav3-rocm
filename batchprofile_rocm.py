import os, time
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

BSZ = int(os.environ.get("BSZ", "8"))
SKIP = 40
PROF = 8
config = Config.from_directory(os.environ["EXL3_MODEL"])
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768)
model.load()
tokenizer = Tokenizer.from_config(config)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=16)
ids = tokenizer.encode("The history of computing machinery begins in the nineteenth century with", add_bos=True)

def batch_run(n, profile=False):
    jobs = [Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler()) for _ in range(BSZ)]
    for j in jobs: generator.enqueue(j)
    t0 = time.perf_counter(); steps = 0
    if profile:
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    while True:
        if profile and steps >= SKIP:
            evs[steps][0].record()
        rs = generator.iterate()
        if profile and steps >= SKIP:
            evs[steps][1].record()
        steps += 1
        if all(r.get("eos") for r in rs):
            break
    wall = time.perf_counter() - t0
    if profile:
        torch.cuda.synchronize()
        steptimes = [a.elapsed_time(b) for a, b in evs[SKIP:steps]]
        print(f"[b{BSZ}] total {wall:.1f}s; steady steps {len(steptimes)}: "
              f"avg {sum(steptimes)/len(steptimes):.1f} ms/step -> {BSZ*1000/(sum(steptimes)/len(steptimes)):.1f} t/s steady")
        return
    print(f"[b{BSZ}] {n} steps in {wall:.1f}s = {BSZ*n/wall:.1f} t/s aggregate (incl prefill)", flush=True)

batch_run(64)
batch_run(64)
batch_run(SKIP + PROF, profile=True)

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
) as prof:
    # steady-state only: reuse the running... need a fresh batch; skip steps inside profile window
    jobs = [Job(input_ids=ids, max_new_tokens=SKIP + PROF, sampler=GreedySampler()) for _ in range(BSZ)]
    for j in jobs: generator.enqueue(j)
    steps = 0
    while True:
        rs = generator.iterate()
        steps += 1
        if all(r.get("eos") for r in rs):
            break
ka = prof.key_averages()
rows = []
for e in ka:
    dt = getattr(e, "self_device_time_total", 0)
    if dt and ("exl3" in e.key or "gated" in e.key or "paged" in e.key):
        rows.append((dt, e.count, e.key[:64]))
rows.sort(reverse=True)
tot = sum(r[0] for r in rows)
print(f"{'us':>10} {'share':>6} {'calls':>6}  kernel")
for dt, c, k in rows[:14]:
    print(f"{dt:10.0f} {dt/tot*100:5.1f}% {c:6}  {k}")
