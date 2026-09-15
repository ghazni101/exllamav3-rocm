import os, time, threading, queue
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler, CategoricalSampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
MAX_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
MAX_BATCH = int(os.environ.get("EXL3_MAX_BATCH", "16"))

app = FastAPI()
_lock = threading.Lock()
state = {}

# One driver thread owns gen.iterate() so concurrent requests batch instead of
# serializing on _lock. Callers register a per-job result queue keyed by serial;
# enqueue/cancel happen under _gen_cv (which shares _lock).
_gen_cv = threading.Condition(_lock)
_job_queues = {}          # serial -> queue.Queue of raw iterate() result dicts


def _driver_loop():
    while True:
        gen = state.get("generator")
        if gen is None:
            time.sleep(0.1)
            continue
        with _gen_cv:
            while gen.num_remaining_jobs() == 0:
                _gen_cv.wait()
            results = gen.iterate()
        for r in results:
            q = _job_queues.get(r.get("serial"))
            if q is not None:
                q.put(r)


def submit_job(job) -> tuple[int, queue.Queue]:
    """Enqueue under the lock and register the job's result queue.
    Returns (serial, q); the caller drains q until an eos result."""
    with _gen_cv:
        serial = state["generator"].enqueue(job)
        q = queue.Queue()
        _job_queues[serial] = q
        _gen_cv.notify()
    return serial, q


def cancel_job(job, serial):
    """Remove a pending/active job and drop its result queue."""
    with _gen_cv:
        try:
            state["generator"].cancel(job)
        finally:
            _job_queues.pop(serial, None)
            _gen_cv.notify()


def finish_job(serial):
    _job_queues.pop(serial, None)


@app.on_event("startup")
def load_model():
    t0 = time.time()
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=MAX_TOKENS)
    model.load(device="cuda:0")
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          max_batch_size=MAX_BATCH)
    state.update(config=config, model=model, cache=cache,
                 tokenizer=tokenizer, generator=generator)
    state["load_s"] = time.time() - t0
    print(f"[serve] model loaded in {state['load_s']:.1f}s", flush=True)
    threading.Thread(target=_driver_loop, daemon=True).start()


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    add_bos: bool = True


@app.get("/health")
def health():
    return {"ok": "generator" in state, "load_s": state.get("load_s")}


@app.post("/generate")
def generate(req: GenReq):
    tok = state["tokenizer"]
    ids = tok.encode(req.prompt, add_bos=req.add_bos)
    num_prompt_tokens = ids.shape[1]
    sampler = GreedySampler() if req.temperature == 0.0 \
        else CategoricalSampler(temperature=req.temperature)
    job = Job(input_ids=ids, max_new_tokens=req.max_new_tokens, sampler=sampler)
    serial, q = submit_job(job)
    chunks = []
    result = {}
    try:
        while True:
            r = q.get()
            if r.get("text"):
                chunks.append(r["text"])
            if r.get("eos"):
                result = r
                break
    finally:
        finish_job(serial)
    new_tokens = result.get("new_tokens", 0)
    t_prefill = result.get("time_prefill", 0.0)
    t_gen = result.get("time_generate", 0.0)
    return {
        "text": "".join(chunks),
        "prompt_tokens": num_prompt_tokens,
        "new_tokens": new_tokens,
        "eos_reason": result.get("eos_reason"),
        "time_prefill_s": t_prefill,
        "time_generate_s": t_gen,
        "prefill_tok_s": num_prompt_tokens / t_prefill if t_prefill > 0 else None,
        "gen_tok_s": new_tokens / t_gen if t_gen > 0 else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9001")))
