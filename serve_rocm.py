import os, time, threading, queue, traceback
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
import uvicorn

# Serving defaults, applied before exllamav3 reads them at import:
# - RDNA3 GEMV/reconstruct crossover is ~16 rows; 144 (the NVIDIA default) puts
#   17..144-row chat prefills on the decode path (+0.5-1.5 s TTFT).
# - The attention canary (warn mode) logs degenerate all-zero attention output -
#   the signature of the transient triton launcher issue (docs/rocm.md).
os.environ.setdefault("EXL3_RECONSTRUCT_THRESHOLD", "16")
os.environ.setdefault("EXL3_ATTN_CANARY", "1")

from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.sampler import GreedySampler, CategoricalSampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/qwen38-27b")
MAX_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
if MAX_TOKENS % PAGE_SIZE:
    MAX_TOKENS += PAGE_SIZE - (MAX_TOKENS % PAGE_SIZE)
KV_BITS = os.environ.get("EXL3_KV_BITS")
MAX_BATCH = int(os.environ.get("EXL3_MAX_BATCH", "16"))
# Backpressure: pending+active jobs beyond this are rejected with 503 instead of
# queueing unboundedly (each holds prompt tensors and, once active, cache pages).
MAX_QUEUE = int(os.environ.get("EXL3_MAX_QUEUE", "32"))
# Hard ceiling on one request's wall time in /generate; guards against a dead
# driver (which now reports errors, but a hung kernel launch would still block).
REQUEST_TIMEOUT_S = float(os.environ.get("EXL3_REQUEST_TIMEOUT_S", "600"))

app = FastAPI()
_lock = threading.Lock()
state = {}

# One driver thread owns gen.iterate() so concurrent requests batch instead of
# serializing on _lock. Callers register a per-job result queue keyed by serial;
# enqueue/cancel happen under _gen_cv (which shares _lock).
_gen_cv = threading.Condition(_lock)
_job_queues = {}          # serial -> queue.Queue of raw iterate() result dicts


class QueueFull(Exception):
    pass


# Driver watchdog: gen.iterate() can wedge INSIDE a call (a kernel deadlocking
# on-device spins the GPU at 100% while the driver busy-polls the synchronize in
# userspace - no exception ever fires, so the exception containment below never
# sees it, health stays green, and every request hangs forever). The driver
# stamps a heartbeat around each iterate(); a watchdog thread fails the server
# when jobs are waiting but no progress is made for EXL3_DRIVER_STALL_S.
# Recovery is a process exit (docker restarts us); a wedged kernel only dies
# with its process.
DRIVER_STALL_S = float(os.environ.get("EXL3_DRIVER_STALL_S", "120"))
_driver_heartbeat = [0.0]   # time of last progress (iterate returned or wait began)


def _fail_driver(reason: str):
    if "driver_error" in state:
        return
    traceback.print_exc()
    state["driver_error"] = reason
    print(f"[serve] DRIVER WATCHDOG: {reason}", flush=True)
    with _lock:
        for serial, q in list(_job_queues.items()):
            q.put({"serial": serial, "stage": "error", "eos": True,
                   "error": f"driver stalled: {reason}"})
        _job_queues.clear()
    # Leave the process so the container restart policy brings up a clean GPU
    # context; linger briefly so the error results are drained by waiters.
    threading.Timer(5.0, lambda: os._exit(1)).start()


def _watchdog_loop():
    while "driver_error" not in state:
        time.sleep(5)
        if time.time() - _driver_heartbeat[0] < DRIVER_STALL_S:
            continue
        with _lock:
            waiting = len(_job_queues)
        if waiting:
            _fail_driver(
                f"no driver progress for {DRIVER_STALL_S:.0f}s with {waiting} job(s) "
                f"waiting - wedged kernel suspected")


def _driver_loop():
    _driver_heartbeat[0] = time.time()
    while True:
        gen = state.get("generator")
        if gen is None:
            time.sleep(0.1)
            _driver_heartbeat[0] = time.time()
            continue
        try:
            with _gen_cv:
                while gen.num_remaining_jobs() == 0:
                    _driver_heartbeat[0] = time.time()
                    _gen_cv.wait()
                _driver_heartbeat[0] = time.time()
                results = gen.iterate()
                _driver_heartbeat[0] = time.time()
        except Exception as e:
            # One escaped exception (e.g. a sticky HIP error after an OOM) used to
            # kill this thread silently: every in-flight request hung forever while
            # /health stayed green. Instead: deliver an error result to every
            # registered waiter, mark the driver dead (visible in /health), stop.
            # Recovery is a process restart; auto-retrying against a faulted
            # context would just hang the GPU.
            _fail_driver(f"{type(e).__name__}: {e}")
            return
        for r in results:
            q = _job_queues.get(r.get("serial"))
            if q is not None:
                q.put(r)


def submit_job(job) -> tuple[int, queue.Queue]:
    """Enqueue under the lock and register the job's result queue.
    Returns (serial, q); the caller drains q until an eos result."""
    with _gen_cv:
        if len(_job_queues) >= MAX_QUEUE:
            raise QueueFull(
                f"{len(_job_queues)} jobs already queued (EXL3_MAX_QUEUE={MAX_QUEUE})")
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
    if KV_BITS:
        bits = int(KV_BITS)
        cache = Cache(model, max_num_tokens=MAX_TOKENS,
                      layer_type=CacheLayer_quant, k_bits=bits, v_bits=bits)
        print(f"[serve] cache Q{bits} max_num_tokens={MAX_TOKENS} page={PAGE_SIZE}", flush=True)
    else:
        cache = Cache(model, max_num_tokens=MAX_TOKENS)
        print(f"[serve] cache fp16 max_num_tokens={MAX_TOKENS} page={PAGE_SIZE}", flush=True)
    model.load(device="cuda:0")
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          max_batch_size=MAX_BATCH)
    state.update(config=config, model=model, cache=cache,
                 tokenizer=tokenizer, generator=generator)
    state["load_s"] = time.time() - t0
    print(f"[serve] model loaded in {state['load_s']:.1f}s", flush=True)
    threading.Thread(target=_driver_loop, daemon=True).start()
    threading.Thread(target=_watchdog_loop, daemon=True).start()


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    add_bos: bool = True


@app.get("/health")
def health():
    ok = "generator" in state and "driver_error" not in state
    return {"ok": ok, "load_s": state.get("load_s"),
            "driver_error": state.get("driver_error"),
            "queued": len(_job_queues)}


@app.post("/generate")
def generate(req: GenReq):
    if "generator" not in state:
        raise HTTPException(503, "model not loaded yet")
    if "driver_error" in state:
        raise HTTPException(503, f"driver loop is down: {state['driver_error']}; restart required")
    tok = state["tokenizer"]
    ids = tok.encode(req.prompt, add_bos=req.add_bos)
    num_prompt_tokens = ids.shape[1]
    # Impossible requests fail fast instead of asserting deep in page accounting
    if num_prompt_tokens + 1 >= MAX_TOKENS:
        raise HTTPException(400, f"prompt uses {num_prompt_tokens} of {MAX_TOKENS} cache tokens")
    max_new = min(req.max_new_tokens, MAX_TOKENS - num_prompt_tokens - 1)
    if max_new < 1:
        raise HTTPException(400, "no room to generate beyond the prompt")
    sampler = GreedySampler() if req.temperature == 0.0 \
        else CategoricalSampler(temperature=req.temperature)
    job = Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler)
    try:
        serial, q = submit_job(job)
    except QueueFull as e:
        raise HTTPException(503, str(e))
    chunks = []
    result = {}
    try:
        while True:
            try:
                r = q.get(timeout=REQUEST_TIMEOUT_S)
            except queue.Empty:
                cancel_job(job, serial)
                raise HTTPException(503, f"no result within {REQUEST_TIMEOUT_S:.0f}s; job cancelled")
            if r.get("stage") == "error":
                raise HTTPException(500, f"generation failed: {r.get('error')}")
            if r.get("text"):
                chunks.append(r["text"])
            if r.get("eos"):
                result = r
                held = r.get("held") or {}
                # Same rule as serve_openai: flush held tail text only on a length
                # stop (healing tail); stop_token holds the stop token itself,
                # stop_string's rem_held_text starts with the stop string
                if held.get("text") and result.get("eos_reason") == "max_new_tokens":
                    chunks.append(held["text"])
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
