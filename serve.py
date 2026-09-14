import os, time, threading
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler, CategoricalSampler

MODEL_DIR = os.environ.get("EXL3_MODEL", "/models/Qwen3.8-27B-SC_4.00bpw_H5_V6")
MAX_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))

app = FastAPI()
_lock = threading.Lock()
state = {}

@app.on_event("startup")
def load_model():
    t0 = time.time()
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=MAX_TOKENS)
    model.load(device="cuda:0")
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    state.update(config=config, model=model, cache=cache,
                 tokenizer=tokenizer, generator=generator)
    state["load_s"] = time.time() - t0
    print(f"[serve] model loaded in {state['load_s']:.1f}s", flush=True)

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
    with _lock:
        gen = state["generator"]
        tok = state["tokenizer"]
        ids = tok.encode(req.prompt, add_bos=req.add_bos)
        num_prompt_tokens = ids.shape[1]
        sampler = GreedySampler() if req.temperature == 0.0 else CategoricalSampler(temperature=req.temperature)
        job = Job(input_ids=ids, max_new_tokens=req.max_new_tokens,
                  sampler=sampler)
        gen.enqueue(job)
        chunks = []
        result = {}
        while True:
            for r in gen.iterate():
                if r.get("text"):
                    chunks.append(r["text"])
                if r.get("eos"):
                    result = r
                    break
            if result:
                break
        new_tokens = result.get("new_tokens", 0)
        t_prefill = result.get("time_prefill", 0.0)
        t_gen = result.get("time_generate", 0.0)
        return {
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
