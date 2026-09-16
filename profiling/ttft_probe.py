#!/usr/bin/env python3
"""TTFT probe for exllamav3-rocm-serve (TabbyAPI, streaming completions).

Measures time-to-first-streamed-token and total wall time for:
  - short prompt (decode-dominated)
  - cold 3.2k prompt (unique nonce each run -> true prefill)
  - warm 3.2k prompt (same text repeated -> prefix cache hit if enabled)
  - 1.5k prompt (single chunk round at chunk_size 2048)
  - two concurrent decodes (batching sanity)
"""
import json, sys, time, urllib.request, threading

URL = "http://192.168.1.200:9001"
label = sys.argv[1] if len(sys.argv) > 1 else "ttft"

def post(path, payload, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def stream_ttft(prompt, max_tokens, timeout=600):
    payload = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(URL + "/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("text", "")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
    total = time.perf_counter() - t0
    return ttft, total, ntok

base = "The history of computing machinery begins in the nineteenth century with "
filler = "Machine learning models have grown rapidly in scale over the past decade. "
nonce = str(time.time())
long_prompt = (base + nonce + " ") + filler * 130   # ~3.2k tokens
mid_prompt = (base + nonce + "-b ") + filler * 60   # ~1.5k tokens

results = {"label": label, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}

def ptok(prompt):
    try:
        return post("/v1/token/encode", {"text": prompt})[0].get("length")
    except Exception:
        return None

def record(name, ttft, total, ntok, prompt=None, extra=None):
    r = {"name": name, "ttft_s": round(ttft, 3) if ttft else None,
         "total_s": round(total, 3), "chunks": ntok}
    if prompt is not None:
        r["prompt_tokens"] = ptok(prompt)
    if extra:
        r.update(extra)
    results["runs"].append(r)
    print(f"[{label}] {name:16s} ttft={r['ttft_s']}s total={r['total_s']}s chunks={ntok} ptok={r.get('prompt_tokens')}", flush=True)

# warmup (loads/JIT)
record("warmup", *stream_ttft("Hello, tell me a short story.", 32))

for i in range(3):
    record(f"decode_short_{i}", *stream_ttft("Write a detailed essay about the history of computing.", 256))

for i in range(3):
    p = (base + nonce + f"-c{i} ") + filler * 130
    record(f"prefill3k_cold_{i}", *stream_ttft(p, 8))

for i in range(3):
    record(f"prefill3k_warm_{i}", *stream_ttft(long_prompt, 8))

record("prefill1k5", *stream_ttft(mid_prompt, 8))

# two concurrent short decodes
out = {}
def worker(i):
    out[i] = stream_ttft("Count the letter e in this sentence and explain your reasoning.", 192)
ths = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
t0 = time.perf_counter()
for t in ths: t.start()
for t in ths: t.join()
wall = time.perf_counter() - t0
results["runs"].append({"name": "concurrent_2x_decode", "wall_s": round(wall, 3),
                        "ttfts": [round(out[i][0], 3) for i in range(2)],
                        "totals": [round(out[i][1], 3) for i in range(2)]})
print(f"[{label}] concurrent_2x    wall={wall:.2f}s ttfts={[round(out[i][0],2) for i in range(2)]}", flush=True)

with open(f"/tmp/ttft_{label}.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
