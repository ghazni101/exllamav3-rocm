#!/usr/bin/env python3
"""A/B benchmark for exllamav3-rocm-serve (TabbyAPI OAI endpoint).

Measures decode tok/s and prefill tok/s via client-side wall time on
/v1/completions with fixed prompts, temperature=0, max_tokens fixed.
Usage: python3 bench_ab.py <label> [--url http://192.168.1.200:9001]
"""
import json, sys, time, urllib.request

URL = "http://192.168.1.200:9001"
label = sys.argv[1] if len(sys.argv) > 1 else "run"
if "--url" in sys.argv:
    URL = sys.argv[sys.argv.index("--url") + 1]

def post(path, payload, timeout=600):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    dt = time.perf_counter() - t0
    return body, dt

def wait_health(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        try:
            # TabbyAPI may not expose /health; try model list
            with urllib.request.urlopen(URL + "/v1/model/list", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False

def completion(prompt, max_tokens):
    ptok = post("/v1/token/encode", {"text": prompt})[0].get("length")
    body, dt = post("/v1/completions", {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    })
    ch = body["choices"][0]
    ctok = max_tokens if ch.get("finish_reason") == "length" else None
    if ctok is None:
        ctok = post("/v1/token/encode", {"text": ch.get("text", "")})[0].get("length")
    return {
        "wall_s": dt,
        "completion_tokens": ctok,
        "prompt_tokens": ptok,
        "finish_reason": ch.get("finish_reason"),
        "tok_s": ctok / dt if dt else None,
    }

results = {"label": label, "url": URL, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

if not wait_health():
    print(json.dumps({"label": label, "error": "server not healthy"}))
    sys.exit(1)

# Warmup (also triggers any lazy autotune)
completion("Hello, tell me a short story.", 32)

# Decode runs: short prompt, fixed gen length
decode = []
for i in range(3):
    r = completion("Write a detailed essay about the history of computing.", 256)
    decode.append(r)
    print(f"[{label}] decode{i}: {r['completion_tokens']} tok in {r['wall_s']:.2f}s = {r['tok_s']:.1f} tok/s", flush=True)

# Prefill runs: long prompt (~3200 tok), tiny gen. Unique nonce prefix per run
# defeats the KV prefix cache so each run measures true cold prefill.
import random, string
def mkprompt():
    nonce = "".join(random.choices(string.ascii_letters, k=16))
    return (nonce + " The following is a technical document. " +
            " ".join(f"Section {i}: " + "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor " * 4
                     for i in range(60)))
prefill = []
for i in range(3):
    r = completion(mkprompt(), 8)
    r["prefill_tok_s"] = (r["prompt_tokens"] or 0) / r["wall_s"] if r["wall_s"] else None
    prefill.append(r)
    print(f"[{label}] prefill{i}: {r['prompt_tokens']} tok in {r['wall_s']:.2f}s = {r['prefill_tok_s']:.1f} tok/s", flush=True)

def med(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

results["decode_tok_s"] = [r["tok_s"] for r in decode]
results["decode_median_tok_s"] = med([r["tok_s"] for r in decode])
results["prefill_tok_s"] = [r["prefill_tok_s"] for r in prefill]
results["prefill_median_tok_s"] = med([r["prefill_tok_s"] for r in prefill])
results["prompt_tokens_prefill"] = prefill[0]["prompt_tokens"]

print(json.dumps(results, indent=1))
