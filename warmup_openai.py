#!/usr/bin/env python3
"""TTFT warmup: compile Triton/FLA, autotune chat-sized prefills, hit chat+think.

Works against both server entrypoints:
  - serve_openai.py (has /v1/*): full warmup incl. chat template + streaming TTFT
  - serve_rocm.py (the Dockerfile CMD, /generate only): warms the same prefill/
    decode kernel shapes via /generate; chat-template paths are reported skipped.

Exits nonzero on failure (health never OK, HTTP error, driver down) so an
orchestrator gating on warmup is not fooled by a quiet skip.
"""
import json, os, sys, time, urllib.request, urllib.error

PORT = os.environ.get("PORT", "9001")
URL = f"http://127.0.0.1:{PORT}"
OPTIONAL = os.environ.get("EXL3_WARMUP_OPTIONAL") == "1"


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_health(minutes=15):
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            body = get("/health", timeout=5)
            if body.get("ok"):
                if body.get("driver_error"):
                    print(f"[warmup] FATAL: driver loop is down: {body['driver_error']}")
                    return False
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def server_kind():
    """'openai' (serve_openai, /v1 routes) | 'rocm' (serve_rocm, /generate) | None."""
    try:
        get("/v1/models", timeout=5)
        return "openai"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "rocm"   # /health worked; /v1 absent -> serve_rocm entrypoint
        raise
    except Exception:
        return None


def completion(kind, prompt, max_tokens):
    if kind == "openai":
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
        path = "/v1/completions"
    else:
        payload = {"prompt": prompt, "max_new_tokens": max_tokens, "temperature": 0.0}
        path = "/generate"
    t0 = time.perf_counter()
    post(path, payload)
    return time.perf_counter() - t0


def chat(msg, max_tokens, thinking):
    t0 = time.perf_counter()
    post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    })
    return time.perf_counter() - t0


def chat_stream_ttft(msg, thinking=False, max_tokens=8):
    payload = {
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        buf = b""
        while True:
            chunk = r.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                line = ev.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return time.perf_counter() - t0
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
                if delta.get("content") or delta.get("reasoning_content"):
                    return time.perf_counter() - t0
    return time.perf_counter() - t0


def main():
    if not wait_health():
        print("[warmup] FATAL: server did not become healthy", flush=True)
        sys.exit(0 if OPTIONAL else 1)
    time.sleep(2)
    kind = server_kind()
    if kind is None:
        print("[warmup] FATAL: cannot classify server (neither /v1/models nor /health responds)", flush=True)
        sys.exit(0 if OPTIONAL else 1)
    print(f"[warmup] server kind: {kind}", flush=True)

    try:
        # Compile FLA chunk + prefill attention on a long prompt first so later
        # chat-sized prefills reuse those kernels instead of JITing on first user token.
        filler = "Machine learning models have grown rapidly in scale over the past decade. "
        dt = completion(kind, "The history of computing begins here. " + filler * 150, 2)
        print(f"[warmup] 2k prefill (chunk compile): {dt:.2f}s", flush=True)

        dt = completion(kind, "Hello, tell me a short story about a robot.", 24)
        print(f"[warmup] short decode (BC graph): {dt:.2f}s", flush=True)

        # Unique prefixes so paged-cache hits do not skip the prefill we want to autotune.
        for ntok in (16, 32, 48, 64, 96, 128, 192, 256, 512):
            prompt = f"warmup-len-{ntok} " + ("word " * ntok)
            dt = completion(kind, prompt, 2)
            print(f"[warmup] prefill ~{ntok} toks: {dt:.2f}s", flush=True)

        if kind != "openai":
            print("[warmup] /v1 routes absent (serve_rocm entrypoint); "
                  "chat-template + streaming TTFT warmup skipped", flush=True)
            print("[warmup] done", flush=True)
            return

        dt = chat("Reply with the word ping only.", 8, thinking=True)
        print(f"[warmup] chat thinking: {dt:.2f}s", flush=True)
        dt = chat("Reply with the word pong only.", 8, thinking=False)
        print(f"[warmup] chat no-think: {dt:.2f}s", flush=True)

        ttft = chat_stream_ttft("Say hi in one word.", thinking=False)
        print(f"[warmup] stream chat TTFT (no-think): {ttft:.3f}s", flush=True)
        ttft = chat_stream_ttft("Say hi in one word again.", thinking=True)
        print(f"[warmup] stream chat TTFT (thinking): {ttft:.3f}s", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[warmup] FATAL: HTTP {e.code} from server: {e.read()[:200]!r}", flush=True)
        sys.exit(1)
    print("[warmup] done", flush=True)


if __name__ == "__main__":
    main()
