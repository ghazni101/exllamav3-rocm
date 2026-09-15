# OpenAI-compatible serving layer on top of serve_rocm.py's model load.
# Endpoints: GET /v1/models, POST /v1/chat/completions, POST /v1/completions.
# Mirrors vLLM/SGLang/llama.cpp conventions:
#   - reasoning split on first </think> into reasoning_content vs content;
#     thinking enabled + no </think> => entire output is reasoning_content
#   - reasoning_effort maps OpenAI values onto the template's
#     enable_thinking / reasoning_effort (xhigh|medium|low) kwargs
#   - tools rendered via chat_template.jinja; <tool_call> output parsed into
#     OpenAI tool_calls[] with finish_reason "tool_calls"
#   - stream_options.include_usage emits a final usage chunk; first chunk
#     carries delta.role="assistant"
#   - client disconnect cancels the job (streaming and non-streaming)
import os, re, time, uuid, json, queue, threading, asyncio
import jinja2, jinja2.sandbox, jinja2.ext
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Union

from serve_rocm import app, state, _lock, MODEL_DIR, submit_job, cancel_job, finish_job
from exllamav3 import Job
from exllamav3.generator.sampler import ComboSampler, GreedySampler

# Browser frontends (e.g. SvelteKit UIs on another port) hit this API
# cross-origin; without CORS headers the browser blocks every request.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.environ.get("EXL3_MODEL_NAME", os.path.basename(MODEL_DIR.rstrip("/")))
MAX_TOKENS = int(os.environ.get("EXL3_CACHE_TOKENS", "32768"))
THINK_CLOSE = "</think>"
THINK_TAIL = len(THINK_CLOSE) - 1  # hold-back for partial-tag chunk boundaries

# --- chat template (jinja2, mirroring transformers' environment) ----------

def _raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)

_jinja_env = jinja2.sandbox.ImmutableSandboxedEnvironment(
    trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
)
_jinja_env.globals["raise_exception"] = _raise_exception
_jinja_env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)

_template_src = open(os.path.join(MODEL_DIR, "chat_template.jinja")).read()
_chat_template = _jinja_env.from_string(_template_src)

# OpenAI reasoning_effort -> template kwargs. The template accepts
# reasoning_effort in {xhigh, medium, low}; "high" maps to its maximum.
_EFFORT_MAP = {
    "none": {"enable_thinking": False},
    "minimal": {"reasoning_effort": "low"},
    "low": {"reasoning_effort": "low"},
    "medium": {"reasoning_effort": "medium"},
    "high": {"reasoning_effort": "xhigh"},
    "xhigh": {"reasoning_effort": "xhigh"},
}


def render_chat(messages: list[dict], template_kwargs: dict) -> str:
    return _chat_template.render(
        messages=messages,
        add_generation_prompt=True,
        **template_kwargs,
    )


def _normalize_messages(req: "ChatReq") -> list[dict]:
    """OpenAI wire format -> template format. tool_call.arguments arrives as a
    JSON string; the template iterates it as a mapping, so parse it."""
    out = []
    for m in req.messages:
        d = m.model_dump(exclude_none=True)
        if d.get("tool_calls"):
            for tc in d["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
        out.append(d)
    return out


def _template_kwargs(req: "ChatReq") -> dict:
    kw = dict(_EFFORT_MAP.get(req.reasoning_effort or "", {}))
    if req.tools and req.tool_choice != "none":
        kw["tools"] = [t.model_dump(exclude_none=True) for t in req.tools]
    kw.update(req.chat_template_kwargs or {})  # explicit kwargs win
    return kw


# --- request models --------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Union[str, list, None] = None
    name: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    n: Optional[int] = 1
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = None
    parallel_tool_calls: Optional[bool] = True
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[dict] = None
    user: Optional[str] = None


class CompletionReq(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict] = None
    stop: Optional[Union[str, list[str]]] = None
    seed: Optional[int] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    n: Optional[int] = 1
    user: Optional[str] = None


# --- generation core -------------------------------------------------------

def _make_sampler(r) -> object:
    if (r.temperature or 0) == 0.0:
        return GreedySampler()
    logit_bias = None
    if r.logit_bias:
        try:
            logit_bias = {int(k): float(v) for k, v in r.logit_bias.items()}
        except (ValueError, TypeError):
            logit_bias = None
    return ComboSampler(
        temperature=r.temperature,
        top_p=r.top_p,
        top_k=r.top_k or 0,
        min_p=r.min_p or 0.0,
        rep_p=r.repetition_penalty or 1.0,
        pres_p=r.presence_penalty or 0.0,
        freq_p=r.frequency_penalty or 0.0,
        logit_bias=logit_bias,
    )


def _stop_conditions(r) -> list:
    # Job does not auto-stop on EOS; chat.py adds config.eos_token_id_list
    # explicitly — do the same so <|im_end|> ends the turn.
    conds = list(state["config"].eos_token_id_list or [])
    if r.stop:
        conds += [r.stop] if isinstance(r.stop, str) else list(r.stop)
    return conds


def _run_job(prompt_ids, max_new_tokens, sampler, stop_conds, seed, out_q, cancel_ev):
    """Worker thread: registers the job with the shared generator driver, then
    translates raw iterate() results into ("text", s) / ("done", result) on out_q.
    The driver thread owns gen.iterate() so concurrent requests batch."""
    serial = None
    try:
        job = Job(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            seed=seed,
            stop_conditions=stop_conds or None,
        )
        serial, jq = submit_job(job)
        out_q.put(("job", job))
        result = {}
        while not cancel_ev.is_set():
            try:
                r = jq.get(timeout=0.5)
            except queue.Empty:
                continue
            if r.get("text"):
                out_q.put(("text", r["text"]))
            if r.get("eos"):
                result = r
                break
        if cancel_ev.is_set() and not result:
            # Aborted mid-flight: remove the job so it doesn't keep
            # generating (and holding pages) inside the generator.
            cancel_job(job, serial)
        else:
            finish_job(serial)
        out_q.put(("done", result))
    except Exception as e:
        if serial is not None:
            finish_job(serial)
        out_q.put(("err", e))


def _finish_reason(eos_reason: str, tool_calls: list | None = None) -> str:
    if tool_calls:
        return "tool_calls"
    return "length" if eos_reason == "max_new_tokens" else "stop"


def _usage(prompt_tokens: int, new_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": new_tokens,
        "total_tokens": prompt_tokens + new_tokens,
    }


def _start_generation(prompt_text, req, max_tokens_default, seed_offset: int = 0):
    """Encode prompt, spawn worker, return (queue, cancel_event, prompt_tokens).
    Default max_new_tokens = remaining cache (vLLM: context_window - prompt)."""
    tok = state["tokenizer"]
    ids = tok.encode(prompt_text, encode_special_tokens=True)
    n_prompt = ids.shape[1]
    if n_prompt >= MAX_TOKENS:
        raise ValueError(f"prompt uses {n_prompt} of {MAX_TOKENS} cache tokens")
    max_new = getattr(req, "max_completion_tokens", None) or req.max_tokens or max_tokens_default
    # n>1: vary the seed per choice so identical seeds don't clone outputs
    seed = None if req.seed is None else req.seed + seed_offset
    max_new = min(max_new, MAX_TOKENS - n_prompt)
    if max_new < 1:
        raise ValueError(f"prompt uses {n_prompt} of {MAX_TOKENS} cache tokens; no room to generate")
    q = queue.Queue()
    cancel_ev = threading.Event()
    t = threading.Thread(
        target=_run_job,
        args=(ids, max_new, _make_sampler(req), _stop_conditions(req), seed, q, cancel_ev),
        daemon=True,
    )
    t.start()
    return q, cancel_ev, n_prompt


def _collect(q):
    """Non-streaming: drain queue, return (full_text, result_dict)."""
    chunks, result = [], {}
    while True:
        typ, payload = q.get()
        if typ == "err":
            raise payload
        if typ == "done":
            result = payload
            break
        if typ == "text":
            chunks.append(payload)
    return "".join(chunks), result


async def _collect_with_abort(q, cancel_ev, request: Request):
    """_collect with client-disconnect abort. Returns (text, result) or
    raises ClientDisconnect."""
    task = asyncio.create_task(asyncio.to_thread(_collect, q))
    while not task.done():
        if await request.is_disconnected():
            cancel_ev.set()  # worker cancels its own job on exit
            task.cancel()
            raise ClientDisconnect()
        await asyncio.sleep(0.5)
    return await task


class ClientDisconnect(Exception):
    pass


# --- think-tag splitting -----------------------------------------------------

def split_thinking(text: str, thinking_enabled: bool) -> tuple[str, str]:
    """vLLM reasoning-parser semantics: with thinking enabled, everything up to
    the first </think> is reasoning; if the tag never arrives (cut mid-thought)
    the whole output is reasoning. With thinking disabled there is no split."""
    if not thinking_enabled:
        return "", text
    if THINK_CLOSE in text:
        reasoning, _, content = text.partition(THINK_CLOSE)
        return reasoning, content
    return text, ""


class ThinkSplitter:
    """Streaming variant: routes text chunks to reasoning vs content deltas.
    Holds back THINK_TAIL chars while in reasoning so a </think> split across
    chunk boundaries is still detected."""

    def __init__(self):
        self.in_thinking = True  # generation prompt ends with "<think>\n"
        self.buf = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Returns (reasoning_delta, content_delta) safe to emit now."""
        self.buf += text
        if not self.in_thinking:
            out, self.buf = self.buf, ""
            return "", out
        idx = self.buf.find(THINK_CLOSE)
        if idx >= 0:
            reasoning = self.buf[:idx]
            content = self.buf[idx + len(THINK_CLOSE):]
            self.buf = ""
            self.in_thinking = False
            return reasoning, content
        # No tag yet: emit all but the tail that could be a partial tag
        safe = self.buf[:-THINK_TAIL] if len(self.buf) > THINK_TAIL else ""
        self.buf = self.buf[len(safe):]
        return safe, ""

    def flush(self) -> tuple[str, str]:
        out, self.buf = self.buf, ""
        return (out, "") if self.in_thinking else ("", out)


# --- tool-call parsing (Qwen <tool_call> format -> OpenAI tool_calls) --------

_TOOLCALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)</parameter>", re.S)


def parse_tool_calls(text: str) -> tuple[str, list]:
    """Extract <tool_call> blocks. Returns (remaining_content, tool_calls).
    Arguments are assembled into a JSON object string per OpenAI spec."""
    calls = []
    for m in _TOOLCALL_RE.finditer(text):
        name = m.group(1).strip()
        body = m.group(2)
        args = {}
        for pm in _PARAM_RE.finditer(body):
            args[pm.group(1).strip()] = pm.group(2).strip()
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    if not calls:
        return text, []
    content = _TOOLCALL_RE.sub("", text).strip()
    return content, calls


# --- SSE streaming -----------------------------------------------------------

def _chunk(cid, created, kind, delta=None, text=None, finish=None):
    if kind == "chat":
        return {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        }
    return {
        "id": cid, "object": "text_completion", "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text or "", "finish_reason": finish}],
    }


def _usage_chunk(cid, created, kind, usage):
    return {
        "id": cid, "created": created, "model": MODEL_NAME,
        "object": "chat.completion.chunk" if kind == "chat" else "text_completion",
        "choices": [],
        "usage": usage,
    }


async def _sse_stream(q, cancel_ev, request: Request, kind: str, split_think: bool,
                      include_usage: bool, n_prompt: int):
    """Drain the worker queue into OpenAI SSE chunks.
    On client disconnect the StreamingResponse cancels this generator; the
    finally clause sets cancel_ev and the worker cancels its own job."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if kind == "chat" else f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    splitter = ThinkSplitter() if (split_think and kind == "chat") else None
    new_tokens = 0
    try:
        # First chunk carries the role, per OpenAI/vLLM convention
        if kind == "chat":
            yield f"data: {json.dumps(_chunk(cid, created, kind, delta={'role': 'assistant'}))}\n\n"
        while True:
            if await request.is_disconnected():
                return
            try:
                typ, payload = await asyncio.to_thread(q.get, True, 0.5)
            except queue.Empty:
                continue
            if typ == "job":
                continue
            if typ == "err":
                yield f"data: {json.dumps({'error': {'message': str(payload), 'type': 'server_error'}})}\n\n"
                return
            if typ == "done":
                if splitter:
                    r, c = splitter.flush()
                    if r:
                        yield f"data: {json.dumps(_chunk(cid, created, kind, delta={'reasoning_content': r}))}\n\n"
                    if c:
                        yield f"data: {json.dumps(_chunk(cid, created, kind, delta={'content': c}))}\n\n"
                finish = _finish_reason(payload.get("eos_reason", "")) if payload else "stop"
                new_tokens = payload.get("new_tokens", new_tokens) if payload else new_tokens
                yield f"data: {json.dumps(_chunk(cid, created, kind, finish=finish))}\n\n"
                if include_usage:
                    yield f"data: {json.dumps(_usage_chunk(cid, created, kind, _usage(n_prompt, new_tokens)))}\n\n"
                yield "data: [DONE]\n\n"
                return
            # text chunk
            if splitter:
                r, c = splitter.feed(payload)
                if r:
                    yield f"data: {json.dumps(_chunk(cid, created, kind, delta={'reasoning_content': r}))}\n\n"
                if c:
                    yield f"data: {json.dumps(_chunk(cid, created, kind, delta={'content': c}))}\n\n"
            elif kind == "chat":
                yield f"data: {json.dumps(_chunk(cid, created, kind, delta={'content': payload}))}\n\n"
            else:
                yield f"data: {json.dumps(_chunk(cid, created, kind, text=payload))}\n\n"
    finally:
        cancel_ev.set()


# --- endpoints ---------------------------------------------------------------

@app.get("/v1/models")
@app.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "exllamav3",
        }],
    }

def _thinking_enabled(req: ChatReq) -> bool:
    kw = _template_kwargs(req)
    return kw.get("enable_thinking", True) is not False


def _err(msg, status=400):
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status)


# llama.cpp-style probes some frontends use for engine detection / server info
@app.get("/props")
def props():
    return {
        "model_alias": MODEL_NAME,
        "default_generation_settings": {"n_ctx": MAX_TOKENS},
        "total_slots": 1,
        "chat_template": "chat_template.jinja",
        "bos_token": "",
        "eos_token": "<|im_end|>",
        "build_info": "exllamav3-rocm",
    }


@app.get("/v1/chat/completions")
def chat_completions_get():
    return _err("chat completions requires POST", 405)

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatReq, request: Request):
    n = req.n or 1
    if n != 1 and req.stream:
        return _err("n>1 not supported for streaming")
    try:
        prompt = render_chat(_normalize_messages(req), _template_kwargs(req))
    except Exception as e:
        return _err(f"chat template: {e}")

    think = _thinking_enabled(req)
    include_usage = bool(req.stream_options and req.stream_options.include_usage)

    if req.stream:
        try:
            q, cancel_ev, n_prompt = _start_generation(prompt, req, MAX_TOKENS)
        except Exception as e:
            return _err(str(e))
        return StreamingResponse(
            _sse_stream(q, cancel_ev, request, "chat", think, include_usage, n_prompt),
            media_type="text/event-stream",
        )

    # Non-streaming: n sequential jobs (no parallel sampling in this generator)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    choices = []
    total_prompt = total_new = 0
    for i in range(n):
        try:
            q, cancel_ev, n_prompt = _start_generation(prompt, req, MAX_TOKENS, seed_offset=i)
            text, result = await _collect_with_abort(q, cancel_ev, request)
        except ClientDisconnect:
            return JSONResponse(status_code=499, content={})
        except ValueError as e:
            return _err(str(e))
        except Exception as e:
            return JSONResponse({"error": {"message": str(e), "type": "server_error"}}, 500)
        total_prompt += n_prompt
        total_new += result.get("new_tokens", 0)

        reasoning, content = split_thinking(text, think)
        content, tool_calls = parse_tool_calls(content)
        message = {"role": "assistant", "content": content or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        choices.append({
            "index": i,
            "message": message,
            "finish_reason": _finish_reason(result.get("eos_reason", ""), tool_calls),
        })

    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": choices,
        "usage": _usage(total_prompt, total_new),
    }


@app.post("/v1/completions")
@app.post("/completions")
async def completions(req: CompletionReq, request: Request):
    n = req.n or 1
    if n != 1 and req.stream:
        return _err("n>1 not supported for streaming")
    include_usage = bool(req.stream_options and req.stream_options.include_usage)

    if req.stream:
        try:
            q, cancel_ev, n_prompt = _start_generation(req.prompt, req, MAX_TOKENS)
        except Exception as e:
            return _err(str(e))
        return StreamingResponse(
            _sse_stream(q, cancel_ev, request, "completion", False, include_usage, n_prompt),
            media_type="text/event-stream",
        )

    cid = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    choices = []
    total_prompt = total_new = 0
    for i in range(n):
        try:
            q, cancel_ev, n_prompt = _start_generation(req.prompt, req, MAX_TOKENS, seed_offset=i)
            text, result = await _collect_with_abort(q, cancel_ev, request)
        except ClientDisconnect:
            return JSONResponse(status_code=499, content={})
        except ValueError as e:
            return _err(str(e))
        except Exception as e:
            return JSONResponse({"error": {"message": str(e), "type": "server_error"}}, 500)
        total_prompt += n_prompt
        total_new += result.get("new_tokens", 0)
        choices.append({
            "index": i,
            "text": text,
            "finish_reason": _finish_reason(result.get("eos_reason", "")),
        })

    return {
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": choices,
        "usage": _usage(total_prompt, total_new),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_openai:app", host="0.0.0.0", port=int(os.environ.get("PORT", "9001")))
