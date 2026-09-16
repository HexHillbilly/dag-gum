"""Semantic Context Proxy — production mode: coherent persona voice injection.

Retrieves a line from the brain corpus relevant to the prompt and injects it as
a coherent persona directive, then generates the response. This is the mode the
research established as the actual voice-transfer mechanism
(scripts/benchmark.py phases 1-4).

Backends (env `SCP_BACKEND`):
  local   (default) serve the cached HF model directly via transformers —
          reuses ~/.cache/huggingface, no Ollama and no download.
  ollama  forward to a local Ollama server (TARGET_LLM_URL).
  avg     like local, but the generation loop is driven by the AVG
          (Active Variety Governor) mid-generation controller: SCP conditions
          pre-generation voice, AVG suppresses repetition / early-EOS during
          decoding. Requires the AVG package on sys.path.

Config (env vars):
  SCP_PERSONA_NAME   persona name (default "Daisy")
  SCP_PERSONA_DESC   persona description (default: the Daisy voice)
  SCP_MODE           "persona" (default) | "markov" (v1.0 behavior)
  SCP_BACKEND        "local" (default) | "ollama" | "avg"
  SCP_LOCAL_MODEL    HF model id for the local/avg backends
                     (default "microsoft/Phi-3-mini-4k-instruct" — instruction-
                     tuned; the cached Qwen2.5-1.5B *base* cannot follow
                     directives and must NOT be used here)
  SCP_FRAMING        "explain" (default) | "harden" | "persona" — which persona
                     directive to inject. "explain" (v4) = "explain the request
                     thoroughly and accurately, in your own voice and terms"
                     (the personalized-explainer framing); "harden" (v2) =
                     "weave its spirit" without the leaky tail; "persona" (v1)
                     = original (with the leaky "answer naturally" tail).
  SCP_QUERY_TAIL_CHARS  max chars of the user message to embed for retrieval
                     (default 1200 — embed the tail/question, not a pasted dump)
  SCP_AVG_BAND_LOW   spectral-PR collapse threshold for the avg backend
                     (default 6.95 = Phi-3-mini-4k; 7.8703 = llama3-8b 4-bit)
  SCP_QUANTIZE       quantization for the local/avg backends: none (default,
                     bf16) | 8bit | 4bit — use 4bit for 8B models on 12GB VRAM
  SCP_GATE           "1" enables the persona-drop tool gate (router + drop +
                     two-stage resume; see gate.py). Default off. Router is
                     verified on Phi-3-mini only — re-verify on llama3-8b
                     before enabling in production.
  SCP_MODEL_ID       model id advertised on GET /v1/models (default llama3-8b)
"""
import json
import os
import re
import time

import faiss
import httpx
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import (
    markov_synthesize,
    retrieve,
    wrap_explain_in_terms,
    wrap_misfire,
    wrap_persona,
    wrap_persona_hardened,
)

import gate  # persona-drop tool gate (SCP_GATE=1)

INDEX_FILE = "brain.index"
MAP_FILE = "brain_lines.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TARGET_LLM_URL = "http://localhost:11434/v1/chat/completions"

PERSONA_NAME = os.environ.get("SCP_PERSONA_NAME", "Daisy")
PERSONA_DESC = os.environ.get(
    "SCP_PERSONA_DESC",
    "a naive, warm 2000-era language-learning chatbot with no hardcoded language, "
    "who speaks in lowercase and makes her own sentences by recombining phrases she has heard",
)
MODE = os.environ.get("SCP_MODE", "persona")
BACKEND = os.environ.get("SCP_BACKEND", "local")
LOCAL_MODEL_ID = os.environ.get("SCP_LOCAL_MODEL", "microsoft/Phi-3-mini-4k-instruct")
DEVICE = os.environ.get("SCP_DEVICE", "auto")
SERVED_MODEL = os.environ.get("SCP_MODEL_ID", "llama3-8b")
FRAMING = os.environ.get("SCP_FRAMING", "explain")  # explain (v4) | harden (v2) | persona (v1)
QUERY_TAIL_CHARS = int(os.environ.get("SCP_QUERY_TAIL_CHARS", "1200"))
AVG_BAND_LOW = float(os.environ.get("SCP_AVG_BAND_LOW", "6.95"))
QUANTIZE = os.environ.get("SCP_QUANTIZE", "none")  # none | 8bit | 4bit (local/avg backends)
GATE = os.environ.get("SCP_GATE", "0") == "1"  # persona-drop tool gate (router+drop+resume)
GATE_ROUTER = os.environ.get("SCP_GATE_ROUTER", "auto")  # auto | fewshot | defined
if GATE_ROUTER == "defined":
    _router_prompt = gate.ROUTER_DEFINED
elif GATE_ROUTER == "fewshot":
    _router_prompt = gate.ROUTER_FEWSHOT
else:  # auto: defined for ollama (llama3-8b), fewshot for local/avg (Phi-3-mini)
    _router_prompt = gate.ROUTER_DEFINED if BACKEND == "ollama" else gate.ROUTER_FEWSHOT

print(f"[*] Booting the Semantic Context Proxy (mode={MODE}, persona={PERSONA_NAME}, "
      f"backend={BACKEND}, framing={FRAMING})...")

if not os.path.exists(INDEX_FILE) or not os.path.exists(MAP_FILE):
    print("[!] Brain not found. Run compile_brain.py first.")
    raise SystemExit(1)

index = faiss.read_index(INDEX_FILE)
with open(MAP_FILE, "r", encoding="utf-8") as f:
    text_map = json.load(f)

embedder = SentenceTransformer(MODEL_NAME)

_local_model = _local_tok = None
if BACKEND in ("local", "avg"):
    print(f"[*] Loading model for backend '{BACKEND}': {LOCAL_MODEL_ID} (quant={QUANTIZE})")
    _local_tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
    if QUANTIZE in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_8bit=(QUANTIZE == "8bit"), load_in_4bit=(QUANTIZE == "4bit")
        )
        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID, quantization_config=quant_config, device_map="auto"
        )
        print(f"[*] Model quantized ({QUANTIZE}), device_map=auto")
    else:
        _local_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, dtype=torch.bfloat16)
        device = DEVICE if DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            _local_model = _local_model.to(device)
        except torch.cuda.OutOfMemoryError:
            print(f"[!] CUDA OOM on {device} — falling back to CPU (slow).")
            device = "cpu"
            _local_model = _local_model.to("cpu")
        print(f"[*] Model on: {device}")
    _local_model.eval()
    if _local_tok.pad_token_id is None:
        _local_tok.pad_token_id = _local_tok.eos_token_id

_governor = None
if BACKEND == "avg":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # AVG package parent (repo root IS the AVG package)
    from AVG.governor.controller import ActiveVarietyGovernor

    _governor = ActiveVarietyGovernor(_local_model, _local_tok, band_low=AVG_BAND_LOW)
    print(f"[*] AVG governor armed (band_low={AVG_BAND_LOW}; "
          f"suppression/eos_guard defaults 5.0/True)")

app = FastAPI()


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model list so OpenWebUI can validate/discover us."""
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "scp",
            }
        ],
    }


def _query_tail(user_msg: str, max_chars: int = QUERY_TAIL_CHARS) -> str:
    """Embed the TAIL of a long message, not the head: MiniLM truncates a query
    to ~256 tokens from the front, so a pasted dump would otherwise dominate the
    retrieval and the actual question (usually last) would never be seen."""
    if len(user_msg) <= max_chars:
        return user_msg
    return user_msg[-max_chars:]


def _framing_wrapper():
    if FRAMING == "explain":
        return wrap_explain_in_terms
    if FRAMING == "harden":
        return wrap_persona_hardened
    return wrap_persona


def inject(user_msg: str) -> str:
    """Return the wrapped prompt for a user message."""
    matched, _ = retrieve(_query_tail(user_msg), index, text_map, embedder, k=3)
    if not matched:
        return user_msg
    if MODE == "markov":
        misfire, _ = markov_synthesize(matched)
        print(f"[MARKOV INJECTS]: {misfire}")
        return wrap_misfire(user_msg, misfire)
    seed = matched[0]
    print(f"[PERSONA SEED]: {seed}")
    return _framing_wrapper()(user_msg, seed, PERSONA_NAME, PERSONA_DESC)


def generate_local(wrapped: str, max_new_tokens: int = 256) -> str:
    msgs = [{"role": "user", "content": wrapped}]
    text = _local_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = _local_tok(text, return_tensors="pt").to(_local_model.device)
    with torch.no_grad():
        out = _local_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=_local_tok.pad_token_id,
        )
    return _local_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def generate_avg(wrapped: str, max_new_tokens: int = 256) -> str:
    """Governed generation: SCP persona (pre-gen) + AVG governor (mid-gen)."""
    msgs = [{"role": "user", "content": wrapped}]
    templated = _local_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    result = _governor.generate(
        templated,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        intervene=True,
    )
    return (result.get("gen_only_text") or "").strip()


def generate_raw(prompt: str, max_new_tokens: int = 64) -> str:
    """Persona-free generation on the active backend (router + tool call).

    Greedy/deterministic: a trigger and a structured call should be stable.
    """
    if BACKEND in ("local", "avg"):
        msgs = [{"role": "user", "content": prompt}]
        text = _local_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = _local_tok(text, return_tensors="pt").to(_local_model.device)
        with torch.no_grad():
            out = _local_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=_local_tok.pad_token_id,
            )
        return _local_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # ollama: raw persona-free completion
    body = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(TARGET_LLM_URL, json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[!] raw ollama generate failed: {exc}")
        return ""


def generate_wrapped(wrapped: str, max_new_tokens: int = 256) -> str:
    """Generate from a (persona-wrapped) prompt on the active backend."""
    if BACKEND == "local":
        return generate_local(wrapped, max_new_tokens)
    if BACKEND == "avg":
        return generate_avg(wrapped, max_new_tokens)
    body = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": wrapped}],
        "stream": False,
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(TARGET_LLM_URL, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _extract_openwebui_search(messages) -> str:
    """Pull OpenWebUI's injected <source> web-search/RAG context out of the
    incoming messages (present when the model-card web search toggle is ON).

    OpenWebUI runs its own search and injects results as <source id=...>...</source>
    tags (verified in the open-webui:main container, retrieval/utils.py +
    middleware.py apply_source_context_to_messages). Returns the joined source
    bodies, or "" when OpenWebUI injected nothing.
    """
    bodies = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, str):
            bodies += re.findall(r"<source[^>]*>(.*?)</source>", content, re.S)
    text = " | ".join(b.strip() for b in bodies if b.strip())
    return text[:2000]


def gate_tool_flow(user_msg: str, preinjected: str = "") -> str:
    """Phases 2+3: persona-free tool call -> execute -> two-stage resume.

    Phase 1 (the router decision) happens in the endpoint so it can fall through
    to the persona path on "chat".

    `preinjected` is OpenWebUI's already-injected web-search context (its
    model-card web search toggle ON). When present we skip our own web_search and
    ground the resume on OpenWebUI's results — no double search, and the proxy's
    search behaviour follows the OpenWebUI toggle.
    """
    if preinjected:
        result = preinjected
        print(f"[GATE RESULT (OpenWebUI web search)]: {result[:200]}")
    else:
        call = gate.emit_tool_call(user_msg, generate_raw)
        print(f"[GATE TOOL CALL]: {call}")
        result = gate.execute_tool(call)
        print(f"[GATE RESULT (proxy web search)]: {result[:200]}")
    prompt = gate.build_resume_prompt(user_msg, result)
    matched, _ = retrieve(_query_tail(user_msg), index, text_map, embedder, k=3)
    if matched:
        # resume measured with harden (v2), not explain (v4) — see gate.py header
        prompt = wrap_persona_hardened(prompt, matched[0], PERSONA_NAME, PERSONA_DESC)
    return generate_wrapped(prompt)


def completion_response(content: str) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


def _completion_sse_bytes(content: str) -> bytes:
    """Wrap a single completion as an OpenAI-style SSE stream (one delta chunk
    + a [DONE] sentinel), so OpenWebUI's stream=true client gets what it wants."""
    cid = f"chatcmpl-{int(time.time() * 1000)}"
    base = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": SERVED_MODEL,
    }
    chunk = {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]}
    done = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    return (
        f"data: {json.dumps(chunk)}\n\n"
        f"data: {json.dumps(done)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")


@app.post("/v1/chat/completions")
async def intercept_and_inject(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])

    if messages and messages[-1].get("role") == "user" and messages[-1].get("content"):
        last_user_msg = messages[-1]["content"]
        print(f"\n[USER PROMPT]: {last_user_msg}")

        # ---- persona-drop tool gate (SCP_GATE=1) ----
        if GATE:
            decision = gate.classify_tool_intent(last_user_msg, generate_raw, prompt=_router_prompt)
            print(f"[GATE]: {decision}")
            if decision == "tool":
                injected = _extract_openwebui_search(messages)
                content = gate_tool_flow(last_user_msg, preinjected=injected)
                print(f"[GATED SAYS]: {content}")
                if payload.get("stream"):
                    async def _gate_sse():
                        yield _completion_sse_bytes(content)

                    return StreamingResponse(_gate_sse(), media_type="text/event-stream")
                return JSONResponse(content=completion_response(content))

        wrapped = inject(last_user_msg)
        if BACKEND == "local":
            content = generate_local(wrapped)
            print(f"[DAISY SAYS]: {content}")
            return JSONResponse(content=completion_response(content))
        if BACKEND == "avg":
            content = generate_avg(wrapped)
            print(f"[DAISY SAYS]: {content}")
            if payload.get("stream"):
                async def _avg_sse():
                    yield _completion_sse_bytes(content)

                return StreamingResponse(_avg_sse(), media_type="text/event-stream")
            return JSONResponse(content=completion_response(content))
        messages[-1]["content"] = wrapped
        payload["messages"] = messages

    if payload.get("stream") and BACKEND == "ollama":
        # Stream the SSE reply straight back to the client. OpenWebUI defaults
        # to stream=true; buffering the stream and json()-ing it produced a
        # JSONDecodeError -> HTTP 500 -> "keeps thinking". The client must be
        # created INSIDE the generator: the route's async-with client closes
        # the moment we return the StreamingResponse (before the stream is
        # consumed), which raised "Cannot send a request, client closed".
        async def sse():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", TARGET_LLM_URL, json=payload, timeout=120.0
                ) as upstream:
                    upstream.raise_for_status()
                    async for chunk in upstream.aiter_bytes():
                        yield chunk

        return StreamingResponse(sse(), media_type="text/event-stream")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(TARGET_LLM_URL, json=payload, timeout=120.0)
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except httpx.RequestError as exc:
            print(f"[!] Target LLM connection failed: {exc}")
            return {"error": "Target LLM unreachable.", "mutated_payload": payload}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
