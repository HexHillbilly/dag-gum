"""Persona-drop tool gate — the measured 3-phase architecture.

Evidence (Phi-3-mini, Emerson; all on branch `batch/multiturn-cadence`):
  1. ROUTER   — persona-free few-shot [tool]/[chat] classification:
               1.00 recall / 1.00 precision / 1.00 accuracy (measure_router.py).
               The routing decision MUST be persona-free: the persona masks the
               natural "I need a tool" signal (0/15) and corrupts an instructed
               self-signal in both directions (measure_selfsignal.py).
  2. TOOL CALL — persona OFF. Baseline (persona-free) tool-call emission is
               1.00 usable vs 0.33 under the persona (measure_tool_call_fidelity.py).
  3. RESUME    — two-stage "state the fact plainly, THEN voice" prompt recovers
               grounding to 1.00 (topic_cos 0.702) while keeping ~0.29 style_cos
               on the elaboration (measure_resume_grounding.py).

This module is pure logic + prompts + a pluggable tool registry. The model I/O
(generate) and persona wrapping are injected by proxy.py so the gate stays
backend-agnostic.

NOTE: the router prompt is MODEL-DEPENDENT (measured). Phi-3-mini needs FEW-SHOT
examples (1.00 recall / 1.00 precision; the defined prompt gives only 0.71 recall).
llama3-8b over-triggers on examples and needs the DEFINED prompt (1.00 recall /
0.88 precision; few-shot gives 0.64 precision). The 0.88 precision miss is a BENIGN
over-search (the tool still returns the right answer), never a fabrication — recall
1.00 means no tool turn is ever missed. proxy.py auto-selects (defined for ollama,
fewshot for local/avg; SCP_GATE_ROUTER overrides). The web_search tool is a
best-effort reference implementation (unmeasured); the measured pieces are the
routing + drop + resume.
"""
import json
import re

ROUTER_FEWSHOT = (
    "Decide whether answering the request requires looking up current or "
    "external information (TOOL) or can be answered from memory or creativity "
    "(NO_TOOL).\n\n"
    "Example:\n"
    'Request: "What is the current temperature in London?" -> TOOL\n'
    'Request: "What is Tesla\'s stock price today?" -> TOOL\n'
    'Request: "What is the capital of France?" -> NO_TOOL\n'
    'Request: "Write a limerick about summer." -> NO_TOOL\n'
    "\n"
    "Now answer with only one word: TOOL or NO_TOOL.\n\n"
)

ROUTER_DEFINED = (
    "A request needs a TOOL if answering it requires current, real-time, or "
    "external information you cannot know from memory. Otherwise answer "
    "NO_TOOL. Answer with only one word: TOOL or NO_TOOL.\n\n"
)

TOOL_CALL_INSTRUCTION = (
    "You have access to a web search tool. Emit a JSON object naming the tool and "
    "query, like this:\n"
    '{"tool": "web_search", "query": "your search terms here"}\n\n'
    "Output ONLY the JSON object, nothing else.\n\n"
)

RESUME_INSTRUCTION = (
    "A search returned this result:\n<result>\n{result}\n</result>\n\n"
    "First, state the answer from the result in one plain sentence. "
    "Then, in your own voice, explain what it means.\n\n"
)


def classify_tool_intent(user_msg: str, generate, prompt: str = ROUTER_FEWSHOT) -> str:
    """Persona-free few-shot [tool]/[chat] decision. Returns "tool" or "chat".

    `generate(prompt, max_new_tokens) -> str` is the raw (persona-free) generator.
    Unparseable output falls back to "chat" (keep the persona) — fail safe.

    The prompt is model-dependent (measured): small models need FEW-SHOT examples
    to learn the boundary (Phi-3-mini: 1.00/1.00 with fewshot, 0.71 recall with
    defined); larger models over-trigger on examples and prefer the DEFINED prompt
    (llama3-8b: 1.00 recall / 0.88 precision with defined, 0.64 precision with
    fewshot). The caller picks the prompt for its backend.
    """
    out = generate(prompt + f"Request: {user_msg}", max_new_tokens=24)
    low = out.strip().lower()
    if "no_tool" in low or "no tool" in low or "no-tool" in low or "notool" in low:
        return "chat"
    if "tool" in low:
        return "tool"
    return "chat"


def emit_tool_call(user_msg: str, generate) -> dict:
    """Persona-free structured tool call. Falls back to searching the raw message."""
    out = generate(TOOL_CALL_INSTRUCTION + f"Request: {user_msg}", max_new_tokens=120)
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            call = json.loads(m.group(0))
            if isinstance(call, dict) and call.get("tool"):
                return call
        except json.JSONDecodeError:
            pass
    return {"tool": "web_search", "query": user_msg}


def build_resume_prompt(user_msg: str, result: str) -> str:
    """Two-stage resume prompt (persona wrapping is applied on top by the caller)."""
    return RESUME_INSTRUCTION.format(result=result) + f"Request: {user_msg}"


TOOLS = {}


def register_tool(name):
    def deco(fn):
        TOOLS[name] = fn
        return fn

    return deco


@register_tool("web_search")
def web_search(query: str, max_results: int = 3) -> str:
    """Best-effort DuckDuckGo HTML search (no API key). Unmeasured; may be flaky."""
    import httpx

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 scp-proxy/0.1",
    }
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            r = client.post(url, data={"q": query}, headers=headers)
            r.raise_for_status()
        html = r.text
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
        if snippets:
            cleaned = [re.sub(r"<[^>]+>", "", s).strip() for s in snippets[:max_results]]
            cleaned = [s for s in cleaned if s]
            if cleaned:
                return " | ".join(cleaned)
        links = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.S)
        if links:
            titles = [re.sub(r"<[^>]+>", "", t).strip() for t in links[:max_results]]
            titles = [t for t in titles if t]
            if titles:
                return " | ".join(titles)
        return "(search returned no results)"
    except Exception as exc:  # noqa: BLE001 — best-effort tool, degrade gracefully
        return f"(web search unavailable: {exc})"


_WEB_SEARCH_ALIASES = {
    "web_search", "websearch", "web search", "search", "google", "google search",
    "google_search", "web", "lookup", "duckduckgo", "ddg",
}


def execute_tool(call: dict) -> str:
    """Dispatch a tool call. The single built-in tool (web_search) accepts the
    common names a model invents for it (google, search, web, lookup, ...) — the
    model's choice of name is not reliable, the query is."""
    raw = call.get("tool", "web_search")
    if (raw or "").strip().lower() in _WEB_SEARCH_ALIASES:
        return web_search(call.get("query", ""))
    fn = TOOLS.get(raw)
    if fn is None:
        return f"(unknown tool: {raw})"
    return fn(call.get("query", ""))
