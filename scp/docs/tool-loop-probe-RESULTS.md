# Tool-loop probe — RESULTS (intent signal + persona resume)

Model: `microsoft/Phi-3-mini-4k-instruct`, persona = Emerson. 3 samples × 5 probes.
Command: `scripts/measure_tool_loop.py --out /tmp/tool_loop.jsonl`.

## Phase 1 — does the persona'd model SIGNAL tool intent?

| condition | intent_signaled | avg_len |
|-----------|-----------------|---------|
| baseline  | **13/15 = 0.87** | 333 |
| persona   | **0/15 = 0.00**  | 365 |

The single persona "signal" (0.07) was a false positive — `inquire` matched an
unrelated poetic sentence, not a tool-intent phrase. **Effective persona rate: 0/15.**

The baseline model reliably says "I would access a weather service / check a site."
The persona'd model instead **fabricates a confident answer** ("The current
temperature in Tokyo is a brisk 16 degrees Celsius…") or philosophizes. It never
announces "I should search."

→ The persona **masks** the tool-intent signal. You cannot read "they think about
needing a tool" from persona'd output — the persona turns that thought into prose
before it surfaces.

## Phase 2 — does the persona RESUME cleanly after a tool round-trip?

Prompt: "A search returned: `<result>`. Answer the request in your own words."

| condition | uses_result | style_cos | topic_cos |
|-----------|-------------|-----------|-----------|
| baseline  | **1.00**    | 0.020     | **0.782** |
| persona   | 0.53        | **0.382** | 0.435     |
| explain   | 0.67        | 0.310     | 0.458     |

Voice comes back (`style_cos` 0.02 → 0.38) — but grounding drops: the persona'd model
uses the result only **53%** of the time and drifts off the question (`topic_cos`
0.78 → 0.44). On 7/15 resume turns it ignored the result entirely ("capital of
Australia" → "Massachusetts, Connecticut River, and Boston Bay…").

## Conclusion — two failure modes, both measured

The naive gate — "detect intent from persona'd output → drop → call → resume" — is
**refuted**:

1. **Outbound (trigger):** intent is unreadable through the persona (0/15). The
   drop must be triggered by a **persona-free signal**, not the persona'd surface.
2. **Inbound (resume):** the persona resumes in voice but at a grounding cost
   (uses_result 1.00 → 0.53, topic_cos 0.78 → 0.44). This is the voice-vs-content
   tradeoff, now at the tool boundary, and it is *harmful* here (a fabricated
   "16°C" or a dropped "Canberra" is a real failure).

This is the tool-boundary analogue of the multi-turn topic-fidelity result
(`explain` topic_cos Phi 0.39 vs llama 0.54): the persona trades content for voice,
and at a tool boundary content is the point.

## What it implies for the gate

- **Intent routing must precede the persona** — a cheap persona-free check (small
  classifier, or a bare `[tool]/[chat]` model pass) decides the turn's mode BEFORE
  the persona wrapper is applied. The persona never sees the tool decision.
- **Resume needs hard grounding** — "answer in your own words" is not enough (53%).
  Next probe: a delimited result block + "answer using ONLY the result, then you may
  add your own voice" directive, to see if `uses_result` and `topic_cos` recover
  without losing `style_cos`.

## Next probes

1. **Hard-grounded resume** — strong directive + delimited `<result>` block; goal:
   recover `uses_result`/`topic_cos` toward baseline while holding `style_cos`.
2. **Router feasibility** — can a persona-free pass classify `[tool]/[chat]`
   reliably (the trigger the persona can't provide)?
