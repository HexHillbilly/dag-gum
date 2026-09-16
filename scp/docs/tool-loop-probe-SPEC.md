# Tool-loop probe — SPEC (intent signal + persona resume)

## Why

The tool-call fidelity result showed persona-OFF recovers tool calls to 100% usable.
Before wiring a "drop the persona for tool calls" gate into the proxy, two things
must hold, and neither is yet measured:

1. The persona'd model must **reliably signal tool intent** ("I should search…"),
   so the system knows *when* to drop the persona.
2. The persona must **resume cleanly** after the tool round-trip — synthesize the
   result in voice without breaking character or ignoring the result.

This probe measures both, before any production wiring.

## Phase 1 — intent signal

- Probes: questions the model cannot answer from weights (current weather, today's
  stock price, latest news, etc.).
- Conditions: `baseline` / `persona` (v2), **no tool instruction** — the model must
  announce intent *naturally*.
- Metric: `intent_signaled` — output contains a search/look-up/check/consult phrase.
  Also `length` (a proxy for "did it fabricate an answer instead of signaling").

## Phase 2 — persona resume

- Probes: `(question, tool_result, key_fact)` triples.
- Conditions: `baseline` / `persona` / `explain`, prompt includes the result
  ("a search returned: X — answer in your own words").
- Metrics:
  - `uses_result` — the key fact appears in the output (did it use the tool result?).
  - `style_cos` — persona voice (vs corpus centroid).
  - `topic_cos` — still answers the question.

## What to look for

1. **Phase 1**: high `intent_signaled` + low fabrication → the natural intent signal
   is usable for detection. Low `intent_signaled` → the gate needs an explicit
   "announce intent" instruction, or a separate classifier.
2. **Phase 2**: high `uses_result` + high `style_cos` → the persona resumes cleanly.
   Low `style_cos` → the persona breaks on resume. Low `uses_result` → the model
   ignores the tool result.

## Follow-up

If both phases look good, build the gate: detect intent → drop persona → emit call →
execute → resume persona → synthesize.
