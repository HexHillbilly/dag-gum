# Tool-call fidelity — RESULTS (Emerson, Phi-3-mini)

## Question

Does SCP persona injection corrupt structured tool-call output? (The precondition for
a "drop the persona for tool calls" gate.)

## Setup

- 5 search-tool probes, 3 conditions (`baseline` / `persona` / `explain`), 3 samples
  (sampling), `microsoft/Phi-3-mini-4k-instruct`.
- Prompt = tool instruction ("output ONLY a JSON object") + the request; the persona
  conditions wrap the same request with Emerson framing.
- Metrics: `usable` (extractable correct call), `clean` (output is only the JSON),
  `extra_chars` (non-JSON chatter).

## Results

| condition | usable | clean | extra_chars |
|---|---|---|---|
| baseline | 1.00 | 0.00 | 10.0 |
| persona | 0.33 | 0.00 | 265.6 |
| explain | 0.40 | 0.13 | 229.7 |

## Finding

Persona injection **destroys** tool-call fidelity: `usable` drops 1.00 → 0.33
(persona) / 0.40 (explain). The model does not decorate the JSON — it abandons it and
writes persona prose ("As Emerson, I might express the sentiment…"). The "output ONLY
JSON" format constraint is overridden by the "weave your spirit" / "in your own terms"
framing ~2/3 of the time.

Baseline's `clean=0` is harmless markdown (```json``` fences, ~10 chars) — the JSON is
still 100% usable. The persona's corruption is the `usable` collapse, not the fence
noise.

## Conclusion

The "drop the persona for tool calls" gate is **essential**, not optional: without it,
60–67% of tool calls never fire. Next: build the gate (detect tool intent → drop
persona → emit call → execute → resume) and measure the recovery.
