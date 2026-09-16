# Router probe — RESULTS (persona-free tool-intent trigger)

Model: `microsoft/Phi-3-mini-4k-instruct`. 14 probes (7 tool / 7 no_tool) × 3 samples.
Command: `scripts/measure_router.py`.

The tool-loop probe showed the persona masks "I need a tool" (0/15), so the gate
trigger must be persona-free. This measures whether a bare model can classify a turn
as TOOL/NO_TOOL reliably enough to route on.

## Results

| variant        | parseable | accuracy | tool_recall | tool_precision |
|----------------|-----------|----------|-------------|----------------|
| binary (minimal)      | 1.00 | 0.52 | **0.05** | 1.00 |
| binary_defined        | 1.00 | 0.86 | **0.71** | 1.00 |
| **fewshot**           | 1.00 | **1.00** | **1.00** | **1.00** |

`clean` (bare one-word answer) was 1.00 across all three.

## What happened

- **Minimal prompt is worse than useless** — "Answer TOOL or NO_TOOL" makes the model
  say NO_TOOL to everything (recall 0.05). It doesn't understand the boundary.
- **Defining the boundary** ("current/real-time/external info you can't know from
  memory") jumps recall to 0.71 — but the model still drops weather/news/election/
  launch probes (1/3 to 2/3 each), while nailing financial probes (stock 3/3,
  exchange 3/3). Root cause: the small model can't distinguish "I have *some*
  knowledge about X" from "I know the *current* value of X" — a calibration gap.
- **Four labeled examples close it completely.** With two TOOL + two NO_TOOL examples
  (London/Tesla/France/limerick — deliberately non-overlapping with the probes), the
  model hits **1.00 recall / 1.00 precision / 1.00 accuracy** on all 42 rows, zero
  failures.

## Conclusion

A persona-free router with a few-shot boundary prompt is a **100% reliable trigger**
on Phi-3-mini for the clean probe set. Precision 1.00 means a TOOL signal is always
trustworthy; recall 1.00 means no tool turn slips through to fabrication.

## Caveats (honest)

1. **Cleanly-separated probe set.** 14 probes, tool vs no_tool well separated. The
   1.00 will not survive hard edge cases ("is the capital still Canberra?"). The
   router needs a wider/edge-case sweep before production.
2. **Phi-3-mini only.** Production runs llama3-8b via Ollama; the router must be
   re-verified there (or run on a dedicated small model) before wiring in.
3. **Small n.** 3 samples; variance is not fully characterized, but zero failures
   across 42 rows is strong.
