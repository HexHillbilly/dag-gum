# Multi-turn cadence benchmark — SPEC

Status: **ready to run** (script committed; results pending GPU availability).

## Why this exists

The single-turn harness (`scripts/measure_explain_in_terms.py`) established the
joint claim — voice + topic — on *isolated* generations. The live production
proxy (llama3-8b + Daisy persona) then showed something single-turn probes cannot
measure: cadence held **at length**, complex concepts held **across turns**, and
cross-turn reference happened **while staying in voice**. The owner's framing of
the baseline failure mode: *"normally all or nothing, even if you have a giant
model"* — you get the voice **or** the content, never both.

This benchmark reproduces that observation under controlled conditions and turns
the anecdote into a number.

## Hypothesis

SCP steers voice by **conditioning** (injecting the persona's own words as
context), not by **constraining** (commanding the voice). Conditioning leaves the
sampler's budget free for content, so voice and thread coexist over turns. A
constraint-based persona (a hard identity directive) forces a tug-of-war — the
model must trade voice for thread or vice versa — which is the "all or nothing"
failure. If that contrast shows up over a multi-turn trajectory, the mechanism
claim generalizes beyond the single-turn results.

## Design

- **Threads** (`THREADS` in the script): 2 threads × 6 turns. Each turn after the
  first embeds a referent to the prior answer's expected content, so answering
  correctly *requires* holding the thread (e.g. "You mentioned the compressor —
  why must the air be compressed before combustion?").
- **Conditions** (per thread, per turn):
  - `baseline` — plain conversation, no persona.
  - `persona` — v2 "weave its spirit" context injection (retrieved line).
  - `explain` — v4 "explain in your own terms" (production default).
  - `hard_prompt` — hard identity directive, **no retrieval** (the pure
    constraint control; see `wrap_hard_prompt`).
  - `hard_prompt_retrieval` — the same hard directive **plus** a retrieved line,
    to isolate "retrieval" from "framing" (see `wrap_hard_prompt_retrieval`).
- **Metrics** (per turn):
  - `style_cos`  = cosine(output_t, corpus centroid) — voice.
  - `topic_cos`  = cosine(output_t, question_t) — answers *this* turn.
  - `thread_cos` = cosine(output_t, mean(all Q so far + all A so far)) — holds
    the thread (turn 0 ≈ topic_cos; grows into a true "conversation memory" score).
  - `ref_cos`    = cosine(output_t, answer_{t-1}) — builds on the prior turn
    (None on turn 0).
- **Model**: `microsoft/Phi-3-mini-4k-instruct` (the reproducible benchmark model,
  consistent with the single-turn harness). A llama3-8b pass through the production
  stack is a planned follow-up to match the live observation.
- **Regimes**: sampling (default, 2 samples) or greedy; `SEED=42`.

## How to run

```bash
python scripts/measure_multiturn.py \
    --corpus docs/corpora/emerson_essays.txt --out /tmp/multiturn.jsonl
```

## What to look for

1. **Voice stability** — `style_cos` trajectory, turn 1 → 6. Flat/high = cadence
   holds at length. Falling = voice slips.
2. **Thread retention** — `thread_cos` trajectory. Flat/high = concepts hold.
   Falling = content drifts off-thread.
3. **All-or-nothing** — `hard_prompt` should show a tradeoff (style↑ while
   thread↓, or the reverse) where `persona`/`explain` hold both.
4. **Cross-turn reference** — `ref_cos` stays high for the conditioning conditions
   and collapses under the constraint condition.
5. **Retrieval vs framing isolation** — `hard_prompt` vs `hard_prompt_retrieval`
   isolates retrieval; `persona`/`explain` vs `hard_prompt_retrieval` isolates
   framing (all three retrieve a line).

## Interpretation

| pattern | meaning |
|---|---|
| persona/explain hold style + thread high over turns | conditioning hypothesis confirmed |
| hard_prompt trades style for thread (or reverse) | constraint = all-or-nothing confirmed |
| all conditions drift on thread over turns | the thread is too hard / Phi-3-mini too weak |
| hard_prompt holds both too | constraint framing is not the variable — revisit |

## Follow-ups (only if the first pattern holds)

1. Run on llama3-8b (production stack) to match the live observation and confirm
   the effect is not a Phi-3-mini artifact.
