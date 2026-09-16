# Multi-turn cadence benchmark — RESULTS (Emerson, Phi-3-mini)

Status: **first pass** (indicative — 2 samples/condition, weak benchmark model).

## Setup

- 2 threads × 6 turns (progressive follow-ups, each referencing the prior answer).
- 5 conditions: `baseline`, `persona` (v2), `explain` (v4), `hard_prompt`
  (hard directive, no retrieval), `hard_prompt_retrieval` (hard directive + retrieval).
- Model `microsoft/Phi-3-mini-4k-instruct`, sampling (T=0.8, top_p=0.9), 2 samples, SEED=42.
- Metrics per turn: `style_cos` (voice), `topic_cos` (answers this turn),
  `thread_cos` (holds whole thread), `ref_cos` (builds on prior turn).

## Overall (mean over all turns)

| condition | style | topic | thread | ref |
|---|---|---|---|---|
| baseline | −0.022 | 0.701 | 0.635 | 0.454 |
| persona | 0.338 | 0.495 | 0.575 | 0.504 |
| explain | 0.439 | 0.320 | 0.501 | 0.482 |
| hard_prompt | 0.349 | 0.470 | 0.532 | 0.470 |
| hard_prompt_retrieval | 0.227 | 0.498 | 0.589 | 0.514 |

## Trajectory (style_cos / thread_cos, turn 0 → 5)

| condition | style t0→t5 | thread t0→t5 |
|---|---|---|
| baseline | −0.01 → −0.02 (flat ~0) | 0.78 → 0.59 |
| persona | 0.31 → 0.42 (**+0.11**) | 0.58 → 0.55 (−0.03) |
| explain | 0.33 → 0.54 (**+0.21**) | 0.60 → 0.31 (**−0.29**) |
| hard_prompt | 0.32 → 0.34 (flat) | 0.53 → 0.53 (0.00) |
| hard_prompt_retrieval | 0.23 → 0.24 (flat) | 0.53 → 0.62 (+0.09) |

## Findings

1. **Cadence holds at length — CONFIRMED.** Voice (`style_cos`) is stable or
   rising over 6 turns in every persona condition; it never collapses. This
   reproduces the production observation (the "all or nothing" voice-collapse
   did not appear).

2. **The "constraint = all-or-nothing" hypothesis was NOT confirmed.** The hard
   directive (`hard_prompt`) held both voice (0.349) and thread (0.532) about as
   well as the soft `persona` framing. The predicted collapse did not show up
   where expected.

3. **The real tension is in `explain` (v4).** `explain` has the strongest voice
   (0.439, rising to 0.54) but the worst thread retention (0.60 → 0.31). By turn 5
   it abandons the question entirely and writes pure persona essays (turn-5
   `topic_cos` ≈ −0.02). The single-turn winner is the weakest multi-turn holder:
   the repeated "in your own voice and terms" cue + a fresh retrieved line each
   turn progressively over-weights voice against the thread.

4. **`persona` (v2) is the balanced multi-turn choice** on Phi-3-mini: voice rises
   (+0.11) while thread holds (−0.03).

5. **The hard directive is counterproductive for voice here.** `hard_prompt_retrieval`
   has the lowest `style_cos` (0.227) — "MUST… never break character" + a retrieved
   example *damped* voice transfer relative to the soft framing, while giving the
   best thread retention (0.589). On Phi-3-mini, soft conditioning beats commanding.

## Caveats

- **Phi-3-mini (3.8B) is the weak benchmark model.** The production observation —
   cadence *and* complex concepts both holding across a long chat, with cross-chat
   reference — was on **llama3-8b**. The thread drift seen here (especially `explain`)
   may be a small-model limit, not a mechanism property.
- 2 samples/condition → indicative, not statistically locked.

## llama3-8b (production model class) — follow-up result

Re-ran the same benchmark on `NousResearch/Meta-Llama-3-8B-Instruct` (4-bit), the
SCP stack's llama3-8b — the model class the live production observation came from.

| condition | style | topic | thread | ref |
|---|---|---|---|---|
| baseline | −0.041 | 0.732 | 0.627 | 0.478 |
| persona | 0.326 | 0.473 | 0.627 | 0.591 |
| explain | 0.227 | 0.600 | 0.617 | 0.504 |
| hard_prompt | 0.258 | 0.585 | 0.639 | 0.568 |
| hard_prompt_retrieval | 0.251 | 0.584 | 0.606 | 0.508 |

`explain` (v4) thread trajectory turn 0→5: **0.69 → 0.51** (drift −0.18), vs
Phi-3-mini's 0.60 → 0.31 (−0.29). `explain` `topic_cos` = **0.600** on llama3-8b vs
**0.320** on Phi-3-mini — the stronger model keeps answering while in voice; the
weaker one dissolves into pure persona essays (turn-5 topic −0.02).

**Finding (refined at n=4): the model-size effect is on TOPIC fidelity, not thread
drift.** Doubling to n=4 (240 gens/model): the `explain` `topic_cos` gap is
significant — Phi-3-mini **0.390±0.032** vs llama3-8b **0.536±0.029**, **z=3.4
(p<0.001)** — the stronger model keeps answering the question under v4, the weaker
drifts off it. The thread drift is NOT model-size-dependent: both drift nearly
identically (Phi-3-mini +0.124 vs llama3-8b +0.133, turn 0→5); the n=2 thread
divergence was sample noise. The effect is also **v4-specific**: the topic gap
(llama−phi) is baseline +0.051 / persona (v2) **+0.002** / explain (v4) **+0.146**.
v2 erases the model gap; v4 amplifies it. The production observation ("holding
complex concepts throughout") is exactly this topic-fidelity axis — the axis where
llama3-8b beats Phi-3-mini under v4.

**Secondary (negative) result:** the "constraint = all-or-nothing" hypothesis still
did not reproduce — `hard_prompt` held both voice and thread on BOTH models. The
"all or nothing" the owner describes is therefore not a prompt-directive effect; it
is consistent with the logit-level constraint finding already established
(logit-bias skew does NOT transfer voice — see Phase B), not prompt framing.

## Follow-up (remaining)

- Optionally raise `--samples` (2 → 4+) for statistical confidence on the
  model-size finding; the direction is already clear at 2 samples.
