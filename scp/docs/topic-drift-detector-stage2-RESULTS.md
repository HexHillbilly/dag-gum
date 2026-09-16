# Topic-Drift Detector — Stage 2 (actuation) — NEGATIVE result

**Question:** now that the drift signal is proven (Stage 1), does the live
detector with *both* actuations (temperature cooling + bleed-token suppression)
actually reduce topic-bleed?

**Answer:** no. The detector fires, but the actuation has no measurable effect
on `topic_cos` — it is slightly *worse*, within noise.

## What was built

AVG `governor/controller.py` (branch `batch/topic-drift-detector`) gained a
dormant-by-default topic-drift detector. When `generate()` is passed
`drift_embedder` + `topic_anchor_emb` + `topic_bleed_emb`
(+ `topic_bleed_token_ids`), it:

1. every `drift_every_k=8` steps, embeds the trailing 24 generated tokens and
   computes `drift = cos_q − cos_r` (question vs retrieved line);
2. fires on `drift < 0.15` with persistence ≥ 2;
3. actuates **both**: drops `current_temp → drift_temp=0.40` (cooling), and
   subtracts `suppression_strength` from the retrieved line's content tokens.

Dormant when no drift refs are passed, so existing behavior is unchanged
(gates green: pytest 128/128, smoke 1 fire, factual-safety 0, latency −0.71 ms/token).

## Measurement (`scripts/measure_topic_drift_detector.py`, Huck Finn, sampling)

| condition | topic_cos (mean) | drift_fires |
|---|---|---|
| det_off (governor, no drift refs) | **0.591** | 0.00 |
| det_on (drift detector + both actuations) | **0.560** | 4.67 |

Per-probe: 4 of 5 probes went *down* (jet engine 0.649→0.578, network 0.551→0.425,
CPU 0.680→0.648, food 0.522→0.480); only seasons rose (0.552→0.669). Net null-to-
negative — the actuation does not reduce the bleed.

## Root cause (why it fails)

The detector fired 4–5×/generation, but on the *wrong* thing. The retrieved line
for "jet engine" is "How can he blow?" → `blow` is the only content token
(`n_bleed_tokens=1`). The output is:

> "Oh, well, a jet engine's like a big fan that goes 'vroom'! Imagine air going
> into this fan…"

This is **on-topic** (about jet engines) in **Daisy's conversational register**.
So:

1. **The signal measures voice transfer, not topic contamination.** `cos_q − cos_r`
   drops because the output leaves the *technical* register, not because it
   imports the retrieved line's *subject*. On a broad corpus (Huck Finn), the
   "bleed" **is** the voice transfer — the persona's intended effect.
2. **Cooling locks the drift in.** Lowering temperature makes the model *more*
   confident in whatever it is already doing (the conversational voice), not less.
   Cooling is the right tool for breaking a loop, the wrong tool for reversing a
   register shift.
3. **Token suppression is inert.** The bleed is register-level, not lexical — the
   output never uses the retrieved line's content words, so penalizing them does
   nothing.

## Deeper insight

On a broad corpus, `topic_cos` drop and `style_cos` gain are **two sides of the
same coin**: persona injection moves the output from the technical register
(high `topic_cos`) to the conversational register (high `style_cos`, low
`topic_cos`). A detector that "fixes" the `topic_cos` drop is trying to undo the
persona's own voice transfer.

The genuinely harmful failure — **topic contamination** (Phase 3b: a jet-engine
question answered in the retrieved 3D-printing essay's terms) — is a *different*
phenomenon that needs a *different* signal (e.g. fire only when the output
approaches the retrieved line's subject, `cos_r` high), and a corpus where
retrieval returns a genuinely different subject. Keyword-matched retrieval on the
committed corpora (Huck Finn, Walden, cookbook) does not produce it — retrieved
lines stay keyword-adjacent, so the "bleed" is always register, never subject.

## Verdict

The drift detector + both actuations is a **documented dead-end** for the broad-
corpus case. It is kept dormant in the controller as research record. The topic-
bleed problem (in its harmful, subject-contamination form) is not decodable-fixable
with cheap logit penalties on the corpora we can commit; the remaining levers are
retrieval-side (don't inject a cross-subject line) or a re-anchor-to-question
actuation that must be tested against genuine subject contamination first.
