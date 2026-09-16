# Track 2 Results — does AVG governance suppress topic-bleed?

**Date:** 2026-08-15 · **Model:** `microsoft/Phi-3-mini-4k-instruct` ·
**Corpus:** Huckleberry Finn (5,748 sentences, public domain) · **Probes:** 5 ·
**Regimes:** greedy + sampling (temp 0.8, top_p 0.9, 3 seeds) ·
**Tool:** `scripts/benchmark_topic_governed.py`

## Question

Phase 3b found that persona injection bleeds the retrieved line's topic into the
answer (topic_cos drops). Track 2 tests whether the AVG governed backend
(SCP persona pre-gen + AVG mid-gen logit penalties) suppresses that drift.

## Result (mean topic_cos)

| condition | greedy | sampling |
|---|---|---|
| baseline | 0.631 | 0.625 |
| persona | 0.574 | 0.510 |
| persona_governed | 0.547 | 0.501 |

(style_cos / rep_rate, sampling: baseline 0.041 / 0.054 · persona 0.184 / 0.023 ·
persona_governed 0.207 / 0.012.)

## Finding (negative)

AVG governance does **not** restore topic fidelity — `persona_governed` is
slightly *lower* than `persona` in both regimes, not a recovery. This is
consistent with the mechanism: AVG's logit penalties target loops/degeneracy
(repeated-token suppression, spectral rank collapse, non-ASCII spam), not
coherent topic drift.

Side observations (consistent with AVG's actual job): governance slightly lowers
rep_rate (0.023 → 0.012 sampling, the loop-suppression doing its thing) and leaves
style_cos roughly unchanged (0.184 → 0.207) — voice transfer is unaffected.

## Implication

Topic-bleed is a **distinct failure mode from degeneracy**. It needs a different
mechanism than AVG's logit penalties — e.g. a topic-fidelity directive that
decouples voice from retrieved content (the Phase 3b suggestion), or a
topic-drift detector. The "sandwich" integration itself is sound (persona +
governed generation runs cleanly); AVG just isn't the right tool for topic drift.

## Topic-fidelity directive (also negative)

A `wrap_persona_topic` variant adds an explicit topic-boundary clause
("answer only the question's own topic — do not bring in the line's subject
matter"). Result (mean):

| condition | topic_cos greedy | topic_cos sampling | style_cos sampling |
|---|---|---|---|
| persona | 0.574 | 0.510 | 0.184 |
| persona_topic | 0.566 | 0.515 | 0.075 |

The directive does **not** restore topic fidelity (topic_cos barely moves) and
*suppresses* voice transfer (style_cos 0.184 → 0.075, −59%). Prompt-side
decoupling fails: telling the model to ignore the seed line's subject makes it
more generic without fixing the topic bleed. The decode-side topic-drift
detector is the remaining lever.

