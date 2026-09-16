# Topic-Drift Detector — Stage 1 (signal measurement)

**Question:** is there a cheap decode-time signal that separates topic-bleed from
on-topic generation, before we touch `controller.py`?

**Context.** Phase 3b: with persona injection, the retrieved line's *topic* bleeds
into the answer. Track 2 showed AVG's logit penalties do NOT fix this (coherent
drift, not a loop), and the prompt-side topic-fidelity directive also failed
(kills voice, doesn't fix topic). A decode-time **topic-drift detector** is the
remaining lever. Stage 1 proves the signal exists; Stage 2 builds the detector.

## Signal under test

For a trailing window of `W=24` generated tokens, at each step `t`:

    cos_q   = cosine(embed(trailing), embed(question))
    cos_r   = cosine(embed(trailing), embed(retrieved_line))
    drift   = cos_q - cos_r          # >0 anchored to question, ~0/- drifting

Measured **post-hoc** from the generated token ids (no controller change).

## Method

`scripts/measure_topic_drift.py` — Phi-3-mini-4k-instruct, sampling
(temperature 0.8, top_p 0.9, 3 samples), `wrap_persona` v1, 5 probes (4 far +
1 on-topic control), 3 corpora:

| corpus | why | expected bleed |
|---|---|---|
| `huckleberry_finn.txt` | broad, conversational | modest |
| `walden.txt` | single-author, formal prose | modest |
| `farmer_cookbook.txt` | narrow single-topic | strong (predicted) |

## Result (sampling, mean)

| corpus | topic_cos base→persona | drift base→persona |
|---|---|---|
| Huck Finn | 0.710 → 0.539 | **0.322 → 0.041** |
| Walden | 0.710 → 0.541 | **0.301 → 0.114** |
| Cookbook | 0.710 → 0.655 | **0.274 → 0.123** |

The drift signal separates baseline from persona on **all three** corpora.

**Early separation (Huck Finn trajectory, mean by step):** baseline drift is
already +0.374 at step 8 and holds +0.24…+0.42 through step 80; persona is −0.012
at step 8 and stays ~0.0. **The gap is fully established by step 8** — early
enough to intervene.

**Both components contribute.** From baseline → persona, `cos_q` drops (0.60 →
0.31, the output leaves the question) AND `cos_r` rises (0.22 → 0.32, it
approaches the retrieved line). The drift `cos_q − cos_r` collapses from ~+0.4 to
~0.0.

## Interpretation

- A trailing-window embedding distance to the **question** (with the retrieved
  line as a reference) is a clean, early, reliable topic-drift signal.
- A candidate detector fires when `drift < θ` (persistence ≥ 2 steps), i.e. the
  output stops being more-question-than-retrieved-line.
- The cookbook's *weakest* bleed (0.655) is counter to the "narrow = strong bleed"
  prior: its retrieval is keyword-matched fragments (`=Heat=`, recipe quantities),
  which transfer less topic than coherent prose lines. Bleed strength is driven by
  the retrieved line's *coherence*, not the corpus's breadth.

## Bug fixed en route

`measure_topic_drift.py` originally double-sliced the generated ids
(`generate()` already strips the prompt, then the caller sliced again by
`prompt_len`), which truncated every output to its last ~30 tokens and produced a
spurious "generation collapse" (0–15 token outputs) that masked the real signal.
Fixed: `generate()` returns generated tokens only; trajectory and output decode
consume them directly. Also fixed `build_book_corpus.py`, whose `__main__` was
silently ignoring `--start-after` (the SCP copy predates the SSP argparse fix).

## Next (Stage 2)

Build the live detector in the governor: recompute the drift score every K=8
steps, fire on `drift < θ` with persistence ≥ 2, and pick an actuation (cheapest
candidates: temperature cooling on drift, or suppressing the retrieved line's
distinctive content tokens). Actuation choice is the decision point — see owner.
