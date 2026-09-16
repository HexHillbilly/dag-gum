# Real-generation degeneracy — sampling regime, extended horizon

**Date:** 2026-08-14
**Script:** `scripts/measure_real_degeneracy.py` (extended with `--band-low`, horizon-tagged output)
**Question:** does the governor's help on *natural* degeneration hold under sampling at longer
horizons and across model scale? The 256-token 1.5B run (`REAL_DEGENERACY_RESULTS.md`) found
natural degeneration rare (8%) and the rescue suppression-only. Two follow-ups were flagged:
longer horizons (512–1024) and a smaller model (0.5B). This run tests both.

## Method

Same 40 real-prose prompts (`data/t2s_bench/valid_subset_200.jsonl`), sampling (temp 0.8), RAW
vs GOVERNED, at 512 tokens — on Qwen2.5-1.5B (production) and Qwen2.5-0.5B (smaller, with
`--band-low 7.51`). Metrics: tail/full distinct-2, distinct-3, tokens-before-degeneration,
teacher-forced perplexity, collapse-fire counts.

## Results

| metric | 1.5B @ 256 (baseline) | 1.5B @ 512 | 0.5B @ 512 |
|---|---|---|---|
| natural degen (raw tail d2 < 0.5) | 3/40 (8%) | 3/40 (8%) | 2/40 (5%) |
| rescue on degenerate subset | 3/3 (100%) | 3/3 (100%) | 2/2 (100%) |
| tail distinct-2 Δ | +0.073 | +0.079 | +0.057 |
| full distinct-2 Δ | +0.045 | +0.001 | −0.031 |
| distinct-3 Δ | +0.034 | −0.012 | −0.051 |
| tokens-before-degen Δ | +23.9 | +67.4 | +23.7 |
| **perplexity Δ (lower = better)** | −0.54 | −0.44 | −0.74 |
| collapse fires | 0/40 | 0/40 | 0/40 |

## Findings

1. **Natural degeneration is rare at every scale and horizon tested.** 8% at both 256 and 512
   tokens (1.5B); 5% at 512 (0.5B). The expected "longer horizon → more degeneration" did NOT
   materialize on real prose, and the "smaller model → more degeneration" expectation is
   *reversed*: 0.5B is slightly *more* diverse on real prose (raw tail d2 0.919 vs 0.895).
   The "0.5B degenerates sooner" observation was specific to synthetic word-loop fixtures, not
   real prose.

2. **The governor rescues 100% of natural degeneration at both scales, suppression-only**
   (collapse detector fires 0/40 everywhere). Natural degeneration is *soft* (tail d2 0.3–0.5)
   and never hardens to the d2<0.30 / PR<band_low collapse the detector targets; unconditional
   token suppression catches it first.

3. **The diversity gain is tail-concentrated, not full-sequence — the 256-token "+0.045 full
   d2" was a short-horizon artifact.** At 512 tokens, full-sequence distinct-2 is flat (1.5B
   +0.001) or slightly negative (0.5B −0.031), and distinct-3 is slightly negative on both.
   The robust, real effect is the *tail*: the governor breaks imminent repetition (tail d2
   +0.057 to +0.079) without changing the bulk of the healthy text.

4. **No perplexity cost — actually lower (better) at every scale and horizon.** −0.44 to −0.74,
   lower on 24/40 to 29/40 prompts. Suppression nudges away from common-word repetition without
   making the text less likely under the model.

## Bottom line (refined)

The governor's real-generation benefit is: (a) **100% rescue** of the rare natural-degeneration
tail cases, (b) **tail diversity** (breaking imminent loops), with (c) **no perplexity cost**.
The full-sequence diversity boost reported at 256 tokens does not persist at 512 — it was a
short-horizon artifact. Natural degeneration on real prose is rare (~5–8%) across scale and
horizon, so the governor's real-world value is a safety net on the tail, not a global
diversity enhancer.

## Artifacts

- `docs/gate23/qwen2.5-1.5b_real_degeneracy_512.jsonl`
- `docs/gate23/qwen2.5-0.5b_real_degeneracy_512.jsonl`
