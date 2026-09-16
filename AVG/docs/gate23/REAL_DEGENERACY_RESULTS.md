# Real-generation degeneracy validation (Qwen2.5-1.5B)

**Date:** 2026-08-13
**Script:** `scripts/measure_real_degeneracy.py`
**Question:** every prior rescue was measured on *synthetic* word-loops. Does the governor help
*natural* degeneracy — the gradual repetition/rambling a small model drifts into on long
continuations of real prose?

**Method:** 40 real prose prompts (`data/t2s_bench/valid_subset_200.jsonl`), 256-token
continuations under sampling (temp 0.8), RAW (`intervene=False`) vs GOVERNED (`intervene=True`).
Measures tail/full distinct-2, distinct-3, tokens-before-degeneration, teacher-forced perplexity,
and collapse-fire counts.

## Results

| metric | raw | governed | Δ |
|---|---|---|---|
| natural degeneration rate (raw tail d2 < 0.5) | 3/40 (8%) | — | — |
| rescue on degenerate subset | — | **3/3 (100%)** | — |
| tail distinct-2 | 0.908 | 0.980 | **+0.073** |
| full distinct-2 | 0.717 | 0.762 | **+0.045** |
| distinct-3 | 0.817 | 0.851 | +0.034 |
| tokens-before-degeneration | 232.1 | 256.0 | +23.9 |
| **perplexity (teacher-forced)** | 4.51 | 3.97 | **−0.54** |
| collapse fires (bigram + spectral) | 0.00 mean | 0/40 engaged | — |

## Findings

1. **Natural degeneration is rare at this length.** Only 3/40 (8%) real-prose prompts
   degenerate (tail d2 < 0.5) within 256 tokens on the 1.5B production model. The model is
   mostly fine; the governor's rescue opportunity is a tail case.
2. **When it does degenerate, the governor rescues 100% (3/3).** And it is **suppression-only** —
   `fires=0` everywhere, because natural repetition is *softer* than a synthetic word-loop
   (the bigram/spectral collapse detectors require a hard d2<0.30 / PR<band_low collapse that
   natural rambling doesn't reach). The unconditional token suppression is what catches it,
   consistent with the cross-model finding that suppression is the universal actuation.
3. **Diversity rises *for free* — no perplexity cost.** Governed text is more diverse
   (full d2 +0.045, tail d2 +0.073) *and* slightly *lower* perplexity (−0.54, within noise but
   skewing lower, ~60% of prompts). The suppression nudges away from common-word repetition
   without making the text less likely under the model — i.e., it's variety, not distortion.

## Caveats / follow-ups

- 256 tokens is a short horizon; 1.5B rarely degenerates that early. A stronger signal needs
  longer generations (512–1024) or a smaller model (0.5B degenerates sooner). The headline here
  is directional: the governor *does* rescue natural degeneracy when it occurs, and its
  diversity increase is quality-neutral.
- Perplexity is teacher-forced self-perplexity (the model judging its own continuations) — a
  necessary-but-not-sufficient quality check. A downstream coherence/readability metric or human
  read remains a follow-up.

## Artifacts

- `scripts/measure_real_degeneracy.py` (reusable).
- `docs/gate23/qwen2.5-1.5b_real_degeneracy.jsonl` (per-prompt rows).
