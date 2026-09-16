# Real-generation degeneracy — greedy regime (the prod repetition regime)

**Date:** 2026-08-14
**Script:** `scripts/measure_real_degeneracy.py --greedy`
**Why this run:** the sampling-regime validation (`REAL_DEGENERACY_SAMPLING_REGIME.md`) found
natural degeneration rare (5–8%) — but that was temp-0.8 sampling, the *diversity-friendly*
regime. Prod repetition ("I hit repetitions many times in prod") happens under **greedy**
(or low-temp) decoding, the hard-collapse regime. This run measures the governor in that regime.

## Results (40 real-prose prompts, greedy, 512 tokens)

| metric | 1.5B | 0.5B |
|---|---|---|
| natural degen (tail d2 < 0.5) | 5/40 (12%) | 2/40 (5%) |
| rescue on degenerate subset | 5/5 (100%) | 2/2 (100%) |
| tail distinct-2 | 0.783 → 0.971 (**+0.188**) | 0.841 → 0.961 (+0.120) |
| full distinct-2 | 0.229 → 0.290 (**+0.061**) | 0.226 → 0.255 (+0.029) |
| distinct-3 | 0.272 → 0.339 (+0.067) | 0.284 → 0.297 (+0.013) |
| tokens-before-degen | 455.5 → 512.0 | 479.1 → 512.0 |
| perplexity (teacher-forced) | 1.19 → 1.38 (+0.19) | 1.16 → 1.36 (+0.20) |
| collapse fires (bigram / spectral) | 0/40 / 0/40 | 0/40 / 0/40 |

## Findings

1. **Greedy is the repetition regime.** Full-sequence distinct-2 collapses to **0.23** (vs
   ~0.61 under sampling), with individual prompts hitting **0.002** — near-total repetition.
   This is the prod failure mode; the sampling-regime "8% rare" was measuring the wrong regime.

2. **The governor substantially helps under greedy.** Rescue 100%, and diversity rises across
   the board — full d2 +0.061, tail d2 +0.188, d3 +0.067 (1.5B). A much bigger effect than
   sampling (where full d2 was flat). The governor's real-world value is *here*, not in the
   sampling regime.

3. **Small perplexity cost (+0.19) — the price of breaking repetition, not degrading quality.**
   Raw greedy ppl (1.19) is *artificially* low: confident repetition. The governor breaks it,
   raising ppl to 1.38 (still very low). Under sampling ppl *lowered*; under greedy it rises
   slightly because repetition is "cheap confidence." Teacher-forced self-ppl is a weak quality
   signal here precisely because the model scores its own repetition as likely.

4. **The collapse detector STILL never fires — even under greedy with full d2 0.23.** Natural
   greedy repetition is *coherent* (repeating phrases while staying grammatical), not the
   incoherent token-loop (`d2<0.30 AND ctr<0.30`) or low-dimensional hidden-state collapse
   (`PR<band_low`) the dual-predicate detector targets. The unconditional token suppression is
   the **entire** real-world rescue mechanism; the spectral/bigram detector is a
   synthetic-fixture-specific tool.

## Bottom line

The governor's real-world value is in the **greedy regime**: it substantially breaks the
coherent repetition that dominates prod decoding, at a small perplexity cost — and it does so
with **unconditional suppression alone**. The collapse detector never fires on real prose
across 200 prompt-runs and 5 model×regime×horizon configs (256/512 tokens, sampling/greedy,
1.5B/0.5B).

**Caveat (macro-loops):** "suppression-only" holds for *natural* prose, not for adversarial
inputs. On crafted macro-loop motifs the spectral detector is load-bearing — it fires the
kickstart that breaks deep loops suppression can't (17%→97% rescue;
`SPECTRAL_MARGINAL_RESCUE_RESULTS.md`). The detector earns its complexity as macro-loop defense
for a served model, even though natural text never reaches it.
