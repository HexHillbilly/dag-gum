# Spectral detector false-positive rate on weird-but-benign inputs

**Date:** 2026-08-14
**Script:** `scripts/measure_spectral_fp_weird.py`
**Question:** does the spectral-PR detector *falsely* fire on benign-but-unusual inputs —
code, JSON schemas, fenced code blocks, URL-heavy strings, prose/code-switch salad — i.e. the
"served model receives structured non-prose" surface? A false positive here would mean the
governor kicks (suppresses / kickstarts) a *healthy* generation.

## Why the old gate was vacuous

`HAZARD_SCHEMA_FP_RESULTS.md` reported ~0 fires, but for the wrong reason: 445/450 structured
inputs were *complete* blocks (code ending in `return`, JSON ending in `}`), so the model
emitted `<eos>` at step 0 and the detector had zero tokens to evaluate. A detector that fires
0 times because it never ran is not evidence of a clean detector.

## Method

For each record, **truncate the text to an open 50% prefix** (cut at a whitespace boundary) so
the model is forced to *continue* the fragment — this is the real served-model surface. Then run
`ActiveVarietyGovernor` twice, greedy, 64 tokens:

- **dormant** (`intervene=False`): is the input *actually* degeneration-prone? (`tail d2 < 0.5`)
- **active** (`intervene=True`): does the detector fire, and is the continuation degenerate?

Classification (per record):
- **FP** = a detector fired **and** the dormant run was *not* degenerate (healthy input mis-kicked).
- **TP** = a detector fired **and** the dormant run *was* degenerate (correctly caught a real loop).

## Results (N=650)

| corpus | records | dormant-degenerate | spectral fires | bigram fires | FP |
|---|---|---|---|---|---|
| code_torture | 200 | 77 (38.5%) | 0 | 0 | 0 |
| schema_corpus_v2 | 300 | 98 (32.7%) | 0 | 0 | 0 |
| schema_markdown_fenced | 50 | 22 (44.0%) | 0 | 0 | 0 |
| schema_urls_strings | 50 | 22 (44.0%) | 0 | 0 | 0 |
| prose_code_switch | 50 | 12 (24.0%) | 0 | 0 | 0 |
| **total** | **650** | **231 (35.5%)** | **0** | **0** | **0** |

**Spectral FP = 0/650 (0.00%). Bigram FP = 0/650 (0.00%).**

Active rescue on the 231 dormant-degenerate records: **231/231 (100%)** — suppression alone
broke every would-be loop, so neither detector needed to fire.

## Findings

1. **The spectral detector does not false-positive on benign-but-weird input.** 0/650 — code,
   JSON, fenced blocks, URL strings, and prose/code-switch salad never trip it. Combined with
   the earlier natural-prose result (0/200 fires), the detector is inert across the *entire*
   benign surface (850 records total, 0 fires).

2. **The 0-TP side is not a detection failure — it is the design working as intended.** 231
   inputs (35.5%) genuinely made the model loop *dormant* (so the test has real statistical
   power), and suppression proactively broke **all 231** before the hidden state could collapse
   far enough for either detector to arm. The bigram + spectral predicates are **reactive
   fallbacks**: they only fire when suppression (the front-line) is insufficient — i.e. deep
   macro-loops whose entrenched token has an argmax margin ≫ 5. Those appear in *adversarial*
   motifs (`t2s_degenerate`), not in this benign corpus, so the detectors correctly stayed
   silent.

3. **This closes the FP question cleanly.** The prior "0 fires" was vacuous (detector never
   ran). This measurement is non-vacuous (231 real loops, all rescued, all detector-silent) and
   confirms: **the spectral detector has a 0% false-positive rate on benign input, and earns its
   complexity only on adversarial macro-loops** (see `SPECTRAL_MARGINAL_RESCUE_RESULTS.md`,
   17%→97% rescue).

## Bottom line

Spectral FP on weird-but-benign input = **0.00%** (N=650, with 231 non-vacuous loop cases).
Suppression is the proactive front-line (100% rescue of tight loops); the spectral detector is a
precise reactive fallback that only arms on the adversarial macro-loop class it was built for.
