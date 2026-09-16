# DS-024 — Gate 2.2 AUROC measurement results

> MEASUREMENT REPORT. This document reports measured values only.
> No threshold is created or modified, no gate script is touched,
> and no green/red verdict is offered. Ratification belongs to
> human review and humans.

## Gate context (quoted from the task greenlight line)

> "AUROC >= 0.92 vs a degenerate corpus, SEED=42, engagement evidence,
> STOP-DON'T-NEGOTIATE."

This report MEASURES and REPORTS. It does not render the verdict.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| transformers | 5.14.1 |
| numpy | 2.4.6 |
| SEED | 42 |
| Control selection seed | 42 |
| Analysis window | last 24 token positions |
| Layers | 0 (embeddings datum) .. 28 |
| bmm Triton override | deregistered |
| Wall clock (s) | 126.0 |

Corpus sha256 (qwen_degenerate):

```text
2e0f21a02fc05308c00ac64a18f35e18c9f580839abe885716c81d6a39f991bf
```

## Determinism smoke

Two qwen_degenerate records were replayed twice (teacher-forced, 
single forward pass each). The full metric tuple (29 layers x 3 
per-layer signals + 2 text-level signals) must be identical across 
the two runs.

| record_id | n_tokens | identical |
|---|---|---|
| 0 | 168 | True |
| 1 | 168 | True |
|  | ALL | True |

## Numerical notes

- All three per-layer SVD signals force float32 internally (`core/metrics.py`). SVD is computed on-device for the 24x1536 analysis window (24*1536 < 1,000,000).
- `singular_value_spectrum_entropy(normalize=True)` divides by `log(rank + eps)` with `eps=1e-8`; for a rank-1 analysis window (e.g. the layer-0 embeddings of a fully repeated 24-token tail) `rank + eps` rounds to `1.0` in float32, so the normalized entropy is `inf`. Non-finite counts are reported in every distribution table (`n_non_finite` column); rank-based AUROC handles `inf` correctly (top rank for `+inf`).

## Primary contrast (qwen_degenerate vs prose)

### Per-layer signal AUROC (degenerate = positive class)

#### PR — best layer: 0 (AUC 0.0000, degenerate-lower, sep 1.0000)

| layer | AUC | direction | separation |
|---|---|---|---|
| 0 | 0.0000 | degenerate-lower | 1.0000 |
| 1 | 0.0000 | degenerate-lower | 1.0000 |
| 2 | 0.0000 | degenerate-lower | 1.0000 |
| 3 | 0.0000 | degenerate-lower | 1.0000 |
| 4 | 0.0000 | degenerate-lower | 1.0000 |
| 5 | 0.0000 | degenerate-lower | 1.0000 |
| 6 | 0.0000 | degenerate-lower | 1.0000 |
| 7 | 0.0000 | degenerate-lower | 1.0000 |
| 8 | 0.0000 | degenerate-lower | 1.0000 |
| 9 | 0.0000 | degenerate-lower | 1.0000 |
| 10 | 0.0000 | degenerate-lower | 1.0000 |
| 11 | 0.0000 | degenerate-lower | 1.0000 |
| 12 | 0.0000 | degenerate-lower | 1.0000 |
| 13 | 0.0000 | degenerate-lower | 1.0000 |
| 14 | 0.0000 | degenerate-lower | 1.0000 |
| 15 | 0.0000 | degenerate-lower | 1.0000 |
| 16 | 0.0000 | degenerate-lower | 1.0000 |
| 17 | 0.0000 | degenerate-lower | 1.0000 |
| 18 | 0.0000 | degenerate-lower | 1.0000 |
| 19 | 0.0000 | degenerate-lower | 1.0000 |
| 20 | 0.0000 | degenerate-lower | 1.0000 |
| 21 | 0.0000 | degenerate-lower | 1.0000 |
| 22 | 0.0000 | degenerate-lower | 1.0000 |
| 23 | 0.0000 | degenerate-lower | 1.0000 |
| 24 | 0.0000 | degenerate-lower | 1.0000 |
| 25 | 0.0000 | degenerate-lower | 1.0000 |
| 26 | 0.0000 | degenerate-lower | 1.0000 |
| 27 | 0.0000 | degenerate-lower | 1.0000 |
| 28 | 0.0000 | degenerate-lower | 1.0000 |

Best-layer distributions (PR, layer 0):

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 1.9453 | 1.9855 | 1.0000 | 2.9838 | 0 |
| prose | 18.5760 | 18.9806 | 16.1567 | 20.7872 | 0 |

#### SVSE — best layer: 1 (AUC 0.0000, degenerate-lower, sep 1.0000)

| layer | AUC | direction | separation |
|---|---|---|---|
| 0 | 0.5200 | degenerate-higher | 0.5200 |
| 1 | 0.0000 | degenerate-lower | 1.0000 |
| 2 | 0.0000 | degenerate-lower | 1.0000 |
| 3 | 0.0000 | degenerate-lower | 1.0000 |
| 4 | 0.0000 | degenerate-lower | 1.0000 |
| 5 | 0.0000 | degenerate-lower | 1.0000 |
| 6 | 0.0000 | degenerate-lower | 1.0000 |
| 7 | 0.0000 | degenerate-lower | 1.0000 |
| 8 | 0.0000 | degenerate-lower | 1.0000 |
| 9 | 0.0000 | degenerate-lower | 1.0000 |
| 10 | 0.0000 | degenerate-lower | 1.0000 |
| 11 | 0.0000 | degenerate-lower | 1.0000 |
| 12 | 0.0000 | degenerate-lower | 1.0000 |
| 13 | 0.0000 | degenerate-lower | 1.0000 |
| 14 | 0.0000 | degenerate-lower | 1.0000 |
| 15 | 0.0000 | degenerate-lower | 1.0000 |
| 16 | 0.0000 | degenerate-lower | 1.0000 |
| 17 | 0.0000 | degenerate-lower | 1.0000 |
| 18 | 0.0000 | degenerate-lower | 1.0000 |
| 19 | 0.0000 | degenerate-lower | 1.0000 |
| 20 | 0.0000 | degenerate-lower | 1.0000 |
| 21 | 0.0000 | degenerate-lower | 1.0000 |
| 22 | 0.0000 | degenerate-lower | 1.0000 |
| 23 | 0.0000 | degenerate-lower | 1.0000 |
| 24 | 0.0000 | degenerate-lower | 1.0000 |
| 25 | 0.0000 | degenerate-lower | 1.0000 |
| 26 | 0.0000 | degenerate-lower | 1.0000 |
| 27 | 0.0000 | degenerate-lower | 1.0000 |
| 28 | 0.0000 | degenerate-lower | 1.0000 |

Best-layer distributions (SVSE, layer 1):

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 0.2485 | 0.2554 | 0.0661 | 0.3730 | 0 |
| prose | 0.9635 | 0.9684 | 0.9468 | 0.9764 | 0 |

#### ER — best layer: 2 (AUC 0.0000, degenerate-lower, sep 1.0000)

| layer | AUC | direction | separation |
|---|---|---|---|
| 0 | 0.0200 | degenerate-lower | 0.9800 |
| 1 | 0.0027 | degenerate-lower | 0.9973 |
| 2 | 0.0000 | degenerate-lower | 1.0000 |
| 3 | 0.0000 | degenerate-lower | 1.0000 |
| 4 | 0.0000 | degenerate-lower | 1.0000 |
| 5 | 0.0000 | degenerate-lower | 1.0000 |
| 6 | 0.0000 | degenerate-lower | 1.0000 |
| 7 | 0.0000 | degenerate-lower | 1.0000 |
| 8 | 0.0000 | degenerate-lower | 1.0000 |
| 9 | 0.0000 | degenerate-lower | 1.0000 |
| 10 | 0.0000 | degenerate-lower | 1.0000 |
| 11 | 0.0000 | degenerate-lower | 1.0000 |
| 12 | 0.0000 | degenerate-lower | 1.0000 |
| 13 | 0.0000 | degenerate-lower | 1.0000 |
| 14 | 0.0000 | degenerate-lower | 1.0000 |
| 15 | 0.0000 | degenerate-lower | 1.0000 |
| 16 | 0.0000 | degenerate-lower | 1.0000 |
| 17 | 0.0000 | degenerate-lower | 1.0000 |
| 18 | 0.0000 | degenerate-lower | 1.0000 |
| 19 | 0.0000 | degenerate-lower | 1.0000 |
| 20 | 0.0000 | degenerate-lower | 1.0000 |
| 21 | 0.0000 | degenerate-lower | 1.0000 |
| 22 | 0.0000 | degenerate-lower | 1.0000 |
| 23 | 0.0000 | degenerate-lower | 1.0000 |
| 24 | 0.0000 | degenerate-lower | 1.0000 |
| 25 | 0.0000 | degenerate-lower | 1.0000 |
| 26 | 0.0000 | degenerate-lower | 1.0000 |
| 27 | 0.0000 | degenerate-lower | 1.0000 |
| 28 | 0.0000 | degenerate-lower | 1.0000 |

Best-layer distributions (ER, layer 2):

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 2.2800 | 2.0000 | 1.0000 | 3.0000 | 0 |
| prose | 21.6200 | 22.0000 | 20.0000 | 23.0000 | 0 |

### Text-level signal AUROC

| signal | AUC | direction | separation |
|---|---|---|---|
| ctr | 0.8618 | degenerate-higher | 0.8618 |
| distinct_2 | 0.0000 | degenerate-lower | 1.0000 |

Text-level distributions:

#### ctr

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 0.9584 | 1.0000 | 0.7500 | 1.0000 | 0 |
| prose | 0.8739 | 0.8846 | 0.8006 | 0.9235 | 0 |

#### distinct_2

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 0.1296 | 0.1304 | 0.0870 | 0.1739 | 0 |
| prose | 0.9813 | 1.0000 | 0.9565 | 1.0000 | 0 |

## Secondary contrast (t2s_degenerate vs prose)

### Per-layer signal AUROC (degenerate = positive class)

#### PR — best layer: 0 (AUC 0.0000, degenerate-lower, sep 1.0000)

| layer | AUC | direction | separation |
|---|---|---|---|
| 0 | 0.0000 | degenerate-lower | 1.0000 |
| 1 | 0.0026 | degenerate-lower | 0.9974 |
| 2 | 0.0000 | degenerate-lower | 1.0000 |
| 3 | 0.0000 | degenerate-lower | 1.0000 |
| 4 | 0.0000 | degenerate-lower | 1.0000 |
| 5 | 0.0000 | degenerate-lower | 1.0000 |
| 6 | 0.0000 | degenerate-lower | 1.0000 |
| 7 | 0.0000 | degenerate-lower | 1.0000 |
| 8 | 0.0000 | degenerate-lower | 1.0000 |
| 9 | 0.0000 | degenerate-lower | 1.0000 |
| 10 | 0.0000 | degenerate-lower | 1.0000 |
| 11 | 0.0000 | degenerate-lower | 1.0000 |
| 12 | 0.0000 | degenerate-lower | 1.0000 |
| 13 | 0.0000 | degenerate-lower | 1.0000 |
| 14 | 0.0000 | degenerate-lower | 1.0000 |
| 15 | 0.0000 | degenerate-lower | 1.0000 |
| 16 | 0.0000 | degenerate-lower | 1.0000 |
| 17 | 0.0000 | degenerate-lower | 1.0000 |
| 18 | 0.0000 | degenerate-lower | 1.0000 |
| 19 | 0.0000 | degenerate-lower | 1.0000 |
| 20 | 0.0000 | degenerate-lower | 1.0000 |
| 21 | 0.0000 | degenerate-lower | 1.0000 |
| 22 | 0.0000 | degenerate-lower | 1.0000 |
| 23 | 0.0000 | degenerate-lower | 1.0000 |
| 24 | 0.0000 | degenerate-lower | 1.0000 |
| 25 | 0.0000 | degenerate-lower | 1.0000 |
| 26 | 0.0000 | degenerate-lower | 1.0000 |
| 27 | 0.0000 | degenerate-lower | 1.0000 |
| 28 | 0.0005 | degenerate-lower | 0.9995 |

Best-layer distributions (PR, layer 0):

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 3.9742 | 3.9651 | 1.0000 | 6.8808 | 0 |
| prose | 18.5760 | 18.9806 | 16.1567 | 20.7872 | 0 |

#### SVSE — best layer: 2 (AUC 0.0000, degenerate-lower, sep 1.0000)

| layer | AUC | direction | separation |
|---|---|---|---|
| 0 | 0.2519 | degenerate-lower | 0.7481 |
| 1 | 0.0039 | degenerate-lower | 0.9961 |
| 2 | 0.0000 | degenerate-lower | 1.0000 |
| 3 | 0.0000 | degenerate-lower | 1.0000 |
| 4 | 0.0000 | degenerate-lower | 1.0000 |
| 5 | 0.0000 | degenerate-lower | 1.0000 |
| 6 | 0.0000 | degenerate-lower | 1.0000 |
| 7 | 0.0000 | degenerate-lower | 1.0000 |
| 8 | 0.0000 | degenerate-lower | 1.0000 |
| 9 | 0.0000 | degenerate-lower | 1.0000 |
| 10 | 0.0000 | degenerate-lower | 1.0000 |
| 11 | 0.0000 | degenerate-lower | 1.0000 |
| 12 | 0.0000 | degenerate-lower | 1.0000 |
| 13 | 0.0000 | degenerate-lower | 1.0000 |
| 14 | 0.0000 | degenerate-lower | 1.0000 |
| 15 | 0.0000 | degenerate-lower | 1.0000 |
| 16 | 0.0000 | degenerate-lower | 1.0000 |
| 17 | 0.0000 | degenerate-lower | 1.0000 |
| 18 | 0.0000 | degenerate-lower | 1.0000 |
| 19 | 0.0000 | degenerate-lower | 1.0000 |
| 20 | 0.0000 | degenerate-lower | 1.0000 |
| 21 | 0.0000 | degenerate-lower | 1.0000 |
| 22 | 0.0000 | degenerate-lower | 1.0000 |
| 23 | 0.0000 | degenerate-lower | 1.0000 |
| 24 | 0.0000 | degenerate-lower | 1.0000 |
| 25 | 0.0000 | degenerate-lower | 1.0000 |
| 26 | 0.0000 | degenerate-lower | 1.0000 |
| 27 | 0.0000 | degenerate-lower | 1.0000 |
| 28 | 0.0001 | degenerate-lower | 0.9999 |

Best-layer distributions (SVSE, layer 2):

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 0.4382 | 0.4680 | 0.1992 | 0.6286 | 0 |
| prose | 0.9643 | 0.9678 | 0.9479 | 0.9747 | 0 |

#### ER — best layer: 2 (AUC 0.0000, degenerate-lower, sep 1.0000)

| layer | AUC | direction | separation |
|---|---|---|---|
| 0 | 0.0100 | degenerate-lower | 0.9900 |
| 1 | 0.0063 | degenerate-lower | 0.9937 |
| 2 | 0.0000 | degenerate-lower | 1.0000 |
| 3 | 0.0000 | degenerate-lower | 1.0000 |
| 4 | 0.0000 | degenerate-lower | 1.0000 |
| 5 | 0.0000 | degenerate-lower | 1.0000 |
| 6 | 0.0000 | degenerate-lower | 1.0000 |
| 7 | 0.0000 | degenerate-lower | 1.0000 |
| 8 | 0.0000 | degenerate-lower | 1.0000 |
| 9 | 0.0000 | degenerate-lower | 1.0000 |
| 10 | 0.0000 | degenerate-lower | 1.0000 |
| 11 | 0.0000 | degenerate-lower | 1.0000 |
| 12 | 0.0000 | degenerate-lower | 1.0000 |
| 13 | 0.0000 | degenerate-lower | 1.0000 |
| 14 | 0.0000 | degenerate-lower | 1.0000 |
| 15 | 0.0000 | degenerate-lower | 1.0000 |
| 16 | 0.0000 | degenerate-lower | 1.0000 |
| 17 | 0.0000 | degenerate-lower | 1.0000 |
| 18 | 0.0000 | degenerate-lower | 1.0000 |
| 19 | 0.0000 | degenerate-lower | 1.0000 |
| 20 | 0.0000 | degenerate-lower | 1.0000 |
| 21 | 0.0000 | degenerate-lower | 1.0000 |
| 22 | 0.0000 | degenerate-lower | 1.0000 |
| 23 | 0.0000 | degenerate-lower | 1.0000 |
| 24 | 0.0000 | degenerate-lower | 1.0000 |
| 25 | 0.0000 | degenerate-lower | 1.0000 |
| 26 | 0.0000 | degenerate-lower | 1.0000 |
| 27 | 0.0000 | degenerate-lower | 1.0000 |
| 28 | 0.0000 | degenerate-lower | 1.0000 |

Best-layer distributions (ER, layer 2):

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 4.1500 | 4.0000 | 1.9000 | 7.0000 | 0 |
| prose | 21.6200 | 22.0000 | 20.0000 | 23.0000 | 0 |

### Text-level signal AUROC

| signal | AUC | direction | separation |
|---|---|---|---|
| ctr | 0.8438 | degenerate-higher | 0.8438 |
| distinct_2 | 0.0000 | degenerate-lower | 1.0000 |

Text-level distributions:

#### ctr

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 0.9640 | 1.0000 | 0.8543 | 1.0000 | 0 |
| prose | 0.8739 | 0.8846 | 0.8006 | 0.9235 | 0 |

#### distinct_2

| class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| degenerate | 0.2213 | 0.2174 | 0.0870 | 0.3478 | 0 |
| prose | 0.9813 | 1.0000 | 0.9565 | 1.0000 | 0 |

## Engagement-evidence arm

Governor: `ActiveVarietyGovernor(model, tokenizer=tokenizer, use_code_filter=True)`, calibrated on `'The quick brown fox jumps over the lazy dog.'`.

20 seeded qwen_degenerate prompts (seed 42), decoding per each corpus record's own params, `max_new_tokens=128`.

Selected record ids: `[3, 4, 11, 13, 14, 17, 27, 28, 29, 31, 35, 54, 64, 69, 71, 75, 77, 81, 86, 94]`

### Per-prompt engagement results

| id | decisions | injections | intervened | mean_delta | max_delta | layers |
|---|---|---|---|---|---|---|
| 3 | 1 | 126 | True | 0.4669 | 0.4902 | [26] |
| 4 | 1 | 126 | True | 0.4650 | 0.4839 | [26] |
| 11 | 1 | 25 | True | 0.4652 | 0.4772 | [26] |
| 13 | 1 | 126 | True | 0.4666 | 0.4866 | [26] |
| 14 | 1 | 126 | True | 0.4666 | 0.4866 | [26] |
| 17 | 1 | 126 | True | 0.4665 | 0.4832 | [26] |
| 27 | 1 | 126 | True | 0.4641 | 0.4860 | [26] |
| 28 | 1 | 126 | True | 0.4641 | 0.4860 | [26] |
| 29 | 1 | 126 | True | 0.4641 | 0.4860 | [26] |
| 31 | 1 | 126 | True | 0.4640 | 0.4807 | [26] |
| 35 | 1 | 126 | True | 0.4646 | 0.4792 | [26] |
| 54 | 1 | 126 | True | 0.4662 | 0.4823 | [26] |
| 64 | 1 | 126 | True | 0.4628 | 0.4782 | [26] |
| 69 | 1 | 126 | True | 0.4649 | 0.4803 | [26] |
| 71 | 1 | 126 | True | 0.4649 | 0.4803 | [26] |
| 75 | 0 | 0 | False | 0.0000 | 0.0000 | [] |
| 77 | 0 | 0 | False | 0.0000 | 0.0000 | [] |
| 81 | 7 | 49 | True | 0.5132 | 0.7491 | [26] |
| 86 | 0 | 0 | False | 0.0000 | 0.0000 | [] |
| 94 | 1 | 126 | True | 0.4654 | 0.4769 | [26] |

### Word-loop positive-control reference

| Field | Value |
|---|---|
| Prompt | `word word word word word word word word word word` |
| temperature / top_p / max_new_tokens | 0.7 / 0.9 / 64 |
| decisions | 0 |
| injections | 0 |
| intervened | False |
| mean_delta | 0.0000 |
| max_delta | 0.0000 |
| layers | [] |

### Engagement-arm observations (strictly factual)

- On 20 seeded qwen_degenerate prompts, the governor fired on 17/20 prompts, logging 1964 total residual injections (mean 98.2 per prompt, max 126).
- Prompt-level injection counts ranged from 0 to 126.
- The word-loop positive-control probe logged 0 injection(s) across 64 generated tokens (mean delta 0.0000 L2).
- qwen_degenerate vs prose: pr best AUC 0.0000 at layer 0 (degenerate-lower); svse best AUC 0.0000 at layer 1 (degenerate-lower); er best AUC 0.0000 at layer 2 (degenerate-lower); ctr AUC 0.8618 (degenerate-higher); distinct_2 AUC 0.0000 (degenerate-lower).
- t2s_degenerate vs prose: pr best AUC 0.0000 at layer 0 (degenerate-lower); svse best AUC 0.0000 at layer 2 (degenerate-lower); er best AUC 0.0000 at layer 2 (degenerate-lower); ctr AUC 0.8438 (degenerate-higher); distinct_2 AUC 0.0000 (degenerate-lower).

## Data

- Per-record per-layer signal values: `docs/gate22/gate22_results.jsonl`
