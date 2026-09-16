# DS-027 — Gate 2.3b acceptance scoring results (v2)

> MEASUREMENT REPORT. This document reports measured values only.
> No threshold is created or modified, no gate script is touched,
> and no green/red verdict and no layer recommendation are offered.
> Earliest-passing layer selection is the human's act per the greenlight.

## Greenlight acceptance criteria (quoted from gate-2.3b)

> 4. Acceptance (all required): AUROC >= 0.98 on held-out contrasts;
> 0 false positives on prose + hazard + schema corpora (including
> Stage-1 routing measurement on structured-legitimate text); minimum
> recall >= 0.95 on the held-out hard-degenerate slice.

This report MEASURES and REPORTS per layer. It does not render the verdict.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Control selection seed | 42 |
| Analysis window | last 24 token positions |
| Candidate layers | ['2', '12', '26'] |
| bmm Triton override | deregistered |
| Wall clock (s) | 83.9 |

## Held-out slice characterization (Distinct-2)

Distinct-2 is the offline text-level diversity measure (last-24-token
window) used to curate the held-out degenerate slice. It is reported
here as strictly factual slice characterization; it is not a decision
signal in the RFC-004 Stage-2 detector.

| set | mean | median | p10 | p90 |
|---|---|---|---|---|
| heldout_degenerate | 0.1643 | 0.1739 | 0.1304 | 0.2174 |
| heldout-100 prose | 0.9700 | 1.0000 | 0.9087 | 1.0000 |

The full grid/curation profile is in `docs/gate23/HELDOUT_V2_PROFILE.md`.

## Determinism smoke

Two heldout-degenerate records were replayed twice (teacher-forced,
single forward pass each). The per-layer metric tuple (PR/SVSE/v_traj
at layers {2,12,26} + is_code + distinct_2) must be identical across
the two runs.

| record_id | n_tokens | identical |
|---|---|---|
| 0 | 156 | True |
| 1 | 163 | True |
|  | ALL | True |

## Frozen thresholds (restated from Part A freeze)

Thresholds are read from `docs/gate23/FROZEN_THRESHOLDS.md`; nothing
here re-derives them.

| layer | signal | T | band_low | band_high |
|---|---|---|---|---|
| 2 | PR | 10.9548 | 8.2161 | 13.6935 |
| 2 | SVSE | 0.6708 | 0.5031 | 0.8385 |
| 12 | PR | 11.1736 | 8.3802 | 13.9670 |
| 12 | SVSE | 0.7709 | 0.5782 | 0.9637 |
| 26 | PR | 10.7727 | 8.0795 | 13.4658 |
| 26 | SVSE | 0.7801 | 0.5850 | 0.9751 |

## Per-layer acceptance scoring (fusion rule)

Fusion rule (RFC-004 + greenlight parenthetical): PR primary. If PR is
below its band -> fire. If PR is inside its ambiguity band -> fire only
if SVSE agrees (SVSE below T_SVSE) and not both-inside-bands (SVSE below
its own lower band edge). If PR is above its band -> no fire.

| layer | PR AUC | PR dir | PR sep | SVSE AUC | SVSE dir | SVSE sep | recall(deg) | FP prose | FP schema_md | FP schema_urls | FP prose_code | FP schema_v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.0000 | degenerate-lower | 1.0000 | 0.0000 | degenerate-lower | 1.0000 | 1.0000 | 0 | 0 | 0 | 0 | 0 |
| 12 | 0.0000 | degenerate-lower | 1.0000 | 0.0000 | degenerate-lower | 1.0000 | 0.9900 | 0 | 0 | 0 | 0 | 0 |
| 26 | 0.0000 | degenerate-lower | 1.0000 | 0.0000 | degenerate-lower | 1.0000 | 0.9800 | 0 | 0 | 0 | 0 | 0 |

- AUROC direction is with heldout-degenerate as the positive class;
  degenerate-LOWER means AUC < 0.5 (PR/SVSE lower for degenerate).
  Separation = max(AUC, 1-AUC); the `AUROC >= 0.98` criterion is read
  against separation (equivalently |1-AUC| for a degenerate-lower signal).
- recall(deg) = fraction of the 100 heldout-degenerate records that fire.
- FP <set> = number of records in that legitimate set that fire under the
  fusion rule (any fire on a legitimate set is a false positive).

## Stage-1 routing measurement (is_code_syntax_context)

`is_code_syntax_context(text)` is the Stage-1 context router as-is; the
rate is the fraction of records in each set flagged as code/schema
context (routed to Path-2 immunity and therefore not eligible for Stage-2).

| set | n_records | routing_rate |
|---|---|---|
| heldout_degenerate | 100 | 0.0000 |
| heldout_prose | 100 | 0.0000 |
| schema_markdown_fenced | 50 | 1.0000 |
| schema_urls_strings | 50 | 1.0000 |
| prose_code_switch | 50 | 0.0000 |
| schema_corpus_v2 | 300 | 0.0000 |

## Held-out contrast distributions (per layer)

### Layer 2

| signal | class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|---|
| PR | degenerate | 2.9687 | 3.0510 | 2.0816 | 3.9885 | 0 |
| PR | prose | 20.0009 | 20.6013 | 18.1523 | 21.3357 | 0 |
| SVSE | degenerate | 0.3718 | 0.3850 | 0.2701 | 0.4645 | 0 |
| SVSE | prose | 0.9600 | 0.9678 | 0.9403 | 0.9741 | 0 |

### Layer 12

| signal | class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|---|
| PR | degenerate | 3.4223 | 3.3322 | 2.3648 | 4.3968 | 0 |
| PR | prose | 19.4082 | 19.7959 | 18.3145 | 20.2886 | 0 |
| SVSE | degenerate | 0.4615 | 0.4539 | 0.3677 | 0.5380 | 0 |
| SVSE | prose | 0.9576 | 0.9629 | 0.9459 | 0.9668 | 0 |

### Layer 26

| signal | class | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|---|
| PR | degenerate | 3.4967 | 3.3081 | 2.3407 | 4.3279 | 0 |
| PR | prose | 18.4269 | 18.5856 | 17.3945 | 19.2658 | 0 |
| SVSE | degenerate | 0.4682 | 0.4477 | 0.3601 | 0.5593 | 0 |
| SVSE | prose | 0.9495 | 0.9517 | 0.9375 | 0.9586 | 0 |

## v_traj SHADOW log (RFC-004 §7.4)

v_traj is the independent review candidate shadow metric: normalized mean cosine
step-distance between consecutive hidden-state deltas over the last-24
window. It is LOGGED ONLY, decides NOTHING, and is NOT scored against
thresholds.

| layer | mean | median | p10 | p90 | n_non_finite |
|---|---|---|---|---|---|
| 2 | 0.7460 | 0.7350 | 0.7075 | 0.8096 | 0 |
| 12 | 0.7195 | 0.7136 | 0.6691 | 0.7884 | 0 |
| 26 | 0.7233 | 0.7239 | 0.6824 | 0.7715 | 0 |

Per-record v_traj values are in `docs/gate23/gate23_results_v2.jsonl`.

## Data

- Per-record per-layer values: `docs/gate23/gate23_results_v2.jsonl`
