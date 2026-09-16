# DS-034 — Spectral PR liveness on t2s_degenerate (MEASUREMENT ONLY)

> MEASUREMENT REPORT. This probe tests the EXISTING spectral PR instrument on t2s_degenerate: does PR cross the frozen threshold on records where the bigram detector is silent? Zero new code, zero new architecture, no threshold edits, no controller changes, no new detection logic. No verdict is offered.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture | tests/fixtures/t2s_degenerate.jsonl (night-001; N=100), record[\"text\"] input |
| Analysis window | last 24 token positions |
| Candidate layers | {2, 12, 26} (RFC-004) |
| PR instrument | participation_ratio() from core/metrics.py (existing) |
| Thresholds | docs/gate23/FROZEN_THRESHOLDS.md (ds-025 freeze), T_PR per layer; not re-derived |
| DS-033 classification | REUSED from docs/gate23/production_cross_fixture_results.jsonl (do NOT re-compute) |
| bmm Triton override | deregistered |
| Wall clock (s) | 6.1 |

## Determinism smoke

Two t2s_degenerate records replayed twice (teacher-forced, single forward pass each). The per-layer PR tuple at layers {2,12,26} must be identical across the two runs.

| record_id | n_tokens | PR layer2 run A | PR layer2 run B | identical |
|---|---|---|---|---|
| 0 | 199 | 7.1053 | 7.1053 | True |
| 1 | 160 | 4.0927 | 4.0927 | True |
|  | ALL |  |  | True |

## Frozen thresholds (restated from Part A freeze)

Thresholds are read from `docs/gate23/FROZEN_THRESHOLDS.md`; nothing here re-derives them. Criterion for "PR fires": PR < T_PR(layer).

| layer | signal | T_PR | band_low | band_high |
|---|---|---|---|---|
| 2 | PR | 10.954796 | 8.216097 | 13.693495 |
| 12 | PR | 11.173562 | 8.380172 | 13.966953 |
| 26 | PR | 10.772658 | 8.079493 | 13.465822 |

## Table 1 — Per-layer PR distributions on detection-gap records

Detection-gap records: 93 (predicate_true_steps == 0 in DS-033; bigram detector never fired).

| layer | n | mean PR | median PR | p10 PR | p90 PR | #PR<T | PR<T rate |
|---|---|---|---|---|---|---|---|
| 2 | 93 | 4.2875 | 4.9477 | 1.5590 | 6.9640 | 93 | 1.0000 |
| 12 | 93 | 4.7183 | 5.2982 | 2.3830 | 7.2480 | 93 | 1.0000 |
| 26 | 93 | 4.7449 | 5.2623 | 2.3692 | 7.3101 | 93 | 1.0000 |

## Table 2 — Detection-gap records: PR liveness summary

| metric | value |
|---|---|
| Total detection-gap records | 93 |
| PR fires at any layer | 93 |
| PR fires at layer 2 | 93 |
| PR silent at all layers | 0 |

## Table 3 — Per-layer PR comparison: detection-gap vs rescued

Rescued records (n=2) are the non-detection-gap reference where the bigram predicate fired and ΔDistinct-2 moved.

| layer | class | n | mean PR | median PR |
|---|---|---|---|---|
| 2 | detection-gap | 93 | 4.2875 | 4.9477 |
| 2 | rescued | 2 | 1.0558 | 1.0558 |
| 12 | detection-gap | 93 | 4.7183 | 5.2982 |
| 12 | rescued | 2 | 1.0783 | 1.0783 |
| 26 | detection-gap | 93 | 4.7449 | 5.2623 |
| 26 | rescued | 2 | 1.1075 | 1.1075 |

## Non-detection-gap records (comparison log)

The 7 non-detection-gap records (predicate_true_steps > 0 in DS-033) are logged per-record for comparison; the primary question concerns the 93 detection-gap blind spots.

| record_id | class | pred_true_steps | PR L2 | PR L12 | PR L26 | fires any | first fire |
|---|---|---|---|---|---|---|---|
| 28 | rescued | 2 | 1.0877 | 1.1012 | 1.1524 | True | 2 |
| 34 | actuation-gap | 1 | 1.0525 | 1.2013 | 1.2270 | True | 2 |
| 42 | actuation-gap | 1 | 4.0666 | 4.2339 | 4.3462 | True | 2 |
| 48 | actuation-gap | 1 | 3.0722 | 3.3935 | 3.3330 | True | 2 |
| 57 | actuation-gap | 1 | 2.1370 | 2.4828 | 2.4422 | True | 2 |
| 64 | actuation-gap | 1 | 4.0550 | 4.4738 | 4.4068 | True | 2 |
| 86 | rescued | 1 | 1.0240 | 1.0554 | 1.0626 | True | 2 |

## Interpretation (measurement, not a verdict)

Per the DS-033 locked triad decision rule, this probe feeds the human decision:

- If **PR fires where the bigram predicate does not** → dual-predicate fix (bigram OR spectral → logit-penalty path).
- If **PR is also silent** → open macro-syntax window design.

This report MEASURES per-layer PR and the fire rate on detection-gap records. It does not render the verdict.

## Notes / caveats

- MEASUREMENT ONLY: no threshold edits, no controller changes, no new detection logic. Uses the EXISTING participation_ratio instrument and EXISTING frozen thresholds.
- DS-033 detection-gap classification is REUSED from production_cross_fixture_results.jsonl (join on fixture + record_id). Do NOT re-compute or re-classify.
- Input text is `record["text"]` from t2s_degenerate.jsonl (the whole degenerate text). Analysis window = last 24 token positions, matching the Gate 2.3 / ds-025 convention.
- Hidden states are indexed as `out.hidden_states[int(layer)]`, identical to score_gate23.py (ds-025/027) so that the measured PR is directly comparable to the frozen thresholds.
- The criterion for "PR fires" is `PR < T_PR(layer)` (the frozen midpoint threshold), per the DS-034 task parentheticals.
- Per-record results are in `spectral_pr_t2s_results.jsonl`.
