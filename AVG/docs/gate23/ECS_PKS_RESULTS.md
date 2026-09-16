# DS-038 — ECS/PKS liveness on t2s_degenerate (MEASUREMENT ONLY)

> MEASUREMENT REPORT. This probe characterizes whether ECS (External Context Score) and PKS (Parametric Knowledge Score) separate t2s_degenerate from healthy prose strongly enough to justify investment in AARF actuation. Zero controller edits, zero new architecture, zero actuation design. No verdict is offered.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture A | tests/fixtures/t2s_degenerate.jsonl (night-001; N=100), record[\"text\"] input |
| Fixture B | data/t2s_bench/valid_subset_200.jsonl heldout prose-100 (ds-027 complement of Gate-2.2 control-100; N=100) |
| Analysis window | last 24 token positions |
| Candidate layers | {2, 6, 14, 21, 27} |
| Attention heads / layer | 12 (real Qwen2.5-1.5B config; task spec's '28 heads' conflated layer count with head count) |
| ECS prompt boundary | prompt_len = seq_len - 24 (the analysis window is the 'generated' region; everything before is the 'prompt' context) |
| PKS top-k | k = 1000 + tail-mass bucket (float32) |
| Max seq len (prefix truncation) | 4096 tokens (4 prose records truncated) |
| Attention capture | forward hooks on candidate-layer self_attn sub-modules (output_attentions=True OOMs on long prose; hooks return identical attention weights) |
| bmm Triton override | deregistered |
| Wall clock (s) | 42.7 |

## Determinism smoke (FIRST)

Two t2s_degenerate records replayed twice (teacher-forced, single forward pass each). Per-head ECS and per-layer PKS must be identical to 6 decimal places. STOP if not.

| record_id | n_tokens | ECS L2H0 run A | ECS L2H0 run B | PKS L2 run A | PKS L2 run B | identical |
|---|---|---|---|---|---|---|
| 0 | 199 | 0.747163 | 0.747163 | 0.189167 | 0.189167 | True |
| 1 | 160 | 0.762771 | 0.762771 | 0.304710 | 0.304710 | True |
|  | ALL |  |  |  |  | True |

## Table 1 — Per-layer ECS distributions (degenerate vs prose)

ECS = fraction of attention that the last-24 query positions give to prompt tokens (prompt_len = seq_len - 24). Degenerate records are expected to have LOWER ECS. Fires when ECS < T_ECS(L, H). Fire rate is computed on degenerate records only.

| layer | n_heads | mean ECS deg | mean ECS prose | separation | #heads fire | fire rate (deg) |
|---|---|---|---|---|---|---|
| 2 | 12 | 0.7311 | 0.7014 | 0.9788 | 11 | 0.4625 |
| 6 | 12 | 0.7539 | 0.7838 | 0.9814 | 12 | 0.4383 |
| 14 | 12 | 0.7790 | 0.8124 | 0.9715 | 12 | 0.4892 |
| 21 | 12 | 0.9117 | 0.8174 | 0.9986 | 9 | 0.0775 |
| 27 | 12 | 0.8414 | 0.8465 | 0.9358 | 12 | 0.1517 |

## Table 2 — Per-layer PKS distributions (degenerate vs prose)

PKS = JSD between post-attention and post-FFN projected distributions (top-1000 + tail bucket). Degenerate records are expected to have HIGHER PKS. Fires when PKS > T_PKS(L). Fire rate is computed on degenerate records only.

| layer | mean PKS deg | mean PKS prose | separation | #layers fire | fire rate (deg) |
|---|---|---|---|---|---|
| 2 | 0.2074 | 0.1918 | 0.5944 | 1 | 0.3500 |
| 6 | 0.1086 | 0.1469 | 0.8028 | 1 | 0.2300 |
| 14 | 0.0801 | 0.0698 | 0.7112 | 1 | 0.5600 |
| 21 | 0.0394 | 0.1213 | 0.9393 | 1 | 0.1200 |
| 27 | 0.0163 | 0.0724 | 0.9252 | 1 | 0.0900 |

## Table 3 — Top-5 most separated ECS heads

Separation = max |AUC - 0.5| (rank-based Mann-Whitney, degenerate as the positive class). AUC < 0.5 → degenerate-lower.

| layer | head | ECS deg mean | ECS prose mean | separation | AUC |
|---|---|---|---|---|---|
| 21 | 3 | 0.8318 | 0.5326 | 0.9986 | 0.9986 |
| 6 | 6 | 0.5452 | 0.9162 | 0.9814 | 0.0186 |
| 2 | 1 | 0.8987 | 0.6973 | 0.9788 | 0.9788 |
| 21 | 7 | 0.8805 | 0.6793 | 0.9728 | 0.9728 |
| 14 | 0 | 0.6514 | 0.9355 | 0.9715 | 0.0285 |

## Table 4 — Top-5 most separated PKS layers

| layer | PKS deg mean | PKS prose mean | separation | AUC |
|---|---|---|---|---|
| 21 | 0.0394 | 0.1213 | 0.9393 | 0.0607 |
| 27 | 0.0163 | 0.0724 | 0.9252 | 0.0748 |
| 6 | 0.1086 | 0.1469 | 0.8028 | 0.1972 |
| 14 | 0.0801 | 0.0698 | 0.7112 | 0.7112 |
| 2 | 0.2074 | 0.1918 | 0.5944 | 0.5944 |

## Table 5 — Per-layer per-head ECS thresholds (frozen)

Thresholds are read from `docs/gate23/FROZEN_ECS_PKS_THRESHOLDS.md` (Gate 2.3 Part A rule: T = (p90_deg + p10_prose)/2, band = [0.75*T, 1.25*T]); nothing here re-derives them.

| layer | head | T_ECS | band_low | band_high |
|---|---|---|---|---|
| 2 | 0 | 0.671564 | 0.503673 | 0.839454 |
| 2 | 1 | 0.741784 | 0.556338 | 0.927230 |
| 2 | 2 | 0.920455 | 0.690341 | 1.150569 |
| 2 | 3 | 0.933113 | 0.699834 | 1.166391 |
| 2 | 4 | 0.790850 | 0.593138 | 0.988563 |
| 2 | 5 | 0.819301 | 0.614475 | 1.024126 |
| 2 | 6 | 0.758536 | 0.568902 | 0.948170 |
| 2 | 7 | 0.526950 | 0.395212 | 0.658687 |
| 2 | 8 | 0.411139 | 0.308354 | 0.513924 |
| 2 | 9 | 0.766235 | 0.574676 | 0.957794 |
| 2 | 10 | 0.601374 | 0.451031 | 0.751718 |
| 2 | 11 | 0.716920 | 0.537690 | 0.896150 |
| 6 | 0 | 0.734034 | 0.550526 | 0.917543 |
| 6 | 1 | 0.966390 | 0.724792 | 1.207987 |
| 6 | 2 | 0.736515 | 0.552387 | 0.920644 |
| 6 | 3 | 0.937721 | 0.703290 | 1.172151 |
| 6 | 4 | 0.607376 | 0.455532 | 0.759219 |
| 6 | 5 | 0.621855 | 0.466391 | 0.777319 |
| 6 | 6 | 0.726620 | 0.544965 | 0.908275 |
| 6 | 7 | 0.981565 | 0.736174 | 1.226957 |
| 6 | 8 | 0.622125 | 0.466594 | 0.777657 |
| 6 | 9 | 0.992640 | 0.744480 | 1.240800 |
| 6 | 10 | 0.537478 | 0.403109 | 0.671848 |
| 6 | 11 | 0.762941 | 0.572206 | 0.953676 |
| 14 | 0 | 0.800429 | 0.600322 | 1.000536 |
| 14 | 1 | 0.785026 | 0.588770 | 0.981283 |
| 14 | 2 | 0.932380 | 0.699285 | 1.165475 |
| 14 | 3 | 0.961933 | 0.721450 | 1.202417 |
| 14 | 4 | 0.980157 | 0.735117 | 1.225196 |
| 14 | 5 | 0.996775 | 0.747581 | 1.245969 |
| 14 | 6 | 0.934490 | 0.700868 | 1.168113 |
| 14 | 7 | 0.638039 | 0.478529 | 0.797548 |
| 14 | 8 | 0.518003 | 0.388502 | 0.647503 |
| 14 | 9 | 0.267940 | 0.200955 | 0.334925 |
| 14 | 10 | 0.794667 | 0.596000 | 0.993333 |
| 14 | 11 | 0.819742 | 0.614807 | 1.024678 |
| 21 | 0 | 0.901046 | 0.675785 | 1.126308 |
| 21 | 1 | 0.814318 | 0.610739 | 1.017898 |
| 21 | 2 | 0.844427 | 0.633320 | 1.055533 |
| 21 | 3 | 0.659885 | 0.494914 | 0.824857 |
| 21 | 4 | 0.795431 | 0.596573 | 0.994288 |
| 21 | 5 | 0.909800 | 0.682350 | 1.137250 |
| 21 | 6 | 0.935490 | 0.701618 | 1.169363 |
| 21 | 7 | 0.730686 | 0.548014 | 0.913357 |
| 21 | 8 | 0.795031 | 0.596273 | 0.993789 |
| 21 | 9 | 0.880943 | 0.660707 | 1.101179 |
| 21 | 10 | 0.769872 | 0.577404 | 0.962340 |
| 21 | 11 | 0.942781 | 0.707086 | 1.178476 |
| 27 | 0 | 0.886424 | 0.664818 | 1.108030 |
| 27 | 1 | 0.897895 | 0.673421 | 1.122368 |
| 27 | 2 | 0.883934 | 0.662950 | 1.104917 |
| 27 | 3 | 0.890329 | 0.667747 | 1.112912 |
| 27 | 4 | 0.889896 | 0.667422 | 1.112370 |
| 27 | 5 | 0.885031 | 0.663773 | 1.106288 |
| 27 | 6 | 0.880565 | 0.660424 | 1.100706 |
| 27 | 7 | 0.885352 | 0.664014 | 1.106689 |
| 27 | 8 | 0.867352 | 0.650514 | 1.084190 |
| 27 | 9 | 0.645363 | 0.484022 | 0.806703 |
| 27 | 10 | 0.290329 | 0.217747 | 0.362911 |
| 27 | 11 | 0.892977 | 0.669733 | 1.116221 |

## Frozen PKS thresholds

| layer | T_PKS | band_low | band_high |
|---|---|---|---|
| 2 | 0.228893 | 0.171670 | 0.286116 |
| 6 | 0.135540 | 0.101655 | 0.169425 |
| 14 | 0.079733 | 0.059800 | 0.099666 |
| 21 | 0.088151 | 0.066113 | 0.110189 |
| 27 | 0.031608 | 0.023706 | 0.039510 |

## Prose false-positive fire rate (auxiliary)

For a gate to be usable, the frozen thresholds must NOT fire on healthy prose. This table reports the prose fire rate (fraction of prose records where ECS < T or PKS > T).

| layer | prose FP rate (ECS, per head-record) | prose FP rate (PKS, per record) |
|---|---|---|
| 2 | 0.5433 | 0.0800 |
| 6 | 0.4683 | 0.6200 |
| 14 | 0.2692 | 0.1800 |
| 21 | 0.4675 | 0.8700 |
| 27 | 0.2442 | 0.9300 |

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no new architecture, no actuation design, no thresholds beyond the frozen derivation.
- The same 100 t2s records are used for both calibration and scoring in this initial probe (the separation magnitude is the primary result). A held-out scoring pass on a different degenerate fixture (e.g., qwen_degenerate detection-gap records) is a future measurement.
- The task spec stated '28 attention heads per layer'; the actual Qwen2.5-1.5B config has num_attention_heads=12 (the spec conflated layer count with head count). All ECS per-head metrics use the real 12-head config.
- `output_attentions=True` OOMs on long prose records (attention weights are O(seq_len²); the longest prose record tokenizes to 7265 tokens). Forward hooks on the candidate-layer self_attn sub-modules capture the identical attention weights (Qwen2Attention returns them in eager mode) without materializing the other 23 layers' attention.
- 4 prose records exceed MAX_SEQ_LEN=4096 tokens and were prefix-truncated (standard truncation; keeps the last-24 analysis window of the truncated sequence).
- ECS prompt boundary: prompt_len = seq_len - 24 (the Gate 2.3 analysis window is the 'generated' region). This is a documented modeling choice for teacher-forced replay, which has no explicit prompt/generation split.
- All PKS math is float32 (per the DS-036 bf16 dead-zone finding). ECS attention sums are computed in float32.
- Per-record per-head per-layer values are in `ecs_pks_results.jsonl`.
