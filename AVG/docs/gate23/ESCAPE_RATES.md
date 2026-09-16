# night-013 — Retroactive Escape-Rate Pass (DATA ANALYSIS ONLY)

> Contextualizing secondary metrics. The rescue-rate gate criterion (ΔDistinct-2 > 0 vs dormant) is unchanged. Gross/net escape rates are computed from existing per-record JSONL; zero GPU, zero model loading, zero re-running.

## Method

- **SEED:** 42 (set for reproducibility; the pass is deterministic).
- **Source:** every probe with per-record JSONL in `docs/gate23/` that contains the fields `byte_identical` (or `byte-id`), `n_generated` (or `n_gen`), and `ΔDistinct-2` (or `ΔD2`), spanning DS-026 through night-012.
- **Gross escape:** `1.0 - byte_identical_rate`. Any divergence at any position counts as escape — including 1-token EOS bailouts.
- **Net escape:** gross escape AND `n_generated >= 24` (the standard Distinct-2 analysis window).
- **Rescue (unchanged):** `ΔDistinct-2 > 0` vs dormant. The primary gate criterion from every prior probe.
- **Four-bucket histogram:** counts over *escaped* continuations: `n=1` (pure EOS bailout), `2-23` (escaped but too short for Distinct-2 analysis), `24-127` (net escape with analyzable text), `n=128` (escaped AND filled the full budget).

### Field-name normalization and multi-condition probes

Probes across the arc used slightly different field names. The script auto-detects and normalizes: byte-identity (`byte-id`, `byte_identical`, `byte-identical`, `token_match`, `byte_identical_to_dormant`), n_generated (`n_gen`, `n_generated`, `n_gen_active`), and ΔDistinct-2 (`ΔD2`, `delta_d2`, `delta_distinct_2`). The measurement is the record's own `active` continuation when present (preferred over reused baselines). For the two multi-condition probes the documented canonical reference condition is used:

- **DS-030** (`penalty_decomposition_results.jsonl`): arm **D** — production suppression-only (cooldown=8, penalty=-5.0), the DS-030 primary-actuator finding reused throughout the arc as the suppression-only reference.
- **DS-031** (`cooldown_sweep_results.jsonl`): cell **(5,-5)** — the DS-031 headline best-quality-at-equal-rescue finding.

### Skipped / excluded sources

Prose-100 FP probes are excluded (they measure false positives on healthy text, not escape from degenerate loops). Sources that lack per-record escape fields, or whose required fields could not be resolved, are logged and skipped.

- `dual_predicate_fp_results.jsonl`
- `ecs_pks_results.jsonl`
- `fp_dual_hardening_results.jsonl`
- `gate23_results.jsonl`
- `gate23_results_v2.jsonl`
- `greens_function_results.jsonl`
- `jlens_alignment_results.jsonl`
- `layer2_pr_latency_results.jsonl`
- `layer_effect_persistent_results.jsonl`
- `layer_effect_results.jsonl`
- `live_ecs_results.jsonl`
- `multistep_perturbation_results.jsonl`
- `online_pr_parity_results.jsonl`
- `pr_ecs_causal_results.jsonl`
- `residual_layer_sweep_results.jsonl`
- `spectral_pr_t2s_results.jsonl`

## Trend tables

Rows ordered chronologically (DS-026 earliest, night-012 latest). `IQR (esc)` is (p25, p75) of n_generated over escaped continuations. The `n=1 | 2-23 | 24-127 | n=128` buckets sum to the gross-escape count.

### Fixture: `t2s_degenerate`

| probe | n | rescue_rate | gross_escape | net_escape | median_n_gen (esc) | IQR (esc) | n=1 | 2-23 | 24-127 | n=128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-032 | 100 | 0.1500 | 0.1500 | 0.1000 | 76 | (17.5, 128) | 0 | 5 | 5 | 5 |
| DS-033 | 100 | 0.1500 | 0.1500 | 0.1000 | 95 | (21.5, 128) | 0 | 5 | 4 | 6 |
| DS-035 | 100 | 0.2000 | 0.2200 | 0.2200 | 29.5 | (27.25, 128) | 0 | 0 | 15 | 7 |
| night-004 | 100 | 0.1300 | 0.1400 | 0.1100 | 93 | (44.75, 128) | 0 | 3 | 6 | 5 |
| night-006 | 100 | 0.1300 | 0.1400 | 0.1000 | 110 | (25.5, 128) | 0 | 4 | 4 | 6 |
| night-007 | 100 | 0.2400 | 0.2500 | 0.2100 | 29 | (26, 128) | 0 | 4 | 14 | 7 |
| night-008 | 100 | 0.2300 | 0.2400 | 0.2000 | 29.5 | (26.75, 128) | 0 | 4 | 13 | 7 |

### Fixture: `qwen_degenerate`

| probe | n | rescue_rate | gross_escape | net_escape | median_n_gen (esc) | IQR (esc) | n=1 | 2-23 | 24-127 | n=128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-032 | 100 | 0.6800 | 0.7000 | 0.5400 | 51 | (25, 128) | 7 | 9 | 28 | 26 |
| DS-033 | 100 | 0.6800 | 0.7000 | 0.5400 | 65 | (25, 128) | 7 | 9 | 25 | 29 |
| DS-035 | 100 | 0.2000 | 0.2200 | 0.2200 | 29 | (27, 106) | 0 | 0 | 17 | 5 |
| night-004 | 100 | 0.6500 | 0.6700 | 0.5300 | 49 | (26.5, 128) | 7 | 7 | 30 | 23 |
| night-006 | 100 | 0.6500 | 0.6700 | 0.5600 | 75 | (28.5, 128) | 7 | 4 | 26 | 30 |
| night-007 | 100 | 0.6100 | 0.6700 | 0.5600 | 66 | (28.5, 128) | 7 | 4 | 26 | 30 |
| night-008 | 100 | 0.6100 | 0.6700 | 0.5600 | 66 | (28.5, 128) | 7 | 4 | 26 | 30 |

### Fixture: `heldout_degenerate_v2`

| probe | n | rescue_rate | gross_escape | net_escape | median_n_gen (esc) | IQR (esc) | n=1 | 2-23 | 24-127 | n=128 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-030 | 50 | 0.9600 | 0.9600 | 0.9000 | 128 | (109.75, 128) | 3 | 0 | 11 | 34 |
| DS-031 | 50 | 0.9600 | 0.9600 | 0.9000 | 128 | (128, 128) | 3 | 0 | 7 | 38 |
| DS-032 | 100 | 1.0000 | 1.0000 | 0.7300 | 128 | (12.25, 128) | 19 | 8 | 8 | 65 |
| DS-035 | 50 | 0.9600 | 1.0000 | 0.9000 | 28 | (28, 128) | 0 | 5 | 30 | 15 |
| night-004 | 50 | 1.0000 | 1.0000 | 0.5000 | 59 | (1, 128) | 19 | 6 | 2 | 23 |
| night-006 | 50 | 0.9600 | 0.9600 | 0.9000 | 128 | (124.5, 128) | 3 | 0 | 9 | 36 |
| night-007 | 50 | 1.0000 | 1.0000 | 0.9400 | 128 | (114, 128) | 3 | 0 | 11 | 36 |
| night-008 | 50 | 0.9600 | 0.9600 | 0.9000 | 128 | (124.5, 128) | 3 | 0 | 9 | 36 |

## Key observations (computed, contextualizing)

The rescue-rate ceilings quoted across the arc — qwen 0.61 (night-007/008), t2s 0.23 (night-008) — are compared below with gross/net escape at the same probe. No gate verdict is offered; these are contextualizing metrics only.

- **`t2s_degenerate`** — at the rescue-ceiling probe `night-008` (max rescue 0.2300): net escape 0.2000 vs gross 0.2400; 4/100 escaped records are too short for Distinct-2 analysis (0 pure-EOS).
- **`qwen_degenerate`** — at the rescue-ceiling probe `night-007` (max rescue 0.6100): net escape 0.5600 vs gross 0.6700; 11/100 escaped records are too short for Distinct-2 analysis (7 pure-EOS).
- **`heldout_degenerate_v2`** — at the rescue-ceiling probe `DS-032` (max rescue 1.0000): net escape 0.7300 vs gross 1.0000; 27/100 escaped records are too short for Distinct-2 analysis (19 pure-EOS).

## Per-record data

Per-record per-probe escape flags are written to `docs/gate23/escape_rates.jsonl` (one JSON object per line, chronological probe order).

## Caveats

- These are contextualizing metrics, not a gate. The rescue-rate gate criterion is unchanged.
- DS-032 `heldout_degenerate_v2` spans two regimes (Part 1 greedy combined, Part 2 sampling suppression-only); the pair is aggregated into one row.
- DS-030/DS-031 rows use the documented canonical reference condition so the trend table has one row per probe per fixture.
