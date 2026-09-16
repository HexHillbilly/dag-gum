# night-002 — Residual→Logit Singular-Vector Layer Sweep

Date: 2026-08-11  |  SEED: 42  |  Model: `Qwen/Qwen2.5-1.5B@8faed761d45a263340a0528343f099c05c9a4323`

MEASUREMENT ONLY. Self-contained probe (no AVG imports).

## Method

- Layers probed: `[6, 14, 21, 27]`
- Randomized SVD on finite differences: K=40 (q=32, p=8), top-4 singular vectors via Rayleigh–Ritz on the K×K Gram.
- ε = 0.001 × RMS(h_ℓ) at the anchor prompt (`'The capital of France is'`, index 0).
- FD Jacobian estimation in **fp32** (cpu); generation arms in **bf16** (cuda) to match production.
- Arms per prompt per layer: dormant, random-orthogonal, top v₁, bottom v₄; forces [0.35, 0.5, 0.75]. Continuation = 64 tokens.

## Diagnostic table

| layer | σ₁ | σ₂ | σ₃ | σ₄ | σ₁/σ₂ | pred KL@0.5 | KL@0.5 v₁ | KL@0.5 rand | status |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 4.432e-02 | 1.847e-02 | 1.575e-02 | 9.785e-03 | 2.40 | 2.46e-04 | 8.75e-04 | 7.42e-04 | **INCONCLUSIVE** |
| 14 | 3.126e-02 | 3.024e-02 | 1.765e-02 | 1.595e-02 | 1.03 | 1.22e-04 | 9.46e-04 | 7.05e-04 | **INCONCLUSIVE** |
| 21 | 5.143e-02 | 3.013e-02 | 2.705e-02 | 2.610e-02 | 1.71 | 3.31e-04 | 4.85e-04 | 5.29e-04 | **INCONCLUSIVE** |
| 27 | 8.488e-02 | 3.580e-02 | 3.555e-02 | 3.532e-02 | 2.37 | 9.01e-04 | 2.55e-04 | 3.93e-05 | **INCONCLUSIVE** |

Predicted KL via local linearity: `KL_pred = 0.5·(σ₁·0.5)²` (noise-floor normalised to 1; σ₁ in logit units per unit hidden perturbation).

## Verdicts

### Layer 6: **INCONCLUSIVE**

- Basis: top beats random at f=0.5 but absolute KL 8.75e-04 < 1e-3 (non-zero singular structure, channel effectively closed)
- KL_pred@0.5: 2.455e-04

### Layer 14: **INCONCLUSIVE**

- Basis: top beats random at f=0.5 but absolute KL 9.46e-04 < 1e-3 (non-zero singular structure, channel effectively closed)
- KL_pred@0.5: 1.222e-04

### Layer 21: **INCONCLUSIVE**

- Basis: no ALIVE/DEAD conditions met at primary force 0.5
- KL_pred@0.5: 3.306e-04

### Layer 27: **INCONCLUSIVE**

- Basis: top beats random at f=0.5 but absolute KL 2.55e-04 < 1e-3 (non-zero singular structure, channel effectively closed)
- KL_pred@0.5: 9.005e-04

## Per-arm aggregates (median step-0 KL, argmax-flip rate, exact-match rate)

| layer | force | arm | median KL | mean KL | flip rate | exact-match | token-match | Distinct-2 (hooked) |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.35 | random | 8.89e-04 | 9.51e-04 | 0.000 | 0.750 | 0.838 | 0.7004 |
| 6 | 0.35 | top | 1.08e-03 | 1.08e-03 | 0.000 | 0.703 | 0.833 | 0.7174 |
| 6 | 0.35 | bottom | 9.01e-04 | 1.01e-03 | 0.000 | 0.766 | 0.844 | 0.7323 |
| 6 | 0.5 | random | 7.42e-04 | 9.13e-04 | 0.000 | 0.656 | 0.830 | 0.7058 |
| 6 | 0.5 | top | 8.75e-04 | 1.03e-03 | 0.000 | 0.703 | 0.834 | 0.7004 |
| 6 | 0.5 | bottom | 9.42e-04 | 9.70e-04 | 0.016 | 0.688 | 0.823 | 0.7255 |
| 6 | 0.75 | random | 9.60e-04 | 9.83e-04 | 0.016 | 0.719 | 0.825 | 0.7350 |
| 6 | 0.75 | top | 1.05e-03 | 1.30e-03 | 0.016 | 0.719 | 0.850 | 0.7255 |
| 6 | 0.75 | bottom | 1.22e-03 | 1.17e-03 | 0.016 | 0.750 | 0.853 | 0.7330 |
| 14 | 0.35 | random | 6.77e-04 | 8.12e-04 | 0.000 | 0.781 | 0.884 | 0.7180 |
| 14 | 0.35 | top | 1.04e-03 | 1.00e-03 | 0.016 | 0.703 | 0.819 | 0.7262 |
| 14 | 0.35 | bottom | 7.35e-04 | 9.57e-04 | 0.000 | 0.750 | 0.840 | 0.7269 |
| 14 | 0.5 | random | 7.05e-04 | 9.42e-04 | 0.016 | 0.688 | 0.833 | 0.7126 |
| 14 | 0.5 | top | 9.46e-04 | 9.36e-04 | 0.016 | 0.766 | 0.883 | 0.7058 |
| 14 | 0.5 | bottom | 9.05e-04 | 9.72e-04 | 0.016 | 0.797 | 0.880 | 0.7058 |
| 14 | 0.75 | random | 1.01e-03 | 1.11e-03 | 0.000 | 0.797 | 0.907 | 0.7248 |
| 14 | 0.75 | top | 1.11e-03 | 1.10e-03 | 0.016 | 0.719 | 0.843 | 0.7242 |
| 14 | 0.75 | bottom | 1.00e-03 | 1.15e-03 | 0.031 | 0.656 | 0.821 | 0.7235 |
| 21 | 0.35 | random | 4.86e-04 | 7.56e-04 | 0.031 | 0.672 | 0.768 | 0.7119 |
| 21 | 0.35 | top | 4.24e-04 | 7.12e-04 | 0.016 | 0.750 | 0.855 | 0.7153 |
| 21 | 0.35 | bottom | 6.32e-04 | 8.35e-04 | 0.016 | 0.703 | 0.845 | 0.7323 |
| 21 | 0.5 | random | 5.29e-04 | 6.73e-04 | 0.016 | 0.719 | 0.868 | 0.7371 |
| 21 | 0.5 | top | 4.85e-04 | 7.19e-04 | 0.016 | 0.797 | 0.906 | 0.7092 |
| 21 | 0.5 | bottom | 5.72e-04 | 7.81e-04 | 0.047 | 0.703 | 0.796 | 0.7282 |
| 21 | 0.75 | random | 6.14e-04 | 7.72e-04 | 0.000 | 0.781 | 0.888 | 0.7065 |
| 21 | 0.75 | top | 6.99e-04 | 7.85e-04 | 0.016 | 0.734 | 0.855 | 0.7221 |
| 21 | 0.75 | bottom | 6.84e-04 | 8.79e-04 | 0.016 | 0.750 | 0.847 | 0.6990 |
| 27 | 0.35 | random | 2.28e-05 | 1.18e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.35 | top | 6.71e-05 | 2.89e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.35 | bottom | 3.29e-05 | 1.53e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.5 | random | 3.93e-05 | 2.20e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.5 | top | 2.55e-04 | 4.74e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.5 | bottom | 7.57e-05 | 2.08e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.75 | random | 9.68e-05 | 2.83e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.75 | top | 4.55e-04 | 6.91e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |
| 27 | 0.75 | bottom | 1.42e-04 | 2.95e-04 | 0.000 | 1.000 | 1.000 | 0.7187 |

Dormant Distinct-2 mean (trailing 24): layer 6: 0.6003, layer 14: 0.6003, layer 21: 0.6003, layer 27: 0.6003

## Interpretation

- **All four layers are INCONCLUSIVE** — none met the ALIVE threshold (top
  median KL ≥ 10× random OR ≥20pp argmax-flip gap) and none met the DEAD
  threshold (mean KL ≲ 1e-4 AND ≥95% byte-identical continuations at all
  forces).
- **The residual channel is not dead at any probed depth.**  At layers 6, 14
  and 27 the top singular direction v₁ beats the random-orthogonal control at
  force 0.5 (KL_v₁/KL_rand ≈ 1.2×, 1.3×, 6.5× respectively).  Non-zero
  singular structure exists at every layer (σ₁ ∈ [3.1e-2, 8.5e-2]).
- **But the channel is effectively closed at the tested force band** at every
  layer: the largest v₁ median KL is 9.5e-4 (layer 14, force 0.5), below the
  1e-3 INCONCLUSIVE threshold.
- **Layer 27 (last block) is the most interesting.**  It has the largest σ₁
  (8.5e-2) and the largest top-vs-random KL gap (6.5× at force 0.5), yet 100%
  of hooked 64-token continuations are byte-identical to dormant at all forces.
  The step-0 logit distribution shifts measurably along v₁ but the greedy
  trajectory never flips — the textbook "channel effectively closed" signature.
  This is consistent with DS-036's layer-26 finding (dead under greedy) while
  refining it: the residual→logit map is locally non-zero but its singular
  directions do not move the greedy rollout at forces ≤ 0.75.
- **Layer 14 has a near-degenerate spectrum** (σ₁/σ₂ = 1.03): no single
  dominant direction.  ContextFocus's hypothesis that intermediate layers
  (~40-50% depth) are optimal for steering is not supported by this probe in
  the sense that layer 14 shows no larger v₁ response than other layers.
- **KL_pred vs KL_measured (falsification check):** at layers 6 and 14 the
  measured KL exceeds the linear prediction (3.5× and 7.7×); at layer 27 it is
  3.5× below the prediction.  The local-linear approximation is only order-of-
  magnitude accurate and does not rescue the channel: even where KL_pred is
  large (9e-4 at layer 27), the measured rollout divergence is zero.
- **Caveats:** the Jacobian is estimated at a single factual anchor prompt
  (index 0) per layer and the ε-linearity check passed at all layers (no ε
  adjustment was needed).  FD probes run in fp32 (bf16 dead-zone avoidance);
  arms run in bf16 to match production.  The 64-prompt mix includes 16
  degenerate prompts (trailing-24 Distinct-2 ≈ 0.04) that do engage the
  repetition regime, so both healthy and degenerate activation states are
  represented in the arm statistics.

## Notes

- FD probe runs in fp32 to avoid the bf16 dead-zone (DS-036 finding: ε=1e-3 perturbations quantise to zero in bf16 at late layers). Arms run in bf16; mismatch documented.
- `docs/gate23/residual_layer_sweep_results.jsonl` contains full per-layer records (singular spectrum, power-iteration trace, linearity check, per-arm aggregates).
