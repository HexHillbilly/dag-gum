# DS-028 — Persistent-regime layer effect sweep (per-layer effect)

> MEASUREMENT REPORT. This document reports measured values only.
> No threshold is created or modified beyond the ds-025 frozen PR
> values, no gate script is touched, no governor/controller.py edit,
> and no green/red verdict and no layer recommendation are offered.
> Joint detection x effect layer selection is the human's act per
> RFC-004 §3.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Records | 50 seeded heldout-degenerate v2 (random.Random(42).sample, sorted) |
| Decoding | greedy (do_sample=False), max_new_tokens=128 |
| Analysis window | trailing 24 generated positions |
| Frozen PR thresholds | docs/gate23/FROZEN_THRESHOLDS.md (ds-025) |
| Persistence hysteresis | below_ctr >= 2 (controller.py:497) |
| Force ramp | base=0.45, +0.05/consecutive, max=0.75, clamp=[0.35, 0.75] (controller.py:120-146,566) |
| Perturbation | orthogonal-style, (controller.py fallback path, no SAE) |
| bmm Triton override | deregistered |
| KL horizon | 8 steps after each fire |
| Wall clock (s) | 483.4 |

## Determinism smoke

One heldout-degenerate record, dormant condition (no hooks),
generated twice with the same seed; the generated token ids must
match exactly.

| record id | run A tokens | run B tokens | identical |
|---|---|---|---|
| 0 | 128 | 128 | True |
|  |  | ALL | True |

## Method summary

- Per record, per candidate layer {2, 12, 26}: two greedy
  generations from the record's mutated_prompt.
- Dormant: no hooks. Hooked: a PERSISTENT forward hook on layer L
  computes PR over the trailing 24 generated positions at each
  decoding step; when PR < T_PR(L) (frozen) it increments
  below_ctr (else resets below_ctr AND consecutive_interventions
  to 0); when below_ctr >= 2 it fires, applying ONE bounded
  orthogonal-style perturbation at
  clamp(min(0.75, 0.45 + (consecutive-1)*0.05), [0.35, 0.75]).
  The hook stays alive for the entire generation (does NOT clear).
- Distinct-2 is measured on the trailing 24 generated tokens
  (ds-025 convention); CTR is the coherent-token ratio of the
  decoded continuation.
- Per-fire KL is the mean next-token KL(hooked || dormant) over
  the 8 generation steps following each fire (the fire step's
  forward pass and the next 7), same horizon as ds-026.

## Frozen PR thresholds used (from ds-025, read-only)

| layer | PR T |
|---|---|
| 2 | 10.954796 |
| 12 | 11.173562 |
| 26 | 10.772658 |

The hook fires when PR < T_PR(L) (the frozen PR primary threshold)
and below_ctr >= 2.

## Per-layer effect on continuation distinct_2

Distinct-2 over the trailing 24 generated tokens. Dormant is
layer-independent (no hooks) and is the same across layers for a
given record. delta = hooked - dormant.

| layer | n | dorm mean | dorm med | dorm p10 | dorm p90 | hook mean | hook med | hook p10 | hook p90 | Δ mean | Δ med | Δ p10 | Δ p90 | #Δ>0 | #fired≥1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 0.1696 | 0.1739 | 0.1304 | 0.2174 | 0.1696 | 0.1739 | 0.1304 | 0.2174 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 |
| 12 | 50 | 0.1696 | 0.1739 | 0.1304 | 0.2174 | 0.1696 | 0.1739 | 0.1304 | 0.2174 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 |
| 26 | 50 | 0.1696 | 0.1739 | 0.1304 | 0.2174 | 0.1696 | 0.1739 | 0.1304 | 0.2174 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 |

- #Δ>0 = number of records where hooked distinct-2 > dormant
  distinct-2; #fired≥1 = number of records with at least one
  persistent-regime fire.

## Per-layer effect on continuation CTR

Coherent-token ratio of the decoded continuation. delta = hooked - dormant.

| layer | n | dorm mean | dorm med | dorm p10 | dorm p90 | hook mean | hook med | hook p10 | hook p90 | Δ mean | Δ med | Δ p10 | Δ p90 | #Δ>0 | #fired≥1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 0.5508 | 0.6641 | 0.3247 | 0.6667 | 0.5508 | 0.6641 | 0.3247 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 |
| 12 | 50 | 0.5508 | 0.6641 | 0.3247 | 0.6667 | 0.5508 | 0.6641 | 0.3247 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 |
| 26 | 50 | 0.5508 | 0.6641 | 0.3247 | 0.6667 | 0.5508 | 0.6641 | 0.3247 | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 50 |

## Intervention count distribution (per layer, per record)

Number of persistent-regime fires per record (0 if a record never
satisfied below_ctr >= 2 with PR below the frozen threshold).

| layer | n_records | total_fires | #records≥1 fire | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 5150 | 50 | 103.0000 | 103.0000 | 103.0000 | 103.0000 | 103 | 103 |
| 12 | 50 | 5150 | 50 | 103.0000 | 103.0000 | 103.0000 | 103.0000 | 103 | 103 |
| 26 | 50 | 5150 | 50 | 103.0000 | 103.0000 | 103.0000 | 103.0000 | 103 | 103 |

## Applied strength distribution (per fire)

Strength = clamp(min(0.75, 0.45 + (consecutive-1)*0.05),
[0.35, 0.75]) — the production force ramp. First fire in a run
uses 0.45; each consecutive fire adds +0.05 up to 0.75.

| layer | n_fires | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 5150 | 0.7398 | 0.7500 | 0.7500 | 0.7500 | 0.4500 | 0.7500 |
| 12 | 5150 | 0.7398 | 0.7500 | 0.7500 | 0.7500 | 0.4500 | 0.7500 |
| 26 | 5150 | 0.7398 | 0.7500 | 0.7500 | 0.7500 | 0.4500 | 0.7500 |

## Fire step distribution (per fire)

0-based decoding step at which each persistent-regime fire was
applied (the step's forward pass predicted the next generated
token).

| layer | n_fires | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 5150 | 75.0000 | 75.0000 | 34.0000 | 116.0000 | 24.0000 | 126.0000 |
| 12 | 5150 | 75.0000 | 75.0000 | 34.0000 | 116.0000 | 24.0000 | 126.0000 |
| 26 | 5150 | 75.0000 | 75.0000 | 34.0000 | 116.0000 | 24.0000 | 126.0000 |

## PR@fire distribution (per fire)

Participation ratio over the trailing 24 generated hidden states
at the step that triggered the fire (always below the frozen
T_PR(L)).

| layer | n_fires | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 5150 | 2.9836 | 3.0858 | 2.0871 | 4.0400 | 2.0058 | 6.4565 |
| 12 | 5150 | 3.6966 | 3.5926 | 2.5426 | 4.8595 | 2.2843 | 7.8022 |
| 26 | 5150 | 3.4034 | 3.3281 | 2.3827 | 4.4560 | 2.3146 | 7.2548 |

## Per-fire mean next-token KL (hooked vs dormant, 8 steps after each fire)

KL(hooked || dormant) over the 8 generation steps following each
fire (the fire step's forward pass plus the next 7). Per-fire KL
is averaged over those steps; the table aggregates per layer over
all fires where KL was measured.

| layer | n_fires+KL | note | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|---|---|
| 2 | 5150 |  | 1.30e-05 | 7.89e-06 | 2.28e-06 | 3.00e-05 | 4.03e-08 | 1.15e-04 |
| 12 | 5150 |  | 1.16e-05 | 6.78e-06 | 2.05e-06 | 2.60e-05 | 0.00e+00 | 1.24e-04 |
| 26 | 5150 |  | 6.70e-06 | 3.57e-06 | 8.72e-07 | 1.62e-05 | 0.00e+00 | 9.64e-05 |

KL is a measurement of how much the bounded perturbation changed
the model's next-token distribution; it is not a decision signal
and it does not by itself imply a layer choice.

## Per-record JSONL

Per-record per-layer results (dormant and hooked continuations,
intervention count, per-fire step/PR@fire/strength/below_ctr/
consecutive_interventions, per-fire KL) are in
`layer_effect_persistent_results.jsonl`.

## Notes / caveats

- MEASUREMENT ONLY. No verdict, no layer recommendation.
- The hook is PERSISTENT: it stays alive for the entire
  generation and may fire multiple times when PR stays below the
  frozen T_PR(L) with below_ctr >= 2.
- The frozen T_PR(L) is the ds-025 PR primary threshold; no band
  or SVSE fusion is used in this effect measurement.
- The perturbation uses the production fallback orthogonalisation
  (random noise projected orthogonal to the residual, no SAE),
  same as ds-026 and the controller.py SAE-fallback path.
- Per-fire KL windows may overlap when fires are consecutive; each
  fire is measured independently over the same 8-step horizon.
