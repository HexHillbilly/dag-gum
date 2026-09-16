# DS-026 — Layer effect measurement (per-layer intervention effect)

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
| Perturbation | orthogonal-style, 0.45 L2, clamp [0.35, 0.75] (controller.py:120-146,566) |
| bmm Triton override | deregistered |
| KL horizon | 8 steps after each fire |
| Wall clock (s) | 439.0 |

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
- Dormant: no hooks. Hooked: a forward hook on layer L computes
  PR over the trailing 24 generated positions at each decoding
  step; when PR < T_PR(L) (frozen), it applies ONE bounded
  orthogonal-style perturbation (0.45 L2, clamp [0.35, 0.75]) to
  the current hidden state, then clears (at most one fire).
- Distinct-2 is measured on the trailing 24 generated tokens
  (ds-025 convention); CTR is the coherent-token ratio of the
  decoded continuation.
- KL is the mean next-token KL(hooked || dormant) over the 8
  generation steps following each application (the application
  step's forward pass and the next 7). Where logits could not be
  captured cleanly it is reported as NOT MEASURED.

## Frozen PR thresholds used (from ds-025, read-only)

| layer | PR T |
|---|---|
| 2 | 10.954796 |
| 12 | 11.173562 |
| 26 | 10.772658 |

The hook fires when PR < T_PR(L) (the frozen PR primary threshold).

## Per-layer effect on continuation distinct_2

Distinct-2 over the trailing 24 generated tokens. Dormant is
layer-independent (no hooks) and is the same across layers for a
given record. delta = hooked - dormant.

| layer | n | dorm mean | dorm med | hook mean | hook med | Δ mean | Δ med | #Δ>0 | #fired |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 0.1696 | 0.1739 | 0.1696 | 0.1739 | 0.0000 | 0.0000 | 0 | 50 |
| 12 | 50 | 0.1696 | 0.1739 | 0.1696 | 0.1739 | 0.0000 | 0.0000 | 0 | 50 |
| 26 | 50 | 0.1696 | 0.1739 | 0.1696 | 0.1739 | 0.0000 | 0.0000 | 0 | 50 |

- #Δ>0 = number of records where hooked distinct-2 > dormant
  distinct-2; #fired = number of records where the hook fired.

## Per-layer effect on continuation CTR

Coherent-token ratio of the decoded continuation. delta = hooked - dormant.

| layer | n | dorm mean | dorm med | hook mean | hook med | Δ mean | Δ med | #Δ>0 | #fired |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 0.5508 | 0.6641 | 0.5508 | 0.6641 | 0.0000 | 0.0000 | 0 | 50 |
| 12 | 50 | 0.5508 | 0.6641 | 0.5508 | 0.6641 | 0.0000 | 0.0000 | 0 | 50 |
| 26 | 50 | 0.5508 | 0.6641 | 0.5508 | 0.6641 | 0.0000 | 0.0000 | 0 | 50 |

## Applied-force L2 stats (per layer, records where the hook fired)

The perturbation is a single 0.45 L2 orthogonal-style impulse
(production base_strength), clamped to [0.35, 0.75]. Because the
base strength is inside the clamp, before/after clamp are equal.

| layer | n_fired | L2 before | L2 after | fire step min | fire step med | fire step max | PR@fire mean | PR@fire min |
|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 0.4500 | 0.4500 | 23.0000 | 23.0000 | 23.0000 | 3.1341 | 2.1358 |
| 12 | 50 | 0.4500 | 0.4500 | 23.0000 | 23.0000 | 23.0000 | 4.2103 | 2.7936 |
| 26 | 50 | 0.4500 | 0.4500 | 23.0000 | 23.0000 | 23.0000 | 3.8154 | 2.7806 |

## Mean next-token KL (hooked vs dormant, 8 steps after each fire)

KL(hooked || dormant) over the 8 generation steps following the
application (the application step's forward pass plus the next 7).
Per-record KL is averaged over those steps; the table aggregates
per layer over records where the hook fired and KL was measured.

| layer | n_fired+KL | note | mean | median | min | max |
|---|---|---|---|---|---|---|
| 2 | 50 |  | 2.88e-05 | 2.13e-05 | 4.78e-06 | 9.30e-05 |
| 12 | 50 |  | 2.53e-05 | 1.74e-05 | 5.06e-06 | 7.27e-05 |
| 26 | 50 |  | 7.06e-06 | 4.19e-06 | 6.99e-08 | 3.20e-05 |

KL is a measurement of how much the bounded perturbation changed
the model's next-token distribution; it is not a decision signal
and it does not by itself imply a layer choice.

## Per-record JSONL

Per-record per-layer results (dormant and hooked continuations,
intervention count, force L2, per-fire KL) are in
`layer_effect_results.jsonl`.

## Notes / caveats

- MEASUREMENT ONLY. No verdict, no layer recommendation.
- The hook fires at most once per record per layer (spec: ONE
  perturbation, then clears).
- The frozen T_PR(L) is the ds-025 PR primary threshold; no band
  or SVSE fusion is used in this effect measurement.
- The perturbation uses the production fallback orthogonalisation
  (random noise projected orthogonal to the residual, no SAE).
