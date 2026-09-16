# night-012 — Multi-step PR→ECS causal probe (MEASUREMENT ONLY)

> MEASUREMENT REPORT. This probe tests whether a MULTI-STEP 0.50-L2 isotropic-noise perturbation at layer 2 (steps 23-30, 8 consecutive steps) re-inflates the trailing-window PR across T_PR (10.954796) and, if so, whether L6H6 ECS rises in response. One record, one intervention, one diagnostic. PURELY DIAGNOSTIC: no thresholds, no actuation, no controller change. No verdict is offered; the human interprets the result.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Perturbation | isotropic Gaussian noise at layer 2, scaled to 0.5 L2, applied to the current token's hidden state |
| Perturbation steps | [23, 24, 25, 26, 27, 28, 29, 30] (8 consecutive steps) |
| Accumulation point | step 30 (trailing window: 8 perturbed / 24 positions) |
| Frozen T_PR(2) | 10.954796 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |
| ECS head | L6H6 (night-010 live ECS reference) |
| Selected record | 5 (REUSED from night-011 Phase 0; do NOT re-run) |
| night-011 Phase 0/1 | REUSED from docs/gate23/pr_ecs_causal_results.jsonl |
| bmm Triton override | deregistered |
| Wall clock (s) | 2.9 |

## Baseline (REUSED, not re-measured)

night-011 Phase 0 selected record 5 on the t2s_degenerate fixture. The single-token 0.50-L2 perturbation at layer 2 moved PR by only ~0.008 (Phase-1 STOP). This probe tests the multi-step version of the same perturbation.

| Quantity | Value |
|---|---|
| night-011 Phase-1 baseline PR (steps 23-25) | s23=5.2763, s24=5.2704, s25=5.2658 |
| night-011 Phase-1 perturbed PR (steps 23-25) | s23=5.2849, s24=5.2811, s25=5.2755 |
| night-010 live L6H6 ECS mean (steps 24-40) | 0.1126 |
| night-010 live L6H6 ECS range (steps 24-43) | [0.0661, 0.1747] |
| night-011 baseline ECS reference (task criterion) | 0.17 |

## Phase 1 — PR pre-check (multi-step perturbation)

Greedy generation from record[\"text\"] through step 30 with KV-cache. At steps 23-30, TWO forward passes are run per step: (a) the unperturbed baseline trajectory, and (b) the self-consistent perturbed trajectory with 0.50-L2 isotropic Gaussian noise applied to the current token's hidden state at layer 2. By step 30, the perturbed trajectory's trailing 24-token window contains 8 perturbed positions and 16 unperturbed positions.

Continue gate: perturbed PR must cross T_PR (10.954796) at step 30 (the accumulation point).

| step | PR base | PR perturbed | delta | n_pert/24 | pert > base | pert > T_PR |
|---|---|---|---|---|---|---|
| 20 | — | — | — | 0 | N | N |
| 21 | — | — | — | 0 | N | N |
| 22 | — | — | — | 0 | N | N |
| 23 | 5.2763 | 5.2849 | 0.0086 | 1 | Y | N |
| 24 | 5.2704 | 5.2896 | 0.0192 | 2 | Y | N |
| 25 | 5.2658 | 5.2947 | 0.0289 | 3 | Y | N |
| 26 | 5.2689 | 5.3068 | 0.0378 | 4 | Y | N |
| 27 | 5.2695 | 5.3178 | 0.0483 | 5 | Y | N |
| 28 | 5.2694 | 5.3294 | 0.0600 | 6 | Y | N |
| 29 | 5.3084 | 5.3716 | 0.0632 | 7 | Y | N |
| 30 | 5.3184 | 5.3901 | 0.0718 | 8 | Y | N |

Continue gate (perturbed PR > T_PR at step 30): **STOP**

### Phase-1 STOP (measured values reported)

The multi-step perturbation did NOT cross T_PR (10.954796) at step 30 (the accumulation point). The perturbed PR trajectory across the 8-step window rose steadily (perturbed PR range [5.2849, 5.3901] vs baseline PR at step 30 = 5.3184), reaching 5.3901 at step 30 — still far below T_PR. Even with 8 of 24 trailing-window positions carrying the 0.50-L2 perturbation, the injected energy in the minor singular-value components is insufficient to re-inflate the manifold across T_PR. Per the task specification, the perturbation is NOT adjusted; the human decides whether to test longer perturbation windows.

Per the task specification, the perturbation is NOT adjusted. The human decides whether to test longer perturbation windows. Phase 2 was not run.

## Notes / caveats

- MEASUREMENT ONLY: no thresholds, no actuation, no controller change [1]. This is a diagnostic probe; it offers no verdict.
- The perturbation is isotropic Gaussian noise scaled to 0.50 L2 (L2 norm 0.50, the AVG controller's L2-force convention), added to the current token's hidden state at layer 2. It adds energy to minor singular-value components, increasing PR by making the trailing window more isotropic.
- The perturbed branch is a SELF-CONSISTENT trajectory: its KV-cache and greedy token stream accumulate the perturbed states, so by step 30 the trailing 24-token window contains 8 perturbed positions and 16 unperturbed positions.
- The 24-token PR ring buffer is filled only during decoding steps (the layer-2 hook is registered after pre-fill), matching the night-010 / DS-034b / night-011 hook pattern.
- All runs seeded (SEED=42); seed provenance and engagement evidence are printed in stdout.
- night-011 Phase 0/1 data are REUSED from `docs/gate23/pr_ecs_causal_results.jsonl` (record selection and baseline PR). night-010 live L6H6 ECS is REUSED from `docs/gate23/live_ecs_results.jsonl`. Do NOT re-generate.
- The full machine-readable payload is in `multistep_perturbation_results.jsonl`.
