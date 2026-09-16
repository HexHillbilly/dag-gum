# night-011 — Causal probe: does PR compression cause L6H6 attention detachment? (MEASUREMENT ONLY)

> MEASUREMENT REPORT. This probe tests the FORWARD direction of the macro-loop dynamics model: if you artificially increase PR (re-inflate the manifold) at step 23, does L6H6 ECS rise in response? One record, one intervention, one diagnostic. Existence proof only — not general causation. PURELY DIAGNOSTIC: no thresholds, no actuation, no controller change. No verdict is offered; the human interprets the result.

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
| Intervention point | step 23 (first step where the 24-token ring buffer is full for both PR and ECS measurement) |
| Frozen T_PR(2) | 10.954796 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |
| ECS head | L6H6 (night-010 live ECS reference) |
| Selected record | 5 (auto-selected by Phase-0 ranking scan on night-010 JSONL) |
| night-010 JSONL | REUSED from docs/gate23/live_ecs_results.jsonl (per-step PR and L6H6 ECS); do NOT re-generate |
| bmm Triton override | deregistered |
| Wall clock (s) | 0.0 |

## Phase 0 — Automated record selection

Records ranked by composite = 0.4 × stability + 0.4 × severity + 0.2 × pattern. PR and L6H6 ECS are extracted from the night-010 JSONL for steps 20-40. PR is None at steps 20-23 in the night-010 JSONL (24-token ring-buffer warm-up); the joint window used for both PR and ECS statistics is steps 24-40.

Stability = 1.0 − sqrt((std(PR)/1.0)² + (std(ECS)/0.05)²). Severity = 1.0 − |mean(PR) − 5.5|/5.0, clamped to [0, 1]. Pattern = 1.0 if DS-033 detection-gap AND generated text has structural delimiters; 0.5 if detection-gap only; 0.0 otherwise.

| record_id | mean PR | std PR | mean ECS | std ECS | stability | severity | pattern | composite |
|---|---|---|---|---|---|---|---|---|
| 5 | 5.3017 | 0.0240 | 0.1126 | 0.0318 | 0.3626 | 0.9603 | 0.5 | 0.6292 |
| 7 | 5.2284 | 0.0063 | 0.0826 | 0.0355 | 0.2901 | 0.9457 | 0.5 | 0.5943 |
| 3 | 2.6093 | 0.0320 | 0.0512 | 0.0122 | 0.7544 | 0.4219 | 0.5 | 0.5705 |
| 2 | 2.2723 | 0.0400 | 0.0939 | 0.0134 | 0.7299 | 0.3545 | 0.5 | 0.5337 |
| 0 | 7.6789 | 0.0286 | 0.1151 | 0.0430 | 0.1396 | 0.5642 | 0.5 | 0.3815 |
| 6 | 6.5367 | 0.0674 | 0.1150 | 0.0614 | -0.2298 | 0.7927 | 0.5 | 0.3252 |
| 9 | 3.2249 | 0.0394 | 0.1247 | 0.0524 | -0.0491 | 0.5450 | 0.5 | 0.2984 |
| 8 | 3.8246 | 0.0542 | 0.1275 | 0.0598 | -0.1968 | 0.6649 | 0.5 | 0.2872 |
| 1 | 4.1940 | 0.0186 | 0.1558 | 0.0925 | -0.8507 | 0.7388 | 0.5 | 0.0553 |
| 4 | 7.2018 | 0.0409 | 0.2306 | 0.1001 | -1.0024 | 0.6596 | 0.5 | -0.0371 |

**Selected record: 5** (composite 0.6292). Rationale: Top-ranked record 5 has the highest composite (0.6292). It is a DS-033 detection-gap record (predicate_true_steps == 0) with no structural delimiters in its generated text (pattern=0.5). Its mean PR (5.3017) and stability (0.3626) / severity (0.9603) combination outranks the other 9 records.

## Phase 1 — PR pre-check (determinism smoke)

Greedy generation from record[\"text\"] to step 25. At steps 23, 24, 25, two forward passes are run from the same KV-cache state: (a) unperturbed baseline PR, (b) perturbed PR after applying isotropic Gaussian noise at 0.50 L2 to the current token's hidden state at layer 2. Continue to Phase 2 only if the perturbed PR exceeds the baseline AND crosses T_PR (10.954796) at all three steps.

| step | PR base | PR perturbed | delta | pert > base | pert > T_PR |
|---|---|---|---|---|---|
| 23 | 5.2763 | 5.2849 | 0.0086 | Y | N |
| 24 | 5.2704 | 5.2811 | 0.0106 | Y | N |
| 25 | 5.2658 | 5.2755 | 0.0097 | Y | N |

Continue gate (all 3 steps: pert > base AND pert > T_PR): **STOP**

### Phase-1 STOP (measured values reported)

The perturbation did NOT cross T_PR (10.954796) at all three steps. PR moved only marginally (perturbed PR range [5.2755, 5.2849] vs baseline range [5.2658, 5.2763]). With a single-token 0.50-L2 isotropic-noise injection at layer 2, the trailing 24-token window gains too little energy in the minor singular-value components to re-inflate the manifold across T_PR. Per the task specification, the perturbation is NOT adjusted; the human decides whether to test multi-step perturbation.

Per the task specification, the perturbation is NOT adjusted. The human decides whether to test multi-step perturbation. Phase 2 was not run.

## Notes / caveats

- MEASUREMENT ONLY: no thresholds, no actuation, no controller change [1]. This is a diagnostic probe; it offers no verdict.
- The perturbation is isotropic Gaussian noise scaled to 0.50 L2 (L2 norm 0.50, the AVG controller's L2-force convention), added to the current token's hidden state at layer 2. It adds energy to minor singular-value components, increasing PR by making the trailing window more isotropic.
- The 24-token PR ring buffer is filled only during decoding steps (the layer-2 hook is registered after pre-fill), matching the night-010 / DS-034b hook pattern.
- All runs seeded (SEED=42); seed provenance and engagement evidence are printed in stdout.
- Per-step PR and L6H6 ECS for the ranking scan are REUSED from `docs/gate23/live_ecs_results.jsonl` (night-010). Do NOT re-generate.
- The full machine-readable payload is in `pr_ecs_causal_results.jsonl`.
