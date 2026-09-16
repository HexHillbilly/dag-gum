# DS-031 — Token-suppression cooldown sweep

> MEASUREMENT REPORT. This document reports measured values only.
> No threshold is created or modified, no gate script is touched,
> no governor/controller.py edit, and no green/red verdict and no
> "this is the right setting" language are offered. Cooldown/penalty
> selection is the human's act.

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
| Dormant | REUSED from DS-028 (docs/gate23/layer_effect_persistent_results.jsonl, join on record_id) |
| Reused cell | (8,-5) REUSED from DS-030 Condition D (docs/gate23/penalty_decomposition_results.jsonl, join on record_id) |
| New cells | (3,-3), (3,-5), (5,-2), (5,-3), (5,-5) |
| top_p | 0.85 (controller.py:138,663) |
| bmm Triton override | deregistered |
| Wall clock (s) | 780.8 |

## Determinism smoke

One heldout-degenerate record, cell (5,-3), generated twice
on the same model load; generated token ids must match exactly.

| cell | run A tokens | run B tokens | identical |
|---|---|---|---|
| (5,-3) | 128 | 128 | True |
|  |  | ALL | True |

## Method summary

- Per record, per cell: one greedy generation from the record's
  mutated_prompt; dormant is reused from DS-028.
- Suppression-only logic is identical to DS-030 Condition D, only
  cooldown and penalty change per cell: repeated-token ids from
  compute_token_distinct_2_fast set/reset a cooldown dict entry to
  the cell cooldown; before argmax each entry with steps_left>0 is
  penalised by the cell penalty and decremented, else removed.
- top_p=0.85 when the suppression dict is non-empty after the
  decrement/removal step (controller.py:762; kickstart counter 0).
- NO kickstart, NO vocabulary cache, NO trailing_ctr computation.
- Distinct-2 on trailing 24 generated tokens (ds-025 convention);
  CTR is the coherent-token ratio of the decoded continuation.
- Δ vs dormant = cell − dormant (per record).

## Primary aggregation table

ΔDistinct-2 is the per-record delta vs the DS-028 dormant
continuation; #byte-id counts records whose decoded continuation is
byte-identical to dormant; EOS rate is the fraction of records that
terminated early (n_generated < max_new_tokens); n>=24 D2 is the
distinct-2 mean over records that reached the analysis window;
#n_gen=128 counts records that generated the full budget.

| cell | n | ΔD2 mean | ΔD2 med | #ΔD2>0 | #byte-id | EOS rate | mean n_gen | n≥24 D2 | #n_gen=128 |
|---|---|---|---|---|---|---|---|---|---|
| (3,-3) | 50 | 0.0600 | 0.0000 | 6 | 44 | 0.0000 | 128.00 | 0.2296 | 50 |
| (3,-5) | 50 | 0.7009 | 0.7826 | 48 | 2 | 0.2400 | 112.44 | 0.8622 | 38 |
| (5,-2) | 50 | 0.0017 | 0.0000 | 1 | 49 | 0.0000 | 128.00 | 0.1713 | 50 |
| (5,-3) | 50 | 0.0774 | 0.0000 | 6 | 44 | 0.0000 | 128.00 | 0.2470 | 50 |
| (5,-5) | 50 | 0.7374 | 0.7826 | 48 | 2 | 0.2000 | 114.56 | 0.9010 | 40 |
| (8,-5)* | 50 | 0.7435 | 0.7826 | 48 | 2 | 0.2800 | 111.86 | 0.9075 | 36 |

* (8,-5) is REUSED from DS-030 Condition D (not re-run).

## Continuation length distribution per cell

| cell | n_gen=1 | n_gen=2..23 | n_gen=24..127 | n_gen=128 | mean n_gen | median n_gen |
|---|---|---|---|---|---|---|
| (3,-3) | 0 | 0 | 0 | 50 | 128.00 | 128.00 |
| (3,-5) | 3 | 0 | 9 | 38 | 112.44 | 128.00 |
| (5,-2) | 0 | 0 | 0 | 50 | 128.00 | 128.00 |
| (5,-3) | 0 | 0 | 0 | 50 | 128.00 | 128.00 |
| (5,-5) | 3 | 0 | 7 | 40 | 114.56 | 128.00 |
| (8,-5)* | 3 | 0 | 11 | 36 | 111.86 | 128.00 |

## Distinct-2 for records reaching n>=24 vs n<24

Distinct-2 is measured over the trailing 24 generated tokens;
records with fewer than 24 generated tokens use the cold-start
fallback (Distinct-2 = 1.0 when n<4). Splitting by n>=24 / n<24
separates prose-recovery continuations from early-EOS continuations.

| cell | subset | n | d2 mean | d2 med |
|---|---|---|---|---|
| (3,-3) | n>=24 | 50 | 0.2296 | 0.1739 |
| (3,-3) | n<24 | 0 | nan | nan |
| (3,-5) | n>=24 | 47 | 0.8622 | 0.9565 |
| (3,-5) | n<24 | 3 | 1.0000 | 1.0000 |
| (5,-2) | n>=24 | 50 | 0.1713 | 0.1739 |
| (5,-2) | n<24 | 0 | nan | nan |
| (5,-3) | n>=24 | 50 | 0.2470 | 0.1739 |
| (5,-3) | n<24 | 0 | nan | nan |
| (5,-5) | n>=24 | 47 | 0.9010 | 0.9565 |
| (5,-5) | n<24 | 3 | 1.0000 | 1.0000 |
| (8,-5)* | n>=24 | 47 | 0.9075 | 0.9565 |
| (8,-5)* | n<24 | 3 | 1.0000 | 1.0000 |

## Records 19/20 canary

DS-030 showed suppression-only (8,-5) failed to rescue records 19
and 20 (both byte-identical to dormant under D); only the combined
B path broke their loops. This table reports, per cell, whether the
canary pair is rescued without kickstart (ΔD2 > 0). Whether kickstart
is still required for the 4% tail is a design decision for the human.

| record_id | cell | ΔD2 | byte-id? | n_gen | EOS? | rescued? |
|---|---|---|---|---|---|---|
| 19 | (3,-3) | 0.0000 | Y | 128 | N | N |
| 19 | (3,-5) | 0.0000 | Y | 128 | N | N |
| 19 | (5,-2) | 0.0000 | Y | 128 | N | N |
| 19 | (5,-3) | 0.0000 | Y | 128 | N | N |
| 19 | (5,-5) | 0.0000 | Y | 128 | N | N |
| 19 | (8,-5)* | 0.0000 | Y | 128 | N | N |
| 20 | (3,-3) | 0.0000 | Y | 128 | N | N |
| 20 | (3,-5) | 0.0000 | Y | 128 | N | N |
| 20 | (5,-2) | 0.0000 | Y | 128 | N | N |
| 20 | (5,-3) | 0.0000 | Y | 128 | N | N |
| 20 | (5,-5) | 0.0000 | Y | 128 | N | N |
| 20 | (8,-5)* | 0.0000 | Y | 128 | N | N |

## Suppression activity per cell

Aggregated over the 50 records of each cell. suppression_steps =
steps with at least one suppression penalty; n_suppression_applications
= total (token, step) penalty applications; mean cooldown occupancy =
mean number of ids in the cooldown dict at the end of each step.

| cell | supp steps | n supp appl | cooldown occupancy | top_p steps | max cooldown |
|---|---|---|---|---|---|
| (3,-3) | 127.78 | 520.32 | 4.07 | 127.78 | 4.56 |
| (3,-5) | 109.84 | 554.42 | 4.71 | 109.84 | 8.16 |
| (5,-2) | 127.96 | 498.48 | 3.89 | 127.96 | 3.98 |
| (5,-3) | 127.98 | 526.38 | 4.11 | 127.98 | 4.64 |
| (5,-5) | 112.06 | 567.96 | 4.78 | 112.06 | 8.26 |
| (8,-5)* | 110.26 | 619.30 | 5.35 | 110.26 | 9.16 |

## Per-record detail table for cell (5,-5)

Selection rule (display only, not a verdict): among cells with
#ΔD2>0 >= 48 (the (8,-5) rescue count), the cell with the lowest EOS
rate; ties broken by lowest mean suppression applications per record,
then lowest |penalty|, then lowest cooldown.

Selected cell: `(5,-5)`.

| record_id | n_gen | D2 | ΔD2 | CTR | ΔCTR | byte-id? | EOS? | rescued? | supp steps | supp appl | cooldown occ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 83 | 1.0000 | 0.8261 | 0.7377 | 0.4044 | N | Y | Y | 83 | 418 | 5.04 |
| 2 | 97 | 1.0000 | 0.8696 | 0.9310 | 0.2670 | N | Y | Y | 87 | 175 | 1.80 |
| 3 | 97 | 1.0000 | 0.8696 | 0.9310 | 0.2670 | N | Y | Y | 87 | 175 | 1.80 |
| 4 | 97 | 1.0000 | 0.8696 | 0.9310 | 0.2670 | N | Y | Y | 87 | 175 | 1.80 |
| 5 | 97 | 1.0000 | 0.8696 | 0.9310 | 0.2670 | N | Y | Y | 87 | 175 | 1.80 |
| 6 | 128 | 1.0000 | 0.8696 | 0.7368 | 0.0728 | N | N | Y | 126 | 719 | 5.62 |
| 11 | 1 | 1.0000 | 0.8696 | 0.0000 | -0.6641 | N | Y | Y | 1 | 3 | 3.00 |
| 13 | 1 | 1.0000 | 0.8696 | 0.0000 | -0.6641 | N | Y | Y | 1 | 3 | 3.00 |
| 14 | 128 | 0.9130 | 0.7826 | 0.7400 | 0.0759 | N | N | Y | 128 | 690 | 5.39 |
| 16 | 128 | 0.9130 | 0.7826 | 0.7400 | 0.0759 | N | N | Y | 128 | 690 | 5.39 |
| 17 | 128 | 0.9130 | 0.7826 | 0.7400 | 0.0759 | N | N | Y | 128 | 690 | 5.39 |
| 19 | 128 | 0.1304 | 0.0000 | 0.6641 | 0.0000 | Y | N | N | 128 | 384 | 3.00 |
| 20 | 128 | 0.1304 | 0.0000 | 0.6641 | 0.0000 | Y | N | N | 128 | 384 | 3.00 |
| 22 | 128 | 0.9565 | 0.8261 | 0.7174 | 0.3893 | N | N | Y | 128 | 539 | 4.21 |
| 24 | 128 | 0.7391 | 0.3913 | 0.1739 | -0.1594 | N | N | Y | 128 | 1347 | 10.52 |
| 25 | 67 | 1.0000 | 0.8696 | 0.4000 | -0.2641 | N | Y | Y | 67 | 342 | 5.10 |
| 27 | 67 | 1.0000 | 0.8696 | 0.4000 | -0.2641 | N | Y | Y | 67 | 342 | 5.10 |
| 28 | 128 | 0.9565 | 0.8261 | 0.8519 | 0.1878 | N | N | Y | 128 | 735 | 5.74 |
| 29 | 128 | 0.9565 | 0.8261 | 0.8519 | 0.1878 | N | N | Y | 128 | 735 | 5.74 |
| 31 | 128 | 0.9565 | 0.8261 | 0.8519 | 0.1878 | N | N | Y | 128 | 735 | 5.74 |
| 35 | 128 | 0.8261 | 0.6522 | 0.7636 | 0.4303 | N | N | Y | 128 | 1063 | 8.30 |
| 38 | 128 | 0.8261 | 0.6522 | 0.7636 | 0.4303 | N | N | Y | 128 | 1063 | 8.30 |
| 43 | 128 | 1.0000 | 0.8261 | 0.8333 | 0.5000 | N | N | Y | 128 | 578 | 4.52 |
| 46 | 128 | 1.0000 | 0.8261 | 0.8333 | 0.5000 | N | N | Y | 128 | 578 | 4.52 |
| 51 | 1 | 1.0000 | 0.8696 | 0.0000 | -0.6641 | N | Y | Y | 1 | 3 | 3.00 |
| 53 | 128 | 0.9130 | 0.7391 | 0.4651 | -0.2016 | N | N | Y | 128 | 790 | 6.17 |
| 54 | 128 | 0.9130 | 0.7391 | 0.4651 | -0.2016 | N | N | Y | 128 | 790 | 6.17 |
| 57 | 128 | 0.8261 | 0.6522 | 0.5205 | -0.1461 | N | N | Y | 128 | 1236 | 9.66 |
| 58 | 128 | 0.8261 | 0.6522 | 0.5205 | -0.1461 | N | N | Y | 128 | 1236 | 9.66 |
| 62 | 128 | 0.9565 | 0.7826 | 0.5263 | -0.1404 | N | N | Y | 128 | 808 | 6.31 |
| 64 | 128 | 0.9565 | 0.7826 | 0.6979 | 0.3646 | N | N | Y | 128 | 649 | 5.07 |
| 67 | 128 | 0.9130 | 0.7391 | 0.7179 | 0.0513 | N | N | Y | 118 | 410 | 3.20 |
| 68 | 128 | 0.9130 | 0.7391 | 0.7179 | 0.0513 | N | N | Y | 118 | 410 | 3.20 |
| 69 | 128 | 0.9130 | 0.7391 | 0.7179 | 0.0513 | N | N | Y | 118 | 410 | 3.20 |
| 71 | 128 | 0.7391 | 0.5652 | 0.6111 | -0.0556 | N | N | Y | 128 | 813 | 6.35 |
| 75 | 128 | 0.9565 | 0.7826 | 0.5362 | -0.1304 | N | N | Y | 128 | 565 | 4.41 |
| 77 | 128 | 1.0000 | 0.8261 | 0.8661 | 0.1994 | N | N | Y | 124 | 318 | 2.48 |
| 79 | 128 | 1.0000 | 0.8261 | 0.8661 | 0.1994 | N | N | Y | 124 | 318 | 2.48 |
| 81 | 128 | 1.0000 | 0.8261 | 0.9316 | 0.2650 | N | N | Y | 128 | 387 | 3.02 |
| 82 | 128 | 1.0000 | 0.8261 | 0.9316 | 0.2650 | N | N | Y | 128 | 387 | 3.02 |
| 84 | 128 | 0.5652 | 0.3478 | 0.7660 | 0.4413 | N | N | Y | 128 | 970 | 7.58 |
| 86 | 128 | 0.9565 | 0.7391 | 0.8750 | 0.5503 | N | N | Y | 116 | 506 | 3.95 |
| 88 | 128 | 0.9565 | 0.7391 | 0.8750 | 0.5503 | N | N | Y | 116 | 506 | 3.95 |
| 89 | 128 | 1.0000 | 0.7391 | 0.6404 | 0.3174 | N | N | Y | 128 | 703 | 5.49 |
| 90 | 128 | 0.9565 | 0.7391 | 0.8554 | 0.5307 | N | N | Y | 123 | 540 | 4.22 |
| 93 | 128 | 1.0000 | 0.7826 | 0.8491 | 0.5244 | N | N | Y | 125 | 401 | 3.13 |
| 94 | 128 | 1.0000 | 0.8261 | 0.8333 | 0.5000 | N | N | Y | 128 | 578 | 4.52 |
| 95 | 128 | 0.8696 | 0.6522 | 0.7128 | 0.0504 | N | N | Y | 127 | 1006 | 7.86 |
| 97 | 128 | 1.0000 | 0.7826 | 0.8491 | 0.5244 | N | N | Y | 122 | 460 | 3.59 |
| 99 | 128 | 1.0000 | 0.7826 | 0.8491 | 0.5244 | N | N | Y | 122 | 460 | 3.59 |

## Per-record JSONL

Per-record results (dormant plus live cells, deltas, suppression
activity incl. per-step logs) are in `cooldown_sweep_results.jsonl`.

## Notes / caveats

- MEASUREMENT ONLY. No verdict, no mechanism story, no "right setting".
- Cell (8,-5) is REUSED from DS-030 Condition D; it is not re-run.
  Its suppression activity is reconstructed from the DS-030 D
  per-step logs (mean cooldown occupancy computed from per-step
  n_active_cooldown).
- New cells are run live in this process on the same model load;
  determinism smoke (cell (5,-3)) passed before the measurement loop.
- Dormant data is reused from DS-028 and is layer-independent;
  the first layer entry per record was joined on record_id.
- The generation loop breaks on EOS; the EOS token is included in
  n_generated and generated_ids (ds-029/ds-030 convention).
