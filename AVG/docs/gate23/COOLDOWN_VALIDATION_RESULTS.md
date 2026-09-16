# DS-032 — Cooldown (5,-5) validation gate (MEASUREMENT + GATE)

> GATE REPORT. Pass/fail criteria are explicit and quoted from the
> DS-032 task file. A FAIL on any part is a SIGNAL to STOP and
> report — no tuning, no negotiation, no threshold modification [1].
> The human decides whether to accept a partial pass or adjust the
> cooldown target. No controller edit is made by this gate.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Setting under test | cooldown=5, penalty=-5.0 |
| Fixture A | 50 seeded heldout-degenerate v2 (random.Random(42).sample, sorted) |
| Fixture B | qwen_degenerate (100 records) |
| Fixture C | t2s_degenerate (100 records) |
| Decoding | Part 1: greedy; Part 2: do_sample=True temp=0.8; Part 3: greedy |
| max_new_tokens | 128 |
| Analysis window | trailing 24 generated positions |
| Dormant (Part 1) | REUSED from DS-028 (docs/gate23/layer_effect_persistent_results.jsonl, join on record_id) |
| Baselines (Part 1) | DS-030 Condition B (production combined (8,-5)) and Condition D (production (8,-5)) reused from docs/gate23/penalty_decomposition_results.jsonl |
| Dormant (Part 2) | LIVE (do_sample=True, temp=0.8, top_p=0.85) |
| Dormant (Part 3) | LIVE (greedy) for Fixtures B and C |
| top_p | 0.85 (controller.py:138,663) |
| bmm Triton override | deregistered |
| Wall clock (s) | 2098.3 |

## Determinism smoke

Smoke 1: one Fixture A record, Part 1 (combined, greedy),
generated twice on the same model load — token ids must match.

| record_id | condition | run A tokens | run B tokens | identical |
|---|---|---|---|---|
| 0 | combined (greedy) | 1 | 1 | True |
|  |  | ALL | IDENTICAL | True |

Smoke 2: one Fixture A record, Part 2 (suppression-only,
sampling), dormant generated twice with the same seed (seed=42) — token ids must match (sampling with a fixed seed is deterministic).

| record_id | condition | run A tokens | run B tokens | identical |
|---|---|---|---|---|
| 0 | sampling dormant | 128 | 128 | True |
|  |  | ALL | IDENTICAL | True |

## Part 1 — Combined path (suppression + kickstart, greedy)

Fixture A, 50 records, greedy (do_sample=False), max_new_tokens=128. Suppression (cooldown=5, penalty=-5.0) AND kickstart active. Production CTR sub-sampling (every_k=2 or diversity<0.40). Kickstart trigger: trailing_ctr<0.50 AND active_loop_ids -> counter=3, penalties -1e4/-5.0/-2.0 over 3 steps. top_p=0.85 when either mechanism is active. NO residual hooks.

**Pass criteria (Part 1):**

- Rescue count (#ΔD2>0) ≥ 48/50 (matches production combined).
- EOS rate ≤ 0.36 (must be BETTER than production B's 0.54).

Compared against DS-030 Condition B (production combined (8,-5), reused) and Condition D (production (8,-5) suppression-only, reused). Dormant is reused from DS-028.

| condition | n | ΔD2 mean | ΔD2 med | #ΔD2>0 | #byte-id | EOS rate | mean n_gen | n≥24 D2 | #n_gen=128 |
|---|---|---|---|---|---|---|---|---|---|
| dormant (DS-028) | 50 | 0.0000 | 0.0000 | 0 | 50 | 0.0000 | 128.00 | 0.1696 | 50 |
| production combined (8,-5) [DS-030 B] | 50 | 0.8070 | 0.8261 | 50 | 0 | 0.5400 | 64.30 | 0.9530 | 23 |
| production supp-only (8,-5) [DS-030 D] | 50 | 0.7435 | 0.7826 | 48 | 2 | 0.2800 | 111.86 | 0.9075 | 36 |
| live (5,-5) combined | 50 | 0.8000 | 0.8261 | 50 | 0 | 0.5000 | 64.86 | 0.9391 | 25 |

Live (5,-5) combined is measured in this process; the two DS-030 rows are reused from docs/gate23/penalty_decomposition_results.jsonl.

### Part 1 criterion results

| criterion | measured | op | threshold | result | note |
|---|---|---|---|---|---|
| P1 rescue count (#ΔD2>0) | 50.0000 | >= | 48.0000 | PASS | threshold 48/50 |
| P1 EOS rate | 0.5000 | <= | 0.3600 | FAIL | production combined (8,-5) EOS=0.54 |

**Part 1 result: FAIL**

## Part 2 — Sampling regime (suppression-only, do_sample=True)

Fixture A, 50 records, do_sample=True, temperature=0.8, top_p=0.85, max_new_tokens=128. Suppression-only (cooldown=5, penalty=-5.0). NO kickstart. NO residual hooks. Dormant baselines are generated LIVE in this probe (DS-028 was greedy, not sampling). Each generation is seeded (seed_all(SEED)) for deterministic sampling.

**Pass criteria (Part 2):**

- Rescue count (#ΔD2>0) ≥ 40/50 (lower bar than greedy).
- Mean ΔDistinct-2 > 0.50 (substantial rescue, not marginal noise).

| condition | n | ΔD2 mean | ΔD2 med | #ΔD2>0 | #byte-id | EOS rate | mean n_gen | n≥24 D2 | #n_gen=128 |
|---|---|---|---|---|---|---|---|---|---|
| dormant (live sampling) | 50 | 0.0000 | 0.0000 | 0 | 50 | 0.0000 | 128.00 | 0.1696 | 50 |
| live (5,-5) suppression-only | 50 | 0.7889 | 0.8261 | 50 | 0 | 0.2000 | 110.52 | 0.9746 | 40 |

### Part 2 criterion results

| criterion | measured | op | threshold | result | note |
|---|---|---|---|---|---|
| P2 rescue count (#ΔD2>0) | 50.0000 | >= | 40.0000 | PASS | threshold 40/50 |
| P2 mean ΔDistinct-2 | 0.7889 | > | 0.5000 | PASS | threshold 0.5 |

**Part 2 result: PASS**

## Part 3 — Cross-fixture (suppression-only, greedy)

Fixture B (qwen_degenerate, 100) and Fixture C (t2s_degenerate, 100). Greedy (do_sample=False), max_new_tokens=128. Suppression-only (cooldown=5, penalty=-5.0). NO kickstart, NO residual hooks. Dormant continuations are generated LIVE for every record (no pre-existing dormant data for these fixtures).

**Pass criteria (Part 3):**

- Fixture B (qwen_degenerate): rescue ≥ 85/100.
- Fixture C (t2s_degenerate): rescue ≥ 85/100.

| condition | n | ΔD2 mean | ΔD2 med | #ΔD2>0 | #byte-id | EOS rate | mean n_gen | n≥24 D2 | #n_gen=128 |
|---|---|---|---|---|---|---|---|---|---|
| B dormant (live greedy) | 100 | 0.0000 | 0.0000 | 0 | 100 | 0.0000 | 128.00 | 0.1535 | 100 |
| B live (5,-5) suppression-only | 100 | 0.4316 | 0.4783 | 68 | 30 | 0.4400 | 87.24 | 0.5176 | 56 |
| C dormant (live greedy) | 100 | 0.0000 | 0.0000 | 0 | 100 | 0.0100 | 126.75 | 0.2279 | 99 |
| C live (5,-5) suppression-only | 100 | 0.0869 | 0.0000 | 15 | 85 | 0.1100 | 118.46 | 0.2868 | 89 |

### Part 3 criterion results

| criterion | measured | op | threshold | result | note |
|---|---|---|---|---|---|
| P3B qwen_degenerate rescue (#ΔD2>0) | 68.0000 | >= | 85.0000 | FAIL | threshold 85/100 |
| P3C t2s_degenerate rescue (#ΔD2>0) | 15.0000 | >= | 85.0000 | FAIL | threshold 85/100 |

**Part 3 result: FAIL**

## Overall gate result

| Part | Result |
|---|---|
| Part 1 (combined, greedy) | FAIL |
| Part 2 (sampling regime) | PASS |
| Part 3 (cross-fixture) | FAIL |
| **Overall** | **FAIL** |

A FAIL was measured. This is a STOP signal, not a negotiation. No threshold was modified, no retry with different settings was performed. The human decides whether to accept a partial pass or adjust the cooldown target.

## Per-record JSONL

Per-record results (dormant plus live active continuations, deltas, penalty/kickstart activity incl. per-step logs, and the reused DS-030 baselines for Part 1) are in `cooldown_validation_results.jsonl`.

## Notes / caveats

- GATE: pass/fail criteria are explicit and asserted by the script. A red gate is a signal, not a negotiation [1].
- Part 1 dormant is REUSED from DS-028 (layer-independent; first layer per record joined on record_id). DS-030 Conditions B and D are REUSED for Fixture A comparisons (do NOT re-run).
- Part 2 dormant baselines are generated LIVE because DS-028 was greedy, not sampling. Each Part 2 generation is preceded by seed_all(SEED); the determinism smoke confirms sampling with a fixed seed is deterministic.
- Part 3 dormant baselines are generated LIVE (greedy) because no pre-existing dormant data exists for qwen_degenerate or t2s_degenerate. Fixture C (t2s_degenerate) has no `prompt` key; the record's degenerate `text` is used as the generation prompt (the task's `record[\"prompt\"]` maps to `text` for this fixture).
- Part 2 uses temperature=0.8 for BOTH dormant and active arms (task spec) to isolate the suppression mechanism. Production would use 0.70 while suppression is active; this gate follows the task's explicit 0.8 regime.
- The generation loop breaks on EOS; the EOS token is included in n_generated and generated_ids (ds-029/ds-030/ds-031 convention).
- No controller edit, no threshold change, no gate-script touch. This gate tests the PROPOSED setting only.
