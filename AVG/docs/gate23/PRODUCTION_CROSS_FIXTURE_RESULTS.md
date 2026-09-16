# DS-033 — Production (8,-5) instrumented cross-fixture diagnostic (MEASUREMENT ONLY)

> DIAGNOSTIC REPORT. NOT a pass/fail gate. There are no pass/fail criteria in the DS-033 task file. The goal is to classify every cross-fixture failure as detection-gap or actuation-gap per the locked triad diagnostic sequence.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Setting under test | cooldown=8, penalty=-5.0 (production) |
| Fixture B | qwen_degenerate (100 records), record[\"prompt\"] prompt |
| Fixture C | t2s_degenerate (100 records), record[\"text\"] prompt |
| Decoding | greedy (do_sample=False) |
| max_new_tokens | 128 |
| Analysis window | trailing 24 generated positions |
| Dormant | LIVE (greedy) for both fixtures |
| Active | production combined path (suppression + kickstart, cooldown=8, penalty=-5.0) |
| Shadow PR | DIAGNOSTIC ONLY — 55-85% layer span hook; does NOT participate in detection/actuation |
| top_p | 0.85 (controller.py:138,663) |
| bmm Triton override | deregistered |
| Wall clock (s) | 1810.3 |

## Determinism smoke

Smoke 1: one Fixture B record, dormant (greedy), generated twice on the same model load — token ids must match.

| record_id | condition | run A tokens | run B tokens | identical |
|---|---|---|---|---|
| 0 | dormant (greedy) | 1 | 1 | True |
|  |  | ALL | IDENTICAL | True |

Smoke 2: one Fixture B record, active (production combined, greedy), generated twice on the same model load — token ids must match.

| record_id | condition | run A tokens | run B tokens | identical |
|---|---|---|---|---|
| 0 | active (production combined) | 1 | 1 | True |
|  |  | ALL | IDENTICAL | True |

## Table 1 — Rescue metrics per fixture

| fixture | n | ΔD2 mean | #ΔD2>0 | #byte-id | EOS rate | mean n_gen | n≥24 D2 |
|---|---|---|---|---|---|---|---|
| B qwen_degenerate | 100 | 0.4653 | 68 | 30 | 0.4100 | 89.14 | 0.5637 |
| C t2s_degenerate | 100 | 0.0966 | 15 | 85 | 0.1000 | 119.21 | 0.2946 |

Dormant rows (live greedy) for reference:

| fixture | n | ΔD2 mean | #ΔD2>0 | #byte-id | EOS rate | mean n_gen | n≥24 D2 |
|---|---|---|---|---|---|---|---|
| B qwen_degenerate | 100 | 0.0000 | 0 | 100 | 0.0000 | 128.00 | 0.1535 |
| C t2s_degenerate | 100 | 0.0000 | 0 | 100 | 0.0100 | 126.75 | 0.2279 |

## Table 2 — Diagnostic classification per fixture

| class | condition |
|---|---|
| detection-gap | predicate_true_steps == 0 (predicate never fired) |
| actuation-gap | predicate_true_steps > 0 AND ΔDistinct-2 == 0 (predicate fired but no rescue) |
| rescued | ΔDistinct-2 > 0 (with n≥24 D2 ≥ 0.40) |
| mixed | predicate_true_steps > 0, ΔDistinct-2 > 0 but n_generated < 24 or n≥24 D2 < 0.40 |

| fixture | n | detection-gap | actuation-gap | rescued | mixed |
|---|---|---|---|---|---|
| B qwen_degenerate | 100 | 73 | 0 | 25 | 2 |
| C t2s_degenerate | 100 | 93 | 5 | 2 | 0 |

## Table 3 — Per-class predicate and suppression stats

| fixture | class | n | mean predicate_true_steps | mean suppression_steps | mean kickstart_activations | mean shadow PR |
|---|---|---|---|---|---|---|
| B | detection-gap | 73 | 0.0000 | 84.1781 | 0.0959 | 8.9852 |
| B | rescued | 25 | 1.1600 | 102.4400 | 1.1600 | 16.9846 |
| B | mixed | 2 | 1.0000 | 14.0000 | 1.0000 | nan |
| C | detection-gap | 93 | 0.0000 | 119.6882 | 0.0968 | 5.6281 |
| C | actuation-gap | 5 | 1.0000 | 103.0000 | 1.0000 | 3.6896 |
| C | rescued | 2 | 1.5000 | 125.5000 | 3.5000 | 19.0863 |

## Table 4 — Comparison with DS-032 (5,-5) reused data

DS-032 (5,-5) Part 3 cross-fixture rescue rates are REUSED from `docs/gate23/cooldown_validation_results.jsonl` (do NOT re-run). Both DS-033 (8,-5) and DS-032 (5,-5) are greedy; DS-032 was suppression-only, DS-033 is the production combined path.

| fixture | n | (8,-5) rescue rate | (5,-5) rescue rate | delta |
|---|---|---|---|---|
| B qwen_degenerate | 100 | 0.6800 | 0.6800 | 0.0000 |
| C t2s_degenerate | 100 | 0.1500 | 0.1500 | 0.0000 |

## Interpretation (diagnostic, not a verdict)

Per the locked triad decision rule:

- If **detection-gap** dominates → DS-034: test the existing spectral PR on the same fixture (zero new code). If PR fires where the bigram predicate does not → dual-predicate fix. If PR is also silent → open macro-syntax window design.
- If **actuation-gap** dominates → stay in actuator space (suppression design, kickstart steering). Do NOT open new detection architecture.
- **mixed** records are partial/weak recoveries: the predicate fired and ΔD2 moved, but the continuation is too short (n_generated < 24) or has low prose Distinct-2 (n≥24 D2 < 0.40).

The shadow PR column is DIAGNOSTIC ONLY. It does not participate in the collapse predicate or in actuation; it provides post-hoc spectral evidence for the triad to interpret.

## Per-record JSONL

Per-record results (dormant plus live active continuations, deltas, predicate/suppression/kickstart diagnostics, shadow PR, and per-step penalty logs) are in `production_cross_fixture_results.jsonl`.

## Notes / caveats

- MEASUREMENT ONLY: no threshold edits, no controller changes, no verdict on architecture direction. This is a diagnostic, not a gate.
- DS-032 (5,-5) cross-fixture data is REUSED for Table 4 (cooldown_validation_results.jsonl, part==3). Do NOT re-run.
- NEW dormant baselines are generated LIVE (greedy) for both fixtures.
- Fixture C (t2s_degenerate) has no `prompt` key; the record's degenerate `text` is used as the generation prompt (the task's `record[\"prompt\"]` maps to `text` for this fixture).
- The collapse predicate truth count uses the production sub-sampled CTR (trailing_ctr computed only when token_diversity < 0.40 or step % 2 == 0), matching controller.py generate(). Predicate = token_diversity < 0.30 AND trailing_ctr < 0.30 (use_code_filter=False).
- `predicate_true_without_suppression` counts predicate-true steps where the cooldown dict was empty or all entries had steps_left==0 (detection fired but actuator already exhausted).
- `mean_shadow_pr` is the mean Participation Ratio over the trailing 24 generated positions at each step where the buffer had >= 24 generated tokens, computed from a lightweight forward hook on the 55-85% depth span. Records that never reach 24 generated tokens have `mean_shadow_pr` = null.
- The generation loop breaks on EOS; the EOS token is included in n_generated and generated_ids (ds-029/ds-030/ds-031/ds-032 convention).
- `PROSE_D2_MIN = 0.40` is an operational definition of "low prose quality" for the mixed class only; it is NOT a gate threshold and does not affect any controller logic.
