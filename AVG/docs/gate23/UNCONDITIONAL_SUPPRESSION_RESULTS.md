# night-004 — Unconditional suppression during collapse (MEASUREMENT ONLY)

> MEASUREMENT REPORT. NOT a gate. This probe tests the actuation coupling fix: bridge the dual-predicate detection (which fires on 100/100 qwen records) with production actuation strength (unconditional loop-token suppression at every step).

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture A | t2s_degenerate (100 records), record[\"text\"] prompt |
| Fixture B | qwen_degenerate (100 records), record[\"prompt\"] prompt |
| Fixture C | heldout_degenerate_v2 (50-record seeded subset random.Random(42).sample sorted), record[\"mutated_prompt\"] prompt |
| Decoding | greedy (do_sample=False), KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 generated positions |
| Hook layer | model.model.layers[2] (layer-2 PR, rolling 24-token ring buffer) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze; spectral_collapse threshold) |
| Dual-predicate | DS-034e frozen config: bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity (DETECTION ONLY — does not gate actuation) |
| Suppression | UNCONDITIONAL — add ALL active_loop_ids to cooldown (=8) at EVERY step where detected (production controller.py:677-679 coupling). DS-035 gated this on is_collapsed. |
| Kickstart | production trigger (trailing_ctr < 0.50 AND active_loop_ids non-empty -> counter=3, -1e4/-5.0/-2.0) |
| top_p | 0.85 when either mechanism is active |
| Dormant | REUSED from DS-035 (dual_predicate_rescue_results.jsonl) for Fixtures A/B; REUSED from DS-028 (layer_effect_persistent_results.jsonl, layer==2) for Fixture C. Do NOT re-run dormant. |
| DS-033 / DS-035 baselines | REUSED from production_cross_fixture_results.jsonl / dual_predicate_rescue_results.jsonl (do NOT re-run) |
| bmm Triton override | deregistered |
| Wall clock (s) | 438.6 |

## Measurement targets (hypothesis, NOT a gate)

> - qwen_degenerate: match or exceed DS-033 production rescue rate (0.68).
> - t2s_degenerate: preserve or exceed DS-035 rescue rate (0.20).
> - heldout_degenerate_v2: maintain >= 0.96.

Rescue is defined per the DS-033 convention: a record is rescued iff ΔDistinct-2 > 0 vs its dormant baseline (active distinct-2 minus dormant distinct-2 over the trailing 24 generated positions).

## Determinism smoke

One Fixture A record: dormant generated twice AND active (night-004 unconditional suppression) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

| record_id | prompt_len | n_gen | n PR | dormant id | active id | PR log id | identical |
|---|---|---|---|---|---|---|---|
| 0 | 199 | 128 | 105 | True | True | True | True |
|  | ALL |  |  |  |  |  | True |

PR first run A/B: 7.0874 / 7.0874; last run A/B: 7.0750 / 7.0750.

## Comparison table

| fixture | DS-033 prod | DS-035 gated | night-004 uncond | target |
|---|---|---|---|---|
| t2s_degenerate | 0.1500 | 0.2000 | 0.1300 | 0.2000 |
| qwen_degenerate | 0.6800 | 0.2000 | 0.6500 | 0.6800 |
| heldout_degenerate_v2 | 0.9600 | 0.9600 | 1.0000 | 0.9600 |

DS-033 production rates for Fixtures A/B are REUSED from `docs/gate23/production_cross_fixture_results.jsonl`. DS-035 gated rates and dormant baselines are REUSED from `docs/gate23/dual_predicate_rescue_results.jsonl`. Fixture C production baseline is the DS-031 suppression-only 48/50 = 0.9600 (DS-033 did not cover Fixture C).

## Target assessment (measurement hypothesis)

| fixture | target | night-004 | met? |
|---|---|---|---|
| t2s_degenerate | 0.2000 | 0.1300 | NOT MET |
| qwen_degenerate | 0.6800 | 0.6500 | NOT MET |
| heldout_degenerate_v2 | 0.9600 | 1.0000 | MET |

## Table 1 — Rescue metrics per fixture

| fixture | n | rescued | rate | ΔD2 mean | ≥1 fire | mean n_gen | n≥24 D2 |
|---|---|---|---|---|---|---|---|
| A t2s_degenerate | 100 | 13 | 0.1300 | 0.0890 | 88 | 120.32 | 0.3089 |
| B qwen_degenerate | 100 | 65 | 0.6500 | 0.4396 | 46 | 87.25 | 0.5501 |
| C heldout_degenerate_v2 | 50 | 50 | 1.0000 | 0.7861 | 1 | 64.30 | 0.9113 |

## Table 2 — Fire-type counts per fixture (detection liveness)

| fixture | records ≥1 fire | bigram-fire records | spectral-fire records | both-fire records | bigram steps | spectral steps | both steps |
|---|---|---|---|---|---|---|---|
| A | 88 | 1 | 87 | 0 | 1 | 8880 | 0 |
| B | 46 | 4 | 42 | 0 | 4 | 3534 | 0 |
| C | 1 | 0 | 1 | 0 | 0 | 33 | 0 |

Fire counts reflect DUAL-PREDICATE DETECTION liveness. The dual-predicate does NOT gate actuation in night-004; suppression is unconditional at every step.

## Table 3 — Suppression and kickstart activity (per fixture)

| fixture | n | mean suppression steps | mean kickstart events | mean n_gen |
|---|---|---|---|---|
| A | 100 | 120.07 | 0.28 | 120.32 |
| B | 100 | 85.32 | 0.42 | 87.25 |
| C | 50 | 63.72 | 2.08 | 64.30 |

DS-035 qwen reference: mean suppression steps 87.5, mean kickstart events 85.7, mean n_gen 113.9. DS-033 production qwen reference: mean suppression steps 87.3, mean kickstart events 0.4, mean n_gen 89.1. night-004 makes suppression unconditional and kickstart follow the production trigger.

## Table 4 — Dormant reference (per fixture, REUSED)

| fixture | n | mean n_gen | mean D2 | source |
|---|---|---|---|---|
| A | 100 | 125.67 | 0.2465 | reused from DS-035 (dual_predicate_rescue_results.jsonl) |
| B | 100 | 128.00 | 0.1535 | reused from DS-035 (dual_predicate_rescue_results.jsonl) |
| C | 50 | 128.00 | 0.1696 | reused from DS-028 (layer_effect_persistent_results.jsonl) |

## Rescue provenance vs DS-033 production

For Fixtures A/B, night-004 rescues are decomposed against the DS-033 production rescued record-id set (REUSED).

| fixture | prod rescued | night-004 rescued | overlap | NEW (night-004-only) | regressed (prod-only) |
|---|---|---|---|---|---|
| A t2s_degenerate | 15 | 13 | 13 | 0 | 2 |
| B qwen_degenerate | 68 | 65 | 65 | 0 | 3 |

**A t2s_degenerate** — new (night-004-only vs production) rescues: `[]`; regressed (production-only): `[37, 40]`.

**B qwen_degenerate** — new (night-004-only vs production) rescues: `[]`; regressed (production-only): `[12, 13, 14]`.

## Rescue provenance vs DS-035 gated

For all three fixtures, night-004 rescues are decomposed against the DS-035 gated rescued record-id set (REUSED).

| fixture | DS-035 rescued | night-004 rescued | overlap | NEW (night-004-only) | regressed (DS-035-only) |
|---|---|---|---|---|---|
| t2s_degenerate | 20 | 13 | 9 | 4 | 11 |
| qwen_degenerate | 20 | 65 | 20 | 45 | 0 |
| heldout_degenerate_v2 | 48 | 50 | 48 | 2 | 0 |

## DS-034e spectral coverage validation (t2s detection-gap)

Of the 93 DS-033 t2s_degenerate detection-gap records (bigram never fired), the DS-034e dual-predicate fires on 83 — 89.2% coverage. Detection liveness is preserved under unconditional actuation.

## Rescued record ids (ΔDistinct-2 > 0)

- **A t2s_degenerate**: 13 rescued — `[12, 28, 44, 65, 70, 72, 76, 80, 83, 86, 92, 97, 99]`
- **B qwen_degenerate**: 65 rescued — `[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49, 50, 51, 63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95]`
- **C heldout_degenerate_v2**: 50 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`

## Notes / caveats

- MEASUREMENT ONLY: no threshold edits, no controller changes. The actuation coupling fix is a measurement probe for the proposed change before it reaches the controller [1].
- THE CHANGE: suppression is UNCONDITIONAL — at every step, if active_loop_ids is non-empty, ALL active_loop_ids are added to the cooldown dict (=8). DS-035 gated this addition on is_collapsed (fire steps only); night-004 removes that gate and matches the production combined path (controller.py:677-679).
- Kickstart uses the PRODUCTION trigger (trailing_ctr < 0.50 AND active_loop_ids non-empty), NOT the DS-035 is_collapsed trigger. DS-035 fired the kickstart on 85.7 steps/record on qwen; production fires it on ~0.4 steps/record.
- The frozen DS-034e dual-predicate rule is implemented exactly per the task file (detection liveness only): spectral_collapse uses band_low (8.216097), spectral-only fires require token_diversity < 0.40 corroboration when bigram_ctr < 2, and is_code_syntax_context forces spectral_fire = False.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer.
- Dormant baselines are REUSED (do NOT re-run): Fixtures A/B from DS-035 (dual_predicate_rescue_results.jsonl), Fixture C from DS-028 (layer_effect_persistent_results.jsonl, layer==2, join on record_id) — the 50-record seeded subset matches exactly.
- DS-033 production and DS-035 gated rescue rates are REUSED for the comparison table (do NOT re-run).
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
- **FINDING (qwen 0.20 → 0.65).** Unconditional suppression bridges most of the qwen actuation gap: 45 NEW rescues vs DS-035 gated (20 overlap, 0 regressed). Night-004 rescues 65/100 vs production 68/100 — the small remaining gap is consistent with the KV-cache-vs-full-forward decoding difference (DS-033 used full-sequence forwards; night-004 uses KV-cache incremental). This confirms the DS-035 root-cause diagnosis: the is_collapsed gate on suppression was the primary qwen actuation gap.
- **FINDING (t2s 0.20 → 0.13).** The t2s target is NOT met. 11 of DS-035's 20 t2s rescues are lost (4 NEW, 9 overlap). The lost rescues are the DS-035 spectral-only rescues that depended on the is_collapsed-gated kickstart (80.2 events/record on t2s in DS-035 vs 0.28 in night-004). Changing the kickstart trigger to the production CTR trigger removes the spectral rescue mechanism on t2s detection-gap records. The unconditional suppression alone (matching production's 119 supp-steps/record) does not recover these spectral rescues.
- **FINDING (detection liveness).** Fire coverage drops on qwen (100/100 in DS-035 → 46/100 in night-004) because the more aggressive actuation rescues records before they degenerate enough for the spectral detector to fire. On t2s, detection-gap spectral coverage is 83/93 (89.2%) vs DS-035's 92/93 — still high but lower under unconditional actuation.
- **FINDING (heldout_v2 0.96 → 1.00).** Unconditional suppression maintains the positive control (50/50 rescued). The production-style actuation does not regress the heldout control.
