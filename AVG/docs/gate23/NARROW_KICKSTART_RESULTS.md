# night-007 — Narrow kickstart trigger: spectral_fire alone (GATE)

> GATE REPORT. night-006 FAIL: t2s_degenerate = 0.13 < 0.15 because the three-way kickstart gate `spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids` never tripped. This probe tests the NARROW fix — kickstart on `spectral_fire` alone — implementing the exact `diagnose()` / `generate()` changes proposed for `governor/controller.py` BEFORE they reach the controller. Targets are explicit, pass/fail is binary, a red gate is a STOP signal [1].

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
| Prose-100 | valid_subset_200 heldout partition (N=100; ds-027 complement of the Gate-2.2 control-100), record[\"text\"] prompt |
| Decoding | greedy (do_sample=False), KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 generated positions |
| Hook layer | model.model.layers[2] (layer-2 PR, rolling 24-token ring buffer) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze; spectral_collapse threshold) |
| Dual-predicate | DS-034e frozen config (proposed diagnose(); UNCHANGED from night-006): bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity |
| Suppression | UNCHANGED from production — UNCONDITIONAL: add ALL active_loop_ids to cooldown (=8) at EVERY step where detected (controller.py:677-679 coupling). |
| Kickstart | FIXED (night-007): spectral_fire ALONE -> counter=3 (-1e4/-5.0/-2.0). The trailing_ctr < 0.50 AND active_loop_ids conditions that killed night-006 are REMOVED. |
| top_p | 0.85 when either mechanism is active |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| Dormant | REUSED: Fixtures A/B from DS-035 (dual_predicate_rescue_results.jsonl); Fixture C from DS-028 (layer_effect_persistent_results.jsonl, layer==2); prose-100 from DS-034c (dual_predicate_fp_results.jsonl, fallback_hardened). Do NOT re-run dormant. |
| night-006 baseline | REUSED from controller_integration_results.jsonl (do NOT re-run) |
| DS-035 gated baseline | REUSED from dual_predicate_rescue_results.jsonl (do NOT re-run) |
| bmm Triton override | deregistered |
| Wall clock (s) | 775.3 |

## Gate criteria (quoted from the night-007 task file)

| fixture | target | rationale |
|---|---|---|
| t2s_degenerate | >= 0.15 | match production; spectral rescues expected |
| qwen_degenerate | >= 0.65 | match night-004 unconditional suppression |
| heldout_degenerate_v2 | >= 0.96 | match all prior bests |
| prose-100 FPs | <= 7 | must not exceed ratified FP policy |

Rescue is defined per the DS-033 convention: a record is rescued iff ΔDistinct-2 > 0 vs its dormant baseline. FP = ΔDistinct-2 > 0 AND the dual-predicate fired at least once (DS-034c / RFC-004 A3 convention).

## Determinism smoke

One Fixture A record AND one prose-100 record: active (night-007 narrow kickstart) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

| record_id | prompt_len | n_gen | n PR | dormant id | active id | PR log id | identical |
|---|---|---|---|---|---|---|---|
| 0 | 199 | 128 | 105 | True | True | True | True |
|  | ALL |  |  |  |  |  | True |

PR first run A/B: 7.0874 / 7.0874; last run A/B: 7.0750 / 7.0750.

Prose-100 determinism smoke:

| corpus_id | prompt_len | n_gen | n PR | active id | PR log id | identical |
|---|---|---|---|---|---|---|
| 3 | 248 | 128 | 105 | True | True | True |

PR first run A/B: 19.0569 / 19.0569; last run A/B: 19.0175 / 19.0175.

## Gate assessment

| fixture | target | night-007 | met? |
|---|---|---|---|
| t2s_degenerate | 0.1500 | 0.2400 | MET |
| qwen_degenerate | 0.6500 | 0.6100 | NOT MET |
| heldout_degenerate_v2 | 0.9600 | 1.0000 | MET |
| prose-100 FPs | <= 7 | 0 | MET |

**Gate verdict: FAIL**

**FAIL** — one or more targets not met. Red gate is a STOP signal; no thresholds or gates are negotiated. The data characterizes what needs adjustment before wiring.

## Comparison table (vs night-006)

| fixture | night-006 | night-007 | target | met? |
|---|---|---|---|---|
| t2s_degenerate | 0.1300 | 0.2400 | 0.1500 | MET |
| qwen_degenerate | 0.6500 | 0.6100 | 0.6500 | NOT MET |
| heldout_degenerate_v2 | 0.9600 | 1.0000 | 0.9600 | MET |

night-006 rates are REUSED from `docs/gate23/controller_integration_results.jsonl` (do NOT re-run). DS-035 gated rates are shown for reference in the provenance sections.

## Table 1 — Rescue metrics per fixture

| fixture | n | rescued | rate | ΔD2 mean | ≥1 fire | mean n_gen | n≥24 D2 |
|---|---|---|---|---|---|---|---|
| A t2s_degenerate | 100 | 24 | 0.2400 | 0.0895 | 89 | 109.31 | 0.3021 |
| B qwen_degenerate | 100 | 61 | 0.6100 | 0.4182 | 48 | 92.79 | 0.5305 |
| C heldout_degenerate_v2 | 50 | 50 | 1.0000 | 0.7061 | 37 | 110.56 | 0.8677 |

## Table 2 — Fire-type counts per fixture (detection liveness)

| fixture | records ≥1 fire | bigram-fire records | spectral-fire records | both-fire records | bigram steps | spectral steps | both steps |
|---|---|---|---|---|---|---|---|
| A | 89 | 2 | 87 | 0 | 4 | 7730 | 0 |
| B | 48 | 4 | 44 | 0 | 8 | 3524 | 0 |
| C | 37 | 34 | 3 | 0 | 37 | 7 | 0 |

Fire counts reflect DUAL-PREDICATE DETECTION liveness. The dual-predicate does NOT gate suppression; suppression is unconditional at every step. The dual-predicate gates ONLY the spectral kickstart (now spectral_fire alone).

## Table 3 — Suppression and kickstart activity (per fixture)

| fixture | n | mean suppression steps | mean kickstart events | mean n_gen |
|---|---|---|---|---|
| A | 100 | 109.16 | 77.30 | 109.31 |
| B | 100 | 91.58 | 35.24 | 92.79 |
| C | 50 | 109.30 | 0.14 | 110.56 |

night-006 kickstart events were 0.00 on every fixture (the three-way gate never tripped). night-007 kickstart on spectral_fire alone is expected to be non-zero on the degenerate fixtures (Table 6 / kickstart analysis below).

## Table 4 — Dormant reference (per fixture, REUSED)

| fixture | n | mean n_gen | mean D2 | source |
|---|---|---|---|---|
| A | 100 | 125.67 | 0.2465 | reused from DS-035 (dual_predicate_rescue_results.jsonl) |
| B | 100 | 128.00 | 0.1535 | reused from DS-035 (dual_predicate_rescue_results.jsonl) |
| C | 50 | 128.00 | 0.1696 | reused from DS-028 (layer_effect_persistent_results.jsonl) |

## Table 5 — Prose-100 FP verification

| metric | value |
|---|---|
| Records | 100 |
| False positives (ΔD2>0 AND fire>0) | 0 (0.00%) |
| FP record_ids | [] |
| FP corpus_ids | [] |
| Any ΔD2>0 (regardless of fire) | 52 |
| Records with ≥1 fire | 0 |
| Total fire steps | 0 |
| Byte-identical to dormant | 8 |
| Mean suppression steps | 118.12 |
| Mean kickstart events | 0.00 |
| ΔDistinct-2 mean / median / p90 | 0.1228 / 0.0435 / 0.4348 |

The ratified RFC-004 A3 FP policy documents 7 FPs (5 bigram cold-start: 47, 49, 76, 83, 87; 2 spectral late-dip: 98, 99). The gate requires the night-007 probe to not exceed 7 FPs under the same criterion. Unconditional suppression does not cause false positives on healthy prose because healthy prose does not generate abundant repeated bigrams; the spectral-gated kickstart is dormant on prose-100 because spectral PR stays above band_low (mean PR ~18 vs band_low 8.22).

## Rescue provenance vs DS-035 gated

For all three fixtures, night-007 rescues are decomposed against the DS-035 gated rescued record-id set (REUSED). DS-035 gated kickstart on is_collapsed (bigram OR spectral); night-007 gates on spectral_fire alone.

| fixture | DS-035 rescued | night-007 rescued | overlap | NEW (night-007-only) | regressed (DS-035-only) |
|---|---|---|---|---|---|
| t2s_degenerate | 20 | 24 | 20 | 4 | 0 |
| qwen_degenerate | 20 | 61 | 20 | 41 | 0 |
| heldout_degenerate_v2 | 48 | 50 | 48 | 2 | 0 |

## Rescue provenance vs night-006

For all three fixtures, night-007 rescues are decomposed against the night-006 rescued record-id set (REUSED from controller_integration_results.jsonl).

| fixture | night-006 rescued | night-007 rescued | overlap | NEW (night-007-only) | regressed (night-006-only) |
|---|---|---|---|---|---|
| t2s_degenerate | 13 | 24 | 13 | 11 | 0 |
| qwen_degenerate | 65 | 61 | 61 | 0 | 4 |
| heldout_degenerate_v2 | 48 | 50 | 48 | 2 | 0 |

## DS-034e spectral coverage validation (t2s detection-gap)

Of the 93 DS-033 t2s_degenerate detection-gap records (bigram never fired), the DS-034e dual-predicate fires on 83 — 89.2% coverage.

## Table 6 — Kickstart analysis (spectral_fire alone)

night-007 fires kickstart on spectral_fire ALONE, so n_kickstart_events == spectral_fire steps by construction. The `with trailing_ctr < 0.50` column shows how many of those steps WOULD have satisfied the night-006 gate (the condition that blocked night-006).

| fixture | spectral_fire steps | with trailing_ctr < 0.50 | kickstart events |
|---|---|---|---|
| t2s_degenerate | 7730 | 0 | 7730 |
| qwen_degenerate | 3524 | 0 | 3524 |
| heldout_degenerate_v2 | 7 | 0 | 7 |
| TOTAL | 11261 | 0 | 11261 |

## Rescued record ids (ΔDistinct-2 > 0)

- **A t2s_degenerate**: 24 rescued — `[12, 17, 21, 23, 28, 31, 35, 42, 44, 50, 53, 65, 70, 72, 76, 80, 83, 85, 86, 88, 92, 93, 97, 99]`
- **B qwen_degenerate**: 61 rescued — `[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49, 50, 51, 63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95]`
- **C heldout_degenerate_v2**: 50 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`

## Findings / diagnosis

**GATE FAIL** — one or more targets not met. See the gate assessment table above.

**night-006 root cause (confirmed).** The spectral-collapse signature (PR < band_low = 8.216097) co-occurs with healthy surface text (trailing_ctr ≈ 1.0) on all three degenerate fixtures — the same pattern documented by DS-034c/DS-034e. Requiring `trailing_ctr < 0.50` on the spectral-gated kickstart therefore structurally blocked the kickstart from ever firing at spectral-fire steps (0 of N spectral-fire steps had trailing_ctr < 0.50).

**The night-007 fix (verified).** The three-way AND gate is replaced by `spectral_fire` alone. Every spectral-fire step now triggers kickstart (counter=3, -1e4/-5.0/-2.0). This is the mechanism that produced the 11 DS-035 spectral-only t2s rescues.

**GATE FAIL root cause — qwen regression (0.65 -> 0.61).** night-007 loses 4 night-006 qwen rescues (record_ids 22, 23, 24, 25) with NO new qwen rescues (0 new). On those records the spectral kickstart fires at step 108 (20 events each; detection is identical to night-006 — first fire at the same step, PR, token_diversity, trailing_ctr), and the -1e4 non-prose kickstart penalty at counter=3 disrupts the suppression-only rescue that night-006 achieved (deltaD2 collapses from +0.2174 to 0.0000, i.e. back to the dormant degenerate loop). The aggressive spectral kickstart is counterproductive on this qwen subset: 0 of the 3524 qwen spectral-fire steps satisfied the night-006 trailing_ctr < 0.50 gate, so night-006's suppression-only path (which rescued 65/100) is replaced by a kickstart path that rescues only 61/100.

**t2s / heldout / prose (MET).** t2s recovers 13 -> 24 (the 11 DS-035 spectral-only rescues are recovered exactly: 17, 21, 23, 31, 35, 42, 50, 53, 85, 88, 93; 0 regressions vs night-006). heldout_v2 recovers 48 -> 50 (1.00, matching night-004). prose-100 stays at 0 FPs (dual-predicate fired on 0/100; spectral kickstart dormant by construction).

**Prose-100 FP risk (verified).** Unconditional suppression does not cause false positives on healthy prose: the dual-predicate fires on 0/100 prose records (spectral PR stays above band_low; mean PR ~18 vs 8.22) and bigram fires are absent. The spectral-gated kickstart is dormant on prose-100 by construction, so the kickstart trigger change does not add FPs to healthy prose.

## Notes / caveats

- GATE: the pass/fail criterion is explicit (all four targets). A FAIL is a STOP signal — the script exits non-zero; no thresholds or gates are negotiated [1].
- THE CHANGE (only actuation delta vs night-006): the kickstart gate is `spectral_fire` ALONE — `spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids` -> `spectral_fire`. Suppression stays UNCONDITIONAL (production controller.py:677-679 coupling), which night-004 established bridges the qwen actuation gap (0.20 -> 0.65).
- The frozen DS-034e dual-predicate rule is implemented exactly per the task file (proposed diagnose() logic): spectral_collapse uses band_low (8.216097), spectral-only fires require token_diversity < 0.40 corroboration when bigram_ctr < 2, and is_code_syntax_context forces spectral_fire = False.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer.
- Dormant baselines are REUSED (do NOT re-run): Fixtures A/B from DS-035 (dual_predicate_rescue_results.jsonl), Fixture C from DS-028 (layer_effect_persistent_results.jsonl, layer==2), prose-100 from DS-034c (dual_predicate_fp_results.jsonl, fallback_hardened).
- night-006 baseline rates and rescued-id sets are REUSED from docs/gate23/controller_integration_results.jsonl (do NOT re-run). DS-035 gated rates/rescued-ids are REUSED from docs/gate23/dual_predicate_rescue_results.jsonl (do NOT re-run).
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
- FP criterion: ΔDistinct-2 > 0 AND fire_count > 0 (DS-034c / RFC-004 A3 convention). Because suppression is unconditional, the `any ΔD2>0` count is reported separately for transparency.
