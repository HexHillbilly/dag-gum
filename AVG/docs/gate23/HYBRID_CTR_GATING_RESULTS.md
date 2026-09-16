# night-008 — Hybrid CTR gating: spectral_fire AND NOT bigram_fire AND trailing_ctr >= 0.75 (GATE)

> GATE REPORT. night-006 FAIL: t2s_degenerate = 0.13 < 0.15 because the three-way kickstart gate `spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids` never tripped. night-007 fixed t2s coverage (0.24 >= 0.15 MET) with kickstart on `spectral_fire` alone but regressed qwen (0.65 -> 0.61) — the DS-029 interaction pathology. This probe tests the HYBRID fix — `spectral_fire AND NOT bigram_fire AND trailing_ctr >= 0.75` — implementing the exact `diagnose()` / `generate()` changes proposed for `governor/controller.py` BEFORE they reach the controller. Targets are explicit, pass/fail is binary, a red gate is a STOP signal [1].

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
| Dual-predicate | DS-034e frozen config (proposed diagnose(); UNCHANGED from night-006/007): bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity |
| Suppression | UNCHANGED from production — UNCONDITIONAL: add ALL active_loop_ids to cooldown (=8) at EVERY step where detected (controller.py:677-679 coupling). |
| Kickstart | HYBRID (night-008): spectral_fire AND NOT bigram_fire AND trailing_ctr >= 0.75 -> counter=3 (-1e4/-5.0/-2.0). The 0.75 threshold is MATHEMATICALLY DERIVED (p<=11 => <=11/16=0.6875; p>=12 => >=12/16=0.75). night-007 fired kickstart on spectral_fire ALONE. |
| top_p | 0.85 when either mechanism is active |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| Dormant | REUSED: Fixtures A/B from DS-035 (dual_predicate_rescue_results.jsonl); Fixture C from DS-028 (layer_effect_persistent_results.jsonl, layer==2); prose-100 from DS-034c (dual_predicate_fp_results.jsonl, fallback_hardened). Do NOT re-run dormant. |
| night-006 baseline | REUSED from controller_integration_results.jsonl (do NOT re-run) |
| night-007 baseline | REUSED from narrow_kickstart_results.jsonl (do NOT re-run) |
| DS-035 gated baseline | REUSED from dual_predicate_rescue_results.jsonl (do NOT re-run) |
| bmm Triton override | deregistered |
| Wall clock (s) | 836.4 |

## Gate criteria (quoted from the night-008 task file)

| fixture | target | rationale |
|---|---|---|
| t2s_degenerate | >= 0.15 | match production; spectral rescues expected |
| qwen_degenerate | >= 0.65 | match night-004/night-006 unconditional suppression |
| heldout_degenerate_v2 | >= 0.96 | match all prior bests |
| prose-100 FPs | <= 7 | must not exceed ratified FP policy |

Rescue is defined per the DS-033 convention: a record is rescued iff ΔDistinct-2 > 0 vs its dormant baseline. FP = ΔDistinct-2 > 0 AND the dual-predicate fired at least once (DS-034c / RFC-004 A3 convention).

The 0.75 threshold is MATHEMATICALLY DERIVED, not empirically tuned. It is NOT a surface-corroboration threshold in the DS-034c/d/e sense and does not violate the pre-commitment against further surface threshold testing.

## Determinism smoke

One Fixture A record AND one prose-100 record: active (night-008 hybrid CTR gating) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

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

| fixture | target | night-008 | met? |
|---|---|---|---|
| t2s_degenerate | 0.1500 | 0.2300 | MET |
| qwen_degenerate | 0.6500 | 0.6100 | NOT MET |
| heldout_degenerate_v2 | 0.9600 | 0.9600 | MET |
| prose-100 FPs | <= 7 | 0 | MET |

**Gate verdict: FAIL**

**FAIL** — one or more targets not met. Red gate is a STOP signal; no thresholds or gates are negotiated. The data characterizes what needs adjustment before wiring.

## Comparison table (vs night-006 / night-007)

| fixture | night-006 | night-007 | night-008 | target | met? |
|---|---|---|---|---|---|
| t2s_degenerate | 0.1300 | 0.2400 | 0.2300 | 0.1500 | MET |
| qwen_degenerate | 0.6500 | 0.6100 | 0.6100 | 0.6500 | NOT MET |
| heldout_degenerate_v2 | 0.9600 | 1.0000 | 0.9600 | 0.9600 | MET |

night-006 rates are REUSED from `docs/gate23/controller_integration_results.jsonl` and night-007 rates from `docs/gate23/narrow_kickstart_results.jsonl` (do NOT re-run). DS-035 gated rates are shown for reference in the provenance sections.

## Table 1 — Rescue metrics per fixture

| fixture | n | rescued | rate | ΔD2 mean | ≥1 fire | mean n_gen | n≥24 D2 |
|---|---|---|---|---|---|---|---|
| A t2s_degenerate | 100 | 23 | 0.2300 | 0.0890 | 89 | 110.33 | 0.3016 |
| B qwen_degenerate | 100 | 61 | 0.6100 | 0.4182 | 48 | 92.79 | 0.5305 |
| C heldout_degenerate_v2 | 50 | 48 | 0.9600 | 0.7026 | 37 | 114.62 | 0.8640 |

## Table 2 — Fire-type counts per fixture (detection liveness)

| fixture | records ≥1 fire | bigram-fire records | spectral-fire records | both-fire records | bigram steps | spectral steps | both steps |
|---|---|---|---|---|---|---|---|
| A | 89 | 2 | 87 | 0 | 4 | 7832 | 0 |
| B | 48 | 4 | 44 | 0 | 8 | 3524 | 0 |
| C | 37 | 34 | 3 | 0 | 37 | 210 | 0 |

Fire counts reflect DUAL-PREDICATE DETECTION liveness. The dual-predicate does NOT gate suppression; suppression is unconditional at every step. The dual-predicate gates ONLY the kickstart (now the hybrid rule: spectral_fire AND NOT bigram_fire AND trailing_ctr >= 0.75).

## Table 3 — Suppression and kickstart activity (per fixture)

| fixture | n | mean suppression steps | mean kickstart events | mean trailing_ctr @ kickstart | mean n_gen |
|---|---|---|---|---|---|
| A | 100 | 110.18 | 76.26 | 0.9849 | 110.33 |
| B | 100 | 91.58 | 35.24 | 0.9994 | 92.79 |
| C | 50 | 113.36 | 0.00 | nan | 114.62 |

night-006 kickstart events were 0.00 on every fixture (the three-way gate never tripped). night-007 kickstart on spectral_fire alone fired 35.24 times per qwen record on average. night-008 hybrid gating narrows kickstart activation to spectral-only macro-loops with trailing_ctr >= 0.75.

## Dormant reference (per fixture, REUSED)

| fixture | n | mean n_gen | mean D2 | source |
|---|---|---|---|---|
| A | 100 | 125.67 | 0.2465 | reused from DS-035 (dual_predicate_rescue_results.jsonl) |
| B | 100 | 128.00 | 0.1535 | reused from DS-035 (dual_predicate_rescue_results.jsonl) |
| C | 50 | 128.00 | 0.1696 | reused from DS-028 (layer_effect_persistent_results.jsonl) |

## Table 4 — Prose-100 FP verification

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
| Mean trailing_ctr @ kickstart | nan |
| ΔDistinct-2 mean / median / p90 | 0.1228 / 0.0435 / 0.4348 |

The ratified RFC-004 A3 FP policy documents 7 FPs (5 bigram cold-start: 47, 49, 76, 83, 87; 2 spectral late-dip: 98, 99). The gate requires the night-008 probe to not exceed 7 FPs under the same criterion. Unconditional suppression does not cause false positives on healthy prose because healthy prose does not generate abundant repeated bigrams; the hybrid-gated kickstart is dormant on prose-100 because spectral PR stays above band_low (mean PR ~18 vs band_low 8.22).

## Rescue provenance vs DS-035 gated

For all three fixtures, night-008 rescues are decomposed against the DS-035 gated rescued record-id set (REUSED). DS-035 gated kickstart on is_collapsed (bigram OR spectral); night-008 gates on the hybrid rule.

| fixture | DS-035 rescued | night-008 rescued | overlap | NEW (night-008-only) | regressed (DS-035-only) |
|---|---|---|---|---|---|
| t2s_degenerate | 20 | 23 | 19 | 4 | 1 |
| qwen_degenerate | 20 | 61 | 20 | 41 | 0 |
| heldout_degenerate_v2 | 48 | 48 | 46 | 2 | 2 |

## Rescue provenance vs night-006

For all three fixtures, night-008 rescues are decomposed against the night-006 rescued record-id set (REUSED from controller_integration_results.jsonl).

| fixture | night-006 rescued | night-008 rescued | overlap | NEW (night-008-only) | regressed (night-006-only) |
|---|---|---|---|---|---|
| t2s_degenerate | 13 | 23 | 13 | 10 | 0 |
| qwen_degenerate | 65 | 61 | 61 | 0 | 4 |
| heldout_degenerate_v2 | 48 | 48 | 48 | 0 | 0 |

## Rescue provenance vs night-007

For all three fixtures, night-008 rescues are decomposed against the night-007 rescued record-id set (REUSED from narrow_kickstart_results.jsonl).

| fixture | night-007 rescued | night-008 rescued | overlap | NEW (night-008-only) | regressed (night-007-only) |
|---|---|---|---|---|---|
| t2s_degenerate | 24 | 23 | 23 | 0 | 1 |
| qwen_degenerate | 61 | 61 | 61 | 0 | 0 |
| heldout_degenerate_v2 | 50 | 48 | 48 | 0 | 2 |

## DS-034e spectral coverage validation (t2s detection-gap)

Of the 93 DS-033 t2s_degenerate detection-gap records (bigram never fired), the DS-034e dual-predicate fires on 83 — 89.2% coverage.

## Table 5 — Per-record CTR comparison for qwen (kickstart activated vs not)

Records where the hybrid kickstart activated (trailing_ctr >= 0.75) are compared against records where it did not. This documents whether the DS-029 interaction pathology is resolved.

| group | n records | rescue rate | mean ΔD2 | mean trailing_ctr @ kickstart | mean kickstart events |
|---|---|---|---|---|---|
| kickstart activated | 44 | 0.1591 | 0.0069 | 0.9994 | 80.09 |
| kickstart NOT activated | 56 | 0.9643 | 0.7414 | nan | 0.00 |

DS-029 pathology status: 
NOT RESOLVED — the hybrid rule still activates kickstart on 4 of the 4 night-007 qwen regression records: [22, 23, 24, 25].

Kickstart-activated record ids (hybrid rule): [0, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 69, 70, 71, 81, 82, 83, 84, 85, 86, 96, 97, 98, 99]

Kickstart-dormant record ids (hybrid rule did NOT fire): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49, 50, 51, 63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 87, 88, 89, 90, 91, 92, 93, 94, 95]

## Table 6 — Kickstart analysis (hybrid CTR rule)

night-008 fires kickstart on the hybrid rule: spectral_fire AND NOT bigram_fire AND trailing_ctr >= 0.75. The table shows the per-fixture counts of spectral-fire steps, how many WOULD have satisfied the night-007 `spectral_fire`-alone condition, how many satisfy the night-008 hybrid condition, and the actual kickstart events.

| fixture | spectral_fire steps | night-007 (spec alone) | night-008 hybrid (kick events) |
|---|---|---|---|
| t2s_degenerate | 7832 | 7832 | 7626 |
| qwen_degenerate | 3524 | 3524 | 3524 |
| heldout_degenerate_v2 | 210 | 210 | 0 |
| TOTAL | 11566 | 11566 | 11150 |

## Rescued record ids (ΔDistinct-2 > 0)

- **A t2s_degenerate**: 23 rescued — `[12, 17, 21, 23, 28, 35, 42, 44, 50, 53, 65, 70, 72, 76, 80, 83, 85, 86, 88, 92, 93, 97, 99]`
- **B qwen_degenerate**: 61 rescued — `[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49, 50, 51, 63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95]`
- **C heldout_degenerate_v2**: 48 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`

## Findings / diagnosis

**GATE FAIL** — one or more targets not met. See the gate assessment table above.

**night-006 root cause (confirmed).** The spectral-collapse signature (PR < band_low = 8.216097) co-occurs with healthy surface text (trailing_ctr ≈ 1.0) on all three degenerate fixtures — the same pattern documented by DS-034c/DS-034e. Requiring `trailing_ctr < 0.50` on the spectral-gated kickstart therefore structurally blocked the kickstart from ever firing at spectral-fire steps (0 of N spectral-fire steps had trailing_ctr < 0.50).

**night-007 failure mode (verified).** The three-way AND gate replaced by `spectral_fire` alone recovered t2s coverage (0.24 >= 0.15) but regressed qwen (0.65 -> 0.61): kickstart fired 35.24 times per qwen record on average and disrupted the suppression-only rescue on the DS-029 pathology records (22, 23, 24, 25).

**GATE FAIL root cause — the 0.75 trailing_ctr gate provides zero discrimination on qwen.** The hybrid rule (`spectral_fire AND NOT bigram_fire AND trailing_ctr >= 0.75`) filters NONE of the 3524 qwen spectral-fire steps (Table 6: qwen spectral steps 3524 == kick events 3524). The production `trailing_ctr` is `compute_coherent_token_ratio()` — a text-coherence ratio that returns ≈ 1.0 for qwen's coherent-word micro-repetitions (mean trailing_ctr at kickstart activation = 0.9994; Table 5). The mathematical derivation in the task file (p<=11 => <=11/16=0.6875; p>=12 => >=12/16=0.75) describes a repetition-period CTR that the production metric does NOT implement. As a result, the hybrid rule is equivalent to night-007's `spectral_fire` alone on qwen: the same 44 kickstart-activated records, the same 4 DS-029 pathology records (22, 23, 24, 25) still receive kickstart, and qwen stays at 0.61 (< 0.65 target).

**Secondary finding — heldout_v2 drops 1.00 -> 0.96 (still MET).** The hybrid rule filters kickstart to 0 events on Fixture C (Table 6: heldout kick events 0 vs night-007 spectral-only non-zero), losing 2 night-007 heldout rescues. t2s drops 0.24 -> 0.23 (still MET): 206 of 7832 t2s spectral steps were filtered by the hybrid rule, costing 1 t2s rescue.

**Prose-100 FP risk (verified).** Unconditional suppression does not cause false positives on healthy prose: the dual-predicate fires on 0/100 prose records (spectral PR stays above band_low; mean PR ~18 vs 8.22) and bigram fires are absent. The hybrid-gated kickstart is dormant on prose-100 by construction, so the kickstart trigger change does not add FPs to healthy prose.

## Notes / caveats

- GATE: the pass/fail criterion is explicit (all four targets). A FAIL is a STOP signal — the script exits non-zero; no thresholds or gates are negotiated [1].
- THE CHANGE (only actuation delta vs night-007): the kickstart gate is the HYBRID rule — `spectral_fire AND NOT bigram_fire AND trailing_ctr >= 0.75` — instead of night-007's `spectral_fire` alone. Suppression stays UNCONDITIONAL (production controller.py:677-679 coupling), which night-004 established bridges the qwen actuation gap (0.20 -> 0.65).
- The 0.75 threshold is MATHEMATICALLY DERIVED from the geometric relationship between loop period and monitoring window (p<=11 => <=11/16=0.6875; p>=12 => >=12/16=0.75). It is NOT a surface-corroboration threshold in the DS-034c/d/e sense and does NOT violate the pre-commitment against further surface threshold testing.
- The frozen DS-034e dual-predicate rule is implemented exactly per the task file (proposed diagnose() logic): spectral_collapse uses band_low (8.216097), spectral-only fires require token_diversity < 0.40 corroboration when bigram_ctr < 2, and is_code_syntax_context forces spectral_fire = False.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer.
- Dormant baselines are REUSED (do NOT re-run): Fixtures A/B from DS-035 (dual_predicate_rescue_results.jsonl), Fixture C from DS-028 (layer_effect_persistent_results.jsonl, layer==2), prose-100 from DS-034c (dual_predicate_fp_results.jsonl, fallback_hardened).
- night-006 baseline rates/rescued-ids are REUSED from docs/gate23/controller_integration_results.jsonl and night-007 baseline rates/rescued-ids from docs/gate23/narrow_kickstart_results.jsonl (do NOT re-run). DS-035 gated rates/rescued-ids are REUSED from docs/gate23/dual_predicate_rescue_results.jsonl (do NOT re-run).
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
- FP criterion: ΔDistinct-2 > 0 AND fire_count > 0 (DS-034c / RFC-004 A3 convention). Because suppression is unconditional, the `any ΔD2>0` count is reported separately for transparency.
