# night-006 — Controller integration probe: dual-predicate + spectral-gated kickstart (GATE)

> GATE REPORT. This probe tests the exact `diagnose()` / `generate()` changes proposed for `governor/controller.py` BEFORE they reach the controller. Targets are explicit, pass/fail is binary, a red gate is a STOP signal [1].

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
| Dual-predicate | DS-034e frozen config (proposed diagnose()): bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity |
| Suppression | UNCHANGED from production — UNCONDITIONAL: add ALL active_loop_ids to cooldown (=8) at EVERY step where detected (controller.py:677-679 coupling). |
| Kickstart | NEW spectral-gated: spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids -> counter=3 (-1e4/-5.0/-2.0). night-005 finding: the dual-predicate's actuation role is NARROW (macro-loop redirect only). |
| top_p | 0.85 when either mechanism is active |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| Dormant | REUSED: Fixtures A/B from DS-035 (dual_predicate_rescue_results.jsonl); Fixture C from DS-028 (layer_effect_persistent_results.jsonl, layer==2); prose-100 from DS-034c (dual_predicate_fp_results.jsonl, fallback_hardened). Do NOT re-run dormant. |
| DS-033 / DS-035 baselines | REUSED from production_cross_fixture_results.jsonl / dual_predicate_rescue_results.jsonl (do NOT re-run) |
| bmm Triton override | deregistered |
| Wall clock (s) | 751.4 |

## Gate criteria (quoted from the night-006 task file)

| fixture | target | rationale |
|---|---|---|
| t2s_degenerate | >= 0.15 | match production; spectral rescues expected |
| qwen_degenerate | >= 0.65 | match night-004 unconditional suppression |
| heldout_degenerate_v2 | >= 0.96 | match all prior bests |
| prose-100 FPs | <= 7 | must not exceed ratified FP policy |

Rescue is defined per the DS-033 convention: a record is rescued iff ΔDistinct-2 > 0 vs its dormant baseline. FP = ΔDistinct-2 > 0 AND the dual-predicate fired at least once (DS-034c / RFC-004 A3 convention).

## Determinism smoke

One Fixture A record AND one prose-100 record: active (night-006 controller integration) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

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

| fixture | target | night-006 | met? |
|---|---|---|---|
| t2s_degenerate | 0.1500 | 0.1300 | NOT MET |
| qwen_degenerate | 0.6500 | 0.6500 | MET |
| heldout_degenerate_v2 | 0.9600 | 0.9600 | MET |
| prose-100 FPs | <= 7 | 0 | MET |

**Gate verdict: FAIL**

**FAIL** — one or more targets not met. Red gate is a STOP signal; no thresholds or gates are negotiated. The data characterizes what needs adjustment before wiring.

## Comparison table

| fixture | DS-033 prod | DS-035 gated | night-004 uncond | night-006 probe | target |
|---|---|---|---|---|---|
| t2s_degenerate | 0.1500 | 0.2000 | 0.1300 | 0.1300 | 0.1500 |
| qwen_degenerate | 0.6800 | 0.2000 | 0.6500 | 0.6500 | 0.6500 |
| heldout_degenerate_v2 | 0.9600 | 0.9600 | 1.0000 | 0.9600 | 0.9600 |

DS-033 production rates for Fixtures A/B are REUSED from `docs/gate23/production_cross_fixture_results.jsonl`. DS-035 gated rates and dormant baselines are REUSED from `docs/gate23/dual_predicate_rescue_results.jsonl`. night-004 unconditional rates are REUSED from `docs/gate23/unconditional_suppression_results.jsonl`. Fixture C production baseline is the DS-031 suppression-only 48/50 = 0.9600 (DS-033 did not cover Fixture C).

## Table 1 — Rescue metrics per fixture

| fixture | n | rescued | rate | ΔD2 mean | ≥1 fire | mean n_gen | n≥24 D2 |
|---|---|---|---|---|---|---|---|
| A t2s_degenerate | 100 | 13 | 0.1300 | 0.0847 | 89 | 120.28 | 0.2970 |
| B qwen_degenerate | 100 | 65 | 0.6500 | 0.4387 | 48 | 93.81 | 0.5535 |
| C heldout_degenerate_v2 | 50 | 48 | 0.9600 | 0.7026 | 37 | 114.62 | 0.8640 |

## Table 2 — Fire-type counts per fixture (detection liveness)

| fixture | records ≥1 fire | bigram-fire records | spectral-fire records | both-fire records | bigram steps | spectral steps | both steps |
|---|---|---|---|---|---|---|---|
| A | 89 | 2 | 87 | 0 | 4 | 8880 | 0 |
| B | 48 | 4 | 44 | 0 | 8 | 3602 | 0 |
| C | 37 | 34 | 3 | 0 | 37 | 210 | 0 |

Fire counts reflect DUAL-PREDICATE DETECTION liveness. The dual-predicate does NOT gate suppression in night-006; suppression is unconditional at every step. The dual-predicate gates ONLY the spectral kickstart.

## Table 3 — Suppression and kickstart activity (per fixture)

| fixture | n | mean suppression steps | mean kickstart events | mean n_gen |
|---|---|---|---|---|
| A | 100 | 120.08 | 0.00 | 120.28 |
| B | 100 | 92.60 | 0.00 | 93.81 |
| C | 50 | 113.36 | 0.00 | 114.62 |

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

The ratified RFC-004 A3 FP policy documents 7 FPs (5 bigram cold-start: 47, 49, 76, 83, 87; 2 spectral late-dip: 98, 99). The gate requires the night-006 probe to not exceed 7 FPs under the same criterion.

## Rescue provenance vs DS-033 production

For Fixtures A/B, night-006 rescues are decomposed against the DS-033 production rescued record-id set (REUSED).

| fixture | prod rescued | night-006 rescued | overlap | NEW (night-006-only) | regressed (prod-only) |
|---|---|---|---|---|---|
| A t2s_degenerate | 15 | 13 | 13 | 0 | 2 |
| B qwen_degenerate | 68 | 65 | 65 | 0 | 3 |

## Rescue provenance vs DS-035 gated

For all three fixtures, night-006 rescues are decomposed against the DS-035 gated rescued record-id set (REUSED).

| fixture | DS-035 rescued | night-006 rescued | overlap | NEW (night-006-only) | regressed (DS-035-only) |
|---|---|---|---|---|---|
| t2s_degenerate | 20 | 13 | 9 | 4 | 11 |
| qwen_degenerate | 20 | 65 | 20 | 45 | 0 |
| heldout_degenerate_v2 | 48 | 48 | 46 | 2 | 2 |

## DS-034e spectral coverage validation (t2s detection-gap)

Of the 93 DS-033 t2s_degenerate detection-gap records (bigram never fired), the DS-034e dual-predicate fires on 83 — 89.2% coverage.

## Rescued record ids (ΔDistinct-2 > 0)

- **A t2s_degenerate**: 13 rescued — `[12, 28, 44, 65, 70, 72, 76, 80, 83, 86, 92, 97, 99]`
- **B qwen_degenerate**: 65 rescued — `[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49, 50, 51, 63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95]`
- **C heldout_degenerate_v2**: 48 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`

## Findings / diagnosis

**GATE FAIL: t2s_degenerate = 0.13 < 0.15.** The spectral-gated kickstart never fired on any corpus (0 events across all 350 records + smoke). The kickstart gate `spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids` was never satisfiable because spectral-fire steps co-occur with `trailing_ctr >= 0.50` on every fixture.

Per-step evidence (from the night-006 JSONL):

| fixture | spectral_fire steps | with trailing_ctr < 0.50 |
|---|---|---|
| t2s_degenerate | 8880 | 0 |
| qwen_degenerate | 3602 | 0 |
| heldout_degenerate_v2 | 210 | 0 |
| TOTAL | 12692 | 0 |

**Root cause.** The spectral-collapse signature (PR < band_low = 8.216097) co-occurs with healthy surface text (trailing_ctr ≈ 1.0) on all three degenerate fixtures — the same pattern documented by DS-034c/DS-034e (prose FP records 98/99: PR ≈ 6.6–7.5 with trailing_ctr ≈ 1.0). Requiring `trailing_ctr < 0.50` on the spectral-gated kickstart therefore structurally blocks the kickstart from ever firing at spectral-fire steps. This is the mechanism that produced the 11 DS-035 spectral-only t2s rescues (via the is_collapsed-gated kickstart, 80.2 events/record); night-006 loses those rescues (11 DS-035-only regressions, 4 NEW) and stays at the night-004 level (13/100).

**Secondary finding (heldout_v2 1.00 → 0.96).** night-004 achieved 50/50 on Fixture C with the production kickstart trigger (2.08 events/record). The spectral-gated kickstart never fires, so Fixture C drops to 48/50 — still >= 0.96 (MET), but the production kickstart contribution is lost.

**Secondary finding (prose-100).** 0 FPs (MET, <= 7) — the dual-predicate fired on 0/100 prose records. However, unconditional suppression changed the continuation of 52/100 prose records (any ΔD2 > 0); these are not FPs under the ratified criterion (fire_count = 0) but are a non-trivial actuation footprint of the production suppression path on healthy prose.

## Notes / caveats

- GATE: the pass/fail criterion is explicit (all four targets). A FAIL is a STOP signal — the script exits non-zero; no thresholds or gates are negotiated [1].
- THE CHANGE (only actuation delta vs night-004): the kickstart gate is spectral — `spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids`. Suppression stays UNCONDITIONAL (production controller.py:677-679 coupling), which night-004 established bridges the qwen actuation gap (0.20 -> 0.65).
- The frozen DS-034e dual-predicate rule is implemented exactly per the task file (proposed diagnose() logic): spectral_collapse uses band_low (8.216097), spectral-only fires require token_diversity < 0.40 corroboration when bigram_ctr < 2, and is_code_syntax_context forces spectral_fire = False.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer.
- Dormant baselines are REUSED (do NOT re-run): Fixtures A/B from DS-035 (dual_predicate_rescue_results.jsonl), Fixture C from DS-028 (layer_effect_persistent_results.jsonl, layer==2), prose-100 from DS-034c (dual_predicate_fp_results.jsonl, fallback_hardened).
- DS-033 production and DS-035 gated rescue rates are REUSED for the comparison tables (do NOT re-run).
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
- FP criterion: ΔDistinct-2 > 0 AND fire_count > 0 (DS-034c / RFC-004 A3 convention). Because suppression is unconditional, the `any ΔD2>0` count is reported separately for transparency.
