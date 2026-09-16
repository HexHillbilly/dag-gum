# night-016 — KV-cache restructure + dual-predicate → logit-penalty wiring (MEASUREMENT ONLY)

> MEASUREMENT REPORT. NOT a gate. This probe implements ALL proposed fixes INLINE (no controller edits) and measures rescue rates under two decoding regimes (greedy and sampling) on all three degenerate fixtures plus the t2s detection-gap subset (93 records). The CI gate re-run happens as a separate controller commit AFTER this probe validates rescue rates.

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
| Decoding | Arm 1 greedy (do_sample=False); Arm 2 sampling (do_sample=True, temp=0.8, top_p=0.85). Both KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 generated positions |
| Hook layer | model.model.layers[2] (layer-2 PR, rolling 24-token ring buffer) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |
| Dual-predicate | DS-034e frozen config: bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity |
| Actuation | Fix 2: collapse predicate gates logit penalties — suppression gated on is_collapsed; kickstart fires on spectral-only (spectral_fire AND NOT bigram_fire); kickstart retargeted to active_loop_ids (night-015) |
| Fix 1 | KV-cache incremental decoding (pre-fill once, single-token forwards with past_key_values) — hs.shape[1] == 1 on every decoding step |
| Fix 3 | is_code_context recomputed EVERY step when use_code_filter is True (moved outside the CTR sub-sampling gate) |
| Fix 4 | VarietyProfiler.profile() / is_profile_step / hooks_registered dormancy REMOVED from the generate loop; shadow hooks retained as passive telemetry (controller) |
| Fix 5 | kickstart retargeted to active_loop_ids + vocab-range fix (len(tokenizer)); `_cache_vocabulary_subsets` offline diagnostics only |
| Fix 6 | sample_top_p explicit .clone(); finally-block resets per-generation state |
| Residual path | DISABLED in this probe — no diagnose() → apply_interventions() → sae_guided_reset hooks |
| Dormant (restructured) | LIVE — greedy and sampling KV-cache dormant baselines generated fresh for all records |
| Reused baselines | DS-033 (full-seq greedy), night-008 (KV-cache greedy), DS-035 (greedy dual-predicate), night-014 Arm2 (sampling). Do NOT re-run. |
| bmm Triton override | deregistered |
| Wall clock (s) | 1816.2 |

## Determinism smoke

One Fixture C record (heldout_degenerate_v2, seeded subset): greedy dormant generated twice, greedy restructured active twice, sampling dormant twice, sampling restructured active twice, all with SEED=42. Token ids must match; per-step PR log must match to 6 decimal places. STOP if not.

| record_id | prompt_len | n_gen(greedy) | n_gen(sampling) | gd id | ga id | ga PR | sd id | sa id | sa PR | identical |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 28 | 128 | 128 | True | True | True | True | True | True | True |

PR greedy run A/B first: 18.8832 / 18.8832; last: 16.4117 / 16.4117. PR sampling run A/B first: 21.3557 / 21.3557; last: 18.4041 / 18.4041.

## Key findings

1. **Greedy restructured rescue is a step-change over every greedy baseline.** t2s 0.9600 vs DS-033 0.1500 / night-008 0.2300 / DS-035 0.2000 (delta 0.8100 vs DS-033). qwen 0.9800 vs DS-033 0.6800 (delta 0.3000). heldout_v2 1.0000 vs night-008 0.9600.
2. **Sampling restructured holds night-014 Arm 2 on t2s and heldout_v2, with a small qwen regression.** t2s 0.9600 vs Arm2 0.8700 (delta 0.0900); heldout_v2 1.0000 vs Arm2 1.0000; qwen 0.9500 vs Arm2 0.9800 (delta -0.0300). The qwen regression is 3 records (ids 1, 4, 89) where the restructured arm's gated suppression + spectral-only kickstart produced a LOWER distinct-2 than dormant; records 17, 18 are vacuous (dormant D2 already 1.0).
3. **t2s detection-gap rescue under greedy is 92/93 (Concern B, see Table 3).** Of the 93 DS-033 detection-gap records (bigram never fired in DS-033), the restructured greedy arm rescues 92 (0.9892) with 92 spectral-fire records and 279 kickstart events. DS-035's dual-predicate added 11 rescues on the same subset; the restructured KV-cache greedy path rescues 92. The single non-rescue (record 37) was already at D2=1.0 in its greedy dormant baseline (self-escaped to code) — vacuous.
4. **Zero dormant drift.** The LIVE greedy and sampling dormant baselines are token-identical to the DS-035 (greedy) and night-014 (sampling) dormant baselines on all 250 records — human dormant-drift review is vacuous (no drift).
5. **Kickstart EOS-death persists in the restructured arm.** EOS rates are elevated wherever the spectral-only kickstart fires (greedy t2s 0.85, qwen 0.67; sampling t2s 0.64, qwen 0.43). The retargeted kickstart removes the EOS-exclusion quirk but -1e4 on active_loop_ids still leaves EOS as the dominant remaining token when the loop is crushed.
6. **Redundant profiling removed (Concern C / Fix 4).** The VarietyProfiler.profile() step is not called during generation; the layer-2 spectral PR hook (and, in the controller, the shadow-mode hooks) are the only passive telemetry.

## Table 1 — Greedy rescue rates: current vs restructured

Rescue = ΔDistinct-2 > 0 vs the dormant baseline (DS-033 convention). DS-033 (production full-sequence) covered Fixtures A/B only. night-008 (KV-cache greedy, hybrid CTR gating) and DS-035 (greedy dual-predicate) covered all three fixtures. Restructured = this probe's greedy arm (KV-cache, all fixes inline). Delta vs DS-033 for A/B; delta vs night-008 for heldout_v2.

| fixture | DS-033 (full-seq) | night-008 (KV-cache) | DS-035 (greedy dual) | restructured | delta |
|---|---|---|---|---|---|
| t2s_degenerate | 0.1500 | 0.2300 | 0.2000 | 0.9600 | 0.8100 |
| qwen_degenerate | 0.6800 | 0.6100 | 0.2000 | 0.9800 | 0.3000 |
| heldout_degenerate_v2 | nan | 0.9600 | 0.9600 | 1.0000 | 0.0400 |

DS-033 did not cover heldout_v2 (the DS-031 suppression-only production baseline for Fixture C was 48/50 = 0.9600, documented in the DS-035 task file).

## Table 2 — Sampling rescue rates: current vs restructured

Rescue = ΔDistinct-2 > 0 vs the LIVE sampling dormant baseline. night-014 Arm 2 (KV-cache, no kickstart under sampling) is the REUSED comparison baseline (sampling_dual_predicate_results.jsonl, do NOT re-run). Restructured = this probe's sampling arm.

| fixture | night-014 Arm2 (no kickstart) | restructured | delta |
|---|---|---|---|
| t2s_degenerate | 0.8700 | 0.9600 | 0.0900 |
| qwen_degenerate | 0.9800 | 0.9500 | -0.0300 |
| heldout_degenerate_v2 | 1.0000 | 1.0000 | 0.0000 |

## Table 3 — t2s detection-gap rescue (greedy, Concern B)

The DS-033 t2s detection-gap subset = 93 records where the bigram predicate never fired (predicate_true_steps == 0). DS-035's dual-predicate fired on 92/93 of these records and added 11 rescues via the spectral-gated kickstart path. This table re-measures that path under KV-cache greedy with the restructured controller (Fix 1 + Fix 2 + Fix 5).

| subset | n | rescued | rate | spectral fires | kickstart events | ΔD2 mean |
|---|---|---|---|---|---|---|
| t2s detection-gap (DS-033) | 93 | 92 | 0.9892 | 92 | 279 | 0.1061 |

Rescued record ids: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`

## Table 4 — Per-arm escape metrics (night-013 convention)

Gross escape = 1.0 - byte_identical_rate (night-013 convention). Net escape = gross escape AND n_gen >= 24. EOS rate = n_gen < 128 AND last_token == eos_token_id. med n_gen = median generated tokens among escaped continuations. Four-bucket histogram over escaped continuations: n=1 | 2-23 | 24-127 | n=128.

| fixture | arm | rescue | gross | net | EOS | med n_gen | n=1 | 2-23 | 24-127 | n=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| A t2s_degenerate | greedy | 0.9600 | 0.9600 | 0.9600 | 0.8500 | 26.0 | 0 | 0 | 83 | 13 |
| A t2s_degenerate | sampling | 0.9600 | 0.9900 | 0.9900 | 0.6400 | 28.0 | 0 | 0 | 63 | 36 |
| B qwen_degenerate | greedy | 0.9800 | 1.0000 | 1.0000 | 0.6700 | 68.0 | 0 | 0 | 67 | 33 |
| B qwen_degenerate | sampling | 0.9500 | 1.0000 | 1.0000 | 0.4300 | 128.0 | 0 | 0 | 43 | 57 |
| C heldout_degenerate_v2 | greedy | 1.0000 | 1.0000 | 1.0000 | 0.2600 | 128.0 | 0 | 0 | 13 | 37 |
| C heldout_degenerate_v2 | sampling | 1.0000 | 1.0000 | 1.0000 | 0.1400 | 128.0 | 0 | 0 | 7 | 43 |

## Table 5 — Fire-type breakdown per fixture per arm

bigram-only records = bigram fired >= 1 step, spectral never. spectral-only = spectral fired >= 1 step, bigram never. both = both fired >= 1 step.

| fixture | arm | fired | bigram-only | spectral-only | both |
|---|---|---|---|---|---|
| A t2s_degenerate | greedy | 97 | 1 | 96 | 0 |
| A t2s_degenerate | sampling | 96 | 0 | 94 | 2 |
| B qwen_degenerate | greedy | 100 | 0 | 100 | 0 |
| B qwen_degenerate | sampling | 100 | 0 | 100 | 0 |
| C heldout_degenerate_v2 | greedy | 50 | 11 | 26 | 13 |
| C heldout_degenerate_v2 | sampling | 50 | 14 | 26 | 10 |

## Table 6 — Per-arm activity metrics (per fixture)

| fixture | arm | mean supp steps | mean kickstart | mean n_gen | n≥24 D2 | mean ΔD2 |
|---|---|---|---|---|---|---|
| A t2s_degenerate | greedy | 5.48 | 2.91 | 43.50 | 0.3505 | 0.1091 |
| A t2s_degenerate | sampling | 9.43 | 4.71 | 68.64 | 0.5463 | 0.3013 |
| B qwen_degenerate | greedy | 13.51 | 7.32 | 75.13 | 0.5313 | 0.3778 |
| B qwen_degenerate | sampling | 18.74 | 9.67 | 97.52 | 0.6939 | 0.5178 |
| C heldout_degenerate_v2 | greedy | 14.72 | 3.88 | 101.72 | 0.6400 | 0.4704 |
| C heldout_degenerate_v2 | sampling | 18.58 | 6.68 | 113.84 | 0.7052 | 0.5357 |

## Table 7 — Dormant reference (LIVE, per fixture)

| fixture | regime | n | mean n_gen | mean D2 | source |
|---|---|---|---|---|---|
| A t2s_degenerate | greedy | 100 | 125.67 | 0.2465 | live (greedy, KV-cache) |
| A t2s_degenerate | sampling | 100 | 126.81 | 0.2496 | live (sampling, temp=0.8, top_p=0.85, KV-cache) |
| B qwen_degenerate | greedy | 100 | 128.00 | 0.1535 | live (greedy, KV-cache) |
| B qwen_degenerate | sampling | 100 | 127.50 | 0.1761 | live (sampling, temp=0.8, top_p=0.85, KV-cache) |
| C heldout_degenerate_v2 | greedy | 50 | 128.00 | 0.1696 | live (greedy, KV-cache) |
| C heldout_degenerate_v2 | sampling | 50 | 128.00 | 0.1696 | live (sampling, temp=0.8, top_p=0.85, KV-cache) |

## Dormant-drift validation (vs reused baselines)

The LIVE dormant baselines generated by this probe were compared against the reused baselines (DS-035 greedy dormant and night-014 sampling dormant) on all 250 records: token ids are IDENTICAL on every record for both regimes (0/250 disagreements, 0 distinct-2 deltas, 0 n_gen deltas). The restructured regime introduces no dormant drift; rescue deltas are attributable to the active actuation, not to baseline shifts.

## Rescued record ids (ΔDistinct-2 > 0)

- **A t2s_degenerate greedy**: 96 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **A t2s_degenerate sampling**: 96 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **B qwen_degenerate greedy**: 98 rescued — `[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **B qwen_degenerate sampling**: 95 rescued — `[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **C heldout_degenerate_v2 greedy**: 50 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`
- **C heldout_degenerate_v2 sampling**: 50 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no threshold changes, no new actuation design beyond the inline fixes. All fixes are implemented INLINE in this probe before they reach the controller [1].
- The residual intervention path (diagnose() → apply_interventions() → sae_guided_reset hooks) is DISABLED in this probe. The `_make_sae_guided_reset_fn` deprecation warning still fires if the path is ever called (retained per Amendment A4 for cross-model transfer).
- Fix 2 gates token suppression on is_collapsed (was: unconditional in the production combined path). Kickstart fires ONLY on spectral-only (spectral_fire AND NOT bigram_fire), retargeted to active_loop_ids (night-015).
- Fix 1 makes the layer-2 PR hook condition (`hs.shape[1] == 1`) True on every decoding step via KV-cache incremental decoding. This is the Finding #1 blocker fix; all spectral PR instrumentation works as validated by DS-034b (online parity: 0/93 disagreements).
- Fix 3 recomputes is_code_context EVERY step when use_code_filter is True (production default False here, so is_code_context stays False and spectral_fire is not code-suppressed).
- Fix 4 removes the VarietyProfiler.profile() call and the is_profile_step / hooks_registered dormancy management from the generate loop. The layer-2 spectral PR hook is the only hook registered in this probe; the controller's shadow-mode hooks are retained as passive telemetry.
- Dormant baselines for the restructured regime are generated LIVE for both greedy and sampling (KV-cache). batch human review compares these against prior dormant baselines to verify no dormant drift.
- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 (matching night-014/DS-032 Part 2). SEED=42 with seed_all(SEED) before every sampling generation; the determinism smoke confirms reproducible sampling.
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
- **qwen sampling regression (3 records).** Records 1, 4, 89 show a negative ΔD2 under the restructured sampling arm (-0.30 to -0.57) that night-014 Arm 2 did not exhibit. The Fix 2 gating (suppression on is_collapsed instead of unconditional) plus the spectral-only kickstart produced a lower distinct-2 than dormant on these records. Records 17, 18 are vacuous non-rescues (dormant D2 already 1.0).
- **Detection-gap non-rescue (record 37) is vacuous.** The sole non-rescued DS-033 t2s detection-gap record was already at D2=1.0 in its greedy dormant baseline (self-escaped to code); the restructured active is byte-identical to dormant, so ΔD2=0.0 is correct and no collapse existed to detect.
