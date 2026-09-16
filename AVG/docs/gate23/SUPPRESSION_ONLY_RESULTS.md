# night-018 — Suppression-only, kickstart fully disabled (MEASUREMENT ONLY)

> MEASUREMENT REPORT. NOT a gate. This probe tests the cleanest remaining configuration: kickstart FULLY DISABLED (no _kickstart_counter, no kickstart penalty path at all) with token suppression as the sole actuator. If suppression-only holds rescue while dropping EOS to the floor, kickstart can be removed from the production controller entirely. All fixes are implemented INLINE; no controller edits [1].

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
| Dual-predicate | DS-034e frozen config: bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity. Runs DETECTION-ONLY — logged for fire-type breakdown, does NOT gate actuation. |
| Actuation | UNCONDITIONAL token suppression at EVERY step where active_loop_ids is non-empty (cooldown=8, penalty=-5.0). Kickstart FULLY DISABLED. |
| Suppression | cooldown=8 penalty=-5.0 (unchanged production); top_p=0.85 when suppression is active |
| Fix 1 | KV-cache incremental decoding (pre-fill once, single-token forwards with past_key_values) — hs.shape[1] == 1 on every decoding step |
| Fix 3 | is_code_context recomputed EVERY step when use_code_filter is True (moved outside the CTR sub-sampling gate) |
| Fix 4 | VarietyProfiler.profile() / is_profile_step / hooks_registered dormancy REMOVED from the generate loop; shadow hooks retained as passive telemetry (controller) |
| Fix 6 | sample_top_p explicit .clone(); finally-block resets per-generation state |
| Residual path | DISABLED in this probe — no diagnose() → apply_interventions() → sae_guided_reset hooks |
| Kickstart | FULLY DISABLED — no _kickstart_counter, no kickstart penalty path at all |
| Dormant | REUSED from night-016 (greedy + sampling KV-cache dormant, already validated 0/250 drift). Do NOT re-run. |
| Reused baselines | night-016 (kvcache_restructure_results.jsonl, kickstart 1e4 cell); night-014 Arm 2 (sampling_dual_predicate_results.jsonl). Do NOT re-run. |
| NEW runs | 2 arms x 3 fixtures = 6 condition-fixture pairs |
| bmm Triton override | deregistered |
| Wall clock (s) | 849.3 |

## Determinism smoke

One Fixture C record (heldout_degenerate_v2, seeded subset): greedy suppression-only generated twice and sampling suppression-only generated twice, all with SEED=42. Token ids must match; per-step PR log must match to 6 decimal places. STOP if not. Dormant is REUSED from night-016 (not re-run).

| record_id | prompt_len | n_gen(greedy) | n_gen(sampling) | ga id | ga PR | sa id | sa PR | identical |
|---|---|---|---|---|---|---|---|---|
| 0 | 28 | 128 | 128 | True | True | True | True | True |

PR greedy run A/B first: 18.1274 / 18.1274; last: 17.3005 / 17.3005. PR sampling run A/B first: 19.7406 / 19.7406; last: 19.2392 / 19.2392.

## Key findings

1. **Greedy suppression-only rescue/EOS:** t2s 0.1300 (bar ≥ 0.91) / EOS 0.0900 (night-016 0.85); qwen 0.6500 (bar ≥ 0.93) / EOS 0.3700 (night-016 0.67); heldout 0.9600 (bar ≥ 0.95) / EOS 0.2400 (night-016 0.26).
2. **Sampling heldout net escape:** 0.9600 (target ≥ 0.90), EOS 0.3400 (target ≤ 0.34, night-014 Arm 2 0.34).
3. **Sampling t2s/qwen:** t2s rescue 0.8700 / EOS 0.6800 (night-016 0.96/0.64, night-014 Arm 2 0.87/0.68); qwen rescue 0.9800 / EOS 0.3900 (night-016 0.95/0.43, night-014 Arm 2 0.98/0.39).
4. **Kickstart events are all zero** (n_kickstart_events == 0 on every record, both arms, all fixtures) — kickstart is fully disabled; any actuation is suppression-only.

## Table 1 — Primary aggregation (per fixture, per arm)

Rescue = ΔDistinct-2 > 0 vs the REUSED night-016 dormant baseline. Net escape = gross escape AND n_gen >= 24 (night-013 convention). EOS rate = n_gen < 128 AND last_token == eos_token_id (co-primary). med n_gen = median generated tokens among escaped continuations (co-primary). ΔD2 mean = mean ΔDistinct-2 over all records.

| fixture | arm | rescue | net | EOS | med n_gen (esc) | ΔD2 mean |
|---|---|---|---|---|---|---|
| A t2s_degenerate | greedy | 0.1300 | 0.1000 | 0.0900 | 110.0 | 0.0847 |
| A t2s_degenerate | sampling | 0.8700 | 0.5200 | 0.6800 | 28.0 | 0.4675 |
| B qwen_degenerate | greedy | 0.6500 | 0.5600 | 0.3700 | 75.0 | 0.4387 |
| B qwen_degenerate | sampling | 0.9800 | 0.8100 | 0.3900 | 128.0 | 0.7757 |
| C heldout_degenerate_v2 | greedy | 0.9600 | 0.9000 | 0.2400 | 128.0 | 0.7026 |
| C heldout_degenerate_v2 | sampling | 1.0000 | 0.9600 | 0.3400 | 128.0 | 0.7324 |

## Table 2 — Comparison (vs night-016 kickstart, night-014 Arm 2)

night-016 (kickstart) = REUSED kvcache_restructure_results.jsonl 1e4 cell (greedy_active / sampling_active). night-014 Arm 2 = REUSED sampling_dual_predicate_results.jsonl arm2_active (spectral-gated kickstart; greedy arm not run in night-014). night-018 (no kickstart) = this probe.

| fixture | arm | night-016 (kickstart) | night-014 Arm2 | night-018 (no kickstart) |
|---|---|---|---|---|
| A t2s_degenerate | greedy | 0.9600 / EOS 0.8500 | — | 0.1300 / EOS 0.0900 |
| A t2s_degenerate | sampling | 0.9600 / EOS 0.6400 | 0.8700 / EOS 0.6800 | 0.8700 / EOS 0.6800 |
| B qwen_degenerate | greedy | 0.9800 / EOS 0.6700 | — | 0.6500 / EOS 0.3700 |
| B qwen_degenerate | sampling | 0.9500 / EOS 0.4300 | 0.9800 / EOS 0.3900 | 0.9800 / EOS 0.3900 |
| C heldout_degenerate_v2 | greedy | 1.0000 / EOS 0.2600 | — | 0.9600 / EOS 0.2400 |
| C heldout_degenerate_v2 | sampling | 1.0000 / EOS 0.1400 | 1.0000 / EOS 0.3400 | 1.0000 / EOS 0.3400 |

## Table 3 — Four-bucket histogram over escaped continuations

night-013 convention: n=1 | 2-23 | 24-127 | n=128 over escaped continuations (gross escape = not byte-identical to dormant).

| fixture | arm | n=1 | 2-23 | 24-127 | n=128 |
|---|---|---|---|---|---|
| A t2s_degenerate | greedy | 0 | 4 | 4 | 6 |
| A t2s_degenerate | sampling | 6 | 31 | 30 | 22 |
| B qwen_degenerate | greedy | 7 | 4 | 26 | 30 |
| B qwen_degenerate | sampling | 7 | 12 | 20 | 61 |
| C heldout_degenerate_v2 | greedy | 3 | 0 | 9 | 36 |
| C heldout_degenerate_v2 | sampling | 0 | 2 | 15 | 33 |

## Table 4 — Fire-type breakdown (dual-predicate detection-only)

The collapse predicate runs DETECTION-ONLY in this probe — it does NOT gate actuation (suppression is unconditional). bigram-only records = bigram fired >= 1 step, spectral never. spectral-only = spectral fired >= 1 step, bigram never. both = both fired >= 1 step.

| fixture | arm | fired | bigram-only | spectral-only | both | mean supp | mean kick |
|---|---|---|---|---|---|---|---|
| A t2s_degenerate | greedy | 89 | 2 | 87 | 0 | 120.08 | 0.00 |
| A t2s_degenerate | sampling | 31 | 0 | 31 | 0 | 56.99 | 0.00 |
| B qwen_degenerate | greedy | 48 | 4 | 44 | 0 | 92.60 | 0.00 |
| B qwen_degenerate | sampling | 5 | 5 | 0 | 0 | 82.93 | 0.00 |
| C heldout_degenerate_v2 | greedy | 37 | 34 | 3 | 0 | 113.36 | 0.00 |
| C heldout_degenerate_v2 | sampling | 22 | 22 | 0 | 0 | 96.62 | 0.00 |

## Target assessment (hypothesis, NOT a gate)

Suppression-only must hold rescue within 0.05 of the best prior while dropping EOS:

| fixture | arm | target | night-018 | met? |
|---|---|---|---|---|
| t2s_degenerate | greedy | rescue ≥ 0.91, EOS < 0.85 | rescue 0.1300, EOS 0.0900 | NO |
| qwen_degenerate | greedy | rescue ≥ 0.93, EOS < 0.67 | rescue 0.6500, EOS 0.3700 | NO |
| heldout_degenerate_v2 | greedy | rescue ≥ 0.95, EOS ≤ 0.26 | rescue 0.9600, EOS 0.2400 | YES |
| heldout_degenerate_v2 | sampling | net escape ≥ 0.90, EOS ≤ 0.34 | rescue 1.0000, net 0.9600, EOS 0.3400 | YES |

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no threshold changes. All retained night-016 fixes are implemented INLINE in this probe before they reach the controller [1].
- Kickstart is FULLY DISABLED: `_kickstart_counter` is never set; the kickstart penalty block is removed from the generation loop. The `_cache_vocabulary_subsets` call is not needed at runtime (offline diagnostics only per night-015 Fix 5).
- Suppression is UNCONDITIONAL: at EVERY step where active_loop_ids is non-empty, ALL active_loop_ids are added to the cooldown dict (cooldown=8, penalty=-5.0). NOT gated on is_collapsed. This matches night-014's suppression path (net escape 0.96 on heldout_v2).
- The dual-predicate collapse predicate runs DETECTION-ONLY: the predicate is logged for fire-type breakdown but does NOT gate actuation.
- Dormant baselines are REUSED from night-016 (greedy + sampling KV-cache, already validated 0/250 drift). They are NOT re-run; the ΔD2 deltas are computed against the same night-016 dormant baseline, so rescue-rate differences are attributable to the actuation change (kickstart removal) alone.
- night-016 (kickstart) values are REUSED from kvcache_restructure_results.jsonl (1e4 cell). night-014 Arm 2 values are REUSED from sampling_dual_predicate_results.jsonl (arm2_active, spectral-gated kickstart). Do NOT re-run.
- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 (matching night-014/016). SEED=42 with seed_all(SEED) before every sampling generation; the determinism smoke confirms reproducible sampling.
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
