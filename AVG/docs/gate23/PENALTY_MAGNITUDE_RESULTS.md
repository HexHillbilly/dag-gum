# night-017 — Kickstart penalty magnitude sweep (MEASUREMENT ONLY)

> MEASUREMENT REPORT. NOT a gate. This probe sweeps the kickstart penalty magnitude across {1e4, 5.0, 2.0, 1.0} on the night-016 restructured regime (all fixes inline, no controller edits). EOS rate and median n_gen (escaped) are CO-PRIMARY metrics alongside rescue rate. This is the same discipline as DS-031's cooldown sweep, but on the penalty magnitude rather than the cooldown duration.

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
| Suppression | cooldown=8 penalty=-5.0 (unchanged production); top_p=0.85 when either mechanism is active |
| Sweep cells | 1e4 = -1e4/-5.0/-2.0 (REUSED from night-016, do NOT re-run); 5.0 = -5.0/-5.0/-2.0 (NEW); 2.0 = -2.0/-2.0/-1.0 (NEW); 1.0 = -1.0/-1.0/-1.0 (NEW) |
| Fix 1 | KV-cache incremental decoding (pre-fill once, single-token forwards with past_key_values) — hs.shape[1] == 1 on every decoding step |
| Fix 3 | is_code_context recomputed EVERY step when use_code_filter is True (moved outside the CTR sub-sampling gate) |
| Fix 4 | VarietyProfiler.profile() / is_profile_step / hooks_registered dormancy REMOVED from the generate loop; shadow hooks retained as passive telemetry (controller) |
| Fix 5 | kickstart retargeted to active_loop_ids + vocab-range fix (len(tokenizer)); `_cache_vocabulary_subsets` offline diagnostics only |
| Fix 6 | sample_top_p explicit .clone(); finally-block resets per-generation state |
| Residual path | DISABLED in this probe — no diagnose() → apply_interventions() → sae_guided_reset hooks |
| Dormant | REUSED from night-016 (greedy + sampling KV-cache dormant, already validated 0/250 drift). Do NOT re-run. |
| Cell 1e4 | REUSED from night-016 (kvcache_restructure_results.jsonl). Do NOT re-run. |
| NEW runs | 3 cells x 2 arms x 3 fixtures = 18 condition-fixture pairs |
| bmm Triton override | deregistered |
| Wall clock (s) | 2162.6 |

## Determinism smoke

Per NEW cell, one Fixture C record (heldout_degenerate_v2, seeded subset): greedy cell active generated twice and sampling cell active generated twice, all with SEED=42. Token ids must match; per-step PR log must match to 6 decimal places. STOP if not. Dormant is REUSED from night-016 (not re-run).

| cell | record_id | prompt_len | n_gen(greedy) | n_gen(sampling) | ga id | ga PR | sa id | sa PR | identical |
|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 0 | 28 | 128 | 128 | True | True | True | True | True |
| 2.0 | 0 | 28 | 128 | 128 | True | True | True | True | True |
| 1.0 | 0 | 28 | 128 | 128 | True | True | True | True | True |

PR greedy/sampling run A/B first/last values are recorded in the smoke sidecar of penalty_magnitude_results.jsonl.

## Key findings

1. **No cell meets the target across all three fixtures under greedy.** The 1.0 hypothesis (weakest uniform penalty) is NOT confirmed: the 1.0 cell has the lowest EOS rate (t2s 0.2800, qwen 0.3000, heldout 0.0800) but fails the rescue bar on all three fixtures (t2s 0.3100 < 0.91, qwen 0.7000 < 0.93, heldout 0.8000 < 0.95). A -1.0 penalty is too weak to break the loop enough for a distinct-2 improvement under greedy.
2. **The 2.0 cell reduces EOS but fails the rescue bar on t2s and qwen.** t2s rescue 0.7300 (< 0.91), qwen rescue 0.9100 (< 0.93), heldout rescue 1.0000 (bar met). EOS drops to 0.6000 (t2s) and 0.5300 (qwen), but the rescue-rate regression is outside the 0.05 tolerance.
3. **The -1e4 vs -5.0 first-step magnitude is a wash under greedy.** Cells 1e4 and 5.0 have IDENTICAL aggregate rescue, EOS, net-escape, and median n_gen on all three fixtures, and identical rescued/EOS record sets (0/250 set differences). The exact token sequences differ for a small number of records (t2s 2/100, qwen 4/100, heldout 0/50) but the classification outcome is unchanged. Both magnitudes completely remove the loop tokens from argmax contention, leaving EOS as the dominant remaining token whenever the kickstart fires.
4. **Under sampling, weaker penalties increase EOS while holding rescue ~0.96-1.00.** t2s sampling EOS rises 0.6400 → 0.6700 → 0.8000 → 0.8700 across 1e4 → 5.0 → 2.0 → 1.0. The weaker kickstart fails to steer away from the loop, and the continuation drifts to EOS.

## Table 1 — Primary aggregation (per cell, per fixture, per arm)

Rescue = ΔDistinct-2 > 0 vs the REUSED night-016 dormant baseline. Net escape = gross escape AND n_gen >= 24 (night-013 convention). EOS rate = n_gen < 128 AND last_token == eos_token_id (co-primary). med n_gen = median generated tokens among escaped continuations (co-primary). ΔD2 mean = mean ΔDistinct-2 over all records.

| cell | fixture | arm | rescue | net | EOS | med n_gen | ΔD2 mean |
|---|---|---|---|---|---|---|---|
| 1e4 | t2s_degenerate | greedy | 0.9600 | 0.9600 | 0.8500 | 26.0 | 0.1091 |
| 1e4 | t2s_degenerate | sampling | 0.9600 | 0.9900 | 0.6400 | 28.0 | 0.3013 |
| 1e4 | qwen_degenerate | greedy | 0.9800 | 1.0000 | 0.6700 | 68.0 | 0.3778 |
| 1e4 | qwen_degenerate | sampling | 0.9500 | 1.0000 | 0.4300 | 128.0 | 0.5178 |
| 1e4 | heldout_degenerate_v2 | greedy | 1.0000 | 1.0000 | 0.2600 | 128.0 | 0.4704 |
| 1e4 | heldout_degenerate_v2 | sampling | 1.0000 | 1.0000 | 0.1400 | 128.0 | 0.5357 |
| 5.0 | t2s_degenerate | greedy | 0.9600 | 0.9600 | 0.8500 | 26.0 | 0.1091 |
| 5.0 | t2s_degenerate | sampling | 0.9600 | 0.9900 | 0.6700 | 28.0 | 0.2817 |
| 5.0 | qwen_degenerate | greedy | 0.9800 | 1.0000 | 0.6700 | 68.0 | 0.3704 |
| 5.0 | qwen_degenerate | sampling | 0.9500 | 1.0000 | 0.4600 | 128.0 | 0.5152 |
| 5.0 | heldout_degenerate_v2 | greedy | 1.0000 | 1.0000 | 0.2600 | 128.0 | 0.4704 |
| 5.0 | heldout_degenerate_v2 | sampling | 1.0000 | 1.0000 | 0.1400 | 128.0 | 0.5357 |
| 2.0 | t2s_degenerate | greedy | 0.7300 | 0.7500 | 0.6000 | 28.0 | 0.1283 |
| 2.0 | t2s_degenerate | sampling | 0.9700 | 0.9900 | 0.8000 | 27.0 | 0.1796 |
| 2.0 | qwen_degenerate | greedy | 0.9100 | 0.9700 | 0.5300 | 70.0 | 0.3622 |
| 2.0 | qwen_degenerate | sampling | 0.9600 | 1.0000 | 0.5600 | 95.0 | 0.4400 |
| 2.0 | heldout_degenerate_v2 | greedy | 1.0000 | 1.0000 | 0.2600 | 128.0 | 0.4539 |
| 2.0 | heldout_degenerate_v2 | sampling | 1.0000 | 1.0000 | 0.1800 | 128.0 | 0.5191 |
| 1.0 | t2s_degenerate | greedy | 0.3100 | 0.3300 | 0.2800 | 31.0 | 0.0483 |
| 1.0 | t2s_degenerate | sampling | 0.9700 | 0.9900 | 0.8700 | 28.0 | 0.1317 |
| 1.0 | qwen_degenerate | greedy | 0.7000 | 0.7600 | 0.3000 | 128.0 | 0.3043 |
| 1.0 | qwen_degenerate | sampling | 0.9600 | 1.0000 | 0.6100 | 62.5 | 0.3235 |
| 1.0 | heldout_degenerate_v2 | greedy | 0.8000 | 0.8000 | 0.0800 | 128.0 | 0.4583 |
| 1.0 | heldout_degenerate_v2 | sampling | 1.0000 | 1.0000 | 0.2600 | 128.0 | 0.5226 |

## Table 2 — Greedy comparison per fixture (vs night-016 1e4)

night-016 EOS rates (1e4 cell): t2s 0.85, qwen 0.67, heldout_v2 0.26. The best cell minimizes EOS rate while holding rescue rate within 0.05 of the night-016 baseline (t2s ≥ 0.91, qwen ≥ 0.93, heldout ≥ 0.95).

| cell | fixture | rescue | EOS | med n_gen (esc) | vs night-016 EOS |
|---|---|---|---|---|---|
| 1e4 | t2s_degenerate | 0.9600 | 0.8500 | 26.0 | 0.0000 (baseline) |
| 5.0 | t2s_degenerate | 0.9600 | 0.8500 | 26.0 | +0.0000 |
| 2.0 | t2s_degenerate | 0.7300 | 0.6000 | 28.0 | -0.2500 |
| 1.0 | t2s_degenerate | 0.3100 | 0.2800 | 31.0 | -0.5700 |
| 1e4 | qwen_degenerate | 0.9800 | 0.6700 | 68.0 | 0.0000 (baseline) |
| 5.0 | qwen_degenerate | 0.9800 | 0.6700 | 68.0 | +0.0000 |
| 2.0 | qwen_degenerate | 0.9100 | 0.5300 | 70.0 | -0.1400 |
| 1.0 | qwen_degenerate | 0.7000 | 0.3000 | 128.0 | -0.3700 |
| 1e4 | heldout_degenerate_v2 | 1.0000 | 0.2600 | 128.0 | 0.0000 (baseline) |
| 5.0 | heldout_degenerate_v2 | 1.0000 | 0.2600 | 128.0 | +0.0000 |
| 2.0 | heldout_degenerate_v2 | 1.0000 | 0.2600 | 128.0 | +0.0000 |
| 1.0 | heldout_degenerate_v2 | 0.8000 | 0.0800 | 128.0 | -0.1800 |

## Table 3 — Four-bucket histogram over escaped continuations

night-013 convention: n=1 | 2-23 | 24-127 | n=128 over escaped continuations (gross escape = not byte-identical to dormant).

| cell | fixture | arm | n=1 | 2-23 | 24-127 | n=128 |
|---|---|---|---|---|---|---|
| 1e4 | t2s_degenerate | greedy | 0 | 0 | 83 | 13 |
| 1e4 | t2s_degenerate | sampling | 0 | 0 | 63 | 36 |
| 1e4 | qwen_degenerate | greedy | 0 | 0 | 67 | 33 |
| 1e4 | qwen_degenerate | sampling | 0 | 0 | 43 | 57 |
| 1e4 | heldout_degenerate_v2 | greedy | 0 | 0 | 13 | 37 |
| 1e4 | heldout_degenerate_v2 | sampling | 0 | 0 | 7 | 43 |
| 5.0 | t2s_degenerate | greedy | 0 | 0 | 83 | 13 |
| 5.0 | t2s_degenerate | sampling | 0 | 0 | 66 | 33 |
| 5.0 | qwen_degenerate | greedy | 0 | 0 | 67 | 33 |
| 5.0 | qwen_degenerate | sampling | 0 | 0 | 46 | 54 |
| 5.0 | heldout_degenerate_v2 | greedy | 0 | 0 | 13 | 37 |
| 5.0 | heldout_degenerate_v2 | sampling | 0 | 0 | 7 | 43 |
| 2.0 | t2s_degenerate | greedy | 0 | 0 | 58 | 17 |
| 2.0 | t2s_degenerate | sampling | 0 | 0 | 79 | 20 |
| 2.0 | qwen_degenerate | greedy | 0 | 0 | 53 | 44 |
| 2.0 | qwen_degenerate | sampling | 0 | 0 | 56 | 44 |
| 2.0 | heldout_degenerate_v2 | greedy | 0 | 0 | 13 | 37 |
| 2.0 | heldout_degenerate_v2 | sampling | 0 | 0 | 9 | 41 |
| 1.0 | t2s_degenerate | greedy | 0 | 0 | 26 | 7 |
| 1.0 | t2s_degenerate | sampling | 0 | 0 | 86 | 13 |
| 1.0 | qwen_degenerate | greedy | 0 | 0 | 30 | 46 |
| 1.0 | qwen_degenerate | sampling | 0 | 0 | 61 | 39 |
| 1.0 | heldout_degenerate_v2 | greedy | 0 | 0 | 4 | 36 |
| 1.0 | heldout_degenerate_v2 | sampling | 0 | 0 | 13 | 37 |

## Table 4 — Fire-type and activity metrics (per cell, greedy)

bigram-only = bigram fired >= 1 step, spectral never. spectral-only = spectral fired >= 1 step, bigram never. kick = mean kickstart events per record. supp = mean suppression steps per record.

| cell | fixture | fired | bigram-only | spectral-only | both | mean supp | mean kick |
|---|---|---|---|---|---|---|---|
| 1e4 | t2s_degenerate | 97 | 1 | 96 | 0 | 5.48 | 2.91 |
| 1e4 | qwen_degenerate | 100 | 0 | 100 | 0 | 13.51 | 7.32 |
| 1e4 | heldout_degenerate_v2 | 50 | 11 | 26 | 13 | 14.72 | 3.88 |
| 5.0 | t2s_degenerate | 97 | 1 | 96 | 0 | 5.50 | 2.93 |
| 5.0 | qwen_degenerate | 100 | 0 | 100 | 0 | 13.42 | 7.32 |
| 5.0 | heldout_degenerate_v2 | 50 | 11 | 26 | 13 | 14.72 | 3.88 |
| 2.0 | t2s_degenerate | 97 | 1 | 95 | 1 | 29.02 | 26.34 |
| 2.0 | qwen_degenerate | 100 | 0 | 96 | 4 | 21.61 | 12.47 |
| 2.0 | heldout_degenerate_v2 | 50 | 11 | 24 | 15 | 15.82 | 4.26 |
| 1.0 | t2s_degenerate | 97 | 1 | 95 | 1 | 69.21 | 68.04 |
| 1.0 | qwen_degenerate | 100 | 0 | 96 | 4 | 43.05 | 35.81 |
| 1.0 | heldout_degenerate_v2 | 50 | 11 | 26 | 13 | 35.74 | 24.86 |

## Target assessment (hypothesis, NOT a gate)

The 1.0 cell is the strongest candidate: a uniform -1.0 penalty on active_loop_ids is unlikely to crush the loop into EOS-dominance while still nudging away from repeated tokens. The data decides.

| fixture | night-016 rescue | night-016 EOS | min-rescue bar | EOS objective | best cell (by EOS) |
|---|---|---|---|---|---|
| t2s_degenerate | 0.9600 | 0.8500 | 0.9100 | EOS minimized | 1e4 (EOS 0.8500) |
| qwen_degenerate | 0.9800 | 0.6700 | 0.9300 | EOS minimized | 1e4 (EOS 0.6700) |
| heldout_degenerate_v2 | 1.0000 | 0.2600 | 0.9500 | EOS minimized | 1e4 (EOS 0.2600) |

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no threshold changes. All night-016 fixes are implemented INLINE in this probe before they reach the controller [1].
- Cell 1e4 is REUSED from night-016 (kvcache_restructure_results.jsonl); it is NOT re-run. The reused 1e4 records are copied verbatim into penalty_magnitude_results.jsonl with `reused_from` provenance.
- Dormant baselines are REUSED from night-016 (greedy + sampling KV-cache, already validated 0/250 drift). They are NOT re-run; the ΔD2 deltas for all four cells are computed against the same night-016 dormant baseline, so rescue-rate differences are attributable to the penalty magnitude alone.
- The three NEW cells (5.0, 2.0, 1.0) each ran with the night-016 restructured regime: all six fixes inline, suppression cooldown=8 penalty=-5.0, kickstart retargeted to active_loop_ids, top_p=0.85 when either mechanism is active.
- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 (matching night-014/016). SEED=42 with seed_all(SEED) before every sampling generation; the per-cell determinism smoke confirms reproducible sampling.
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
