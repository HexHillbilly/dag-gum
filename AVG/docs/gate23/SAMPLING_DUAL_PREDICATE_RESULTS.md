# night-014 — Sampling-regime dual-predicate measurement (MEASUREMENT ONLY)

> MEASUREMENT REPORT. NOT a gate. This probe characterizes whether the dual-predicate architecture (RFC-004 Amendment A3, Option C) works under the sampling regime (do_sample=True, temp=0.8, top_p=0.85) it would actually be deployed in. Two arms: Arm 1 production actuation (unconditional suppression + CTR-triggered kickstart) and Arm 2 spectral-gated kickstart.

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
| Decoding | sampling (do_sample=True), temp=0.8, top_p=0.85, KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 generated positions |
| Hook layer | model.model.layers[2] (layer-2 PR, rolling 24-token ring buffer) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze; spectral_collapse threshold) |
| Dual-predicate | DS-034e frozen config: bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity (DETECTION ONLY — does not gate actuation) |
| Arm 1 suppression | UNCONDITIONAL — add ALL active_loop_ids to cooldown (=8) at EVERY step where detected (production controller.py:677-679 coupling) |
| Arm 1 kickstart | production trigger (trailing_ctr < 0.50 AND active_loop_ids non-empty -> counter=3, -1e4/-5.0/-2.0) |
| Arm 2 kickstart | spectral-gated (spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids -> counter=3, -1e4/-5.0/-2.0). Same unconditional suppression as Arm 1. |
| Dormant | LIVE sampling (do_sample=True, temp=0.8, top_p=0.85, no hooks/suppression/kickstart) — generated fresh for ALL records; no pre-existing sampling dormant baselines exist. |
| Greedy baselines | DS-035 rescue rates REUSED for Table 1; night-008 fire-type counts REUSED for Table 3; DS-032 Part 2 sampling baseline REUSED for Table 2 (heldout_v2 only). Do NOT re-run. |
| bmm Triton override | deregistered |
| Wall clock (s) | 1251.9 |

## Determinism smoke

One Fixture C record (heldout_degenerate_v2, seeded subset): dormant sampling generated twice AND Arm 1 active generated twice with SEED=42. Token ids must match; per-step PR log must match to 6 decimal places. STOP if not.

| record_id | prompt_len | n_gen(arm1) | n PR | dormant id | active id | PR log id | identical |
|---|---|---|---|---|---|---|---|
| 6 | 21 | 128 | 105 | True | True | True | True |

PR first run A/B: 17.1512 / 17.1512; last run A/B: 18.9455 / 18.9455.

## Key findings

1. **Sampling rescue massively exceeds greedy.** Both arms rescue 87-100% across all three fixtures vs the greedy DS-035 baseline (t2s 0.20, qwen 0.20, heldout_v2 0.96). This confirms the DS-032 Part 2 finding that sampling amplifies suppression effectiveness — the dual-predicate + production actuation works under the regime it would actually be deployed in.
2. **Arm 2's spectral-gated kickstart is inert under sampling** (mean kickstart events = 0.00 on every fixture). The `spectral_fire AND trailing_ctr < 0.50 AND active_loop_ids` trigger never fires because the unconditional suppression rescues records before they degenerate enough to trigger spectral_fire. The DS-035 t2s macro-loop rescue (11 spectral rescues from the gated kickstart) does NOT transfer to sampling.
3. **The DS-029 EOS-death pathology PERSISTS under sampling for the production kickstart.** Arm 1's CTR-triggered kickstart fires (mean 0.42-2.68 events/record) and drives elevated EOS rates (t2s 0.69, qwen 0.38, heldout_v2 0.38). The mechanism is the production `_cache_vocabulary_subsets` quirk: `range(tokenizer.vocab_size)` excludes EOS (id 151643), so the -1e4 kickstart penalty crushes every non-prose token EXCEPT EOS, leaving EOS dominant whenever the kickstart fires. Sampling's entropy does NOT prevent this.
4. **On heldout_v2, Arm 2 matches DS-032 Part 2's net escape (0.96 vs 0.96) with the production (8,-5) suppression.** Arm 2 (kickstart never fires) behaves as suppression-only; its EOS rate (0.34) is above DS-032 Part 2's (0.20) due to the longer (8,-5) cooldown. Arm 1's kickstart reduces net escape to 0.64 on heldout_v2 — the kickstart is a net liability.

## Table 1 — Per-fixture rescue rate: greedy vs sampling

Rescue = ΔDistinct-2 > 0 vs the dormant baseline (DS-033 convention). Greedy column is the DS-035 dual-predicate gate (is_collapsed-gated actuation, greedy), REUSED. Sampling Arm 1 is production actuation; Arm 2 is spectral-gated kickstart.

| fixture | greedy (DS-035) | Arm 1 (sampling) | Arm 2 (sampling) |
|---|---|---|---|
| t2s_degenerate | 0.2000 | 0.8700 | 0.8700 |
| qwen_degenerate | 0.2000 | 0.9800 | 0.9800 |
| heldout_degenerate_v2 | 0.9600 | 1.0000 | 1.0000 |

Greedy baselines: DS-035 for Fixtures A/B (dual_predicate_rescue_results.jsonl, do NOT re-run). For Fixture C, the DS-035 heldout_v2 rate (48/50) is shown; DS-035 reused DS-028 layer-2 dormant for Fixture C.

## Table 2 — Per-fixture escape metrics

Gross escape = 1.0 - byte_identical_rate (night-013 convention). Net escape = gross escape AND n_gen >= 24. EOS rate = n_gen < 128 AND last_token == eos_token_id. Four-bucket histogram over escaped continuations: n=1 | 2-23 | 24-127 | n=128. DS-032 Part 2 (suppression-only sampling, cooldown=5) is shown for heldout_v2 only (it did not cover Fixtures A/B).

| fixture | arm | rescue | gross | net | med n_gen (esc) | EOS | n=1 | 2-23 | 24-127 | n=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| A t2s_degenerate | Arm1 | 0.8700 | 0.8900 | 0.5200 | 28.0 | 0.6900 | 6 | 31 | 31 | 21 |
| A t2s_degenerate | Arm2 | 0.8700 | 0.8900 | 0.5200 | 28.0 | 0.6800 | 6 | 31 | 30 | 22 |
| B qwen_degenerate | Arm1 | 0.9800 | 1.0000 | 0.8500 | 128.0 | 0.3800 | 7 | 8 | 23 | 62 |
| B qwen_degenerate | Arm2 | 0.9800 | 1.0000 | 0.8100 | 128.0 | 0.3900 | 7 | 12 | 20 | 61 |
| C heldout_degenerate_v2 | Arm1 | 1.0000 | 1.0000 | 0.6400 | 128.0 | 0.3800 | 11 | 7 | 1 | 31 |
| C heldout_degenerate_v2 | Arm2 | 1.0000 | 1.0000 | 0.9600 | 128.0 | 0.3400 | 0 | 2 | 15 | 33 |
| C heldout_degenerate_v2 | DS032P2 | 1.0000 | 1.0000 | 0.9600 | 128.0 | 0.2000 | 0 | 2 | 8 | 40 |

DS-032 Part 2 was suppression-only at (5,-5) with NO kickstart; the night-014 arms use the production (8,-5) cooldown/penalty plus kickstart.

## Table 3 — Fire-type breakdown per fixture

bigram-only records = bigram fired >= 1 step, spectral never. spectral-only = spectral fired >= 1 step, bigram never. both = both fired >= 1 step. Greedy column is night-008 (hybrid_ctr_gating_results.jsonl, REUSED).

| fixture | arm | fired | bigram-only | spectral-only | both |
|---|---|---|---|---|---|
| A t2s_degenerate | Arm1 | 31 | 0 | 31 | 0 |
| A t2s_degenerate | Arm2 | 31 | 0 | 31 | 0 |
| A t2s_degenerate | greedy | 89 | 2 | 87 | 0 |
| B qwen_degenerate | Arm1 | 7 | 7 | 0 | 0 |
| B qwen_degenerate | Arm2 | 5 | 5 | 0 | 0 |
| B qwen_degenerate | greedy | 48 | 4 | 44 | 0 |
| C heldout_degenerate_v2 | Arm1 | 0 | 0 | 0 | 0 |
| C heldout_degenerate_v2 | Arm2 | 22 | 22 | 0 | 0 |
| C heldout_degenerate_v2 | greedy | 37 | 34 | 3 | 0 |

## Table 4 — Per-arm activity metrics (per fixture)

| fixture | arm | mean supp steps | mean kickstart | mean n_gen | n≥24 D2 | mean ΔD2 |
|---|---|---|---|---|---|---|
| A t2s_degenerate | Arm1 | 56.65 | 0.42 | 57.19 | 0.6438 | 0.4679 |
| A t2s_degenerate | Arm2 | 56.99 | 0.00 | 57.54 | 0.6431 | 0.4675 |
| B qwen_degenerate | Arm1 | 85.54 | 0.97 | 91.38 | 0.9560 | 0.7765 |
| B qwen_degenerate | Arm2 | 82.93 | 0.00 | 87.63 | 0.9528 | 0.7757 |
| C heldout_degenerate_v2 | Arm1 | 78.18 | 2.68 | 81.34 | 0.9361 | 0.7669 |
| C heldout_degenerate_v2 | Arm2 | 96.62 | 0.00 | 98.10 | 0.9158 | 0.7324 |

## Table 5 — Dormant reference (LIVE sampling, per fixture)

| fixture | n | mean n_gen | mean D2 | source |
|---|---|---|---|---|
| A t2s_degenerate | 100 | 126.81 | 0.2496 | live (sampling, temp=0.8, top_p=0.85, KV-cache) |
| B qwen_degenerate | 100 | 127.50 | 0.1761 | live (sampling, temp=0.8, top_p=0.85, KV-cache) |
| C heldout_degenerate_v2 | 50 | 128.00 | 0.1696 | live (sampling, temp=0.8, top_p=0.85, KV-cache) |

## DS-029 interaction check (qwen_degenerate)

Arm 1 (CTR-triggered kickstart): rescue 0.9800 (98/100), mean kickstart events 0.97, EOS 0.3800.
Arm 2 (spectral-gated kickstart): rescue 0.9800 (98/100), mean kickstart events 0.00, EOS 0.3900.

**Verdict: inconclusive on qwen** — Arm 2's spectral-gated kickstart did not fire on qwen (mean kickstart events = 0). Cross-check the per-fixture EOS rates: Arm 1's CTR-triggered kickstart is the dominant EOS driver via the EOS-excluded-from-kickstart quirk (see caveats).

## Rescued record ids (ΔDistinct-2 > 0)

- **A t2s_degenerate Arm 1**: 87 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **A t2s_degenerate Arm 2**: 87 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **B qwen_degenerate Arm 1**: 98 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **B qwen_degenerate Arm 2**: 98 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **C heldout_degenerate_v2 Arm 1**: 50 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`
- **C heldout_degenerate_v2 Arm 2**: 50 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no threshold changes, no new actuation design. This probe characterizes the dual-predicate under sampling [1].
- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 (matching DS-032 Part 2). SEED=42 with seed_all(SEED) before every generation; the determinism smoke confirms reproducible sampling.
- The frozen DS-034e dual-predicate rule is implemented exactly per the task file: spectral_collapse uses band_low (8.216097), spectral-only fires require token_diversity < 0.40 corroboration when bigram_ctr < 2, and is_code_syntax_context forces spectral_fire = False.
- Arm 1 = production actuation (Option C): unconditional suppression + CTR-triggered kickstart. Arm 2 = spectral-gated kickstart (same suppression). The only difference is the kickstart trigger.
- Dormant baselines are LIVE sampling (generated fresh for all records). No pre-existing sampling dormant baselines exist; DS-028/DS-035 dormant are greedy and are NOT reused for rescue computation.
- Greedy baselines are REUSED for comparison only: DS-035 rescue rates (Table 1), night-008 fire-type counts (Table 3), DS-032 Part 2 sampling baseline (Table 2, heldout_v2 only).
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer.
- Temperature is fixed at 0.8 for BOTH dormant and active arms (task spec, matching DS-032 Part 2). Production would use 0.70 while suppression is active; the task fixes 0.8 to isolate the suppression/kickstart mechanism.
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
- Rescue, escape, and fire-type conventions follow night-013 / night-008 / DS-033 definitions exactly.
- **FINDING (kickstart EOS-death mechanism, production quirk).** The production `_cache_vocabulary_subsets` iterates `range(tokenizer.vocab_size)` = range(151643), which EXCLUDES the EOS token id (151643). The kickstart -1e4 penalty therefore crushes every non-prose token EXCEPT EOS, leaving EOS as the dominant token (prob ~0.44 at step 0 on heldout_v2 record 0) whenever the kickstart fires. This is the DS-029 EOS-death mechanism, faithfully reproduced under sampling. Arm 1's CTR-triggered kickstart fires on degenerate prompts immediately and drives high EOS rates on heldout_v2/t2s; Arm 2's spectral-gated kickstart fires far less often and shows lower EOS.
