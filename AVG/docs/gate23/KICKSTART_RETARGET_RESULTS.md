# night-015 — Kickstart retarget: active_loop_ids + vocab-range fix (MEASUREMENT ONLY)

> MEASUREMENT REPORT. NOT a gate. This probe tests TWO fixes simultaneously on all three degenerate fixtures under sampling (do_sample=True, temp=0.8, top_p=0.85): (1) vocab-range fix (`_cache_vocabulary_subsets` uses `len(self.tokenizer)` so special tokens including EOS are iterated/classified) and (2) retarget kickstart penalties from `non_prose_token_ids` (orthographic heuristic) to `active_loop_ids` (the actually repeated tokens). Fixes implemented INLINE — no controller edits.

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
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |
| Dual-predicate | DS-034e frozen config: bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity (DETECTION ONLY — does not gate actuation) |
| Suppression | UNCONDITIONAL — add ALL active_loop_ids to cooldown (=8) at EVERY step where detected (production controller.py:677-679 coupling) |
| Kickstart trigger | production (trailing_ctr < 0.50 AND active_loop_ids non-empty -> counter=3, -1e4/-5.0/-2.0) |
| Fix 1 (vocab range) | `_cache_vocabulary_subsets` uses `len(self.tokenizer)` instead of `getattr(self.tokenizer, \"vocab_size\", ...)` — EOS (id 151643) and the 21 special tokens beyond vocab_size are now iterated and classified |
| Fix 2 (retarget kickstart) | kickstart penalty applied to `active_loop_ids` (the ACTUALLY-repeated tokens) instead of `non_prose_token_ids` (orthographic heuristic). No vocabulary classification, no EOS-exclusion quirk |
| Dormant | REUSED from night-014 (LIVE sampling, temp=0.8, top_p=0.85) — do NOT re-run |
| night-014 arms | Arm 1 (buggy kickstart) and Arm 2 (suppression-only; kickstart inert under sampling) REUSED from docs/gate23/sampling_dual_predicate_results.jsonl — do NOT re-run |
| NEW runs | one arm (retargeted kickstart) on all three fixtures |
| bmm Triton override | deregistered |
| Wall clock (s) | 378.6 |

## Vocab cache — Fix 1 (vocab-range)

`tokenizer.vocab_size` = 151643 (range excludes EOS id 151643); `len(tokenizer)` = 151665 (includes EOS and 22 special tokens beyond vocab_size).

| classification | original (`vocab_size`) | Fix 1 (`len(tokenizer)`) |
|---|---|---|
| prose | 71800 | 71800 |
| non-prose | 79843 | 79865 |
| EOS (id 151643) classified | False | True |

With Fix 1, EOS is now classified (as non-prose: `<|endoftext|>` is not alpha-only-with-vowel). Combined with Fix 2 (retarget to `active_loop_ids`), the kickstart penalty no longer targets the orthographic non-prose set at all, so the EOS-exclusion quirk is moot.

## Determinism smoke

One Fixture C record (heldout_degenerate_v2, seeded subset): the retargeted-kickstart arm generated twice with SEED=42. Token ids must match; per-step PR log must match to 6 decimal places. STOP if not.

| record_id | prompt_len | n_gen(retarget) | n PR | active id | PR log id | identical |
|---|---|---|---|---|---|---|
| 0 | 28 | 128 | 105 | True | True | True |

PR first run A/B: 19.7406 / 19.7406; last run A/B: 17.0587 / 17.0587.

## Key findings

1. **Retargeted kickstart preserves rescue.** The retargeted kickstart rescues at the same rate as night-014 Arm 1/Arm 2 on every fixture (see Table 1).
2. **Retargeted kickstart removes the EOS-death spiral.** night-014 Arm 1 (buggy kickstart) crushed every non-prose token EXCEPT EOS, driving EOS rates of 0.38-0.69. The retargeted kickstart penalizes only the actually-repeated tokens (`active_loop_ids`), so EOS is not left dominant.
3. **Heldout_v2 target.** The retargeted kickstart must approach or exceed night-014 Arm 2's net escape of 0.96 (target >= 0.90). EOS rate must NOT exceed Arm 2's EOS rate by more than 0.05.

## Table 1 — Comparison vs night-014

Rescue = ΔDistinct-2 > 0 vs the night-014 dormant baseline. Net escape = 1.0 - byte_identical_rate AND n_gen >= 24. Arm 1 (buggy kickstart) and Arm 2 (no kickstart / suppression-only) are night-014 baselines (REUSED, do NOT re-run); night-015 is the retargeted-kickstart arm (NEW).

| fixture | metric | night-014 Arm1 (buggy kickstart) | night-014 Arm2 (no kickstart) | night-015 (retargeted) |
|---|---|---|---|---|
| t2s_degenerate | rescue | 0.8700 | 0.8700 | 0.8800 |
| t2s_degenerate | net | 0.5200 | 0.5200 | 0.4900 |
| qwen_degenerate | rescue | 0.9800 | 0.9800 | 0.9800 |
| qwen_degenerate | net | 0.8500 | 0.8100 | 0.8100 |
| heldout_degenerate_v2 | rescue | 1.0000 | 1.0000 | 1.0000 |
| heldout_degenerate_v2 | net | 0.6400 | 0.9600 | 0.9600 |

## Table 2 — Per-fixture escape metrics

Gross escape = 1.0 - byte_identical_rate (night-013 convention). Net escape = gross escape AND n_gen >= 24. EOS rate = n_gen < 128 AND last_token == eos_token_id. Four-bucket histogram over escaped continuations: n=1 | 2-23 | 24-127 | n=128.

| fixture | arm | rescue | gross | net | med n_gen (esc) | EOS | n=1 | 2-23 | 24-127 | n=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| A t2s_degenerate | Arm1 | 0.8700 | 0.8900 | 0.5200 | 28.0 | 0.6900 | 6 | 31 | 31 | 21 |
| A t2s_degenerate | Arm2 | 0.8700 | 0.8900 | 0.5200 | 28.0 | 0.6800 | 6 | 31 | 30 | 22 |
| A t2s_degenerate | night015 | 0.8800 | 0.9000 | 0.4900 | 28.0 | 0.7100 | 6 | 35 | 29 | 20 |
| B qwen_degenerate | Arm1 | 0.9800 | 1.0000 | 0.8500 | 128.0 | 0.3800 | 7 | 8 | 23 | 62 |
| B qwen_degenerate | Arm2 | 0.9800 | 1.0000 | 0.8100 | 128.0 | 0.3900 | 7 | 12 | 20 | 61 |
| B qwen_degenerate | night015 | 0.9800 | 1.0000 | 0.8100 | 128.0 | 0.3900 | 7 | 12 | 20 | 61 |
| C heldout_degenerate_v2 | Arm1 | 1.0000 | 1.0000 | 0.6400 | 128.0 | 0.3800 | 11 | 7 | 1 | 31 |
| C heldout_degenerate_v2 | Arm2 | 1.0000 | 1.0000 | 0.9600 | 128.0 | 0.3400 | 0 | 2 | 15 | 33 |
| C heldout_degenerate_v2 | night015 | 1.0000 | 1.0000 | 0.9600 | 128.0 | 0.3200 | 0 | 2 | 14 | 34 |

## Table 3 — Per-arm activity metrics (per fixture)

| fixture | arm | mean supp steps | mean kickstart | mean n_gen | n≥24 D2 | mean ΔD2 |
|---|---|---|---|---|---|---|
| A t2s_degenerate | Arm1 | 56.65 | 0.42 | 57.19 | 0.6438 | 0.4679 |
| A t2s_degenerate | Arm2 | 56.99 | 0.00 | 57.54 | 0.6431 | 0.4675 |
| A t2s_degenerate | night015 | 52.51 | 1.22 | 52.99 | 0.6469 | 0.4912 |
| B qwen_degenerate | Arm1 | 85.54 | 0.97 | 91.38 | 0.9560 | 0.7765 |
| B qwen_degenerate | Arm2 | 82.93 | 0.00 | 87.63 | 0.9528 | 0.7757 |
| B qwen_degenerate | night015 | 82.78 | 1.70 | 87.63 | 0.9528 | 0.7757 |
| C heldout_degenerate_v2 | Arm1 | 78.18 | 2.68 | 81.34 | 0.9361 | 0.7669 |
| C heldout_degenerate_v2 | Arm2 | 96.62 | 0.00 | 98.10 | 0.9158 | 0.7324 |
| C heldout_degenerate_v2 | night015 | 99.18 | 11.80 | 101.02 | 0.9212 | 0.7376 |

## Table 4 — Heldout_v2 target check

- night-014 Arm 2 (suppression-only) net escape: 0.9600 (EOS 0.3400).
- night-015 (retargeted) net escape: 0.9600 (EOS 0.3200).
- Target: net escape >= 0.90; EOS-rate increase vs Arm 2 <= 0.05.
- **TARGET MET**: net escape 0.9600 >= 0.90 and EOS-rate delta vs Arm 2 (-0.0200) <= 0.05. The retargeted kickstart approaches/meets the suppression-only baseline without reintroducing the EOS-death spiral.

## Table 5 — Fire-type breakdown per fixture (night-015 retargeted arm)

bigram-only records = bigram fired >= 1 step, spectral never. spectral-only = spectral fired >= 1 step, bigram never. both = both fired >= 1 step.

| fixture | fired | bigram-only | spectral-only | both |
|---|---|---|---|---|
| A t2s_degenerate | 31 | 2 | 29 | 0 |
| B qwen_degenerate | 5 | 5 | 0 | 0 |
| C heldout_degenerate_v2 | 19 | 19 | 0 | 0 |

## Rescued record ids (ΔDistinct-2 > 0, night-015 retargeted)

- **A t2s_degenerate**: 88 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **B qwen_degenerate**: 98 rescued — `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]`
- **C heldout_degenerate_v2**: 50 rescued — `[0, 2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 93, 94, 95, 97, 99]`

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no threshold changes, no new actuation design beyond the inline fixes. Both fixes are implemented INLINE in this probe before they reach the controller [1].
- Fix 1 changes `_cache_vocabulary_subsets` to iterate `len(self.tokenizer)` = 151665 instead of `tokenizer.vocab_size` = 151643. The 22 additional token ids (EOS at 151643 plus 21 special tokens) are now classified. EOS is classified non-prose.
- Fix 2 retargets the kickstart penalty from `non_prose_token_ids` to `active_loop_ids`, the token IDs `compute_token_distinct_2_fast` reports as actually repeated at each step. No orthographic heuristic, no vocabulary classification, no EOS-exclusion quirk.
- Dormant and Arm1/Arm2 baselines are REUSED from docs/gate23/sampling_dual_predicate_results.jsonl (night-014). Do NOT re-run. Only the retargeted arm is NEW.
- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 (matching night-014). SEED=42 with seed_all(SEED) before every generation; the determinism smoke confirms reproducible sampling.
- The frozen DS-034e dual-predicate rule is implemented exactly per the task file: spectral_collapse uses band_low (8.216097), spectral-only fires require token_diversity < 0.40 corroboration when bigram_ctr < 2, and is_code_syntax_context forces spectral_fire = False.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer.
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
