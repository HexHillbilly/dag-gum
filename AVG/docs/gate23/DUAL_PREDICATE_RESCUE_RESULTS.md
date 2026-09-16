# DS-035 — Dual-predicate cross-fixture rescue gate (GATE)

> GATE REPORT (RFC-004 Amendment A3, owner directive). The dual-predicate's entire value proposition is rescue efficacy. Rescue >= 1 on Fixture A or B ratifies A3. Rescue = 0 on both auto-reverts to bigram-only and the dual-predicate is shelved.

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
| Dual-predicate | DS-034e frozen config: bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| Actuation | production logit-penalty path: token suppression cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85 |
| Dormant | LIVE (greedy, KV-cache) for Fixtures A/B; REUSED from DS-028 (layer_effect_persistent_results.jsonl, layer==2, join on record_id) for Fixture C |
| DS-033 baselines | REUSED from production_cross_fixture_results.jsonl (do NOT re-run) |
| bmm Triton override | deregistered |
| Wall clock (s) | 996.0 |

## Gate criteria (quoted from the DS-035 task file / owner directive)

> - Rescue ≥ 1 on Fixture A or B → PASS. A3 ratifies.
> - Rescue = 0 on BOTH → FAIL. A3 auto-reverts to bigram-only [1].
> - Fixture C: rescue rate ≥ 85/100 (positive control — no regression).

Rescue is defined per the DS-033 convention: a record is rescued iff ΔDistinct-2 > 0 vs its dormant baseline (active distinct-2 minus dormant distinct-2 over the trailing 24 generated positions).

## Determinism smoke

One Fixture A record: dormant generated twice AND active (DS-034e dual-predicate) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

| record_id | prompt_len | n_gen | n PR | dormant id | active id | PR log id | identical |
|---|---|---|---|---|---|---|---|
| 0 | 199 | 128 | 105 | True | True | True | True |
|  | ALL |  |  |  |  |  | True |

PR first run A/B: 7.0874 / 7.0874; last run A/B: 7.0750 / 7.0750.

## Verdict

| metric | value |
|---|---|
| t2s_degenerate rescue count | 20/100 (20.00%) |
| qwen_degenerate rescue count | 20/100 (20.00%) |
| heldout_degenerate_v2 rescue count | 48/50 (96.00%) |
| Fixture C rescue rate ≥ 85% | PASS |
| **Gate verdict** | **PASS** |

**PASS** — rescue count ≥ 1 on Fixture A or B. Amendment A3 ratifies; the dual-predicate rescue efficacy is established. FP optimization becomes the next night gate.

## Comparison table

| fixture | production rescue rate [DS-033] | dual-predicate rescue rate | delta |
|---|---|---|---|
| t2s_degenerate | 0.1500 | 0.2000 | 0.0500 |
| qwen_degenerate | 0.6800 | 0.2000 | -0.4800 |
| heldout_degenerate_v2 | 0.9600 | 0.9600 | 0.0000 |

DS-033 production baseline rescue rates for Fixtures A/B are REUSED from `docs/gate23/production_cross_fixture_results.jsonl` (do NOT re-run). DS-033 did not cover Fixture C; the Fixture C production baseline is the DS-031 suppression-only 48/50 = 0.9600 (documented in the DS-035 task file).

## Table 1 — Rescue metrics per fixture

| fixture | n | rescued | rate | ΔD2 mean | ≥1 fire | mean n_gen | n≥24 D2 |
|---|---|---|---|---|---|---|---|
| A t2s_degenerate | 100 | 20 | 0.2000 | 0.0283 | 97 | 111.34 | 0.2680 |
| B qwen_degenerate | 100 | 20 | 0.2000 | 0.0091 | 100 | 113.93 | 0.1626 |
| C heldout_degenerate_v2 | 50 | 48 | 0.9600 | 0.2713 | 50 | 55.62 | 0.3787 |

## Table 2 — Fire-type counts per fixture (active)

| fixture | records ≥1 fire | bigram-fire records | spectral-fire records | both-fire records | bigram steps | spectral steps | both steps |
|---|---|---|---|---|---|---|---|
| A | 97 | 1 | 96 | 0 | 1 | 8018 | 0 |
| B | 100 | 0 | 100 | 0 | 0 | 8570 | 0 |
| C | 50 | 21 | 38 | 4 | 169 | 314 | 74 |

A record can have both bigram and spectral fires (possibly at different steps); the 'both-fire records' column counts records with at least one step where both fired simultaneously.

## Table 3 — Dormant reference (per fixture)

| fixture | n | mean n_gen | mean D2 | source |
|---|---|---|---|---|
| A | 100 | 125.67 | 0.2465 | live (greedy, KV-cache) |
| B | 100 | 128.00 | 0.1535 | live (greedy, KV-cache) |
| C | 50 | 128.00 | 0.1696 | reused from DS-028 (layer_effect_persistent_results.jsonl) |

## Rescue provenance vs DS-033 production

For Fixtures A/B, the dual-predicate rescues are decomposed against the DS-033 production rescued record-id set (REUSED from production_cross_fixture_results.jsonl).

| fixture | prod rescued | dual rescued | overlap | NEW (dual-only) | regressed (prod-only) |
|---|---|---|---|---|---|
| A t2s_degenerate | 15 | 20 | 9 | 11 | 6 |
| B qwen_degenerate | 68 | 20 | 20 | 0 | 48 |

**A t2s_degenerate** — new (dual-only) rescues: `[17, 21, 23, 31, 35, 42, 50, 53, 85, 88, 93]`; regressed (production-only): `[28, 37, 40, 65, 70, 86]`.

**B qwen_degenerate** — new (dual-only) rescues: `[]`; regressed (production-only): `[2, 3, 5, 6, 7, 8, 12, 13, 14, 22, 23, 24, 25, 27, 28, 29, 30, 37, 38, 39, 49, 50, 51, 63, 64, 65, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 90, 91, 92, 93, 94, 95]`.

NEW rescues are records the production bigram-only path missed but the dual-predicate rescued; these are the dual-predicate's incremental rescue efficacy.

## DS-034e spectral coverage validation (t2s detection-gap)

Of the 93 DS-033 t2s_degenerate detection-gap records (bigram never fired), the DS-034e dual-predicate fires on 92 — 98.9% coverage. DS-034e reported 92/93; this run confirms the spectral liveness on the primary blind-spot target.

## Rescued record ids (ΔDistinct-2 > 0)

- **A t2s_degenerate**: 20 rescued — `[12, 17, 21, 23, 31, 35, 42, 44, 50, 53, 72, 76, 80, 83, 85, 88, 92, 93, 97, 99]`
- **B qwen_degenerate**: 20 rescued — `[0, 1, 9, 10, 11, 15, 16, 17, 18, 26, 31, 32, 33, 34, 35, 36, 66, 84, 85, 86]`
- **C heldout_degenerate_v2**: 48 rescued — `[2, 3, 4, 5, 6, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 27, 28, 29, 31, 35, 38, 43, 46, 51, 53, 54, 57, 58, 62, 64, 67, 68, 69, 71, 75, 77, 79, 81, 82, 84, 86, 88, 89, 90, 94, 95, 97, 99]`

## Notes / caveats

- GATE: the pass/fail criterion is explicit and quoted from the task file. Rescue = ΔDistinct-2 > 0 (DS-033 convention). A FAIL is a STOP signal — the script exits non-zero; no thresholds are negotiated or tuned.
- The frozen DS-034e dual-predicate rule is implemented exactly per the task file: spectral_collapse uses band_low (8.216097), spectral-only fires require token_diversity < 0.40 corroboration when bigram_ctr < 2, and is_code_syntax_context forces spectral_fire = False.
- Actuation on is_collapsed uses the production logit-penalty path (token suppression cooldown=8 penalty=-5.0, kickstart counter=3 with -1e4/-5.0/-2.0, top_p=0.85). NO residual hooks.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer.
- Fixture A/B dormant baselines are generated LIVE (greedy, KV-cache, no hooks/penalties). Fixture C dormant is REUSED from DS-028 (layer_effect_persistent_results.jsonl, layer==2, join on record_id) — the 50-record seeded subset matches exactly.
- DS-033 production baseline rescue rates for Fixtures A/B are REUSED from production_cross_fixture_results.jsonl. Fixture C production baseline is 48/50 (DS-031 suppression-only; DS-033 did not cover Fixture C).
- **qwen_degenerate regression (0.68 → 0.20).** The dual-predicate actuation is gated by is_collapsed: suppression tokens are added to the cooldown dict only when the collapse predicate fires, whereas the DS-033 production combined path adds loop-token suppression at every step where active_loop_ids is non-empty. On qwen_degenerate this more conservative actuation rescues fewer records (20 vs 68). All 20 dual rescues overlap with production; 0 new rescues on qwen. The 11 new spectral rescues are on t2s_degenerate (the primary blind-spot target).
- **t2s_degenerate improvement (0.15 → 0.20).** The dual-predicate adds 11 new rescues (records 17, 21, 23, 31, 35, 42, 50, 53, 85, 88, 93) that the production bigram-only path missed — all spectral-only fires. This is the incremental rescue efficacy the gate exists to measure.
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.
