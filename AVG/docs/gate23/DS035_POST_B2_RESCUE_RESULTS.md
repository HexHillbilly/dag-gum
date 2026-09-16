# DS-035 (post-B2) — Cross-Fixture Rescue Gate — GATE REPORT

> GATE REPORT. Measured against the production `ActiveVarietyGovernor` with the
> B3 kickstart-under-sampling change committed (supersedes B1 greedy-only).
> Success bar: sampling rescue >= 85/100 on BOTH qwen_degenerate and
> t2s_degenerate. Auto-revert floor: rescue == 0 on both.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda (RTX 3060 12GB) |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Controller | production ActiveVarietyGovernor (B1 + B2 + B3 committed) |
| Fixture A | t2s_degenerate (100), record["text"] prompt |
| Fixture B | qwen_degenerate (100), record["prompt"] prompt |
| Prose-100 | valid_subset_200 heldout partition (N=100), record["text"] prompt |
| Primary regime | sampling (do_sample=True, temp=0.8, top_p=0.85) |
| Secondary regime | greedy (do_sample=False) |
| Analysis window | trailing 24 generated positions |

## Determinism + liveness

- Determinism smoke (greedy, t2s record 0, 2 runs): token-identical **True**,
  `_layer2_pr` identical **True**.
- Spectral liveness (`_layer2_pr` non-None): **True** (BLOCKER #1 fixed by B2).

## Sampling rescue (PRIMARY)

| fixture | rescued | rate | EOS | mean ΔD2 |
|---|---|---|---|---|
| t2s_degenerate | **97/100** | 0.970 | 0 | +0.712 |
| qwen_degenerate | **97/100** | 0.970 | 0 | +0.796 |

- Success bar (>= 85/100 on both): **PASS**.
- Auto-revert floor (== 0 on both): **not triggered**.

### Detection-gap subset (bigram never fired)

| fixture | records | rescued | rate |
|---|---|---|---|
| t2s_degenerate | 95 | 93 | 0.979 |
| qwen_degenerate | 91 | 88 | 0.967 |

The spectral predicate + sampling kickstart rescues the bigram blind-spot
records at 0.97-0.98.

## Greedy rescue (secondary, not gated)

| fixture | rescued | rate |
|---|---|---|
| t2s_degenerate | 98/100 | 0.98 |
| qwen_degenerate | 98/100 | 0.98 |

## FP ledger (prose-100, sampling)

| metric | value |
|---|---|
| Records | 100 |
| False positives (ΔD2 > 0 AND fire > 0) | **0** (accept <= 7) |
| Records with >= 1 fire | 0 |
| Any ΔD2 > 0 (stochastic, no fire) | 34 |

Prompt-length note: prose-100 `text` fields run 70-7,265 tokens (median 824).
FP scoring used a 512-token prompt truncation to fit the 12GB GPU; the
dual-predicate fires within the first ~24 generated tokens if it fires at all,
so truncation does not mask a false positive. Zero fires = zero FPs.

## Verdict

**PASS** — sampling rescue >= 85/100 on both fixtures (97/100 each). The
A3 escape clause does not auto-revert (rescue >> 0). The dual-predicate now
rescues degenerate loops on the production path under BOTH regimes.

## Delta vs pre-B3 (suppression-only sampling)

| fixture | pre-B3 | post-B3 |
|---|---|---|
| t2s_degenerate | 79/100 | **97/100** |
| qwen_degenerate | 97/100 | 97/100 |

The t2s under-actuation was caused by B1's greedy-only kickstart gate leaving
sampling with only -5.0 suppression. Re-enabling the kickstart under sampling
(B3, docs/B3_KICKSTART_SAMPLING_REENABLE.md) closes the gap with zero EOS
bailout (the B1 retarget to active_loop_ids removed the night-017 EOS-death
mechanism).

## Notes / caveats

- Sampling rescue results measured on the identical committed controller
  (`f890a42`) via scripts/measure_ds035_kickstart_sampling.py.
- Greedy results unchanged by B3 (kickstart was already active under greedy);
  qwen confirmed 98/100 post-B3, t2s 98/100 pre-B3.
- prose-100 FP uses a 512-token prompt truncation (documented above).
- The 3 sampling non-rescues per fixture all have dormant D2 ~= 1.0 (the
  dormant baseline already escapes the loop; nothing to rescue).
