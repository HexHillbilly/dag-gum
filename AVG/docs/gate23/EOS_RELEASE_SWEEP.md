# Bounded EOS-release sweep (0.5B / 1.5B / 3B)

**Date:** 2026-08-13
**Change:** `governor/controller.py` — added `eos_release_after` (default 0) constructor
param. After `N` consecutive loop-free steps, `<eos>` masking is permanently released for
the rest of the generation. Default 0 = disabled = the pre-release behavior (mask `<eos>`
while suppression is armed). Opt-in; no behavior change unless a caller sets `N > 0`.
**Final default: 0** (reverted from a brief 4 — see "Decision" below).
**Script:** `scripts/measure_eos_release.py` (same rescue / EOS-death conventions as
`verify_eos_guard.py`, plus `first_eos` position + `no_eos` fraction to quantify the reclaim).

## Why

With `eos_guard=True` (now the default), degenerate prompts that would otherwise EOS-die now
run to `max_new` (healthy prose, no `<eos>`). The guard masks `<eos>` during every re-armed
suppression window, and a degenerate attractor re-arms suppression repeatedly — so the model
is prevented from terminating for the full window. The bounded release rule unmask `<eos>`
after the episode is over (N clean steps), restoring natural termination without
re-introducing the EOS-death the guard eliminates.

## EOS-death rate by release-N (all three models)

| N | 0.5B (36 loopers) | 1.5B (38) | 3B (40) |
|---|---|---|---|
| guard_off | 3/36 (8%) | 18/38 (47%) | 10/40 (25%) |
| **0 (release off)** | **0/36** | **0/38** | **0/40** |
| 1 | 2/36 | 16/38 | 8/40 |
| 2 | 1/36 | 14/38 | 5/40 |
| 3 | 1/36 | 8/38 | 3/40 |
| 4 | 0/36 | 5/38 (13%) | 2/40 (5%) |
| 6 | 0/36 | 5/38 | 2/40 |
| 8 | 0/36 | 0/38 | 2/40 |
| 12 | 0/36 | 0/38 | 0/40 |

Rescue is 100% (all configs, all models); the release rule never breaks a rescue.

## Per-model full tables

### 0.5B (36 loopers)

| config | rescue | EOS-death | mean ΔD2 | act_tb | 1st_eos | no_eos |
|--------|--------|-----------|----------|--------|---------|--------|
| guard_off | 36/36 | 3/36 | +0.798 | 83.9 | 92.7 | 20/36 |
| release_off | 36/36 | 0/36 | +0.779 | 110.8 | 128.0 | 36/36 |
| release_1 | 36/36 | 2/36 | +0.806 | 88.2 | 101.1 | 23/36 |
| release_2 | 36/36 | 1/36 | +0.808 | 93.8 | 106.8 | 25/36 |
| release_3 | 36/36 | 1/36 | +0.807 | 96.8 | 106.9 | 26/36 |
| release_4 | 36/36 | 0/36 | +0.803 | 97.7 | 112.1 | 28/36 |
| release_6 | 36/36 | 0/36 | +0.803 | 100.0 | 114.4 | 29/36 |
| release_8 | 36/36 | 0/36 | +0.797 | 102.8 | 117.2 | 30/36 |
| release_12 | 36/36 | 0/36 | +0.779 | 108.3 | 125.5 | 35/36 |

### 1.5B (38 loopers — PRODUCTION model)

| config | rescue | EOS-death | mean ΔD2 | act_tb | 1st_eos | no_eos |
|--------|--------|-----------|----------|--------|---------|--------|
| guard_off | 38/38 | 18/38 (47%) | +0.803 | 48.7 | 58.5 | 15/38 |
| release_off | 38/38 | 0/38 | +0.799 | 95.5 | 128.0 | 38/38 |
| release_1 | 38/38 | 16/38 | +0.803 | 55.7 | 65.5 | 17/38 |
| release_2 | 38/38 | 14/38 | +0.809 | 58.6 | 79.1 | 21/38 |
| release_3 | 38/38 | 8/38 | +0.794 | 74.5 | 96.5 | 26/38 |
| release_4 | 38/38 | 5/38 | +0.793 | 78.2 | 102.1 | 26/38 |
| release_6 | 38/38 | 5/38 | +0.794 | 79.4 | 108.4 | 30/38 |
| release_8 | 38/38 | 0/38 | +0.794 | 94.5 | 123.4 | 35/38 |
| release_12 | 38/38 | 0/38 | +0.799 | 95.5 | 126.8 | 37/38 |

### 3B (40 loopers)

| config | rescue | EOS-death | mean ΔD2 | act_tb | 1st_eos | no_eos |
|--------|--------|-----------|----------|--------|---------|--------|
| guard_off | 40/40 | 10/40 (25%) | +0.813 | 40.6 | 50.2 | 6/40 |
| release_off | 40/40 | 0/40 | +0.803 | 93.5 | 128.0 | 40/40 |
| release_1 | 40/40 | 8/40 | +0.813 | 41.0 | 55.9 | 8/40 |
| release_2 | 40/40 | 5/40 | +0.802 | 43.5 | 66.2 | 11/40 |
| release_3 | 40/40 | 3/40 | +0.798 | 48.9 | 75.0 | 15/40 |
| release_4 | 40/40 | 2/40 | +0.798 | 49.3 | 77.7 | 15/40 |
| release_6 | 40/40 | 2/40 | +0.800 | 50.0 | 83.8 | 19/40 |
| release_8 | 40/40 | 2/40 | +0.799 | 52.2 | 86.0 | 20/40 |
| release_12 | 40/40 | 0/40 | +0.803 | 72.3 | 106.1 | 29/40 |

## Findings

1. **The clean release-N is a per-model valve, monotonic in capacity** — 0.5B → 4, 1.5B → 8,
   3B → 12. Larger models have more persistent degenerate episodes, so the "N clean steps"
   needed to judge the episode over before releasing `<eos>` grows with model size. (Candidate
   relationship, 3 points — held as a valve observation, not a law.)
2. **The release rule is effectively a 0.5B-only optimization.** On 0.5B, N=4 reclaims 8/36
   prompts (no_eos 36→28). On 1.5B the reclaim at its clean N=8 is ~nil (38→35), because by the
   time N is large enough to be safe the loop has re-formed. 3B sits in between (N=12 clean,
   40→29 no_eos).
3. **1.5B has the highest baseline EOS-death (47% guard_off)** — higher than 0.5B (8%) and 3B
   (25%). Non-monotonic in capacity, so it's a per-model property, not a scale law; but it means
   the `eos_guard` default flip matters most on the production model.

## Decision (owner, 2026-08-13)

Default **reverted to 0** (release off). The shipped default was briefly N=4 (on 0.5B evidence),
which the cross-model sweep showed reintroduces EOS-death on 1.5B (13%) and 3B (5%) — regressing
the eos_guard fix. The release rule stays available as an **opt-in per-model valve**
(`eos_release_after=N`), and is recorded as a documented negative result for production: bounded
EOS-release does not transfer to larger models at a useful reclaim without reintroducing
EOS-death.

## Artifacts

- `scripts/measure_eos_release.py` — bounded-release sweep (reusable).
- `docs/gate23/qwen2.5-0.5b_eos_release_sweep.jsonl` (324 rows).
- `docs/gate23/qwen2.5-1.5b_eos_release_sweep.jsonl` (342 rows).
- `docs/gate23/qwen2.5-3b_eos_release_sweep.jsonl` (360 rows).
- `governor/controller.py` — `eos_release_after` param (default 0) + release debounce logic.
