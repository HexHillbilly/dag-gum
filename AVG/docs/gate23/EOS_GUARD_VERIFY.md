# EOS-guard wiring + full-governor verification

**Date:** 2026-08-13
**Change:** `governor/controller.py` — added `suppression_strength` (default 5.0) and
`eos_guard` (default True) constructor params; suppression penalty now uses
`self.suppression_strength`; when `eos_guard=True` and suppression is armed, `<eos>` is
masked (−1e9) in the next-token logits. Opt out with `eos_guard=False`.
**Why:** the fixed −5.0 penalty caused EOS-death (early `<eos>` emission when the loop-token
penalty's logit vacuum is filled by `<eos>`). The suppression-valve sweep
(`SUPPRESSION_VALVE_SWEEP.md`) showed this is a fixed-penalty tradeoff; the EOS-guard separates
the failure mode from the rescue.

## Full-governor verification (real `ActiveVarietyGovernor`, kickstart + spectral live)

| config | rescue 0.5B (36 loopers) | EOS-death 0.5B | rescue 3B (40 loopers) | EOS-death 3B |
|--------|--------------------------|----------------|------------------------|--------------|
| guard off @ 5.0 (baseline) | 36/36 (100%) | 3/36 (8%) | 40/40 (100%) | 10/40 (25%) |
| **guard on @ 5.0** | **36/36 (100%)** | **0/36 (0%)** | **40/40 (100%)** | **0/40 (0%)** |
| guard on @ 7.0 | 36/36 (100%) | 0/36 (0%) | 40/40 (100%) | 0/40 (0%) |

- **EOS-death eliminated** at the current strength (5.0): 8% → 0% (0.5B), 25% → 0% (3B).
- **Rescue held at 100%** in every config. No strength bump required — kickstart/spectral
  already backstop the ~9% of cases where pure suppression at 5.0 is too weak.
- **Healthy run length extends** dramatically: active tokens-before-degen 0.5B 84→111, 3B 41→94
  (the model now writes full healthy prose instead of terminating mid-rescue).

## Why the guard is sufficient at 5.0 (vs the sweep)

The sweep (`SUPPRESSION_VALVE_SWEEP.md`) measured PURE suppression and saw rescue *drop* at 5.0
with the guard (72/80, 32/40) — because masking `<eos>` removes the "die early" escape for
cases suppression couldn't break alone. The full governor has kickstart/spectral as a backstop:
they detect the lingering loop and apply a stronger penalty, so rescue returns to 100%. Net:
the guard is purely additive at 5.0 — it removes the failure without opening a new one.

## Gate status

`pytest` 128 passed; factual safety 0 fires / 0.00 L2; degeneracy smoke 1 fire; latency
−0.16 ms/token (budget 7.0). Gate scripts construct `ActiveVarietyGovernor` with default
args, so flipping the default re-exercises the same paths; the guard only masks `<eos>`
while suppression is armed (never on coherent/factual/dormant text), so smoke fire counts
and factual/latency legs are unchanged.

## Decision (owner, 2026-08-13)

Flipped `eos_guard` default to **True**: the governor ships with EOS-death eliminated.
Opt-out is `eos_guard=False` (toggleable). Known tradeoff: degenerate prompts run to
`max_new` (healthy prose) instead of ending early; per-token latency unchanged, total
length rises on degenerate inputs only. A bounded EOS-release rule (unmask `<eos>` after
N consecutive suppression-free steps) is the open follow-up to reclaim that length —
tracked separately, pending an N-sweep + owner sign-off.

## Artifacts

- `scripts/verify_eos_guard.py` — real-governor EOS-guard verification (reusable).
- `docs/gate23/qwen2.5-0.5b_eos_guard_verify.jsonl` / `qwen2.5-3b_eos_guard_verify.jsonl`.
- `governor/controller.py` — `suppression_strength` + `eos_guard` params, EOS mask during active suppression.
