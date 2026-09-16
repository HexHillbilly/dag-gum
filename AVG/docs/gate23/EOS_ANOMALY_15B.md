# 1.5B EOS-death anomaly — kickstart-driven, likely fixture-native

**Date:** 2026-08-13
**Script:** `scripts/probe_eos_anomaly.py` (first-token logit probe; reuses the release-sweep
loopers).
**Trigger:** the bounded-EOS-release sweep surfaced non-monotonic guard_off EOS-death across
scale — 8% (0.5B) / **47% (1.5B)** / 25% (3B).

## Characterization (from `qwen2.5-*_eos_release_sweep.jsonl`)

- `tb_dorm` is **uniformly 24** across all three models and all loopers → the anomaly is NOT a
  metric artifact (the dormant loop onset is identical).
- The 1.5B EOS-death rows emit `<eos>` at generation positions **0–23, mostly 0–6** — the very
  first tokens. Position 0 = immediate give-up.

## Mechanism (logit-level, first token)

`compute_token_distinct_2_fast` has a cold-start fallback: with <4 generated tokens it uses the
trailing 24 **prompt** tokens, so on a degenerate word-loop prompt `active_loop_ids` is already
populated at step 0 — suppression (−5.0) and kickstart (−1e4) both fire on the first token.

| `<eos>` at first token | 0.5B (28 loopers) | 1.5B (28) | 3B (30) |
|---|---|---|---|
| raw (no penalty) | 0 argmax, P≈0.001 | 0 argmax, P≈0.001 | 0 argmax, P≈0.001 |
| + suppression −5.0 | 0 argmax, P≈0.04 | 0 argmax, P≈0.07 | 0 argmax, P≈0.05 |
| + kickstart −1e4 | 7/28 (25%), P≈0.13 | 17/28 (61%), P≈0.28 | 14/30 (47%), P≈0.22 |

**EOS-death is kickstart-driven.** Raw `<eos>` is never near the top; suppression alone never
makes it argmax. The narrow kickstart's −1e4 nukes the loop token, forcing a non-loop token, and
1.5B's next-best token is `<eos>` on 61% of degenerate prompts (vs 25% / 47%). The ordering
matches the sweep's EOS-death exactly.

## Why 1.5B specifically (hypothesis)

The degenerate fixtures (`t2s_degenerate`, `qwen_degenerate`) are **1.5B-native** — they were
built on the original production model. So 1.5B's degenerate attractor on those exact word-loops
is deepest, and its non-loop-token distribution is most peaked on `<eos>`. This is a candidate
explanation, not a law — a controlled check would swap in fixtures native to 0.5B/3B and see
whether the EOS-prone ranking follows the fixture or stays with 1.5B.

## Implication (shipped behavior unaffected)

The EOS-death is already eliminated by `eos_guard=True` (default): the mask fires whenever
suppression is armed, and suppression co-arms with kickstart through `active_loop_ids`, so
`<eos>` is masked at exactly these kickstart steps. This finding is explanatory — it confirms
the earlier note that EOS-death was a kickstart/spectral interaction, not pure suppression, and
pins the specific culprit (the −1e4 kickstart first-step penalty) plus the non-monotonic scale
anomaly.

## Artifacts

- `scripts/probe_eos_anomaly.py` — first-token EOS logit/rank/probability probe (reusable).
