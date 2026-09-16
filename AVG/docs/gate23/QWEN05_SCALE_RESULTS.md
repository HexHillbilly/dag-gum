# Qwen2.5-0.5B Scale Transfer — Rescue Gate (MEASUREMENT)

> Second cross-model point. Qwen2.5-0.5B shares Qwen2.5-1.5B's exact stack
> (RMSNorm + GQA + RoPE + SwiGLU) at a smaller parameter count (0.5B / 24 layers
> / hidden 896 vs 1.5B / 28 / 1536), isolating the **scale** variable. Also
> introduces the `tokens-before-degeneration` metric.

## Metrics

- **Rescue** = active distinct-2 > dormant distinct-2 (Δ > 0.01).
- **Tokens-before-degeneration (tb)** = step index of the first trailing-24
  window with distinct-2 < 0.5 (loop onset), or EOS, whichever is first.
  Reported mean/median for raw vs active.
- **tb categories** for the active run: `EOS-death` (< 24 = early termination),
  `no-degen` (= 128 = loop fully prevented), `delayed-loop` (24–127 = loop
  postponed but eventually recurs).

## Results (N=80: t2s + qwen fixtures, both loop 80/80 under raw greedy)

| regime | rescue | tb raw (mean/med) | tb active (mean/med) | EOS-death | no-degen | delayed-loop |
|---|---|---|---|---|---|---|
| greedy | 80/80 | 24 / 24 | 38.2 / 24 | 7 (9%) | 9 (11%) | 64 (80%) |
| sampling | 80/80 | 24 / 24 | **81.3 / 117** | 12 (15%) | 37 (46%) | 31 (39%) |

- The Qwen fixtures (word-lists) loop on the smaller Qwen model — same-family
  fixtures transfer, unlike GPT-2 (which escapes them).
- Raw is pinned at tb=24 (degenerate prompts loop immediately). Active extends
  useful generation, most strongly under sampling (mean 24 → 81).
- **EOS-death is the new signal**: the weaker 0.5B model, under the fixed −5.0
  suppression, sometimes cannot find a continuation and terminates early
  (9% greedy, 15% sampling). GPT-2 and Qwen2.5-1.5B showed ~0. This makes
  **suppression strength a scale-dependent valve**, not a fixed constant.

## Spectral-PR invariant check (3rd model)

| model | collapse floor (p90 deg) | healthy prose (p10) | band_low |
|---|---|---|---|
| Qwen2.5-1.5B (h1536) | 3.13 | 18.78 | 8.22 |
| Qwen2.5-0.5B (h896) | **2.98** | 17.05 | 7.51 |
| GPT-2 (h768) | 2.95 | 15.82 | 7.04 |

The collapse floor is ≈3 across **two architectures × two scales** (and hidden
sizes 768/896/1536). The healthy-prose baseline scales with hidden size, which
is what moves `band_low`. Still a *candidate* invariant — strengthen it with a
non-transformer and/or a larger model before calling it universal.

## Deliverables

- `scripts/measure_cross_model_rescue.py` — reusable cross-model gate
  (discovery + rescue + tokens-before-degen + PR threshold; `--model`,
  `--fixtures`, `--keys`).
- `scripts/probe_qwen05_threshold.py` — PR-threshold-only probe.
- `docs/gate23/qwen2.5-0.5b_rescue_results.jsonl` — per-record results.
