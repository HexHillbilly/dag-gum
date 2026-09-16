# Auto-tuning pipeline + zero-shot validation (Llama-3.2-1B)

**Date:** 2026-08-13
**Change:** `governor/controller.py` — `band_low` is now a constructor param (default
`8.216097` = the 1.5B-derived frozen value, so behavior is unchanged unless a caller overrides
it). `scripts/auto_tune.py` — a single zero-hand-tuning pipeline that derives `band_low` and
validates rescue + EOS-death for any model.

## Pipeline (`scripts/auto_tune.py`)

1. **Discovery** — generates model-agnostic degenerate candidates (natural prose, repeated-seed,
   and 10 programmatic ~200-token word-loops — the same "synthetic word-loop" shape as the Qwen
   t2s/qwen fixtures) plus any `--fixtures`; collects prompts that degenerate (distinct-2 < 0.5)
   under GREEDY and under SAMPLING (dormant, `intervene=False`).
2. **band_low** — `T = (p90_degenerate + p10_prose) / 2`, `band_low = 0.75 * T` (the exact frozen
   1.5B formula: `0.75*(3.13+18.78)/2 = 8.216`).
3. **Gate** — rescue (greedy + sampling) and EOS-death (sampling), dormant vs active with the
   auto-derived `band_low` and the shared defaults (`suppression_strength=5.0`, `eos_guard=True`).
4. **Valve card** — JSON + MD.

## Zero-shot result — unsloth/Llama-3.2-1B (never measured before; 16 layers, GQA 4:1)

| valve | value |
|---|---|
| band_low (auto-derived) | **8.0421** (degenerate p90 = 2.9733, prose p10 = 18.4723) |
| greedy rescue | 25/26 (96%), 0 EOS-death |
| sampling rescue | 15/15 (100%), 0 EOS-death |
| mean ΔD2 | +0.647 (greedy) / +0.768 (sampling) |

**No hand-tuning**: the governor onboards a held-out model from a different family (Llama) with
the auto-derived `band_low` and the shared suppression/EOS-guard defaults.

## The headline findings

1. **Zero-shot rescue holds**: 96% greedy / 100% sampling, 0 EOS-death — the governor transfers
   to a 4th architecture family (Llama, after Qwen / GPT-2 / Pythia) with zero per-model work
   beyond deriving `band_low`.
2. **Collapse floor ≈ 3 confirmed on a 6th model / 4th architecture.** Llama-3.2-1B degenerate
   min-PR p90 = **2.9733**, joining Qwen1.5B 3.13 / 0.5B 2.98 / GPT-2 2.95 / 3B 2.99 / Pythia
   2.97. The spectral-PR ≈3 collapse floor now has evidence from four architecture families.
3. **Valves are auto-derivable.** `band_low` for Llama (8.04) differs from the frozen 1.5B value
   (8.216) — a per-model valve, and the pipeline recovers it automatically from the prose
   baseline.

## Known-model sanity check (Qwen2.5-0.5B)

`auto_tune.py` recovers `band_low = 7.65` (hand-derived ≈7.51) and gates 96% greedy / 93%
sampling rescue, 0 EOS-death — validating the pipeline against a model whose threshold is known.

## Artifacts

- `scripts/auto_tune.py` — the pipeline (reusable).
- `docs/gate23/llama-3.2-1b_autotune_card.{json,md}` — the zero-shot valve card.
- `docs/gate23/qwen2.5-0.5b_autotune_card.{json,md}` — the known-model sanity card.
- `governor/controller.py` — `band_low` constructor param.
