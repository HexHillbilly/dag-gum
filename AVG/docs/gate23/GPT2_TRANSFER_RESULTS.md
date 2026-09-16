# GPT-2 Cross-Architecture Transfer — Rescue Gate (MEASUREMENT)

> First cross-architecture validation of the AVG governor. Tests whether the
> degeneracy-rescue mechanism transfers from Qwen2.5-1.5B (RMSNorm + GQA +
> RoPE + SwiGLU) to GPT-2 (LayerNorm + full MHA + learned positional
> embeddings + GELU). No controller edits. All claims below are framed as
> *candidate* findings — two models is suggestive, not conclusive.

## Question

Does the governor rescue degenerate loops on a model that shares **none** of
Qwen's architectural components? If so, which parts of the mechanism are
architecture-general vs. per-model "valves"?

## Method

| Field | Value |
|---|---|
| Model | `gpt2` (124M, 12 layers, hidden 768, 12 heads, LayerNorm/MHA/learned-pos/GELU) |
| Controller | production `ActiveVarietyGovernor` (B3), unmodified |
| Fixture | `tests/fixtures/gpt2_degenerate.jsonl` (N=16; curated from a 40-prompt discovery sweep, greedy distinct-2 < 0.5) |
| Regimes | greedy (`do_sample=False`) and sampling (temp 0.8, top_p 0.85) |
| Dormant | raw model generation (no governor) |
| Active | governor `generate(intervene=True)` |
| Rescue predicate | ΔDistinct-2 > 0.01 over trailing 24 generated tokens (active vs dormant) |
| SEED | 42 |

The Qwen `t2s`/`qwen` degenerate fixtures do **not** loop on GPT-2 (GPT-2
escapes a 199-token word-list into prose), so GPT-2-native loop prompts are
used instead — this is itself a finding: degeneracy fixtures are
model-specific.

## Results

| regime | rescue | EOS | mean ΔDistinct-2 |
|---|---|---|---|
| greedy | **16/16 (100%)** | 0 | +0.655 |
| sampling | **11/16** (all 5 "non-rescues" already escape: dormant d2 ≥ 0.96) | 0 | +0.356 |

- **Greedy**: every looping prompt is broken into varied prose (d2 0.04–0.48
  → 0.74–1.00). Zero early termination.
- **Sampling**: GPT-2 loops on fewer prompts under sampling; of the prompts
  that *do* loop (dormant d2 < 0.5), **all are rescued**. The 5 "non-rescues"
  are dormant-escapes (sampling already produces diverse text — nothing to
  rescue), matching the Qwen DS-035 pattern.
- **Mechanism attribution**: fire counters are 0 in 15/16 cases (the one
  exception is the `"I I I I…"` seed, where the bigram predicate fires because
  single-character tokens drive CTR low). The rescue is driven by the
  **token-suppression** path (−5.0 penalty on `active_loop_ids`), which fires
  unconditionally on repeated tokens — pure logit-level, architecture-agnostic.

## Threshold re-derivation (Gate 2.3 Part A)

Layer-2 rolling-24-token spectral PR, degenerate (greedy min-PR) vs healthy
prose (sampling mean-PR):

| | degenerate p90 | prose p10 | T | band_low |
|---|---|---|---|---|
| GPT-2 | 2.95 | 15.82 | 9.38 | **7.04** |
| Qwen (frozen) | 3.13 | 18.78 | 10.95 | 8.22 |

**Candidate invariant:** the collapsed-manifold PR floor is ≈3 across **three
models now** (Qwen2.5-1.5B 3.13, Qwen2.5-0.5B 2.98, GPT-2 2.95 — two
architectures × two scales; see `QWEN05_SCALE_RESULTS.md`). The *healthy-prose*
baseline shifts with hidden size (Qwen 18.8 / 0.5B 17.1 / GPT-2 15.8), which is
what moves the threshold. Still a candidate pending a non-transformer and/or a
larger model.

## False-positive check (suppression on healthy prose)

On healthy prose (sampling), the suppression keeps coherence intact
(CTR 0.84–0.93, unchanged from raw). On prose where raw GPT-2 itself loops
under sampling, the governor rescues it (e.g. d2 0.043→1.000, CTR 0.24→0.92).
GPT-2 is more degenerate than Qwen, and the governor covers it without
degrading normal text.

## Valves — what transfers vs. what is per-model

| Component | Status | Notes |
|---|---|---|
| Spectral-PR collapse signal | **candidate-universal** (PR floor ≈3) | detection mechanism |
| Token suppression (−5.0) | **transfers as-is** | load-bearing actuation |
| `band_low` | **per-model valve** (8.22 → 7.04) | set by healthy-prose baseline |
| Suppression strength / cooldown | transferable, valve | −5.0 works on both so far |
| Kickstart (−1e4/−5/−2) | **Qwen-specific** | never arms on GPT-2 (valid-word loops → CTR high) |
| Hook layer (layer 2) | valve (early-layer choice) | not yet swept on GPT-2 |

Implication for auto-tuning: the actuation carries over untouched, and
`band_low` is derivable from a small healthy-prose calibration pass
(`T ≈ (collapse_floor + healthy_baseline) / 2`).

## Caveats

- **`generate(intervene=False)` is a dead parameter** in B2/B3 — the loop
  applies suppression/kickstart unconditionally. DS-035/036 dormant baselines
  were correct only because they used a *separate raw* function. Needs a fix.
- N=16 (lightweight first pass), not N=100.
- Fixtures are GPT-2-specific (Qwen's don't transfer); a cross-model fixture
  corpus is future work.
- The PR≈3 invariant is a candidate pending a third architecture (and a
  different scale).
- No controller edits were made; the Qwen `band_low=8.216` remains hardcoded
  in `_evaluate_collapse` and is simply not exercised by GPT-2's valid-word
  loops (suppression fires first).

## Deliverables

- `scripts/measure_gpt2_rescue.py` — formal gate + fixture discovery.
- `scripts/probe_gpt2_threshold.py` — spectral-PR threshold re-derivation.
- `tests/fixtures/gpt2_degenerate.jsonl` — GPT-2 loop fixture (N=16).
- `docs/gate23/gpt2_transfer_results.jsonl` — per-record results.
