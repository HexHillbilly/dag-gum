# Pythia-160m (GPT-NeoX) Results — cross-model transfer, model #5

**Date:** 2026-08-13
**Model:** `EleutherAI/pythia-160m` (gpt_neox, 12L, h768, 12 heads, vocab 50304,
**`use_parallel_residual=True`** = parallel attention+MLP blocks, GELU, RoPE, LayerNorm)
**Harness:** `scripts/measure_cross_model_rescue.py`
**Fixture:** `tests/fixtures/pythia_degenerate.jsonl` (N=21 Pythia-native loop prompts,
discovered via `scripts/probe_pythia_loop_discovery.py`)
**Why this model:** third genuinely distinct architecture. Pythia's **parallel
residual blocks** (attention + MLP read the same input, outputs summed) differ structurally
from both Qwen's sequential RMSNorm/GQA/SwiGLU and GPT-2's sequential LayerNorm/MHA/learned-pos.
Same scale as GPT-2 (12L/h768), so the only variable is architecture.

---

## Headline

| regime   | raw loopers | rescue | raw t2d mean | active t2d mean | ΔD2  |
|----------|-------------|--------|--------------|-----------------|------|
| greedy   | 21/21       | **21/21 (100%)** | 30.6    | 128.0 (full)    | +0.576 |
| sampling | 10/21       | **10/10 (100%)** | 36.6    | 128.0 (full)    | +0.696 |

(t2d = tokens-before-degeneration. Rescue is counted **only over regime-loopers** — prompts
that actually loop under that regime's raw decode. 11 of 21 fixtures loop under greedy but
NOT under sampling: sampling's stochasticity breaks the loop before it forms.)

On every prompt that genuinely degenerates, the governor rescues it — **100% in both
regimes** — lifting distinct-2 from ~0.28–0.31 (collapsed) to ~0.89–0.97 (healthy) and
running out the full 128-token window with no loop.

**No EOS-death** (0/21 greedy, 0/10 sampling). Consistent with the EOS-death-grows-with-capacity
pattern: Pythia-160m is small (like GPT-2), so the fixed −5.0 suppression is gentle — it breaks
the loop without over-driving generation into `<eos>`.

## Spectral-PR collapse floor (candidate invariant → 5 models)

| model        | arch        | degenerate minPR p90 | healthy prose p10 | band_low |
|--------------|-------------|----------------------|-------------------|----------|
| Qwen2.5-0.5B | Qwen2.5     | 2.98                 | 17.05             | 7.51     |
| Qwen2.5-1.5B | Qwen2.5     | 3.13                 | 18.78             | 8.22     |
| GPT-2 (124M) | GPT-2       | 2.95                 | 15.82             | 7.04     |
| Qwen2.5-3B   | Qwen2.5     | 2.99                 | 17.58             | 7.71     |
| **Pythia-160m** | **GPT-NeoX** | **2.97**          | **17.58**         | **7.71** |

**Five models, three architectures (Qwen2.5 / GPT-2 / GPT-NeoX), collapse-floor PR ≈ 3.0
(range 2.95–3.13).** GPT-NeoX's parallel residual blocks do not change the floor — the
degenerate attractor still compresses hidden states to a participation ratio of ~3. This is
the strongest evidence yet that collapse-PR≈3 is an architecture-universal degenerate signal,
not a Qwen artifact.

## Mechanism attribution

- **spectral-PR fires: 0/21.** The spectral detector never arms on Pythia — suppression breaks
  the loop in a single step, before PR stays below band_low for the 2 consecutive steps the
  detector requires.
- **bigram fires: 0 in 19/21, 1–2 in 2/21.** Essentially silent.
- **suppression-only: 19/21 records (90%).**

Rescue is again driven entirely by **token-suppression** (−5.0 on `active_loop_ids`), the
actuation that has now transferred to Qwen2.5 (3 scales), GPT-2, Pythia, and Mamba.

## Cross-model summary (5 models)

| model        | arch     | rescue (greedy) | EOS-death (greedy/samp) | PR floor |
|--------------|----------|-----------------|--------------------------|----------|
| Qwen2.5-0.5B | Qwen2.5  | 80/80 (100%)    | 9% / 15%                 | 2.98     |
| Qwen2.5-1.5B | Qwen2.5  | (baseline)      | 0%                       | 3.13     |
| GPT-2 (124M) | GPT-2    | 16/16 (100%)    | 0% / 0%                  | 2.95     |
| Qwen2.5-3B   | Qwen2.5  | 78/78 (100%)    | 6.4% / 24.4%             | 2.99     |
| Pythia-160m  | GPT-NeoX | 21/21 (100%)    | 0% / 0%                  | 2.97     |

**Conclusion:** token-suppression is the universal actuation across **three transformer
architectures + one SSM**; collapse-PR≈3 is the universal degenerate signal across **three
architectures × three scales**. Thresholds (band_low), suppression strength, and EOS-death
propensity remain per-model valves. The mechanism-vs-valve split is now well-evidenced enough
to move to suppression-valve auto-tuning.

## Artifacts

- `docs/gate23/pythia-160m_rescue_results.jsonl` — canonical per-record results (21 records).
- `tests/fixtures/pythia_degenerate.jsonl` — N=21 Pythia-native loop prompts.
- `scripts/probe_pythia_loop_discovery.py` — loop-discovery probe (reusable for the next family).
- `scripts/measure_cross_model_rescue.py` — harness; fixed discovery denominator (`screened`
  counter replaces the hardcoded `*2` assumption).
