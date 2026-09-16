# Mamba (Non-Transformer) Exploratory Test — Suppression Transfer (MEASUREMENT)

> Third cross-model point and first **non-transformer**. `state-spaces/mamba-130m`
> (SSM, d_model 768, 24 layers, no attention, no residual per-token KV) loaded
> into HF `MambaForCausalLM` with the sequential (pure-PyTorch) fallback — no
> `mamba-ssm` CUDA compilation required. Tokenizer: `EleutherAI/gpt-neox-20b`.
> The production governor cannot instantiate on Mamba (no transformer layer
> stack), so this isolates the one genuinely architecture-agnostic component:
> the **token-suppression** actuation.

## Load notes

- The repo stores the *original* mamba config (`model_type` absent) + a single
  `pytorch_model.bin`. Two remaps were required: `vocab_size 50277 → 50280`
  (`pad_vocab_size_multiple=8`) and `backbone.embedding.weight →
  backbone.embeddings.weight`. `load_state_dict(strict=True)` then passes.
- Output cache is `out.cache_params` (a `DynamicCache`), **not**
  `past_key_values`.

## Result

| prompt | greedy d2 | +suppression d2 | Δ |
|---|---|---|---|
| repeated-the | 0.043 | 0.870 | +0.83 |
| cat-mat | 0.304 | 0.913 | +0.61 |
| obama | 0.304 | 0.870 | +0.57 |
| counting | 0.435 | 0.957 | +0.52 |
| t2s | 0.348 | 0.957 | +0.61 |
| news | 0.261 | 0.957 | +0.70 |
| fox | 0.435 | 1.000 | +0.57 |

**7/7 greedy loops rescued** by the −5.0 repeated-token penalty alone.

## Findings

- **Suppression is the universal actuation.** It rescues degenerate loops on a
  transformer (Qwen, GPT-2) *and* a non-transformer (Mamba SSM) — it is pure
  logit-level, with zero architecture dependence.
- **Mamba loops less than GPT-2** (7/14 greedy vs GPT-2's ~16/40), consistent
  with the SSM's selective state resisting the repetition attractor. Sampling
  d2 is also higher (0.30–1.0).
- **Spectral-PR detection does not transfer** to Mamba (no residual per-token
  hidden states / no "layer 2"), so the PR≈3 collapse-floor invariant is
  transformer-specific. A non-transformer governor would be suppression-only,
  which — per this test — is sufficient.

## Deliverables

- `scripts/probe_mamba.py` — Mamba load (remap + sequential fallback) +
  suppression loop test.
