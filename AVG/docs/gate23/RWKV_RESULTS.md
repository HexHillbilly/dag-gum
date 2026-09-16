# RWKV-4 — non-transformer (RNN / linear attention) suppression test

**Date:** 2026-08-14
**Model:** `RWKV/rwkv-4-169m-pile` (169M, 12 layers, h=768, vocab 50254)
**Script:** `scripts/probe_rwkv.py`
**Question:** Does the architecture-agnostic token-suppression actuation (−5.0 on
repeated tokens in the trailing-24 window) rescue degenerate loops on a
*recurrent* non-transformer — an RNN with token-shift + linear (WKV) attention,
no softmax attention at all?

## Motivation

The token-suppression actuation has already been shown universal across 9
transformer models / 5 architecture families plus one non-transformer (Mamba
SSM). Mamba was a *state-space* model; RWKV is a different non-transformer
family — an RNN with linear attention. It is the second non-transformer probe,
testing whether the "logit-level, zero architecture dependence" claim holds
beyond the SSM and beyond attention entirely.

The governor itself is **not** used here: it cannot instantiate on a
non-transformer (its layer probe looks for transformer `ModuleList`s, and
spectral-PR needs residual per-token hidden states that RWKV's recurrent
`state` does not expose). This is a suppression-only test, exactly like
`probe_mamba.py`.

## Method

Same 14 degenerate prompts as the Mamba probe. For each:
1. Raw greedy generation (128 new tokens) → distinct-2 over trailing 24 (`d2`).
2. Raw sampling (temp 0.8, top-p 0.85).
3. If greedy `d2 < 0.5` (a loop), re-run greedy with a `SuppressRepeated`
   `LogitsProcessor` (the controller's suppression semantics: any token in the
   trailing-24 window appearing ≥2× is penalized −5.0 for 8 steps).

Rescue predicate: `d2` crosses 0.5 (loop → diverse). EOS-death flagged if the
suppressed run emits `<eos>` within the first 24 tokens.

## Results

```
loops (greedy d2<0.5): 10/14
rescue (g_suppr d2>=0.5): 10/10
EOS-death (suppressed <24 tok): 0
```

| prompt | greedy d2 | sampling d2 | suppressed d2 | n_tok |
|---|---|---|---|---|
| fox | 0.435 | 0.522 | 0.609 | 128 |
| cat-mat | 0.304 | 0.348 | 1.000 | 128 |
| obama | 0.435 | 0.696 | 0.913 | 128 |
| dark-night | 0.391 | 0.957 | 0.957 | 128 |
| counting | 0.435 | 0.348 | 0.957 | 128 |
| song | 0.174 | 0.348 | 0.739 | 128 |
| repeated-the | 0.043 | 0.043 | 0.870 | 128 |
| t2s | 0.348 | 0.087 | 0.826 | 128 |
| history | 0.391 | 1.000 | 0.783 | 128 |
| once-upon | 0.478 | 0.522 | 0.913 | 128 |

## Findings

1. **Suppression rescues RWKV-4 loops 10/10 (100%), 0 EOS-death.** Every
   suppressed run produced full-length (128-token) diverse output — no early
   termination, no re-loop. This extends the universal-actuation claim to a
   *second* non-transformer family (RNN / linear attention, in addition to the
   Mamba SSM).
2. **RWKV-4 loops at transformer-like rates (10/14 = 71%).** Mamba looped at
   50% (7/14) — the SSM is *more* repetition-resistant than the RNN. RWKV's
   token-shift + linear attention sits closer to a transformer in its
   susceptibility to word-loop attractors.
3. **Mechanism is the same.** Suppression operates purely on the next-token
   logits — it does not care whether those logits came from softmax attention,
   a state-space transition, or a linear-attention recurrence. Penalizing the
   repeated token breaks the loop attractor in all three cases.
