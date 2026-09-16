# Phase 1 Baseline — Results

**Date:** 2026-08-14
**Model:** `microsoft/Phi-3-mini-4k-instruct` (3.8B, bf16, CUDA)
**Corpus:** Daisy memory file (92 sentences)
**Probes:** 10 (conversational + technical + abstract)
**Conditions:** `baseline` (no injection) · `markov` (v1.0: retrieve→Markov "misfire"→chaotic-override wrap) · `rag` (mode B: retrieved lines as delimited grounding) · `persona` (mode A: coherent Daisy voice, seeded with top-1 retrieved line)
**Regimes:** greedy (deterministic) + sampling (temp 0.8 / top_p 0.9, 3 samples)
**Seed:** 42

## Model choice (finding)

The default assumption — benchmark on `Qwen/Qwen2.5-1.5B` (AVG's model) — was
**rejected after evidence**. The cached Qwen is the *base* model, not Instruct:
it is not instruction-tuned (cannot follow "adopt this tone/context"), leaks
Chinese under sampling (reproducing training-data exam questions like `答案`
/ `____（collect）stamps`), and its chat special tokens are mis-mapped (the
model's `<|im_end|>` decodes to `.IsAny`). AVG uses this model *because* it
degenerates; SCP needs a clean instruction-following baseline to measure the
injection's *marginal* effect. Phi-3-mini-4k-instruct (cached) follows the
injection directives correctly with coherent English.

## Summary table (mean over 10 probes)

| condition | regime | style_cos | topic_cos | rep_rate | len |
|---|---|---|---|---|---|
| baseline | greedy | 0.184 | 0.583 | 0.049 | 259 |
| markov | greedy | 0.311 | 0.575 | 0.140 | 237 |
| rag | greedy | 0.225 | 0.613 | 0.033 | 216 |
| persona | greedy | 0.382 | 0.559 | 0.011 | 284 |
| baseline | sampling | 0.239 | 0.549 | 0.047 | 242 |
| markov | sampling | 0.299 | 0.579 | 0.056 | 267 |
| rag | sampling | 0.247 | 0.615 | 0.021 | 274 |
| persona | sampling | 0.355 | 0.510 | 0.009 | 285 |

*`style_cos` = cosine(output embedding, corpus centroid) — the north-star
distribution-matching metric. `topic_cos` = cosine(prompt, output). `rep_rate`
= fraction of repeated word bigrams (degeneracy).*

## Findings

### H3 — Distribution transfer (north star): CONFIRMED
Context injection moves the output toward the target corpus. **The coherent
persona directive (mode A) is the strongest mechanism by far:**

- greedy: +0.198 (+108%) over baseline; sampling: +0.116 (+49%).
- markov (v1.0): +0.127 greedy / +0.060 sampling.
- rag (mode B): +0.041 greedy / +0.008 sampling — grounding does **not** move
  style (expected: facts ≠ voice).

### H2 — Markov synthesis: the weak link, CONFIRMED
- **Fallback rate 40%** (4/10 probes fell back to the verbatim top-1 line).
- **Novelty 0.311** — the "synthesis" barely differs from its source lines.
- Qualitatively it garbles: `"i like to do outside? i like to do outside? i like to
  eat lots of candy."` (broken fragment + induced repetition, rep 0.383).

### H4 — Guardrails: persona is clean, markov degrades
- `rep_rate` stays low everywhere on Phi-3 (no Qwen-style collapse).
- **markov** is the only condition that *raises* repetition (greedy 0.049→0.140,
  ~2.9×).
- **persona** *lowers* repetition (greedy 0.011, sampling 0.009) — the coherent
  directive makes output more varied, not less.
- `topic_cos` holds (0.51–0.62) across all conditions; nothing derails topic.

### H1 — Retrieval: works
Top-1 L2 distance mean 0.906; retrieval returns on-topic lines (MiniLM + FAISS
L2 is sound).

## Qualitative (sampling, seed 42)

- **baseline** — "As an artificial intelligence, I do not have feelings or
  emotions, so I do not experience fear." (generic assistant, refuses persona)
- **persona** — "Oh, hello there! You know, I've always believed that life's a
  grand adventure, full of mysteries and surprises…" (Daisy's voice — the
  resurrection working)
- **markov** — "i like to do outside? i like to do outside? i like to eat lots
  of candy." (garbled fragment + verbatim corpus line + repetition)

## Conclusion & Phase 2 direction

1. **The thesis holds**: a context-injection governor *can* steer a local
   model's output distribution toward a target corpus without touching weights.
2. **Retire the Markov "misfire" synthesis** — it is the weakest injection
   (least style transfer, highest repetition, 40% fallback). The retrieval is
   fine; the synthesis is the problem.
3. **Mode A (coherent persona directive) is the style-setter.** Mode B (rag)
   is the knowledge/grounding half, not a style mechanism.
4. **Phase 2**: component ablation (retrieval-only vs persona-only vs fused
   A+B), then feed a corpus of the *user's* writing (Phase 3) and measure
   distribution transfer against that target.

*Thresholds are not yet set — these are baseline measurements. Any acceptance
threshold must be owner-signed after Phase 2.*
