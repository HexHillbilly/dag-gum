# Phase 2 Results — prompt-injection vs logit-bias (the "skew")

**Date:** 2026-08-14 · **Model:** `microsoft/Phi-3-mini-4k-instruct` · **Corpus:** Daisy memory (92 sentences) · **Probes:** 10 · **Regimes:** greedy + sampling (temp 0.8 / top_p 0.9, 3 samples) · **Seed:** 42

Tests the owner's hypothesis: *skew the output distribution toward the corpus
with a per-message Markov prior* — implemented as a decode-time **logit bias**
(`logits += λ · log P_corpus(t | n-gram)`), a product-of-experts prior over the
LLM's next-token distribution (`style_prior.py`). This is the same bounded
per-step primitive as AVG's logit suppression, sign flipped.

## Summary (mean over 10 probes)

| condition | regime | style_cos | topic_cos | rep_rate | corpus_ppl |
|---|---|---|---|---|---|
| baseline | greedy | 0.184 | 0.583 | 0.049 | 422976 |
| markov | greedy | 0.311 | 0.575 | 0.140 | 155032 |
| rag | greedy | 0.225 | 0.613 | 0.033 | 177627 |
| **persona** | greedy | **0.382** | 0.559 | **0.011** | 216165 |
| logit_bias λ=0.02 | greedy | 0.183 | 0.583 | 0.050 | 401841 |
| logit_bias λ=0.05 | greedy | 0.183 | 0.584 | 0.052 | 384879 |
| logit_bias λ=0.1 | greedy | 0.183 | 0.584 | 0.052 | 384879 |
| logit_bias λ=0.2 | greedy | 0.196 | 0.583 | 0.057 | 394648 |
| baseline | sampling | 0.239 | 0.549 | 0.047 | 418165 |
| markov | sampling | 0.299 | 0.579 | 0.056 | 140845 |
| rag | sampling | 0.247 | 0.615 | 0.021 | 339790 |
| **persona** | sampling | **0.355** | 0.510 | **0.009** | 227815 |
| logit_bias λ=0.02 | sampling | 0.238 | 0.551 | 0.050 | 433664 |
| logit_bias λ=0.05 | sampling | 0.239 | 0.541 | 0.042 | 410492 |
| logit_bias λ=0.1 | sampling | 0.240 | 0.544 | 0.038 | 391217 |
| logit_bias λ=0.2 | sampling | 0.228 | 0.547 | 0.033 | 353040 |

*`style_cos` = cosine(output embedding, corpus centroid) — semantic/voice match.
`corpus_ppl` = perplexity of output under the corpus n-gram model — token-level
fingerprint match (lower = closer).*

## Findings

### The logit-bias "skew" does NOT move style (negative result)
At every λ in [0.02, 0.2], `style_cos` is indistinguishable from baseline
(greedy 0.183–0.196 vs 0.184; sampling 0.228–0.240 vs 0.239). The whole-output
*voice* does not shift.

It **does** nudge the token distribution: `corpus_ppl` falls ~9% at λ≈0.05–0.1
(422976 → 384879 greedy). So the prior is influencing token choice, but too
locally to change semantic/voice embedding.

**Diagnosis (measured):** the bias fires on **42% of decode steps** (708/1680
tokens hit a corpus context), but each firing is a *bigram* continuation — it
boosts a handful of tokens that follow the *previous single token*, not tokens
that encode *style*. Style lives in word choice, phrasing, and sentence rhythm,
which a token-bigram prior over a 92-sentence corpus cannot capture. The prior
is order-1 and the corpus is tiny; both are the problem.

### persona remains the strongest mechanism
`style_cos` +108% greedy / +49% sampling over baseline, with the *lowest*
rep_rate (0.011 / 0.009) — coherent prompt injection both transfers style and
reduces repetition.

### markov (v1.0) remains the weak link
Highest rep_rate (0.140 greedy), 40% fallback, lowest style transfer among the
prompt-injection conditions in sampling.

## Conclusion & Phase 3 direction

1. **The naive "corpus + Markov logit skew" is NOT an effective style-setter**
   at λ ∈ [0.02, 0.2] over a small corpus. The prior is too local (bigram) and
   the corpus too sparse for it to pull the output's voice.
2. **Prompt injection (coherent persona directive) is the working style
   mechanism** — by a wide margin.
3. **The logit-bias idea is not dead, it needs a denser prior**: retest in
   Phase 3 with (a) a real, larger corpus and (b) a
   *higher-order* (trigram+) or *word-level* prior, and add an n-gram-overlap
   metric alongside `style_cos` so local shifts are visible.

*Negative results are kept in the record by design — this one rules out the
token-bigram skew and refines the style-prior design before Phase 3.*
