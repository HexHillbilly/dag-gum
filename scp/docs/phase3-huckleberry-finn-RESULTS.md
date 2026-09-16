# Phase 3 Results — Huckleberry Finn corpus (real style target)

**Date:** 2026-08-14 · **Model:** `microsoft/Phi-3-mini-4k-instruct` · **Corpus:** *Adventures of Huckleberry Finn* (Project Gutenberg #76; 5,748 sentences, 109,206 words) · **Probes:** 10 · **Regimes:** greedy + sampling · **Seed:** 42

Tests whether a *real, dense* style corpus changes the Phase-2 conclusions —
and whether the logit-bias "skew" finally has pull once the prior is no longer
sparse. Persona directive generalized to an arbitrary author voice
(`--persona-name` / `--persona-desc`).

## Summary (mean over 10 probes)

| condition | regime | style_cos | topic_cos | rep_rate | corpus_ppl |
|---|---|---|---|---|---|
| baseline | greedy | 0.055 | 0.583 | 0.049 | 106483 |
| markov | greedy | 0.219 | 0.515 | 0.037 | 9798 |
| rag | greedy | 0.091 | 0.623 | 0.036 | 94768 |
| **persona** | greedy | **0.364** | 0.493 | 0.020 | **4429** |
| logit_bias λ=0.02 | greedy | 0.053 | 0.587 | 0.046 | 95316 |
| logit_bias λ=0.1 | greedy | 0.057 | 0.606 | 0.056 | 74427 |
| logit_bias λ=0.2 | greedy | 0.068 | 0.598 | 0.054 | 44337 |
| baseline | sampling | 0.092 | 0.549 | 0.047 | 102518 |
| markov | sampling | 0.236 | 0.497 | 0.033 | 11396 |
| rag | sampling | 0.082 | 0.594 | 0.014 | 131010 |
| **persona** | sampling | **0.369** | 0.464 | **0.009** | **8016** |
| logit_bias λ=0.2 | sampling | 0.080 | 0.546 | 0.044 | 59896 |

## Findings

### persona: style transfer is now *unambiguous* — the resurrection works
`style_cos` **+561%** greedy (0.055 → 0.364) and **+300%** sampling; `corpus_ppl`
falls **24×** (106K → 4.4K). Qualitatively the model *becomes* Huck Finn:

- *"What do you like to do for fun?"* → **"And then I can paddle over to town
  nights, and slink around and pick up things I want. Ain't nothing better than
  a quiet evening on the river, just me and my raft…"**
- *"Are you afraid of anything?"* → **"Ain't no particular thing to be afraid
  of, but I remember once when I got caught up in that raft with Jim, and the
  current was runnin' mighty fast…"**

First-person dialect, river/raft imagery, "ain't" — a genuine voice transfer,
not a surface nudge.

### logit-bias "skew" still does NOT transfer voice — even with a dense corpus
`style_cos` stays flat (0.053–0.068 greedy vs 0.055 baseline) at every λ. The
output remains **"As an AI, I don't have personal experiences…"** — the tell a
detector would catch is still there.

It *does* shift the raw token fingerprint: `corpus_ppl` falls monotonically
with λ (−58% at λ=0.2). So the token-bigram prior nudges **local token
statistics** but not the **semantic voice**. Style lives in word choice,
phrasing, and sentence structure — not adjacent-token continuations.

### v1.0 bug surfaced by a real corpus
`markovify.Text` **crashes** on dialect/punctuation-heavy Huck lines
(`KeyError: ('___BEGIN__',)`). The 5-line Daisy corpus never triggered it.
Fixed in `core.py` with a try/except fallback (now `markov_fallback` covers
both exceptions and `None` returns; fallback rate 0.20 on this corpus).

## Conclusion — answering the Panagram question

1. **Prompt injection is the working style-setter** — and it's the mechanism to
   build the Panagram test on. A coherent persona directive moves the voice
   dramatically; the logit skew does not.
2. **The logit-bias "skew" is not a voice-transfer primitive.** It nudges token
   stats but leaves the "AI assistant" voice intact, so it would not defeat a
   style-based detector on its own.
3. **The two primitives are complementary, not competing** — this is the merge:
   - prompt/context injection = **global** control (sets the *voice*);
   - logit intervention (AVG / style-prior) = **local** control (token-level
     degeneracy, distribution nudges).
   One framework, two intervention points.

*Next (Phase 4): score persona output vs baseline under a statistical AI-text
detector to quantify the Panagram-style resistance directly. Thresholds remain
owner-signed.*
