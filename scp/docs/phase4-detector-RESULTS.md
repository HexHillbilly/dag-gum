# Phase 4 Results — statistical detection score (GLTR-style)

**Date:** 2026-08-14 · **Reference model:** `microsoft/Phi-3-mini-4k-instruct` (also the generator — see caveat) · **Outputs scored:** the persona run (baseline vs persona, sampling, 10 probes) · **Human reference:** a held-out human-author corpus, chunked 3 sentences/text · **Tool:** `scripts/score_detector.py`

Measures the two signals statistical AI-text detectors key on: **perplexity**
(mean token surprise under a reference LLM — AI text is low, human is high) and
**burstiness** (spread of per-sentence perplexity — human text varies more).

## Result

| group | mean_ppl | burstiness |
|---|---|---|
| baseline (default assistant) | 3.0 | 4.5 |
| persona (target voice) | 11.4 | 37.3 |
| human (held-out reference) | 16.1 | 15.6 |

- **Perplexity:** persona output moves **64% of the way** from baseline toward
  the human reference (3.0 → 11.4; human 16.1). On the primary detection
  signal, style injection makes the output substantially harder to flag as
  low-perplexity LLM text.
- **Burstiness:** persona output **overshoots** the human reference (37.3 vs
  15.6). The injected output oscillates between on-voice and off-voice, which
  is itself an abnormal variance signature.

## Honest reading (answering the Panagram question)

1. **Partial, not clean, evasion.** Style injection moves the perplexity signal
   toward human (64%), but introduces a burstiness tell (overshoot). A
   perplexity-only detector would be partly fooled; a burstiness-aware or
   trained classifier would likely still flag the output.
2. **This is a proxy measurement, not a real detector.** Perplexity under the
   *generator itself* is the GLTR heuristic; Panagram-class tools use a
   separately trained classifier that returns a direct "AI probability." The
   definitive test needs one of those run on the same outputs.
3. **Directionally positive:** the mechanism does shift the statistical
   fingerprint toward the target, consistent with Phase 3's voice-transfer
   result. Refining the injection to *damp* burstiness (e.g. gentler style
   guidance, or a smaller λ-style nudge on top of persona) is the lever to pull
   next.

## Next

- Run a **trained classifier** (RoBERTa AI-text detector) on the same outputs
  for a direct detector score.
- Explore dampening the burstiness overshoot while preserving the perplexity
  shift.
