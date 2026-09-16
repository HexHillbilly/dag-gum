# Semantic Context Proxy — Research Program

> North-star hypothesis, phases, and open questions. This is the living
> research record; update it as measurements land and hypotheses are
> confirmed/retired. Owner-ratified only.

## The one-line thesis

**A Markov/Daisy-style mechanism can act as a *governor* on a modern LLM — and
if it can, so can retrieval (RAG) and any target-distribution conditioner.**

Everything in this repo is a test of that claim. The current `proxy.py` is the
first concrete instantiation: retrieve semantically relevant lines from a
corpus, synthesize a blended string (Markov, in homage to Daisy), and inject it
to shape the model's output distribution — without touching the weights.

## Where this came from

`semantic-context-proxy` is a re-implementation of **DAISY**, Gregory G.
Leedberg's 2000 chatbot. The original chain
of reasoning that motivated this project:

1. If **Daisy / a Markov generator** works as a governor on a model…
2. …then an **LLM or RAG** could work as a governor too.
3. A governor that injects a **target distribution** is a **style setter**:
   load a corpus of a user's writing and shape the model's output toward that
   statistical fingerprint.
4. Style-matching is the inverse of **detection**: if the output's fingerprint
   matches a human's, it should resist statistical AI-text detectors (e.g.
   Panagram) and statistical watermarking (e.g. Anthropic's). A detector and a
   watermark are both fingerprint tests; style-setting is the operation that
   moves the fingerprint.

## Relationship to AVG (the merge)

AVG and SCP are two intervention points of the **same framework: weight-free
steering of a local model.**

| | AVG (Active Variety Governor) | SCP (Semantic Context Proxy) |
|---|---|---|
| Intervention axis | **generation** (mid-decode) | **distribution** (pre-decode) |
| Mechanism | hidden-state / logit penalty when collapse is detected | context injection shaping output distribution |
| Direction | collapse → variety (repair) | output → target corpus (conditioning) |
| Target | break repetition attractors | match a persona / a writer's style |

Daisy itself was a distribution-matcher: `Response` samples from the observed
bigram distribution, `Percent` (rarity = importance) picks the salient words,
and `BestResponse` selects the candidate closest to the target. SCP = Daisy +
a modern LLM as the generator.

## The falsifiable core

The single claim to prove or kill:

> **Injecting retrieved corpus context measurably moves the model's output
> distribution toward the target corpus, without destroying coherence or
> topic adherence — and the *mechanism* (retrieval vs. synthesis) can be
> isolated to find which component carries the effect.**

Decomposed:

- **H1 — Retrieval quality.** Does FAISS top-k return lines semantically
  relevant to the prompt? *(metric: prompt↔top-k embedding cosine / L2)*
- **H2 — Synthesis quality.** Does the Markov blend over k retrieved lines
  produce coherent novel strings, or fall back / garble? *(metric: fallback
  rate, novelty vs. source lines)*
- **H3 — Distribution transfer (north star).** Does injection move the output
  toward the target corpus's statistical fingerprint? *(metric: output↔corpus
  centroid cosine, bigram-perplexity under the corpus model, stylometric
  overlap)*
- **H4 — Guardrails.** Does injection preserve coherence and topic adherence?
  *(metric: repetition rate, prompt↔output cosine)*

## Phases

- **Phase 1 (current) — Baseline.** Measure the *existing* `proxy.py` v1.0
  behavior (retrieve→Markov→"chaotic misfire" wrap) against a no-injection
  baseline. Quantify H1/H2 so we know what the current mechanism actually
  produces.
- **Phase 2 — Component ablation.** Isolate retrieval-only (verbatim top-k,
  no Markov), persona-only (coherent Daisy voice), and RAG-only (delimited
  grounding) to find which component carries H3.
- **Phase 3 — Style setter.** Feed a corpus of the *user's* writing and
  measure distribution transfer (H3) against that target. This is the
  Panagram-resistance / watermark-defeat hypothesis, tested as a measurement
  first, not a product claim.
- **Phase 4 — Detection-evasion measurement.** Score Phase-3 output with a
  statistical AI-text detector and any available watermark check; report
  deltas. *(Owner-gated; evidence-first, no claims without numbers.)*

## Red lines (mirroring AVG)

- Thresholds are **owner-signed after baseline measurement**, never pre-tuned.
- A red measurement is STOP-and-report, never a reason to tune.
- Detection-evasion claims (Phase 4) require real detector scores quoted
  verbatim; no vibes.
