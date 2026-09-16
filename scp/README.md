# Semantic Context Proxy (SCP)

**Weight-free voice & style steering for local LLMs — via semantic retrieval and
context injection.**

---

## Executive summary

SCP makes a small local language model "write like" a given voice or author —
without retraining it. Give it a corpus of text (a chatbot's memory, an author's
works, a brand's writing), and it retrieves the lines most relevant to each prompt
and injects them so the model generates new text in that voice.

The core result: **prompt injection transfers voice, always.** Across public-domain
corpora of very different size and character — a 92-line chatbot memory and a
109,000-word novel — the target-voice metric rises **up to 5.6× over baseline** with no loss
of topic coherence.

## Headline results

| corpus | target-voice gain (`style_cos`, greedy) | corpus perplexity drop |
|---|---|---|
| Daisy MEM.DSY (92 lines) | +108% | — |
| Huckleberry Finn (109K words) | +561% | 24× |

Also established:

- **Retrieval works** — MiniLM + FAISS L2 returns on-topic lines (top-1 L2 ≈ 0.9–1.5).
- **The v1.0 Markov path is the weak link** — a 40% fallback rate, low novelty, and
  degeneracy-prone (it loops at ~10× the persona path's repetition rate). It is
  kept as a legacy Daisy-homage mode, not the production path.
- **A logit-bias "skew" does not transfer voice** — it nudges the token fingerprint
  (corpus perplexity −32% to −58%) but leaves the voice metric flat. Voice lives in
  word choice and sentence structure, not adjacent-token continuation.
- **Detection evasion is partial** — style injection moves output 64% of the way
  toward human perplexity (the primary GLTR signal) but overshoots burstiness.

## Methodology

All results are reproducible via the seeded scripts in `scripts/`:

- **A/B benchmark** (`scripts/benchmark.py`) — baseline vs markov vs rag vs persona
  vs logit_bias, under a fixed seed.
- **Voice scoring** — a corpus-centroid cosine metric (`style_cos`) on the output,
  with a topic-coherence control.
- **Detection scoring** (`scripts/score_detector.py`) — a GLTR-style perplexity +
  burstiness proxy (a trained classifier supersedes it in the sibling `ssp/`).

## What it is

```
  user prompt ──► embed ──► FAISS top-k retrieval ──► inject ──► local LLM
                              (semantic relevance)     (voice/style directive)
```

Components:

- `compile_brain.py` — embeds a sentence-per-line corpus and builds a FAISS index.
- `core.py` — retrieval, Markov synthesis (Daisy homage), and the prompt wrappers
  (`wrap_misfire` = v1.0 chaotic override; `wrap_rag` = grounding; `wrap_persona`
  = coherent voice).
- `proxy.py` — the production proxy (FastAPI): persona injection. Two backends:
  `local` (default — serves the cached `Phi-3-mini-4k-instruct` directly via
  transformers, no Ollama) or `ollama` (forwards to a local Ollama server).
- `style_prior.py` — the decode-time logit-bias "skew" (see findings: it is *not*
  the voice mechanism).

## For AI agents

- `core.py` — pure logic: retrieval, synthesis, prompt wrappers (no I/O).
- `proxy.py` — FastAPI server; `SCP_MODE` (`persona` default | `markov` legacy),
  `SCP_BACKEND` (`local` | `ollama` | `avg`).
- `scripts/benchmark.py` — the seeded A/B benchmark; run against any corpus.
- `docs/` — the per-phase research record (`RESEARCH_PROGRAM.md`, `phase*.md`).

## Usage

```bash
# Environment: a virtualenv with torch + transformers + faiss + markovify +
# sentence-transformers in one place:
#   venv/bin/python

# Run the benchmark against any corpus:
python scripts/benchmark.py --corpus docs/corpora/<name>.txt \
    --persona-name "Huckleberry Finn" --persona-desc "..."

# Score outputs for detection (perplexity + burstiness):
python scripts/score_detector.py <results.jsonl> <human_reference.txt>
```

Model: `microsoft/Phi-3-mini-4k-instruct` (instruction-tuned; the cached
`Qwen/Qwen2.5-1.5B` is the *base* model and cannot follow directives — see
`docs/phase1-baseline-RESULTS.md`).

## Origin — DAISY (Gregory G. Leedberg, 2000)

DAISY was a 2000-era chatbot with **no pre-programmed language of any kind**. It
learned by observing what humans typed, remembered word patterns and their
probabilities, and generated its own original sentences by recombining them —
"weeding out the unimportant words and building its response on the important
ones."

DAISY was written by **Gregory G. Leedberg**; the original is available from the
author at <https://leedberg.com/glsoft/daisy/>. This project shares DAISY's
*idea* — learn from observed speech and recombine it — but none of Leedberg's
original source code is included here: it is an independent re-implementation
on a modern stack (embeddings + retrieval + an LLM generator).

## Repository layout

```
core.py, style_prior.py      # pure logic: retrieval / synthesis / wrappers / logit-bias
proxy.py, compile_brain.py   # persona-injection proxy (local/Ollama backend) + index builder
scripts/
  benchmark.py               # seeded A/B benchmark (baseline vs markov vs rag vs persona vs logit_bias)
  score_detector.py          # GLTR-style perplexity + burstiness detector score
  build_book_corpus.py       # Project Gutenberg book -> sentence corpus
docs/
  RESEARCH_PROGRAM.md        # north-star hypothesis + phases
  phase*.md                  # per-phase results (the research record)
  corpora/                   # public-domain book corpora (committed)
```

## License

Code: AGPLv3 + commercial dual-license (see `LICENSE`). The bundled corpora are
derived from Project Gutenberg public-domain texts — see `docs/corpora/README.md`.

## Research & coding credits

This component's research and development were assisted by the following AI
models: **DS Pro, independent review Flash, adversarial critic 4.6, independent review human.**
