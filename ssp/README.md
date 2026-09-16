# Semantic Style Proxy (SSP)

## Executive summary

**Can you make a local AI "write like" someone — an author, a brand voice, or
yourself — without retraining it?**

Yes, but only partly — and measuring it honestly matters more than the headline.
SSP steers a small local model to write in a target style by retrieving a few
relevant source lines and injecting them with a voice descriptor. It also scores
the result against a trained statistical detector, because style-matching and
detection are two sides of one coin.

## Headline results

Measured on a held-out split of the target corpus (retrieval on train lines,
evaluation on unseen lines), `microsoft/Phi-3-mini-4k-instruct`:

| question | answer |
|---|---|
| Does style transfer move the voice? | **Yes** — `style_cos` 0.12 → 0.39 (held-out) |
| Does it hurt topic/coherence? | **No** — `topic_cos` stays ~0.53 |
| Does it reduce AI-detectability? | **Yes, weakly** — P(AI) 0.97 → 0.82 (−15%), honest detector |
| Can the burstiness "tell" be dampened? | **Partly** — a rhythm regularizer reaches −23% |

The honest number, not the naive one: a variance-aware detector penalizes
burstiness too far from human in *either* direction. A naive linear detector
overstated the evasion by ~4×; finding and fixing that is itself one of the
project's results.

## Methodology

- **Style transfer** — retrieve the few corpus sentences most similar to the
  request (MiniLM embeddings + FAISS L2), inject them with a voice descriptor, and
  score the output's `style_cos` (output↔corpus-centroid cosine) on held-out lines.
- **Detection** — a variance-aware trained classifier (`scripts/detector.py`),
  not the naive linear heuristic it supersedes.

Everything runs on one consumer GPU with a 3.8B-parameter model. No fine-tuning,
no cloud, no large training runs.

*On intent: the detection half exists to **understand** how AI-text detectors work
— so they can be built better — not to help evade them.*

## For AI agents

- `core.py` — retrieval / wrappers / logit-bias + rhythm regularizer.
- `scripts/benchmark.py` — A/B harness (`baseline|markov|rag|persona|style|style_rhythm|logit_bias`).
- `scripts/detector.py` — the variance-aware trained detector (the honest score).
- `scripts/score_detector.py` — earlier GLTR-style proxy (superseded by `detector.py`).
- `scripts/build_book_corpus.py` — Gutenberg book → sentence corpus.

## Usage

```bash
# Build a sentence-per-line corpus from a Project Gutenberg book:
python scripts/build_book_corpus.py <raw.txt> docs/corpora/<name>.txt --start-after "<heading>"

# A/B benchmark (the style-transfer gate):
python scripts/benchmark.py --corpus docs/corpora/hemingway.txt \
    --probe-set style --conditions baseline,style,style_rhythm \
    --style-name "Ernest Hemingway" \
    --style-desc "spare, declarative prose with short sentences, concrete nouns and verbs, and no ornament"

# Honest detection score (variance-aware trained classifier):
python scripts/detector.py scripts/results.jsonl docs/corpora/austen.txt docs/corpora/hemingway.txt
```

## Layout

```
core.py, style_prior.py, compile_brain.py   # retrieval / wrappers / logit-bias + rhythm regularizer / index builder
scripts/
  benchmark.py          # A/B harness
  detector.py           # variance-aware trained AI-text detector
  score_detector.py     # earlier GLTR-style proxy (superseded)
  build_book_corpus.py  # Gutenberg book -> sentence corpus
docs/
  RESEARCH_PROGRAM.md   # thesis, hypotheses, phases, results
  corpora/              # public-domain corpora (Hemingway, Austen) + attribution
```

## License

Code: AGPLv3 + commercial dual-license (see `LICENSE`). Bundled corpora are
derived from Project Gutenberg public-domain texts — see `docs/corpora/README.md`.

## Research & coding credits

This component's research and development were assisted by the following AI
models: **DS Pro, independent review Flash, adversarial critic 4.6, independent review human.**
