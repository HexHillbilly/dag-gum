# dag-gum

**A weight-free steering sandwich for small local models.**

---

## Executive summary

Small language models that run on a home computer are cheap and private — but
unreliable. Mid-sentence they get stuck and repeat themselves. Output drifts off
topic. A chosen voice or style falls apart after a few lines.

dag-gum fixes all three at *inference time*, without retraining and without
changing a single model weight. It stacks three independent, optional layers
around a frozen model, like a sandwich:

1. **Before** generation, two retrieval proxies (`scp/`, `ssp/`) condition the
   surface — they pull relevant lines from a corpus and steer output toward a
   target voice or style.
2. **During** generation, a governor (`AVG/`) watches every word and, the instant
   a repetition loop starts to form, gently steers the model back toward fresh
   words.
3. **After** generation, a two-stage synthesis states the grounded fact plainly,
   then voices it.

Each layer is bounded, measurable, and works on its own. Together they make a
1.5B–8B model behave like something far larger.

## Headline results

Verified, not asserted. Every number below is reproducible from the seeded scripts
in the component directories.

- **Degeneracy rescue: 90–100%** across 11 models spanning five architecture
  families — including two that aren't transformers — with no quality cost on
  normal text.
- **Voice/style transfer: up to 5.6× over baseline** in the target-voice metric, verified
  across corpora from a 92-line chatbot memory to a 109,000-word novel.
- **Tool use preserved:** a persona drops tool-call fidelity from 1.00 to 0.33;
  routing tool intent persona-free restores it while keeping the voice.

## Methodology

Every claim is measured under deterministic, reproducible conditions:

- **Voice/style transfer** uses a held-out split (retrieval on training lines,
  evaluation on unseen lines) scored with a corpus-centroid cosine metric
  (`style_cos`) and a topic-coherence control (`topic_cos`).
- **Degeneracy rescue** is measured under deterministic replay (fixed seed, greedy
  decode) on certified degenerate corpora against a frozen spectral threshold.
- **Detection effects** are scored with a variance-aware trained classifier, not a
  naive linear heuristic.

Negative results are kept beside the positive ones — the persona-masks-tool-use
finding, the rejected self-signal, and the voice-vs-content tradeoff are all
documented in the component `docs/`.

## The three layers

| layer | component | what it does |
|---|---|---|
| top bread | `scp/` (Semantic Context Proxy) + `ssp/` (Semantic Style Proxy) | corpus retrieval + prompt-injection voice/style conditioning |
| the meat | `AVG/` (Active Variety Governor) | real-time, closed-loop degeneracy rescue during single-token decode |
| bottom bread | `gum/` | the binder: the `Sandwich` orchestrator + the persona-drop tool gate |

The `gum` also owns the **persona-drop tool gate** — the measured finding that a
persona *destroys* tool use, and the fix: route tool intent persona-free, drop the
persona for the call, then resume in voice with a two-stage "state the fact, then
voice" prompt.

## For AI agents

This is a monorepo of four packages. To orient:

- `AVG/` — mid-generation governor. The repo root *is* the package
  (`AVG.core.*`, `AVG.governor.*`). Entry point: `governor/controller.py`
  (`ActiveVarietyGovernor`). Tests: `tests/`; gate scripts: `scripts/test_*.py`.
- `scp/` — pre-generation context/voice proxy. Pure logic in `core.py`;
  production server in `proxy.py`; measurement in `scripts/`.
- `ssp/` — pre-generation style proxy + detector. `core.py`; detector in
  `scripts/detector.py`.
- `gum/` — the harness. The `Sandwich` orchestrator wires the three layers.

Licensing is uniform (AGPLv3 + commercial) with a per-component `LICENSE`.

## Technical details

```
         ┌───────────────────────────────┐
  top    │  SCP / SSP  — pre-generation  │  retrieve a voice/style line,
  bread  │  conditioning                 │  wrap the prompt
         ├───────────────────────────────┤
   the   │  AVG  — mid-generation        │  suppress repetition / early-EOS
  meat   │  governance                   │  attractors during decode
         ├───────────────────────────────┤
 bottom  │  two-stage synthesis          │  state the grounded fact plainly,
  bread  │  (+ persona-drop tool gate)   │  then voice it
         └───────────────────────────────┘
```

Layout:

```
dag-gum/
  AVG/      # Active Variety Governor (mid-gen)
  scp/      # Semantic Context Proxy (pre-gen context/voice)
  ssp/      # Semantic Style Proxy (pre-gen style + detection)
  gum/      # the harness — wires the three layers
```

Usage:

```python
from sentence_transformers import SentenceTransformer
from scp import core, gate
from gum import Sandwich

# model + tokenizer are yours (transformers / Ollama / llama.cpp — anything with
# a raw generate). governor is optional:
#   from AVG.governor.controller import ActiveVarietyGovernor
sandwich = Sandwich(
    generate=raw_generate,            # (prompt, max_new_tokens) -> str
    retrieve=lambda q, k=1: core.retrieve(q, index, corpus, embedder, k),
    wrap=core.wrap_persona_hardened,  # or wrap_explain_in_terms / ssp.core.wrap_style
    governor=None,                    # optional ActiveVarietyGovernor
    router_prompt=gate.ROUTER_DEFINED,  # ROUTER_FEWSHOT for Phi-3-mini-class
    persona_name="Daisy",
)

print(sandwich.steer("What is the current temperature in Tokyo?"))
```

## License

All code in this repository — `AVG/`, `scp/`, `ssp/`, and `gum/` — is dual-licensed
under **GNU AGPLv3** (open source) **or a commercial license** (available from the
owner, HexHillbilly). The AGPLv3 text is the top-level `LICENSE`; each component
carries its own copy.

Third-party dependencies are under their own licenses (`markovify` is MIT).

### Origin

The SCP layer is an independent, modern re-implementation of the *idea* behind
**DAISY**, a 2000-era chatbot by Gregory G. Leedberg (the original is available
at <https://leedberg.com/glsoft/daisy/>). None of Leedberg's original source
code is included in this repository.

## Research & coding credits

This repository's research and development were assisted by the following AI
models: **DS Pro, independent review Flash, adversarial critic 4.6, independent review human.**
