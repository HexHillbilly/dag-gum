# Active Variety Governor (AVG)

## What is this? (plain English)

Small AI models — the kind that run on a home computer instead of a warehouse of
servers — have a bad habit: mid-sentence they get stuck and start repeating
themselves ("the cat sat on the mat, the cat sat on the mat…"). Big models hide
this; small models do it often enough to be unreliable.

**AVG is a lightweight safety net** that watches a model write, one word at a
time, and the moment it senses a loop forming, gently steers the model back
toward fresh words. It runs in real time, costs almost nothing, and needs no
retraining — it bolts onto a model that already exists.

**What the research shows:**

- **The fix works everywhere.** Tested on 11 models of radically different
  designs — including two that aren't even built like normal language models —
  it rescues the model 90–100% of the time, with no downsides.
- **The warning sign is universal.** The "about to loop" signal is a consistent
  internal pattern, identical across nine transformer models of five different
  designs — not a fluke of one model.
- **It doesn't hurt quality.** On normal text, output is measured to be just as
  good — sometimes better — with no quality cost.
- **New models onboard in one command.** The tool discovers a model's quirks,
  tunes itself, and prints a "health card." No hand-tuning.

**Why it matters.** If small, cheap, locally-run AI is going to be useful outside
a data center — privately, offline, on-device — it has to be dependable. AVG is
evidence you can make it dependable with a tiny, portable intervention rather
than a bigger, more expensive model.

---

## The technical picture

AVG is a closed-loop, inference-time controller for transformer language
models. During single-token autoregressive decoding it monitors the
residual stream for dimensionality compression and variety attenuation —
the early signatures of repetition collapse or drift — and, when a joint
set of gated conditions is met, applies bounded, calibrated logit-space
interventions before the next token is sampled. The design follows
Ashby's Law of Requisite Variety: a regulator must keep its own variety
in step with the variety of the disturbances it controls.

> **Research summary:** [RESEARCH.md](RESEARCH.md) is the portfolio-facing summary of the
> cross-model transfer + auto-tuning arc — token suppression is a universal degeneracy-rescue
> actuation (9 models / 5 architecture families + 2 non-transformers), the spectral-PR
> collapse floor ≈3 is explained as the rank of a 4-token loop motif (token-driven, see
> [RESEARCH.md](RESEARCH.md) §1.2), EOS-death is solved (`eos_guard`), and
> the per-model valves are auto-derivable (`scripts/auto_tune.py`, zero-shot validated on 4
> held-out models).

## Current status

- **Governor cost (absolute):** 4.2–5.9 ms/token band, latest
  measurement 5.15 ms/token. Relative-overhead ratios are informational
  only and are not headline figures.
- **Test suite:** 128 tests (pytest unit suite plus gate scripts).
- **Corpora:** three certified corpora (valid prose, degenerate loops,
  schema blocks); byte-level sha256 fingerprints are pinned in
  [docs/CORPUS_PROFILE.md](docs/CORPUS_PROFILE.md).
- **RFC-004 dual-predicate detection:** Gate 2.3b confirmed 0 false
  positives on prose-100, hazard corpora, and schema_corpus_v2 for
  spectral PR. Dual-predicate (bigram OR spectral PR at layer 2) is
  integrated in the controller per Amendment A3. See
  [docs/AVG_RFC_004_TWO_STAGE_DETECTION.md](docs/AVG_RFC_004_TWO_STAGE_DETECTION.md).
- **Residual perturbation path:** Formally deprecated for Qwen2.5-1.5B
  under greedy decoding per Amendment A4. Six measurement probes
  (DS-026 through DS-036) plus independent corroboration (SKOP)
  collectively established that bounded additive residual perturbations
  produce KL ≈ 10⁻⁵ — the unembedding never sees the perturbation.
  A follow-up fp32 re-open (`docs/gate23/FP32_RESIDUAL_RESULTS.md`)
  showed the dead-ness is **precision-independent**: in full fp32 a 0.5-L2
  kick along the Jacobian's optimal direction v₁ yields KL ≈ 5.6e-7 and
  byte-identical output — the residual→output map's dominant direction is
  orthogonal to the output *distribution* (a genuine null space), not a
  bf16 quantization hole. The residual code path remains in the controller
  for cross-model transfer or sampling-regime re-evaluation but carries no
  functional weight in the current architecture.

## Architecture sketch

AVG's control path is dual-resolution, per RFC-003:

- **Fast path (token-level):** a CPU-side integer ring buffer tracks
  bigram diversity of generated tokens. When diversity is healthy, the
  slow path is bypassed for that step.
- **Slow path (macro-trajectory):** a fixed-stride hidden-state ring
  buffer over later model layers feeds trajectory-velocity and
  participation-ratio checks. A PR-entropy shadow logger records
  telemetry without intervening.

**Collapse detection (RFC-004 Amendment A3):** The collapse predicate
is a dual-predicate OR gate:

- **Bigram detector:** `token_diversity < 0.30 AND trailing_ctr < 0.30`
  with persistence ≥2 consecutive steps. Catches micro-repetition
  (tight token loops within the 24-token window).
- **Spectral PR detector:** participation ratio at layer 2 below the
  frozen band_low threshold (8.216), with persistence ≥2 and
  token_diversity < 0.40 corroboration on spectral-only fires.
  Catches macro-loops (12–30 token motifs where local bigram diversity
  stays healthy but the hidden-state manifold collapses).

The spectral PR detector added 11 new rescues on t2s_degenerate records
that the production bigram-only path was blind to (DS-035).

**Actuation (sole demonstrated path):** The logit-penalty path applies
token suppression at cooldown=8, penalty=-5.0 (unconditional, every
step where repeated tokens are detected) and kickstart vocabulary
steering triggered on CTR drops. The residual perturbation path is
formally deprecated under greedy decoding per Amendment A4.

**Track B structural code filter:** the active intervention gate.
No intervention fires when the trailing window contains structural
code syntax (comment markers, code fences, keywords), keeping the
governor silent in code contexts.

Exact parameters (window lengths, stride, displacement bounds) are
specified in the RFCs.

## Measurement arc (2026-08-08 to 2026-08-12)

The actuation architecture was established through a 30+ probe
measurement arc under deterministic replay (SEED=42, Qwen2.5-1.5B,
bf16, greedy decoding unless noted):

- **DS-026/028/029:** Additive residual perturbations produce zero
  measurable effect (KL ≈ 10⁻⁵, byte-identical continuations) across
  single kicks, persistent multi-step kicks with production ramp, and
  joint deployment with the logit-penalty path.
- **DS-030:** Component ablation — token suppression is the primary
  actuator (48/50 rescue on heldout), kickstart is supplementary
  (handles 4% tail), combined path doubles EOS rate.
- **DS-031:** Cooldown sweep identifies (5,-5) as optimal on primary
  fixture but cross-fixture validation FAIL (qwen 68%, t2s 15%).
- **DS-033/034:** Production bigram detector is 93% blind on t2s
  macro-loops. Spectral PR fires on 100% of blind spots at all three
  candidate layers.
- **DS-036 / night-002:** Green's function SVD probe — the optimal
  singular vector v₁ of the empirical Jacobian produces the same null
  result as random orthogonal kicks. The residual channel has no
  viable direction under greedy decoding. Non-zero singular structure
  exists at shallower layers but is effectively closed at the
  production force band.
- **night-003:** J-Lens alignment — the collapse direction c₁ is
  systematically orthogonal to the Jacobian's output-sensitive
  direction v₁ (|cos| ≤ 0.017, below random baseline).
- **DS-038 / night-010:** ECS/PKS evaluated and rejected as binary
  collapse detectors due to prose false-positive rates. Inverted
  late-layer PKS (prose > degenerate) provides mechanistic evidence
  for "update starvation" during macro-loop collapse.
- **DS-032 Part 2 (sampling):** Suppression-only under temp=0.8
  rescues 50/50 heldout records with EOS=0.20 — the strongest
  result in the measurement series. Sampling regime is underexplored.

Full measurement reports are in `docs/gate23/`. A consolidated summary
is in `docs/gate23/ACTUATION_INVESTIGATION_SUMMARY.md`. A standalone
macro-loop dynamics model is in `docs/MACRO_LOOP_DYNAMICS.md`.

## Quickstart

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    # Unit test suite (128 tests)
    python -m pytest tests/ -q

    # One gate command: Path 2 structural code filter active gate
    python scripts/test_path2_code_filter.py

The gate scripts load a Hugging Face model on first run (GPU recommended;
CPU works). `make test` runs the full gate set: factual safety, degeneracy
smoke, Path 2 code filter, latency benchmark, and the unit suite.

## Repository layout

    ├── core/           # Metrics, hooks, SAE loader, causal probe, intervention log
    ├── governor/       # ActiveVarietyGovernor (controller), VarietyProfiler
    ├── scripts/        # Gate scripts, evaluation/demo scripts, corpus builders
    ├── tests/          # pytest unit suite and fixtures (degenerate, schema corpora)
    ├── data/t2s_bench/ # Certified Prose-200 subset and provenance artifacts
    ├── docs/           # RFCs, corpus profile, gate23 measurement reports, design notes
    ├── pyproject.toml  # Package metadata and dependencies
    └── Makefile        # make test entry point

The repository root is the `AVG` package (the root `__init__.py` marks it).
Scripts and tests use a root-as-package import header, so `AVG.core.*` and
`AVG.governor.*` resolve from the repo root.

## Development flow

AVG is developed on night branches by an automated engine. Every night
branch is a proposal: the human review tier reviews it, and a human ratifies it.
Runs are seeded, gate scripts and thresholds are owned by the human tier, and
a red gate is treated as a signal to stop and report — never as a reason to
loosen a threshold. The current standing orders live in
[PROTOCOL.md](PROTOCOL.md).

## License

AVG is dual-licensed. The source is available under the **GNU Affero General
Public License v3 (AGPLv3)** — see [LICENSE](LICENSE) — for open use,
matching the license declared in `pyproject.toml`. For proprietary or hosted
use where AGPL obligations do not fit, a separate commercial license is
available from the project owner.

<!-- TODO(owner): commercial contact -->

### Repository and roadmap

- **Repository:** <https://github.com/HexHillbilly/active-variety-governor>
- **Roadmap:** [RFC-004](docs/AVG_RFC_004_TWO_STAGE_DETECTION.md) defines
  the dual-predicate detection architecture, actuation evidence, and the
  phased gate roadmap. Corpus fingerprints are pinned in
  [docs/CORPUS_PROFILE.md](docs/CORPUS_PROFILE.md).
