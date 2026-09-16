# AGENTS.md — proposed replacement (DS-019)

Status: **proposal** from batch branch `night/ds-019-agents-md-proposal`.
`AGENTS.md` itself is human-edited doctrine and was **not** modified. This file
is a complete proposed replacement: everything still accurate is preserved
verbatim; every changed line is marked with `<!-- ds-019: ... -->`.

**To ratify**: copy the content below the first `---` over `AGENTS.md`, review,
and strip the `<!-- ds-019: ... -->` markers.

---

# Active Variety Governor (AVG) — Agent Orientation

This file is a concise, agent-readable guide to the `active-variety-governor` repository. It reflects the actual state of the codebase as of the last audit.

---

## 1. Project overview

**Active Variety Governor (AVG)** is a real-time, closed-loop inference controller for Large Language Models (LLMs). It is grounded in Ashby’s Law of Requisite Variety and cybernetic feedback loops. Instead of applying static, open-loop steering vectors across all generation steps, AVG monitors internal residual-stream hidden states during single-token (`S=1`) autoregressive decoding and deploys bounded, single-layer interventions only when a repetition attractor or semantic collapse is detected.

Key design invariants documented in the repository:

- **100% factual-safety immunity** on coherent, factual, and code-flavored text (silent, `0.00 L2` displacement) over 256-token windows.
- **4.2-5.9 ms/token absolute governor cost** when dormant (latest measured 5.83 ms/token; guardrail budget `<= 7.0 ms/token` absolute), achieved through vectorized GPU integer proxy operations and dormant hook scheduling. <!-- ds-019: stale "+6.33% passive latency overhead" ratio replaced by canonical 4.2-5.9 ms/token absolute band (latest 5.83); benchmark_latency.py asserts absolute governor cost <= 7.0 ms/token; ratios are informational only -->
- **Surgical perturbation energy**: bounded `0.45–0.75 L2` single-layer directional displacements at the active token position (`[:, -1, :]`).
- **Joint conjunction gate (`AND`)**: requires simultaneous diversity collapse (`Dist-2 < 0.30`) *and* coherence breakdown (`CTR < 0.30`) plus persistence hysteresis (`≥ 2` consecutive steps) before firing.

The project is dual-licensed under **GNU AGPLv3 / Commercial** terms (see `LICENSE`).

---

## 2. Technology stack

- **Language**: Python 3.10+.
- **Deep learning**: PyTorch (`torch>=2.0.0`), Hugging Face `transformers`, `accelerate`.
- **Sparse autoencoders**: `sae-lens>=3.0.0` (optional; a `DummySAE` fallback is always available).
- **Numerics / plotting**: NumPy, Matplotlib.
- **Build backend**: `setuptools>=61.0` / `wheel`.
- **Testing**: `pytest>=7.0.0`.
- **Linting / formatting**: `black`, `ruff` (declared as optional dev dependencies).

The runtime defaults to **CUDA when available**, otherwise CPU, with many SVD-heavy paths falling back to CPU for large matrices and forcing `float32` for numerical stability under FP16/BF16 inference.

---

## 3. Repository structure

```
├── __init__.py                  # 0-byte marker: repo root IS the AVG package
├── core/                        # Low-level utilities and primitives
│   ├── hooks.py                 # PyTorch forward hooks, ring buffers, dormant scheduling
│   ├── sae_loader.py            # SAE factory + SAELens bridge + DummySAE fallback
│   ├── causal_probe.py          # Causal spectator probe (KL/logit-shift causality check)
│   ├── metrics.py               # Dist-2, CTR, SVD-based dimensionality metrics
│   └── intervention_log.py      # Lightweight per-intervention logging
├── governor/                    # High-level controller
│   ├── controller.py            # ActiveVarietyGovernor, dual-gate logic, interventions
│   └── profiler.py              # VarietyProfiler, VarietyProfile, layer statistics
├── scripts/                     # Standalone evaluation / benchmark / demo scripts
│   ├── test_factual_safety.py
│   ├── test_degeneracy_smoke.py
│   ├── test_path2_code_filter.py
│   ├── benchmark_latency.py
│   ├── evaluate_avg.py
│   ├── evaluate_qwen.py
│   ├── profile_model.py          <!-- ds-019: removed stale generate_benchmark_assets_clean.py / plot_real_telemetry.py tree entries — both scripts were removed from the repo (v2.4) -->
│   └── ... (many ablation / analysis scripts)
├── tests/                       # pytest unit tests
│   ├── test_metrics.py
│   ├── test_metrics_edge_cases.py
│   ├── test_code_filter.py
│   ├── test_controller_code_filter.py
│   ├── test_controller_generation_code_filter.py
│   ├── test_dual_resolution_flag.py
│   └── test_shadow_hook_cleanup.py   <!-- ds-019: tests tree updated to current modules (DS-003/005/008/010 added the metrics-edge-case, dual-resolution, and shadow-hook-cleanup tests) -->
├── docs/                        # Design documents
│   ├── AVG_RFC_003_DUAL_RESOLUTION_DESIGN.md
│   ├── CORPUS_PROFILE.md
│   ├── the smoke record
│   └── PATH2_STRUCTURAL_CODE_FILTER_DESIGN.md   <!-- ds-019: docs tree updated to current design documents -->
├── pyproject.toml               # Primary project metadata   <!-- ds-019: removed stale scripts/pyproject.toml entry — duplicate was deleted in the pyproject-drift resolution (ecebde1) -->
├── Makefile                     # `make test` entry point
├── Readme.md                    # Human-facing project overview
└── LICENSE                      # AGPLv3 + commercial dual-license note
```

**Package layout note**: the repository root functions as the `AVG` package because of the empty `__init__.py`. Files in `scripts/` and `tests/` add the repo root to `sys.path` with the standard header `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` so that `from AVG.core...` / `from AVG.governor...` imports resolve when scripts are run directly. <!-- ds-019: import convention — root-as-package / parents[2] / container "/" note; replaces the previous editable-install / checkout-naming description (in the container, `import AVG.*` resolves via the /AVG mount) --> In the Docker development container the repository is mounted at both `/workspace` (working directory) and `/AVG`; from `scripts/` or `tests/`, `parents[2]` resolves to `/` — that is correct, not a bug — and `import AVG.*` resolves via the `/AVG` mount. Never create alias packages, `AVG/` directories, or `sys.modules` workarounds.

---

## 4. Build, install, and run commands

### Install in editable mode

```bash
pip install -e .
```

Optional dev dependencies:

```bash
pip install -e ".[dev]"
```

### Run the validation suite

The `Makefile` defines the primary CI-like entry point:

```bash
make test
```

This executes, in order:

1. `python scripts/test_factual_safety.py`
2. `python scripts/test_degeneracy_smoke.py`
3. `python scripts/benchmark_latency.py`

These scripts default to `Qwen/Qwen2.5-1.5B` and will download model weights from Hugging Face on first run. GPU is optional but strongly preferred.

### Run unit tests directly

```bash
python -m pytest tests/ -v
```

> **Current status**: the unit suite runs green; see “Testing instructions” below for the current test count and environment-gated modules. <!-- ds-019: stale "IdentitySAE collection failure" status removed — test_metrics.py now imports `DummySAE, load_sae` and the IdentitySAE import was fixed (commit 2e92fae) -->

### Other common scripts

- Calibrate a baseline: `python scripts/profile_model.py --model <model> --output avg_baseline.pt`
- Generate with AVG: `python scripts/generate_avg.py --model <model> --prompt "..."`
- Compare governed vs. ungoverned: `python scripts/compare_avg.py`
<!-- ds-019: removed stale commands for generate_benchmark_assets_clean.py / plot_real_telemetry.py — both scripts were removed from the repo (v2.4) -->

---

## 5. Runtime architecture

The generation loop in `ActiveVarietyGovernor.generate()` (`AVG/governor/controller.py`) is the central runtime path:

1. **Hook registration**: `HookManager` registers residual-stream forward hooks on transformer blocks (candidates: `model.layers`, `transformer.h`, `gpt_neox.layers`, `model.decoder.layers`, `transformer.layers`).
2. **Dormant monitoring**: on most steps the hook callbacks passthrough with near-zero overhead. The governor wakes hooks on every `every_k` step or whenever an intervention is registered.
3. **Collapse detection**: each step computes `Dist-2` over generated token bigrams (`compute_token_distinct_2_fast`) and `CTR` (`compute_coherent_token_ratio`) over the trailing 16 decoded tokens.
4. **Profile step**: when collapse indicators pass the dual-gate thresholds, `VarietyProfiler` runs a full forward pass to collect per-layer participation ratio, spectrum entropy, sparsity, reconstruction error, and attenuation.
5. **Diagnosis**: `diagnose()` applies the joint `AND` gate plus a persistence counter (`≥ 2` consecutive collapse steps), then selects the deepest monitored commitment layer (≥ 55% depth).
6. **Intervention**: a registered forward hook applies one of three reset paths:
   - `_monosemantic_clamp` when active feature participation ratio is low and few latents fire.
   - `_attractor_subspace_reset` (preferred polysemantic path): extracts an attractor basis from a 16-token window via SVD and projects a bounded displacement into the strict nullspace.
   - `_polysemantic_reset` fallback: Gram-Schmidt-style orthogonal perturbation guided by decoder directions or pure random noise.
7. **Logit-side kickstart**: when repetition tokens are detected, token suppression and temporary non-prose penalties may be applied to next-token logits, followed by `sample_top_p` nucleus sampling.
8. **Telemetry**: per-step `dist_2`, `ctr`, `fired`, and `delta_l2` records are saved as JSONL via `save_telemetry()`.

All intervention magnitudes are clamped to the `[0.35, 0.75]` L2 range around the active token only.

---

## 6. Code organization and module divisions

### `AVG/core`

- **`hooks.py`**: `HookManager` and `LayerActivationRecord`. Handles residual hook discovery, registration, dormant passthrough, intervention dispatch, and a per-layer ring buffer of the last `history_len` token activations.
- **`sae_loader.py`**: `BaseSAE`, `SAELensWrapper`, `DummySAE`, and loaders (`load_sae`, `load_sae_from_hf_repo`, `load_sae_map_for_model`). Provides a unified interface over SAELens releases, Hugging Face repos, and a random-top-k fallback.
- **`metrics.py`**: token/string metrics (`compute_coherent_token_ratio`, `compute_token_distinct_2` in `controller.py`), SVD-based dimensionality metrics (`participation_ratio`, `singular_value_spectrum_entropy`, `effective_rank`), SAE metrics (`feature_sparsity_ratio`, `residual_reconstruction_error`, `coherence_index`, `variety_attenuation_proxy`).
- **`causal_probe.py`**: `CausalProbe` ablates top-k sparse features and verifies that output logits shift measurably before an intervention is allowed.
- **`intervention_log.py`**: `InterventionLogger`, `InterventionRecord`, and helper functions for measuring residual deltas and next-token KL/logit shifts.

### `AVG/governor`

- **`controller.py`**: `ActiveVarietyGovernor`, `BaselineStats`, `InterventionDecision`, `FailureMode`, `compute_token_distinct_2_fast`, and `sample_top_p`. Contains the full closed-loop generation method.
- **`profiler.py`**: `VarietyProfiler`, `VarietyProfile`, `LayerVarietyStats`. Runs the model through hooks and converts activations into scalar variety/attenuation statistics.

### `scripts`

Stand-alone, mostly executable scripts. Most insert the repo root into `sys.path` so that `AVG.*` imports resolve when run directly. They cover:

- Factual safety probes (`test_factual_safety.py`)
- Degeneracy recovery smoke tests (`test_degeneracy_smoke.py`)
- Latency benchmarks (`benchmark_latency.py`)
- Multi-model evaluation harnesses (`evaluate_avg.py`, `evaluate_qwen.py`, `evaluate_long_horizon.py`, `evaluate_multi_pass.py`) <!-- ds-019: removed stale evaluate_pareto_frontier.py reference — script no longer exists in the repo -->
- SAE / feature analysis (`analyze_psychon_binding.py`, `analyze_psychon_binding_gpt2.py`) <!-- ds-019: removed stale ablation_feature_resolution.py / positive_control_monosemantic.py references — scripts no longer exist in the repo -->
<!-- ds-019: removed stale "Asset generation and plotting" bullet — generate_benchmark_assets_clean.py / plot_real_telemetry.py / gen_plot.py no longer exist in the repo (v2.4) -->

---

## 7. Development conventions

- **Import style**: almost every Python file starts with `from __future__ import annotations`. Scripts that are meant to be run directly add `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` or `parents[2]` to resolve the `AVG` package.
- **Package naming**: the installed package is `AVG` (all caps), not `avg` or `active_variety_governor`.
- **Type hints**: pervasive use of `typing` generics (`Dict`, `List`, `Optional`, `Tuple`, `Callable`, `Sequence`) and dataclasses.
- **Numerical safety**: SVD paths force `float32`; large matrices fall back to CPU; clamping with small `eps` values (`1e-8`) is standard.
- **Device/dtype discipline**: tensors are explicitly moved to target devices/dtypes before operations, especially around SAE encode/decode and unembedding.
- **Fail-soft**: SAE loading failures silently fall back to `DummySAE`; diagnosis returns early if the baseline is uncalibrated.
- **Constants embedded in code**: key thresholds (`0.30`, `0.35`, `0.75`, persistence counter `2`, cooldown `8`, `history_len=16`) are currently hard-coded rather than configuration-driven.
- **Comment conventions**: module-level docstrings summarize purpose; ASCII diagrams and section banners (`# ========================================================================`) are used for major algorithm tracks.

---

## 8. Batch development workflow <!-- ds-019: new section — the two-stage batch devflow; points to PROTOCOL.md (docs/DEVFLOW.md does not yet exist in this branch) -->

AVG uses a two-stage batch development flow. The full standing orders live in `PROTOCOL.md`; the summary here keeps the conventions visible to agents:

- **human tier** owns gate scripts, thresholds, and measurement instruments. Every night branch is a *proposal*: human review, then human ratification. No night branch merges itself.
- **automated engine** (`batch_run.sh <task>`) handles volume work — file sweeps, fixture/corpus generation, test generation — in disposable clones that never push, never create remotes, and never leave the clone.

Standing rules for batch engines:

- **Red-gate doctrine**: a red gate is a signal, not a negotiation. Stop, log the evidence, report. Never loosen a threshold to make a gate pass.
- **Stale-premise rule**: prove-and-report, never manufacture work. If a task premise is stale (the described issue no longer exists), verify with evidence and report back rather than manufacturing work to fit the premise.
- **Engine-tier rule**: gate scripts, thresholds, and measurement instruments stay with the human review tier regardless of which engine proposes the change.
- **Seeded runs**: all night runs are seeded; outputs print seed provenance and engagement evidence. Gate outputs are quoted verbatim in the final report.
- **Import convention**: the repo root IS the `AVG` package; scripts/tests use `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`. In the container the repo mounts at both `/workspace` and `/AVG`; from `scripts/` or `tests/`, `parents[2]` resolves to `/` — correct, not a bug — and `import AVG.*` resolves via the `/AVG` mount.

---

## 9. Testing instructions

- Run `make test` for the documented validation suite. Be aware it downloads `Qwen/Qwen2.5-1.5B` and takes significant time/resources.
- Run `python -m pytest tests/ -v` for the unit-test suite. The four `transformers`-free modules run green (66 tests) in the CPU-only DS container; three modules (`test_controller_generation_code_filter.py`, `test_dual_resolution_flag.py`, `test_shadow_hook_cleanup.py`) additionally require `transformers` and are environment-gated in that container. <!-- ds-019: test count and collection status updated — "40 tests" was stale after DS-003/005/008/010 added modules; 66 tests pass in the transformers-free subset here -->
- Individual scripts accept `--model`, `--tokens`, `--max-dormant-overhead-pct`, `--sae-path`, `--every-k`, etc. Use `--help` for per-script options.

**Gate discipline:** Gates are seeded and deterministic. A red gate is a signal, not a negotiation — never fix it by loosening thresholds. Recalibration is legitimate only when the measurement instrument itself changes; it requires before/after data in the commit message and an explicit human decision — never as a response to a red gate.

### Known test / script issues

<!-- ds-019: removed stale issue "tests/test_metrics.py imports IdentitySAE" — the import was fixed (commit 2e92fae) and the module now imports `DummySAE, load_sae` -->
1. `scripts/compare_avg.py` imports `from AVG.scripts.profile_model import make_synthetic_trusted`, but `AVG.scripts` is not a package in this layout (`profile_model.py` lives under `scripts/`, not `AVG/scripts/`).
<!-- ds-019: removed stale issue "scripts/pyproject.toml is an inconsistent duplicate" — the file was deleted in the pyproject-drift resolution (ecebde1) -->

---

## 10. Security considerations <!-- ds-019: renumbered after new section 8 inserted -->

- **License exposure**: AVG is AGPLv3-licensed with a commercial dual-license option. Any SaaS or networked use of modified versions has source-disclosure obligations under the AGPL. Do not strip or alter `LICENSE` / `Readme.md` licensing language without explicit authorization.
- **Model downloads**: scripts automatically download pretrained models and SAE dictionaries from Hugging Face / SAELens. Ensure the environment has appropriate network controls and cache policies.
- **Code execution**: the project loads arbitrary Hugging Face models and runs `torch.inference_mode()` generation. Treat prompts and model inputs as untrusted in production deployments.
- **No secrets**: the repository does not contain API keys or credentials. Do not add `.env` files or tokens to the repo.
- **Pickle / `torch.load`**: `scripts/profile_model.py` and `generate_avg.py` use `torch.save` / `torch.load` for baseline payloads. Only load baselines from trusted sources.
- **In-place tensor mutation**: interventions modify residual tensors in forward hooks. Verify that the registered hook returns the mutated tensor to avoid silent no-ops.

---

## 11. Useful entry points for agents <!-- ds-019: renumbered after new section 8 inserted -->

| Task | File |
|------|------|
| Closed-loop generation | `AVG/governor/controller.py` → `ActiveVarietyGovernor.generate()` |
| Add a new collapse metric | `AVG/core/metrics.py` and `AVG/governor/controller.py` → `diagnose()` |
| Add a new SAE backend | `AVG/core/sae_loader.py` → subclass `BaseSAE` |
| Change hook discovery logic | `AVG/core/hooks.py` → `register_residual_hooks()` |
| Change intervention force bounds | `AVG/governor/controller.py` → `_monosemantic_clamp`, `_attractor_subspace_reset`, `_polysemantic_reset` |
| Add a new failure mode | `AVG/governor/controller.py` → `FailureMode` + `diagnose()` |
| Benchmark / smoke test | `scripts/test_factual_safety.py`, `scripts/test_degeneracy_smoke.py`, `scripts/benchmark_latency.py` |
