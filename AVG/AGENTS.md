# Active Variety Governor (AVG) — Agent Orientation

This file is a concise, agent-readable guide to the `active-variety-governor` repository. It reflects the actual state of the codebase as of the last audit.

---

## Environment bootstrap

The system Python is externally managed (PEP 668). Create a venv on
every clone before installing packages:

    python3 -m venv /tmp/night_venv
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /tmp/night_venv/pyvenv.cfg
    /tmp/night_venv/bin/python -m ensurepip
    /tmp/night_venv/bin/pip install --no-cache-dir "transformers>=4.40.0" "accelerate>=0.27.0"

The system torch installation lives at /usr/local/lib/python3.11/dist-packages.
Using --system-site-packages avoids re-downloading torch (2GB+). The
sed command fixes a known bug where the flag doesn't take effect during
venv creation. ensurepip bootstraps pip if it's absent from the venv.

HF cache is read-only at ~/.cache/huggingface. Set HF_HOME for any
run that downloads or caches model files:

    export HF_HOME=/tmp/hf_cache

GPU is an RTX 3060 12GB. Verify with `nvidia-smi`.
Use `python3`, not `python`. All scripts run from the venv:
    /tmp/night_venv/bin/python scripts/...

---

## 1. Project overview

**Active Variety Governor (AVG)** is a real-time, closed-loop inference controller for Large Language Models (LLMs). It is grounded in Ashby’s Law of Requisite Variety and cybernetic feedback loops. Instead of applying static, open-loop steering vectors across all generation steps, AVG monitors internal residual-stream hidden states during single-token (`S=1`) autoregressive decoding and deploys bounded, single-layer interventions only when a repetition attractor or semantic collapse is detected.

Key design invariants documented in the repository:

- **Universal actuation (logit-penalty only)**: token suppression (−5.0 on
  `active_loop_ids`) rescues degeneracy across 9 transformers / 5 architecture families + 2
  non-transformers (Mamba SSM, RWKV-4 RNN). The residual-perturbation path is formally
  deprecated (KL ≈ 10⁻⁵ bf16 / 5.6e-7 fp32 — a precision-independent null space, not a
  quantization artifact).
- **Dual-predicate collapse detection** (bigram OR spectral PR): bigram = `token_diversity <
  0.30` AND `trailing_ctr < 0.30` (persistence ≥2); spectral = layer-2 PR < `band_low`
  (persistence ≥2, `token_diversity < 0.40` corroboration). `band_low` is a per-model valve
  (default 8.216097 = 1.5B-derived; zero-shot derivable via `scripts/auto_tune.py`).
- **Adaptive EOS-guard** (`eos_guard`, default True): masks `<eos>` during suppression → 100%
  rescue + 0 EOS-death. `eos_release_after` (default 0) is an opt-in bounded release.
- **128 pytest tests green**; latest governor cost ~5.15 ms/token (see `Readme.md`).

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
│   ├── generate_benchmark_assets_clean.py
│   ├── plot_real_telemetry.py
│   ├── profile_model.py
│   └── ... (many ablation / analysis scripts)
├── tests/                       # pytest unit tests
│   ├── test_metrics.py
│   ├── test_code_filter.py
│   ├── test_controller_code_filter.py
│   └── test_controller_generation_code_filter.py
├── docs/                        # Design documents
│   └── PATH2_STRUCTURAL_CODE_FILTER_DESIGN.md
├── pyproject.toml               # Primary project metadata
├── scripts/pyproject.toml       # Stale duplicate metadata — see notes below
├── Makefile                     # `make test` entry point
├── Readme.md                    # Human-facing project overview
└── LICENSE                      # AGPLv3 + commercial dual-license note
```

**Package layout note**: the repository root functions as the `AVG` package because of the empty `__init__.py`. Files in `scripts/` and `tests/` add the repo root to `sys.path` with the standard header `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` so that `from AVG.core...` / `from AVG.governor...` imports resolve when scripts are run directly. The Docker development container additionally relies on the editable install (`pip install -e .`) for top-level `core` / `governor` imports; the checkout directory must be named `AVG` (or the parent of the checkout must be on `sys.path`) for bare `import AVG` to work.

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

> **Current status**: 128 pytest tests pass (`python -m pytest tests/ -q`). The gate scripts
> (`test_factual_safety.py`, `test_degeneracy_smoke.py`, `benchmark_latency.py`,
> `test_path2_code_filter.py`) load `Qwen/Qwen2.5-1.5B` on first run (GPU recommended).

### Other common scripts

- Calibrate a baseline: `python scripts/profile_model.py --model <model> --output avg_baseline.pt`
- Generate with AVG: `python scripts/generate_avg.py --model <model> --prompt "..."`
- Compare governed vs. ungoverned: `python scripts/compare_avg.py`
- Produce benchmark telemetry + plot: `python scripts/generate_benchmark_assets_clean.py`
- Plot existing telemetry: `python scripts/plot_real_telemetry.py`

---

## 5. Runtime architecture

The generation loop in `ActiveVarietyGovernor.generate()` (`AVG/governor/controller.py`) is the central runtime path:

1. **Fast integer token proxy** (every step): `compute_token_distinct_2_fast` over the trailing
   24 generated tokens → `token_diversity` + `active_loop_ids` (repeated tokens).
2. **Suppression arming**: when `intervene=True` and `active_loop_ids` is non-empty, arm
   `_active_suppress_tokens[token] = cooldown_steps` (8). `intervene=False` is a true dormant
   baseline (suppression + kickstart gated off; collapse telemetry still records).
3. **Sub-sampled string decode** (every `every_k` steps or when diversity drops): decode the
   trailing tokens → `trailing_ctr` (coherence) + code-context detection.
4. **Dual-predicate collapse detection** (every step, `_evaluate_collapse`): bigram
   (`token_diversity < 0.30` AND `trailing_ctr < 0.30`, persistence ≥2) OR spectral PR
   (layer-2 PR < `band_low`, persistence ≥2, `token_diversity < 0.40` corroboration). Numeric
   and code contexts are immune.
5. **Kickstart trigger**: when repetition tokens are present and
   (`trailing_ctr < 0.50` OR collapse), arm the kickstart counter (cold-start fallback).
6. **Logit penalties (actuation)**: subtract `suppression_strength` (5.0) from each armed
   `active_suppress_token`'s logit for `cooldown_steps`. `eos_guard` (default True) masks
   `<eos>` during suppression (100% rescue + 0 EOS-death); `eos_release_after` (default 0) is
   an opt-in bounded release.
7. **Sample**: greedy argmax or `sample_top_p` nucleus sampling.

The residual-perturbation path is **deprecated** (KL ≈ 10⁻⁵ bf16 / 5.6e-7 fp32 — a
precision-independent null space). `_make_sae_guided_reset_fn` remains as a SAE-fallback but
carries no functional weight; actuation is logit-penalty only.

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
- Multi-model evaluation harnesses (`evaluate_avg.py`, `evaluate_qwen.py`, `evaluate_long_horizon.py`, `evaluate_pareto_frontier.py`, `evaluate_multi_pass.py`)
- SAE / feature analysis (`analyze_psychon_binding.py`, `analyze_psychon_binding_gpt2.py`, `ablation_feature_resolution.py`, `positive_control_monosemantic.py`)
- Asset generation and plotting (`generate_benchmark_assets_clean.py`, `plot_real_telemetry.py`, `gen_plot.py`)

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

## 8. Testing instructions

- Run `make test` for the documented validation suite. Be aware it downloads `Qwen/Qwen2.5-1.5B` and takes significant time/resources.
- Run `python -m pytest tests/ -q` for the unit-test suite. Collection is clean and the suite runs green (128 tests).
- Individual scripts accept `--model`, `--tokens`, `--max-dormant-overhead-pct`, `--sae-path`, `--every-k`, etc. Use `--help` for per-script options.

**Gate discipline:** Gates are seeded and deterministic. A red gate is a signal, not a negotiation — never fix it by loosening thresholds. Recalibration is legitimate only when the measurement instrument itself changes; it requires before/after data in the commit message and an explicit human decision — never as a response to a red gate.

**Latency gate is idle-GPU-only:** `benchmark_latency.py` measures per-token governor cost, which is only meaningful on an otherwise-idle GPU. A concurrent GPU job (background measurement sweep, another model run) inflates the number and can spuriously trip the 7.0 ms/token budget. Re-run latency on a quiet card before treating a latency red gate as a real regression.

### Known test / script issues

1. `scripts/compare_avg.py` imports `from AVG.scripts.profile_model import make_synthetic_trusted`, but `AVG.scripts` is not a package in this layout (`profile_model.py` lives under `scripts/`, not `AVG/scripts/`).

---

## 9. Security considerations

- **License exposure**: AVG is AGPLv3-licensed with a commercial dual-license option. Any SaaS or networked use of modified versions has source-disclosure obligations under the AGPL. Do not strip or alter `LICENSE` / `Readme.md` licensing language without explicit authorization.
- **Model downloads**: scripts automatically download pretrained models and SAE dictionaries from Hugging Face / SAELens. Ensure the environment has appropriate network controls and cache policies.
- **Code execution**: the project loads arbitrary Hugging Face models and runs `torch.inference_mode()` generation. Treat prompts and model inputs as untrusted in production deployments.
- **No secrets**: the repository does not contain API keys or credentials. Do not add `.env` files or tokens to the repo.
- **Pickle / `torch.load`**: `scripts/profile_model.py` and `generate_avg.py` use `torch.save` / `torch.load` for baseline payloads. Only load baselines from trusted sources.
- **In-place tensor mutation**: interventions modify residual tensors in forward hooks. Verify that the registered hook returns the mutated tensor to avoid silent no-ops.

---

## 10. Useful entry points for agents

| Task | File |
|------|------|
| Closed-loop generation | `AVG/governor/controller.py` → `ActiveVarietyGovernor.generate()` |
| Add a new collapse metric | `AVG/core/metrics.py` and `AVG/governor/controller.py` → `diagnose()` |
| Add a new SAE backend | `AVG/core/sae_loader.py` → subclass `BaseSAE` |
| Change hook discovery logic | `AVG/core/hooks.py` → `register_residual_hooks()` |
| Change collapse predicate | `AVG/governor/controller.py` → `_evaluate_collapse()` |
| Tune suppression / EOS-guard | `AVG/governor/controller.py` → constructor params `suppression_strength`, `eos_guard`, `eos_release_after`, `cooldown_steps`, `band_low` |
| Add a new failure mode | `AVG/governor/controller.py` → `FailureMode` + `diagnose()` |
| Benchmark / smoke test | `scripts/test_factual_safety.py`, `scripts/test_degeneracy_smoke.py`, `scripts/benchmark_latency.py` |
