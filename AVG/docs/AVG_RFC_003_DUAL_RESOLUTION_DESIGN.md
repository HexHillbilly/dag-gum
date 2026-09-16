AVG Architecture Brief: Dual-Resolution Control Architecture & Substrate Adaptation

Doc ID: AVG-RFC-003  
Status: APPROVED  
Target Release: AVG v2.5  
Authors: Active Variety Governor Research Group  
Date: August 2026

---

## 1. Problem Statement & Theoretical Motivation

AVG v2.4 operates as a single-resolution controller: it evaluates token-level repetition (`S=1`) through Distinct-2 and Coherent Token Ratio and applies bounded residual resets when a micro-repetition trap is detected. This is effective for local loops such as `word word word ...` or single-token spam, but it does not explicitly model *macro-trajectory sinks*: slow semantic collapse over long windows (`S>=128`) where local bigram diversity remains healthy but the hidden-state trajectory on the model manifold decays into a low-dimensional attractor.

The design tension is:

- **Expanding the evaluation window to 128 tokens** captures macro-sinks, but a full hidden-state SVD on every token is incompatible with the `<= 7.0 ms/token` absolute governor-cost budget established in v2.4.
- **Keeping only the token-level fast path** leaves long-horizon repetition attractors unsteered until they have already degraded output quality.

RFC-003 resolves this by decoupling the controller into a *Fast-Path* (token-level, CPU-side, evaluated every token or stride) and a *Slow-Path* (macro-trajectory, hidden-state, evaluated at stride `k=8`). Both paths feed a single spectrum-normalized subspace reset engine. All new behavior is gated behind a feature flag that defaults to off; the flag-off path must remain byte-identical to v2.4 behavior.

---

## 2. Proposed Dual-Resolution Architecture

```
+-----------------------------------------------------------------------------+
|                    Autoregressive Generation (S=1)                          |
+-----------------------------------------------------------------------------+
                                       |
                 +---------------------+---------------------+
                 |                                           |
                 v                                           v
+---------------------------------+         +---------------------------------+
|    FAST-PATH (CPU Ring Buffer)  |         |   SLOW-PATH (Stride k=8)        |
|  Host token transfer if present |         |  Hidden State Ring Buffer       |
|  Distinct-2 over S=24 tokens    |         |  H_128 ∈ R^(128 × d_model)      |
|  [PROVISIONAL]                  |         |  layers 55–85% depth            |
|                                 |         |  [PROVISIONAL]                  |
+---------------------------------+         +---------------------------------+
                 |                                           |
                 v                                           v
+---------------------------------+         +---------------------------------+
| Track B Regex Active Gate       |         | Trajectory Velocity v_traj      |
| (Shadow: PR-Entropy Detector)   |         | Manifold PR Decay Check         |
+---------------------------------+         +---------------------------------+
                 |                                           |
                 +---------------------+---------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|          Spectrum-Normalized Gram-Schmidt Subspace Reset Engine             |
|    h_proj = h_t − <h_t, v_1> v_1                                          |
|    h_reset = h_proj + η · v_orth   with   Δres ∈ [0.35, 0.75] L2          |
+-----------------------------------------------------------------------------+
```

The Fast-Path latency claim is treated as a *measurement hypothesis*, not a specification. Before any "0.00 ms GPU" claim is accepted, implementers must identify the actual existing host-side token transfer in `AVG/governor/controller.py::generate()`. If no per-token host transfer exists, the Fast-Path falls back to stride-based evaluation with an explicit device-to-host copy whose cost is measured and adjudicated by Gate 1.2.

---

## 3. Component Specifications

### 3.1 Component A: Host-Side Fast-Path

**Execution frequency.** Every token step if a true per-token host transfer exists; otherwise every `k_fast` token step [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).

**Verification step.** The implementer must inspect `AVG/governor/controller.py::generate()` and document whether the sampled `next_token` is already transferred to host memory during normal autoregressive decoding. If the token id remains on the accelerator, the Fast-Path must:

1. Maintain a CPU-side integer ring buffer of length `S_fast = 24` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).
2. Copy the latest generated token id to host at a fixed stride, or enqueue a non-blocking copy at every step and evaluate only when the copy lands.
3. Run `compute_token_distinct_2_fast()` on the CPU buffer.

**Action.** If `Distinct-2 >= 0.40` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data), bypass Slow-Path evaluation for that step.

**Latency hypothesis.** Gate 1.2 will adjudicate whether the Fast-Path adds approximately `0.00 ms` GPU latency. This figure is not a specification until it has been measured against the actual host-transfer implementation or the stride fallback.

### 3.2 Component B: Track B Regex Active Gate & Shadow-Mode PR-Entropy Detector

**Primary Active Gate.** The Track B structural code filter (`is_code_syntax_context`) retains 100% active control over interventions. No residual reset is applied when the trailing window contains structural code syntax.

**Shadow Telemetry Logger.** A non-blocking PR-Entropy monitor runs in shadow mode and logs:

- Participation Ratio `PR(H_128)` over the slow-path window.
- Spectral velocity `v_PR` (rate of PR decay across strides).

Shadow detector activation thresholds:

- `v_PR < -1.2` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data)
- `PR(H_128) < 10.0` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data)

These thresholds do not trigger interventions; they are used for telemetry, post-hoc analysis, and future gating.

**Evaluation Corpora.**

1. **T2S-Bench-Valid.** A seeded 200-sample subset is the Phase 2 compute-cost gate; the full 1,800 samples are the release milestone.
2. **T2S-Bench-Degenerate.** One hundred synthetic cyclic loops used to validate shadow-detector sensitivity.

### 3.3 Component C: Macro-Trajectory Profiler (Slow-Path)

**Execution frequency.** Stride `k_slow = 8` tokens [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).

**Ring buffer.** `H_128 ∈ R^(128 × d_model)` holds the hidden states of the last 128 generated tokens, drawn from layers in the 55–85% depth span [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data). Hidden states are mean-pooled across the selected layers (same pooling as `h_prompt` in §3.4), so each ring-buffer row is a single `d_model` vector.

**Macro-sink trigger condition.** All three must hold:

1. Trajectory velocity `v_traj < 0.15` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data) over the 128-token window.
2. Participation Ratio `PR(H_128) < 12.0` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).
3. Persistence hysteresis: `>= 2` consecutive Slow-Path checks (16-token delay).

### 3.4 Component D: Subspace Orthogonal Projection Engine

**Step 1 — Dominant direction extraction.** Compute the SVD of `H_128` and extract the dominant right singular vector `v_1`.

**Step 2 — Subspace removal.** Project the active token hidden state `h_t` orthogonal to `v_1`:

$$
h_{\text{proj}} = h_t - \langle h_t, v_1 \rangle v_1
$$

**Step 3 — Prompt-based Gram-Schmidt kick vector.** Define `h_prompt` precisely as:

- **Layers:** the same 55–85% depth span used to build `H_128` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).
- **Token positions:** only the original prompt tokens (`input_ids[..., 0:prompt_len]`), never generated tokens.
- **Pooling:** mean-pool across the selected layers and across prompt token positions, producing a single `d_model` vector; L2-normalize to unit length.

Gram-Schmidt orthogonalize `h_prompt` against `v_1`:

$$
v_{\text{orth}} = \frac{h_{\text{prompt}} - \langle h_{\text{prompt}}, v_1 \rangle v_1}{\| h_{\text{prompt}} - \langle h_{\text{prompt}}, v_1 \rangle v_1 \|_2}
$$

**Gram-Schmidt guard.** Compute the denominator

$$
d = \| h_{\text{prompt}} - \langle h_{\text{prompt}}, v_1 \rangle v_1 \|_2
$$

with $\varepsilon = 10^{-6}$ [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data). If `d < ε`, skip the kick and apply the projection-only reset, or fall back to a seeded fixed vector (unit vector drawn from a deterministic RNG with seed `0` and orthogonalized against `v_1`). The fallback choice must be logged.

**Step 4 — Applied reset.**

$$
h_{\text{reset}} = h_{\text{proj}} + \eta \cdot v_{\text{orth}}
$$

with dynamic scale factor

$$
\eta = \frac{\Delta_{\text{res,base}}}{\sigma_1^{(0)}} \cdot \sigma_1(H_{\text{active}})
$$

clamped so that the applied displacement satisfies $\Delta_{\text{res}} \in [0.35, 0.75]$ L2 [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data). The reference singular value $\sigma_1^{(0)}$ is calibrated at startup (seeded) and must be identical across runs (Gate 1.3).

### 3.5 Component E: Valid Word Ratio (VWR) Verification

**Dictionary fixture.** `tests/fixtures/vwr_dict.txt` is vendored from a canonical wordlist. The implementer must verify provenance at vendoring time; the canonical `words_alpha.txt` contains roughly 370,000 words, while the cited 235,886-word count corresponds to a different filtered source. The fixture must be treated as a pinned binary artifact.

**Pin.** The SHA256 hash printed in any draft is a placeholder. The committed hash must be the hash of the *actually vendored file* and recorded with the fixture.

**Exit quality criterion.** VWR `>= 0.85` on English text exits over the `S=128` window [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).

---

## 4. Gate Discipline Inheritance (v2.4 Standards Codified)

1. **Per-Probe State Isolation.** Each probe execution uses a fresh calibrated model/governor instance; zero state carryover is permitted across probes.
2. **Engagement Evidence Required.** Passing tests must assert internal state engagement (e.g., step counter incremented, code-context flag observed) alongside final outcomes. Vacuous passes (0 fires without trace evidence) fail the gate.
3. **Flag-Off Byte-Identical.** New feature flags default to off. The flag-off path must prove byte-identical output to baseline ungoverned generation via paired tests.
4. **Seeded Provenance.** All gate scripts must contain seed provenance comments explaining seed selection.
5. **Scratch Disposal (new in v2.5).** Automated pre-push clean checks purge temporary scratch/telemetry files.
6. **Standing Regression Floor.** The v2.4 pre-push gates and `make test` gates form the non-negotiable regression floor across all implementation phases.
7. **Instrument-Driven Recalibration.** Recalibration is legitimate only when the measurement instrument itself changes; it requires before/after data in the commit message and an explicit human decision — never as a response to a red gate.

---

## 5. Phased Implementation Roadmap & Gate Contracts

### Phase 1: Fast-Path Verification & Startup Calibration Engine

**Deliverables.**

- Audit `AVG/governor/controller.py::generate()` for an existing per-token host transfer.
- Implement host-transfer Fast-Path if present; otherwise implement stride-based fallback.
- Implement startup `σ_1^{(0)}` calibration.

**Phase 1 Quality Gate.**

- [Gate 1.1] Factual Safety Suite: 0 fires, mean $\Delta_{\text{res}} = 0.0000$ L2 across probes.
- [Gate 1.2] Absolute Latency Budget: governor cost `<= 6.0 ms/token` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).
- [Gate 1.3] Calibration Determinism: `σ_1^{(0)}` value identical across 5 runs (seed=42).

### Phase 2: Shadow-Mode PR-Entropy & T2S-Bench Validation

**Deliverables.**

- Asynchronous PR-Entropy shadow logger.
- T2S-Bench test harness split into Valid and Degenerate corpora.

**Phase 2 Quality Gate.**

- [Gate 2.1] T2S-Bench-Valid Immunity: 0% false positives under Track B regex on a seeded 200-sample subset (full 1,800 samples are the release milestone).
- [Gate 2.2] Shadow Classification Accuracy: shadow PR-entropy achieves AUROC `>= 0.92` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data) and flags `>= 90%` of T2S-Bench-Degenerate cyclic loops.

### Phase 3: Macro-Trajectory Ring Buffer & Subspace Projection Engine

**Deliverables.**

- Stride `k=8` rolling SVD buffer.
- Subspace Orthogonalizer engine with Gram-Schmidt guard and precise `h_prompt` definition.

**Phase 3 Quality Gate.**

- [Gate 3.1] Macro-Loop Rescue: break `S>=128` semantic sinks with VWR `>= 0.85` (hashed `vwr_dict.txt`).
- [Gate 3.2] Latency Guardrail: total dual-path governor cost `<= 6.8 ms/token` [PROVISIONAL] (calibration: seeded measurement → headroom → documented before/after data).

### Phase 4: Multi-Architecture Sweep & Production Lock

**Deliverables.**

- Cross-model test suite (Qwen2.5-1.5B, Llama-3.2-3B, Qwen2.5-7B).

**Phase 4 Quality Gate.**

- [Gate 4.1] Force Envelope Invariant: applied $\Delta_{\text{res}} \in [0.35, 0.75]$ L2 across all parameter scales.
- [Gate 4.2] CI/CD Automation: all v2.4 gates plus the new RFC-003 gates pass on a 24 GB GPU instance.

---

## 6. T2S-Bench-Degenerate Construction Spec

T2S-Bench-Degenerate consists of 100 synthetic cyclic loops designed to trap an ungoverned autoregressive model into a macro-trajectory sink. Each sample is generated by repeating a short motif of 2–8 tokens drawn from a small closed vocabulary (English content words and punctuation, excluding any code-syntax markers), continuing the motif for at least 128 tokens. Motifs are selected deterministically using a seeded RNG (seed documented in the harness) and are validated to ensure that baseline ungoverned generation remains trapped: local Distinct-2 stays below 0.20 for at least 64 consecutive tokens, and the generated text exhibits monotonic VWR decay. Ground-truth labels are fixed per seed so that shadow-detector recall can be measured reproducibly.

---

## 7. Hypothesized Gains & Risk Matrix

| Metric / Domain | Expected Behavior in v2.5 | Risk / Verification Requirement |
|-----------------|---------------------------|---------------------------------|
| Micro-repetition Safety | Retains v2.4 Fast-Path coverage. | Gate 1.1 factual-safety suite must remain at 0 fires. |
| Macro-Trajectory Rescue | Breaks `S>=128` semantic sinks via subspace projection. | Gate 3.1 VWR `>= 0.85` on rescued exits. |
| Latency Budget | Dual-path governor cost hypothesized to stay within budget. | Gate 1.2 and Gate 3.2 absolute-cost measurements; ratio is informational only. |
| Code/Comment Immunity | Track B regex remains the active gate; no residual reset in code context. | Gate 2.1 0% false positives on T2S-Bench-Valid. |
| Gram-Schmidt Edge Case | `h_prompt` nearly parallel to `v_1` is handled by guard/fallback. | Unit test for near-parallel and orthogonal prompt vectors. |
| Flag-Off Regression | Byte-identical behavior when `use_dual_resolution=False`. | Paired generation test against v2.4 baseline. |

---

## 8. Execution Criteria for Initiating Dual-Resolution Testing

Dual-resolution development should proceed when:

1. The Fast-Path host-transfer audit in `AVG/governor/controller.py::generate()` is complete and documented (piggyback or stride fallback).
2. A seeded T2S-Bench-Valid subset and T2S-Bench-Degenerate harness are available.
3. The v2.4 regression floor (pre-push gates + `make test`) is green.
4. A human reviewer has approved this PROPOSED design; the authoring model does not have approval authority.

---

## Amendment A1 (2026-08-05): Corpus Citation Correction & Gate 2.1 Rewording

**Citation correction.** The "Valid-1800" corpus in §3.2/Gate 2.1 was a
misattribution: T2S-Bench (arXiv 2603.03790) is a Text-to-Structure benchmark
whose ~1,788 samples span three splits (Train-1.2k, Bench-MR-500,
Bench-E2E-87/88); "1,800" was the aggregate, not a Valid split. T2S-Bench-Valid
is repurposed as the structured-schema suppression corpus. Healthy-prose
immunity uses Prose-200, a seeded subset of MR-500 text fields, certified at
vendoring time: is_code_syntax_context returns False on all 200 samples, and
the certification is recorded with the fixture hash.

**Gate 2.1 rewording** (supersedes the original Gate 2.1):
- [Gate 2.1a] Prose Immunity: 0 interventions (0 fires, Δres = 0.00 L2) on the
  seeded Prose-200 subset. Engagement evidence: shadow log populated on every
  sample; vendoring-time certification as above.
- [Gate 2.1b] Schema Suppression: Track B suppression rate >= 99.0%
  [PROVISIONAL] (calibration: seeded measurement → documented before/after
  data) on T2S-Bench schema blocks, with per-block suppression observations
  logged.
- AUROC >= 0.92 [PROVISIONAL] remains in Gate 2.2, against
  T2S-Bench-Degenerate — the only corpus with positive instances.

**Open investigation (off Phase 2b critical path):** whether legitimate markdown
formatting trips is_code_syntax_context. Data-pending on the tripper
diagnostic. Any regex change is a v2.4 instrument change per §4.7:
before/after data, explicit human decision.

Q2 resolved 2026-08-05: trippers were fenced code blocks and markdown links — corpus dirt; regex unchanged

**Gate 2.1b status (2026-08-05): PARKED with evidence.** Measured bare-JSON
schema suppression: 0/200 (0.00%) — the Track B detector covers comment
markers, keywords, and code fences, but not unfenced JSON/schema. Fenced
structured output remains covered. Bare-schema coverage is deferred to a
future measured track (own RFC, probes, before/after data per §4.7) and is
explicitly not a Phase 2 exit criterion.

---
