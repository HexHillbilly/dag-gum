# AVG — Research Blueprint

**Active Variety Governor (AVG)** is a closed-loop, inference-time controller for
transformer language models. During single-token autoregressive decoding it monitors token
diversity and residual-stream geometry for the early signatures of **repetition collapse** —
the failure mode where a small model's decoder falls into a loop and emits degenerate text —
and applies a bounded logit-space intervention to break the loop before it is sampled. The
design follows Ashby's Law of Requisite Variety: the regulator's own variety must keep pace
with the variety of the disturbance it controls.

This document is the portfolio-facing summary of the research arc. The per-gate evidence is
in `docs/gate23/`; the controller lives in `governor/controller.py`.

---

## 1. Headline findings

### 1.1 Token suppression is a *universal* degeneracy-rescue actuation

Penalizing repeated tokens in the next-token logits (−5.0 on `active_loop_ids`) breaks
degenerate loops on **every model tested — 9 models across 5 architecture families, plus
two non-transformers (SSM + RNN):**

| model | family | rescue | regime |
|---|---|---|---|
| GPT-2 (124M) | LayerNorm / MHA / learned-pos / GELU | 16/16 (100%) | greedy |
| Qwen2.5-0.5B | RMSNorm / GQA / RoPE / SwiGLU | 80/80 (100%) | greedy + sampling |
| Qwen2.5-1.5B (production) | RMSNorm / GQA / RoPE / SwiGLU | 97–98/100 | greedy + sampling |
| Qwen2.5-3B | RMSNorm / GQA / RoPE / SwiGLU | 78/78 (100%) | greedy + sampling |
| Pythia-160m | GPT-NeoX *parallel* residual blocks | 21/21 + 10/10 | greedy + sampling |
| Llama-3.2-1B | RMSNorm / GQA / RoPE / SwiGLU (held-out) | 96% / 100% | greedy / sampling |
| SmolLM2-1.7B-Instruct | Llama-family (held-out zoo) | 95% / 90% | greedy / sampling |
| TinyLlama-1.1B-Chat | Llama-family (held-out zoo) | 94% / 100% | greedy / sampling |
| Phi-3-mini-4k | MHA / GELU / RoPE (held-out zoo) | 91% / 100% | greedy / sampling |
| Mamba-130m | **SSM (no attention)** | 7/7 | greedy |
| RWKV-4-169m | **RNN / linear attention (no softmax attn)** | 10/10 | greedy |

The actuation is pure logit-level — it has **zero architecture dependence**. It fires
unconditionally on repeated tokens, which is why it rescues valid-word loops on GPT-2 (and
SSM/RNN state loops on Mamba/RWKV) that spectral detection cannot see. This is the universal
*tight-loop* actuation; deep *macro-loops* require the spectral-triggered kickstart (§1.5).

### 1.2 The spectral-PR collapse floor ≈ 3 — *explained* (rank of a 4-token loop motif)

When a model collapses into a loop, the participation ratio of its layer-2 residual stream
compresses to a floor of **≈3** (2.89–3.13), observed across 9 models / 5 architecture families:

| model | degenerate min-PR | architecture |
|---|---|---|
| Qwen2.5-1.5B | 3.13 | Qwen |
| Qwen2.5-0.5B | 2.98 | Qwen |
| GPT-2 | 2.95 | GPT-2 |
| Qwen2.5-3B | 2.99 | Qwen |
| Pythia-160m | 2.97 | GPT-NeoX (parallel) |
| Llama-3.2-1B | 2.97 | Llama |
| SmolLM2-1.7B | 2.89 | Llama (held-out zoo) |
| TinyLlama-1.1B | 2.95 | Llama (held-out zoo) |
| Phi-3-mini-4k | 2.99 | MHA / GELU / RoPE |

**Root cause (2026-08-14):** `participation_ratio` centers the trailing-24 hidden-state window
and returns its effective rank. A loop of `k` distinct tokens has its `k` token embeddings span
`k−1` affine dimensions after centering, and layer-2 hidden states are token-embedding-dominated
(hidden PR tracks the token-embedding PR to within **+0.17**). So **PR ≈ 3 is `k−1` for a
4-token loop** — a hard rank-3 spectrum whose top-3 directions are the affine *differences*
between loop-token states, not identifiable tokens.

The "architecture-independence" is therefore **trivial** — a property of the PR metric and token
embeddings, not a shared internal-attractor invariant (see `docs/gate23/PR_FLOOR_ROOT_CAUSE.md`).
Moreover the loop *period* is itself **fixture-primed**: the t2s fixtures carry a `motif` field
(2–8 tokens, roughly uniform), and the model greedily *continues* that motif (22/24 exact, 2/24 a
sub-motif, 0/24 longer) rather than choosing a loop length. The "≈3 floor" is the 4-token subset
of that distribution — a data artifact, not a preferred loop length. The open question becomes:
*under what conditions does a model spontaneously form a short loop without a primed motif, and
what sets its period?*

Caveat: this is at **layer 2**, where hidden ≈ token. The spectral detector's layer-2 PR is best
described as a continuous token-diversity estimator; the distinct "logit-orthogonal hidden
collapse" (`MACRO_LOOP_DYNAMICS.md` Stage 4) was measured at layer 27 and is a separate signal.
The ≈3 floor is also **specific to short (≤4-token) loop motifs**: natural prose degeneration
under greedy keeps layer-2 PR high (mean 13.6 over 40 prompts) because it is long-range phrase
repetition, which is why the spectral detector fires 0/200 on natural text.

### 1.3 EOS-death is solved (adaptive EOS-guard)

Suppression creates a "logit vacuum": penalizing the loop token leaves `<eos>` as the
next-highest logit, so the model *terminates early* instead of producing content ("EOS-death").
The adaptive **EOS-guard** (`eos_guard`) masks `<eos>` while suppression is armed, forcing a
real non-`<eos>` continuation and restoring natural termination once the loop breaks.

Result: **100% rescue + 0% EOS-death** on every model — the fixed-penalty rescue↔EOS-death
tradeoff is *eliminated*, not tuned around. Shipped as the default. (Root-cause of the residual
EOS-death under the old path: it is **kickstart-driven** — the −1e4 first-step kickstart nukes
the loop token and leaves `<eos>` as argmax — not suppression itself. `docs/gate23/EOS_ANOMALY_15B.md`.)

### 1.4 Valves are per-model — and *auto-derivable*

The things that transfer (the suppression mechanism, the collapse signal) are universal. The
things that don't (the spectral threshold `band_low`, the exact suppression strength, the
EOS-release window) are **per-model "valves"**. Crucially, `band_low` is recoverable from a
small calibration pass over healthy prose: `band_low = 0.75 · (p90_degenerate + p10_prose) / 2`.

`scripts/auto_tune.py` automates the whole onboarding: discover the model's native loops →
derive `band_low` → gate rescue + EOS-death → emit a valve card. **Zero-shot on four held-out
models (never measured before): Llama-3.2-1B 96%/100%, SmolLM2-1.7B 95%/90%, TinyLlama-1.1B
94%/100%, Phi-3-mini-4k 91%/100% greedy/sampling rescue, all 0 EOS-death, every `band_low`
auto-derived (8.04 / 6.00 / 7.95 / 6.95).** No hand-tuning. (Phi-3-mini required
`low_cpu_mem_usage=True` — a 3.8B fp16 load; the fix is baked into the script.)

### 1.5 Input- and regime-dependence: tight loops vs macro-loops, sampling vs greedy

Token suppression (−5) is universal for **tight** loops — short word-loops where the loop token
is only narrowly the argmax. **Macro-loops** (12–30-token motifs where local bigram diversity
stays healthy but the hidden-state manifold collapses) are a *deep* class that −5 cannot
dislodge: the loop token's argmax margin ≫ 5, so suppression leaves the output byte-identical.

For macro-loops the **spectral detector is load-bearing**: it catches the PR collapse
(`PR < band_low`) and fires the **kickstart (−1e4)**, which nukes the loop token. Measured on the
t2s macro-loop fixtures (N=100): **97% rescue with the spectral detector armed, 17% without** —
81 of 100 rescued *only* because spectral is on (`docs/gate23/SPECTRAL_MARGINAL_RESCUE_RESULTS.md`).

This is the adversarial-defense story. Natural prose never reaches the detector (0 fires across
200 prompt-runs — natural repetition is *coherent*, not a hidden-state collapse), but a *served*
model that accepts arbitrary inputs needs it. Real degeneration is also **regime-dependent**:
rare under sampling (5–8%) but rampant under greedy decoding (full distinct-2 collapses to 0.23,
min 0.002), where the governor's suppression-only rescue is at its most valuable
(`docs/gate23/REAL_DEGENERACY_GREEDY.md`).

---

## 2. The valve framework

The research blueprint separates what is **architecture-universal** from what is a
**per-model valve**:

| layer | transfers? | note |
|---|---|---|
| token suppression (actuation) | **universal** (tight loops) | 9 models / 5 arch + SSM + RNN; macro loops need spectral kickstart |
| spectral-PR collapse signal | explained (rank of primed loop) | floor ≈3 = k−1, token-driven + fixture-primed |
| `band_low` (spectral threshold) | per-model valve | auto-derivable from prose baseline |
| `suppression_strength` | shared (≈5.0) | rescue curve near-identical across models |
| `eos_release_after` (bounded release) | per-model valve | 0.5B→4, 1.5B→8, 3B→12 — 0.5B-only benefit |

The practical consequence: **onboarding a new small model is one command**, not a hand-tuned
project.

---

## 3. Methodology

The arc was built under a strict discipline, which is itself part of the deliverable:

- **Evidence before code** — every mechanism was measured inline in a probe harness *before*
  being wired into the controller; the committed controller is the source of truth for a gate
  measurement.
- **investigate → verify → land** — anomalies are root-caused to their mechanism before any fix.
- **Gate doctrine** — stop on any red gate and report; never tune a threshold to force green
  without owner sign-off and before/after data.
- **Documented dead-ends are deliverables** — the residual-perturbation path (KL ≈ 10⁻⁵ in
  bf16 / 5.6e-7 in fp32 — a precision-independent null space, not a quantization artifact; see
  `docs/gate23/FP32_RESIDUAL_RESULTS.md`), the ECS/PKS detectors (prose false-positive rate
  fatal), the bounded EOS-release rule (does not transfer past 0.5B), and the eos-cap guard
  variant (fails) are all recorded so they aren't re-derived.

---

## 4. Reusable artifacts

- `scripts/auto_tune.py` — zero-shot onboarding pipeline (discovery → band_low → gate → card).
- `scripts/measure_cross_model_rescue.py` — cross-model rescue + tokens-before-degeneration gate.
- `scripts/sweep_suppression.py` / `scripts/measure_adaptive_suppression.py` — suppression
  valve curve + EOS-guard variants.
- `scripts/verify_eos_guard.py` — real-governor EOS-guard verification.
- `scripts/probe_eos_anomaly.py` — first-token EOS logit probe.
- `scripts/probe_rwkv.py` — non-transformer (RNN/linear-attn) suppression test.
- `scripts/measure_real_degeneracy.py` — real-prose validation across regimes (`--greedy`,
  tail/full distinct-2/3, tokens-before-degen, teacher-forced perplexity).
- `scripts/measure_spectral_marginal_rescue.py` — spectral detector's marginal rescue on
  macro-loop (adversarial) fixtures (band_low on/off).
- `docs/gate23/` — per-gate reports (`*_RESULTS.md` + `*_results.jsonl`).

---

## 5. Onboarding a new model (one command)

```bash
venv/bin/python scripts/auto_tune.py \
    --model <huggingface-id> \
    --prose data/t2s_bench/valid_subset_200.jsonl \
    --out docs/gate23/
```

Outputs a valve card (`<model>_autotune_card.{json,md}`) with `band_low`, rescue, EOS-death,
and tokens-before-degeneration. Pass `band_low` to `ActiveVarietyGovernor(model, tokenizer,
band_low=...)` and the governor is onboarded.

---

## 6. Reading map

- **This summary** → `RESEARCH.md` (here).
- **Cross-model valve map + architecture cheat-sheet** → the `avg-governor` skill,
  `references/cross-model-transfer.md`.
- **Suppression valve + EOS-guard methodology** → `avg-governor` skill,
  `references/suppression-valve-tuning.md`.
- **Auto-tune + zero-shot result** → `docs/gate23/AUTO_TUNE_PIPELINE.md`.
- **Macro-loop (adversarial) defense + real-degeneration regimes** → `docs/gate23/SPECTRAL_MARGINAL_RESCUE_RESULTS.md` + `docs/gate23/REAL_DEGENERACY_GREEDY.md`.
- **EOS-death root-cause** → `docs/gate23/EOS_ANOMALY_15B.md`.
- **Project overview / setup / license** → `Readme.md`.
