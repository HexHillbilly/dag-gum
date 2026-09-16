# Macro-Loop Degeneration Dynamics in Autoregressive Transformers

**Date:** 2026-08-12
**Model:** Qwen2.5-1.5B @ 8faed761, bf16, SEED=42
**Status:** Descriptive model — measurement-only, not an architectural commitment

## 1. Overview

The Active Variety Governor (AVG) project has identified two distinct
degeneracy mechanisms in autoregressive transformer decoding:
**micro-repetition** (short token sequences repeating within a 24-token
window) and **macro-loops** (extended 12–30 token motifs repeating across
longer spans while local surface diversity remains healthy). Micro-repetition
is well-handled by token-level suppression. Macro-loops are the harder case.

Three measurement probes — night-003 (J-Lens alignment), DS-038 (ECS/PKS
diagnostics), and night-010 (live-generation ECS) — together describe a
coherent mechanistic picture of macro-loop degeneration. This document
captures that picture as a standalone design note.

## 2. The Four-Stage Collapse Model

The ordering below is a **narrative ordering of co-measured phenomena,**
not a proven causal chain. Whether the stages form a cascade or are
independent consequences of entering the attractor basin is unknown
(see §6, Open Question 2).

┌─────────────────────────────────────────────────────────────────────────────┐
│ MACRO-LOOP DEGENERATION DYNAMICS │
├─────────────────────────────────────────────────────────────────────────────┤
│ │
│ 1. Manifold Compression (Spectral PR, Layer 2) │
│ Participation Ratio drops below band_low (8.22). The residual stream │
│ loses effective dimensionality. This is the earliest detectable │
│ signature of collapse. │
│ │
│ 2. Mid-Layer Attention Detachment (exemplar head L6H6) │
│ Exemplar head L6H6 live ECS collapses to ~0.17 during macro-loop │
│ generation. Some mid-layer attention heads detach from the prompt │
│ context and lock onto the active generated token loop. The model │
│ stops attending to what it was asked to generate and instead │
│ attends to what it is already generating. │
│ │
│ 3. Late-Layer FFN Update Starvation (Layers 21–27) │
│ PKS (JSD between pre-FFN and post-FFN LogitLens vocabulary │
│ projections) drops sharply on degenerate text: PKS_deg ≈ 0.016–0.039 │
│ vs PKS_prose ≈ 0.072–0.121. Feed-forward networks stop writing │
│ updates to the residual stream, passing un-updated loop states │
│ forward. This is "update starvation," not parametric override: the │
│ FFNs aren't injecting wrong information — they've stopped injecting │
│ information at all. │
│ │
│ 4. Logit-Orthogonal Hidden Collapse (Layer 27) │
│ The dominant collapse direction c₁ of the trailing 24-token hidden │
│ state window is nearly orthogonal to the Jacobian's output-sensitive │
│ direction v₁ at layer 27 (max |cos(c₁, v₁)| = 0.017, below the │
│ random-orthogonal baseline of ~0.0204 for R^1536). The residual │
│ manifold compresses along an axis the unembedding matrix does not │
│ read from. Surface bigrams remain locally valid while the internal │
│ trajectory collapses onto a low-rank attractor. │
│ │
└─────────────────────────────────────────────────────────────────────────────┘

### 2.1 Stage 1 — Manifold Compression

Spectral PR (participation ratio at layer 2, last-24 token window) drops
below the frozen band_low threshold of 8.216 during t2s macro-loop
generation. Mean PR on detection-gap records: 4.29 at layer 2, well below
the 10.95 midpoint threshold (DS-034). This is the earliest detectable
signature — manifold compression begins before attention detachment or
FFN update starvation becomes measurable.

### 2.2 Stage 2 — Mid-Layer Attention Detachment

During live KV-cache generation on t2s detection-gap records, per-head ECS
(external context score — the fraction of attention mass from generated tokens
to the prompt) was measured at every decoding step across candidate layers
{2, 6, 14, 21, 27}. Exemplar head L6H6 (layer 6, head 6) showed the most
dramatic behavior:

| Measurement | Value |
|---|---|
| Live ECS mean | 0.17 |
| Teacher-forced ECS mean (DS-038) | 0.48 |
| Mean delta | -0.30 |
| Pearson r (live vs teacher-forced) | 0.85 |

At layer 6, this exemplar head is barely attending to the prompt at all
during real generation. The teacher-forced measurement — which treated
everything before the last 24 tokens as "prompt" — massively overestimated
its prompt-attention ratio. Under genuine autoregressive decoding with a
real prompt boundary, this head locks almost entirely onto the expanding
generated-history buffer.

Other mid-layer heads show similar behavior at varying magnitudes, while
some late-layer exemplar heads (L21H3 at 0.72, L2H1 at 0.87) retain
stronger prompt grounding. The detachment is head-specific and concentrated
in mid-depth layers — it is not a uniform property of all heads at a given
depth. The per-head ECS stability table in night-010 shows heads with
standard deviations ranging from 0.005 (L6H9, essentially invariant) to
0.23 (L14H1, highly dynamic), confirming that different heads follow
different trajectories during collapse.

### 2.3 Stage 3 — FFN Update Starvation

PKS (Parametric Knowledge Score) measures the Jensen-Shannon Divergence
between vocabulary distributions projected from hidden states before and
after each FFN layer via LogitLens. The calibrated expectation from the
ReDeEP RAG framework was that degenerate records would show HIGHER PKS —
FFNs over-writing the residual with parametric knowledge. The measurement
showed the opposite:

| Layer | PKS degenerate | PKS prose | Direction |
|---|---|---|---|
| 21 | 0.039 | 0.121 | prose HIGHER |
| 27 | 0.016 | 0.072 | prose HIGHER |

Healthy prose actively rewrites the residual through FFN layers. Macro-loop
degenerate text shows near-zero FFN updates at upper layers. The FFNs aren't
injecting wrong information — they've stopped injecting information entirely.
This is "update starvation": the attractor is so stable that the FFN's
native write goes to near-zero, and the residual passes through unchanged.

This finding is mechanistically consistent with Stage 4: if the hidden state
is already in the repeating basin, there's no gradient signal to drive FFN
updates, and the block's output converges to near-identity.

### 2.4 Stage 4 — Logit-Orthogonal Hidden Collapse

The J-Lens alignment diagnostic (night-003) measured the per-step cosine
similarity between two directions at layer 27 across 80 snapshots (10 records
× 8 steps):

- **c₁:** the dominant right-singular vector of the trailing 24-token hidden
  state window — the direction of manifold compression.
- **v₁:** the top right-singular vector of the residual→logit Jacobian at
  the current hidden state — the direction the unembedding is most sensitive to.

| Quantity | Value |
|---|---|
| |cos(c₁, v₁)| range | 0.0001 – 0.0167 |
| Mean align (borderline) | 0.0034 |
| Mean align (collapsed) | 0.0065 |
| Random-orthogonal baseline (~R^1536) | ~0.0204 |

The collapse direction c₁ and the Jacobian's sensitive direction v₁ are
systematically **more orthogonal than random vectors.** The manifold compresses
along an axis the unembedding doesn't read from. This explains the core
t2s detection-gap signature: spectral PR fires because compression is real,
but surface text stays healthy because the compression is invisible to the
output layer.

A weak rotation toward alignment exists (collapsed mean 0.0065 vs borderline
0.0034) but never leaves the near-orthogonal noise floor. The collapse doesn't
"become visible" to surface metrics — it stays invisible throughout the
entire generation.

## 3. Unified Mechanistic Picture

The four co-measured phenomena form a coherent picture of macro-loop
degeneration:

1. **Manifold compression begins early** (layer 2 PR drops below band_low).
   The hidden state loses effective dimensionality.

2. **Some mid-layer attention heads detach** (exemplar head L6H6 locks onto
   the generated loop rather than the prompt). The model reduces its
   attention to what it was asked to generate.

3. **Late-layer FFNs stop updating** (PKS drops to near-zero at layers
   21-27). The residual stream passes through unchanged because the
   attractor is too stable to generate a meaningful FFN write.

4. **The collapse is output-invisible** (c₁ ⊥ v₁ at layer 27). Surface
   bigrams remain locally diverse because the compression axis is
   orthogonal to the unembedding's sensitive directions.

The practical consequence: a model can be deeply collapsed in hidden-state
space while producing locally coherent text for hundreds of tokens.
Detection requires spectral instruments (PR). Additive residual kicks
under greedy were inert in prior probes (DS-026/028/029). night-003 further
shows the **collapse axis itself** is a near-null direction for residual→logit,
so collapse-aligned residual edits are especially unlikely to move outputs.

### 3.1 Boundary note: teacher-forced vs live ECS

The ECS values reported in DS-038 (teacher-forced replay) were computed
with an artificial `prompt_len = seq_len - 24` boundary — everything before
the last 24 tokens was treated as "prompt." Live-generation ECS (night-010)
uses the genuine prompt boundary tracked by the controller during
autoregressive decoding. Live ECS is systematically lower than
teacher-forced ECS (mean Δ = -0.11 at L21H3, -0.30 at L6H6) because
attention mass bleeds into the expanding generated-history buffer during
real generation. Teacher-forced ECS scores should not be treated as
live-generation levels.

## 4. Measurement Evidence

| Probe | Date | Finding |
|---|---|---|
| night-003 | 2026-08-11 | c₁ ⊥ v₁ at layer 27; collapse direction orthogonal to output-sensitive Jacobian directions |
| DS-038 | 2026-08-12 | ECS/PKS teacher-forced measurement; inverted late-layer PKS (prose > degenerate); prose FP rates disqualify both as binary gates |
| night-010 | 2026-08-12 | Live-generation ECS; systematic downward shift vs teacher-forced; L6H6 attention detachment confirmed (live mean 0.17) |

## 5. Implications for AVG

- **Detection:** Spectral PR (participation ratio at layer 2, frozen threshold
  band_low = 8.22) is the correct instrument for macro-loop detection. It
  measures manifold compression regardless of whether the collapse is surface-visible.

- **Actuation:** The logit-penalty path (token suppression + kickstart vocabulary
  steering) is the sole demonstrated actuator. Additive residual perturbations
  are inert under greedy decoding on this model (DS-026/028/029). Multiplicative
  residual intervention (AARF) is parked pending positive liveness evidence.

- **Surface metrics:** Trailing CTR and token diversity are surface-level
  signals that can remain healthy during deep hidden-state collapse. They
  cannot serve as corroboration gates for spectral detection (confirmed by
  three failed hardening attempts in DS-034c/d/e).

- **ECS/PKS:** Both instruments are maintained as offline diagnostic tools.
  They are not wired into the controller's collapse predicate. ECS provides
  per-head attention distribution visibility; PKS provides per-layer FFN
  update dynamics visibility. Both are useful for understanding model behavior
  but do not meet the gate doctrine's 0-FP standard for prose [1].

## 6. Open Questions

1. **Does the collapse model generalize beyond t2s_degenerate?** The four
   co-measured phenomena were observed on a single macro-loop fixture. Whether
   the same dynamics appear on other macro-structural degeneracy patterns is
   unmeasured.

2. **Are these phenomena causal or correlational?** The four stages are
   measured co-occurring phenomena. Whether manifold compression causes
   attention detachment, or whether both are independent consequences of
   entering the attractor basin, is unknown.

3. **Can multiplicative attention amplification break the attractor?** AARF
   remains parked. The residual→logit singular-vector probe (night-002)
   established that non-zero singular structure exists at every probed depth.
   A blind sweep at exemplar layers with attention amplification would be
   the next residual falsifier — testing whether boosting context-tracking
   heads at mid-depth can redirect the trajectory. This requires its own
   before/after measurement data and explicit human sign-off before any
   actuation design [1].
