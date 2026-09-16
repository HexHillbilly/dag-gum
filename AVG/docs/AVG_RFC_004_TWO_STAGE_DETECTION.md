# AVG RFC-004 v2 — Two-Stage Composite Degeneracy Detection

**Status:** RATIFIED 2026-08-08 by owner. Amended after S4 (independent review) review
and external critic (adversarial critic) attack pass. No threshold herein is active until
Gate 2.3 produces before/after data and the owner signs it.

**Grounds:** RFC-003 (Phase 2 complete); Gate 2.2 measurement
(docs/gate22/GATE22_AUROC_RESULTS.md); ds-022 positive control;
independent review RFC-004 proposal (adopted portions); critic hardenings (v2).

## 1. Measured foundation

Gate 2.2 (Qwen2.5-1.5B @ 8faed761, bf16, SEED=42, last-24-token window,
replay, n=100 vs 100):

| Signal | Degenerate mean | Prose mean | Separation | Direction |
|---|---|---|---|---|
| PR (best layer) | 1.95 (qwen) / 3.97 (t2s) | 18.58 | 1.0000 | degenerate-lower |
| SVSE (best layer) | 0.25 / 0.44 | 0.96 | 1.0000 | degenerate-lower |
| ER (best layer) | 2.28 / 4.15 | 21.62 | 1.0000 | degenerate-lower |
| Distinct-2 (text) | 0.130 / 0.221 | 0.981 | 1.0000 | degenerate-lower |
| CTR (text) | 0.958 / 0.964 | 0.874 | 0.8618 | degenerate-HIGHER |

Engagement: 17/20 qwen_degenerate prompts fired; 1964 injections; layer 26;
Δres in [0.46, 0.75] L2. Probe B byte-exact reproduction (ds-022).

## 2. Quarantine ledger

RESOLVED: "PR<<12.0" (measured ~2-4 vs ~18.6); "Δres band [0.35,0.75]"
(that band is the applied-force clamp, controller.py:566; observed Δres
reaches 0.7861); "14 fires" (14 decisions -> 43 injections); d_model 1536.

STILL QUARANTINED: "RepE Δres ~ 25-35 L2" (no canonical source);
"v_traj < 0.15" (never measured — candidate definition now exists, §7.4).

## 3. Design: two-stage composite detection

### Stage 1 — context routing ONLY
- D_schema (structural delimiter density) routes context class:
  code/schema -> Path-2 immunity path (existing, controller.py:455-458);
  prose -> eligible for Stage 2.
- D_schema is a design intention, not an operational gate. Its numeric
  predicate and error rate on mixed code/prose are UNMEASURED (§7.2).
  ds-012: Track B comment-marker dependence leaves Java/Go/C partially
  blind (66.7/81.8/84.8%) — a known mis-routing risk Gate 2.3 must bound.
- Distinct-2 is demoted to post-hoc / offline analysis. Live text-level
  collapse detection remains with the EXISTING token_diversity instrument
  (current predicate, unchanged by this RFC). Live sliding-window
  Distinct-2 is uninstrumented (§7.3).

### Stage 2 — spectral detector (hidden-state)
- Primary: Participation Ratio, last-24-token window. Measured means
  ~2-4 (degenerate) vs ~18.6 (prose) — state measured means, not derived
  multipliers, when writing thresholds.
- Fusion rule: PR primary. Gate 2.3 defines an ambiguity band around the
  threshold from measured distributions (frozen before scoring). Inside
  the band, fire requires SVSE agreement (below its own frozen band edge).
  Both signals inside their bands: NO FIRE, log only (conservative).
- CTR: dropped from the decision path entirely (0.8618, inverted
  polarity). Diagnostic/shadow logging only — never in a fire predicate.
- Detection layer: decided by Gate 2.3 joint measurement — detection
  AUROC per layer AND downstream effect at the same layer. "Detect early,
  act late" is NOT assumed.

## 4. Gate 2.3 calibration protocol (needs its own greenlight)
1. Calibration corpora: qwen_degenerate (sha256 2e0f21a0...),
   t2s_degenerate, prose-100 (seeded valid_subset_200).
2. Held-out scoring slice: INDEPENDENTLY CONSTRUCTED hard-degenerate set
   (new generation method or held-out prompt/seed split, never the
   calibration data), plus held-out prose, hazard corpora (ds-007),
   schema_corpus_v2 (ds-018).
3. Thresholds derived from Gate 2.2 distributions, frozen BEFORE any
   held-out scoring, including ambiguity-band edges (§3).
4. Acceptance (all required): AUROC >= 0.98 on held-out contrasts;
   0 false positives on prose + hazard + schema corpora (including Stage-1
   routing measurement on structured-legitimate text); minimum recall
   >= 0.95 on the held-out hard-degenerate slice.
5. Stop-don't-negotiate: a miss is a signal, not a tuning session.

## 5. Integration: shadow-mode-first
Stage-2 detector lands log-only. Promotion requires human shadow-log review,
before/after gate, and explicit owner ratification. No change to the
existing collapse predicate in this RFC.

## 6. Substrate-leakage hypothesis — parked
Entry criteria (adversarial critic, faithfully restated): multi-token capture >=16-24,
orthonormal basis (QR/SVD), subtractive removal + bounded nullspace
injection, exact A/B vs blind Gram-Schmidt under the same force budget,
promote only on win. Forced-kick is an instrument change requiring
explicit owner ratification.

## 7. OPEN QUESTIONs (unmeasured — do not quote as fact)
1. Detection layer choice — Gate 2.3 joint measurement (§3, §4).
2. D_schema numeric predicate + error rate on mixed code/prose.
3. Live sliding-window Distinct-2 — uninstrumented.
4. v_traj candidate (independent review proposal, unmeasured/unratified): normalized
   mean cosine step-distance between consecutive hidden-state deltas over
   W=24; hypothesized -> 1.0 in collapse attractors. O(W*d_model), no
   SVD — shadow-log candidate during Gate 2.3. Instrumentation point:
   controller profile steps (every_k=2); no governor/profiler.py exists.
5. CTR composite value — none in the decision path.
6. Cross-model transfer — all values Qwen2.5-1.5B-specific.
7. "RepE Δres ~ 25-35 L2" — no canonical source; quarantined.

## Amendment A2 (2026-08-08): DS-028 persistent-regime joint measurement

Gate 2.3b (ds-027) scored detection with the independently constructed
heldout-degenerate v2 fixture: separation 1.0000 at all three candidate
layers, recall 1.00/0.99/0.98, 0 false positives across all legitimate
corpora (docs/gate23/GATE23_RESULTS_V2.md).

DS-026 measured single-fire effect: ΔDistinct-2 = 0.0000 at every layer.
Single kicks produce byte-identical continuations.

DS-028 (queued) closes the loop: per-layer effect measurement under the
PRODUCTION persistent regime — persistence hysteresis ≥2, force ramp
min(0.75, 0.45 + (consecutive-1)*0.05), hook stays alive for full
generation. This is the joint detection×effect complement to ds-027.
Layer selection remains the human's act (§3, §4).

## Amendment A3 — FP Policy (ratified 2026-08-10)

**Status:** RATIFIED by owner. Dual-predicate FP policy per independent review
directive (Option A).

### Accepted pre-production debt

The dual-predicate architecture (bigram OR spectral PR at layer 2,
band_low = 8.216097, token_diversity < 0.40 corroboration on spectral-only
fires) produces 7 documented false positives on prose-100 under live
KV-cache greedy generation (DS-034c, DS-034d, DS-034e):

- **5 bigram cold-start FPs** (records 47, 49, 76, 83, 87): transient
  early-step token repetition on healthy prose, triggering the production
  bigram path. These are pre-existing in the production controller and
  are not introduced by the dual-predicate.
- **2 spectral late-dip FPs** (records 98, 99): manifold compression
  (PR ≈ 6.6–7.5, below band_low) with healthy surface text (trailing_ctr
  ≈ 1.0, token_diversity ≈ 0.39) at late generation steps (120+). These
  are the measurable incremental tax of the dual-predicate.

Surface corroboration metrics (CTR, token_diversity) are structurally
incompatible with simultaneously filtering prose FPs and preserving
t2s macro-loop detection coverage. Three hardening attempts (DS-034c/d/e)
confirmed this. Per pre-commitment, no further surface threshold will be
tested.

### Disposition

The 7 FPs are accepted as documented pre-production debt. The dual-predicate
proceeds to DS-035 (cross-fixture rescue gate). The FP policy may be
revisited after DS-035 if rescue efficacy justifies additional hardening
investment.

### Escape clause

If DS-035 produces rescue count = 0 on both t2s_degenerate AND
qwen_degenerate, Amendment A3 auto-reverts to bigram-only and the
dual-predicate is shelved. If rescue count ≥ 1 on either fixture, A3
ratifies and FP optimization becomes the next night gate.

## Amendment A4 (2026-08-10): Residual Perturbation Path — Deprecated

**Status:** RATIFIED by owner. Backed by six measurement probes (DS-026,
DS-028, DS-029, DS-030, DS-034b, DS-036) and external corroboration (SKOP).

### Measured finding

The residual perturbation path — bounded orthogonal kicks applied to
hidden states at candidate layers {2, 12, 26} — produces zero measurable
effect on continuations under greedy decoding on Qwen2.5-1.5B. This
finding holds across four distinct perturbation strategies:

1. **Single kicks** (DS-026): one bounded orthogonal perturbation at
   production force (0.45 L2, clamp [0.35, 0.75]). ΔDistinct-2 = 0.0000
   at all three candidate layers. Every continuation byte-identical to
   dormant.

2. **Persistent multi-step kicks with production ramp** (DS-028): hook
   stays alive for entire generation, persistence hysteresis ≥2, force
   ramp min(0.75, 0.45 + (consecutive-1)×0.05). 5,150 fires per layer,
   mean strength 0.74 L2, force saturated to 0.75 for the majority of
   fires. ΔDistinct-2 = 0.0000, ΔCTR = 0.0000 at all three layers.
   Mean per-fire KL: 1.30×10⁻⁵ (L2), 1.16×10⁻⁵ (L12), 6.70×10⁻⁶ (L26).

3. **Joint with logit-penalty path** (DS-029, DS-030): residual
   perturbation deployed simultaneously with token suppression and
   kickstart vocabulary steering. The joint condition is token-identical
   to the logit-penalty-only condition on all 150 (record × layer) pairs.
   The residual perturbation adds zero marginal effect. The logit-penalty
   path alone carries the full rescue.

4. **Optimal singular vector direction v₁** (DS-036): the top right-
   singular vector of the empirical Jacobian M = W_U · J_{26→L} at layer
   26, estimated via finite differences (K=128 QR-orthonormal probe
   directions, ε=10⁻³, fp32 fallback) and power iteration (converged at
   cos > 0.999999 after 6 iterations). A 0.50 L2 kick along v₁ —
   the mathematically optimal direction for logit shift per unit energy —
   produced mean KL = 9.40×10⁻⁶ over the first 10 generated tokens.
   First-token KL = 3.60×10⁻⁵. Continuation byte-identical to dormant.
   Even the optimal singular vector produces the same null result as
   random orthogonal directions.

### Mechanistic explanation: bf16 dead-zone at intervention depth

DS-036 also discovered a bf16 numerical dead-zone at layer 26. At
residual norms of O(250), bf16's absolute precision step for typical
residual components is ~0.03 (derived from bf16's 7-bit mantissa:
the exponent for values in [4, 8) is 2², giving a step of 8 ÷ 128 =
0.0625 at magnitude ~6.4). The DS-026/028 perturbation per component
at η = 0.50 is ~0.013 (0.50 ÷ √1536), well below the bf16 precision
floor. The perturbation is rounded to zero for most dimensions before
the forward pass propagates it. This is not a "washout by later layers"
in the transformer sense — it is numerical absorption at the number
format level, confirmed by the DS-036 probe: ε=10⁻³ produced zero-
shift for 99.2% of probe directions in bf16 but resolvable logit shifts
for all 128 directions in fp32.

The bf16 dead-zone is layer-specific — residual norms grow across the
transformer depth, and the precision floor rises with them. At shallower
layers where residual norms are smaller, smaller-magnitude perturbations
may be resolvable. This is an unmeasured possibility for future work but
does not change the deprecation finding for the production intervention
depth (layer 26).

### Architectural consequence

The residual perturbation path is formally deprecated for Qwen2.5-1.5B
under greedy decoding. The controller's `diagnose()` method retains the
residual intervention code path (it is cheap, dormant-by-default, and
may be re-evaluated on other models or under sampling), but it carries
no functional weight in the current architecture. The logit-penalty path
is the sole demonstrated actuator: token suppression at cooldown=8,
penalty=-5.0, with kickstart vocabulary steering (DS-029, DS-030).

The RFC-003 "Subspace Reset Engine" (Component D, §3.4) and the Phase 3
"Macro-Trajectory Ring Buffer & Subspace Projection Engine" (§5) are
parked indefinitely for this model and decoding regime. They are not
deleted — the architecture documentation preserves them for cross-model
transfer or sampling-regime re-evaluation — but they are not part of
the active controller specification.

### External corroboration: SKOP

Recent work on SKOP (Steering via Key-Orthogonal Projections)
independently validates AVG's dormant-by-default gating architecture.
SKOP demonstrates that continuous steering vectors added at every
generation step distort key-query matrices (Q·K^T) on focus tokens,
degrading performance on MMLU and TruthfulQA — an "attention rerouting
tax" proportional to steering strength and frequency. AVG's closed-loop
gating — remaining at 0.00 L2 force during healthy generation, confirmed
by the CI factual safety suite — avoids this tax entirely. The finding
reinforces the design choice to keep intervention dormant until a
collapse predicate is met, rather than running continuous background
steering.

### Measured evidence (cited)

- DS-026: Single-fire layer effect (docs/gate23/LAYER_EFFECT_RESULTS.md)
- DS-028: Persistent multi-step with production ramp
  (docs/gate23/LAYER_EFFECT_PERSISTENT_RESULTS.md)
- DS-029: Component ablation — residual vs logit-penalty vs joint
  (docs/gate23/COMPONENT_ABLATION_RESULTS.md)
- DS-030: Logit-penalty decomposition — suppression vs kickstart
  (docs/gate23/PENALTY_DECOMPOSITION_RESULTS.md)
- DS-034b: Online PR parity — teacher-forced vs hook-based
  (docs/gate23/ONLINE_PR_PARITY_RESULTS.md)
- DS-036: Green's function SVD probe — residual channel liveness
  (docs/gate23/GREENS_FUNCTION_RESULTS.md)
- SKOP: Steering via Key-Orthogonal Projections (external, cited for
  dormant-by-default validation; not an AVG measurement)

## Amendment A5 (2026-08-12): AARF Evaluation and Residual Intervention Formalism

**Status:** MEASUREMENT COMPLETE for ECS/PKS on t2s vs prose. AARF and
multiplicative residual intervention **documented and parked**. Additive
residual remains deprecated for production under greedy decoding per
DS-026/028/029 (+ precision-floor evidence in DS-036). Dual-predicate
(spectral PR → logit path) unchanged by this amendment.

### 1. Residual Intervention Formalism

The forward pass of a standard pre-norm autoregressive Transformer at
layer l is:

    u_l   = h_l + Attn_l(h_l)           (post-attention residual)
    h_l+1 = u_l + FFN_l(u_l)             (post-FFN residual)

where h_l ∈ R^d_model is the residual-stream representation entering
layer l, Attn_l is the multi-head attention operation, and FFN_l is
the feed-forward network.

**Additive intervention** (DS-026, DS-028, DS-029, DS-036):

    h_l' = h_l + α · v_steer

where v_steer is a direction vector and α ∈ [0.35, 0.75] is the
intervention strength (L2 norm, clamped per controller.py:566).
This mechanism produced KL ≈ 10⁻⁵ under greedy decoding across six
probes. Candidate mechanism (DS-036): at residual norms O(10²),
per-component additive δ at α ∈ [0.35, 0.75] can fall near or below
bf16 resolution (~0.03 per component at layer 26), consistent with
observed KL ≈ 10⁻⁵. Not proven to be the only cause of residual
nulls — later-layer absorption or RMSNorm attenuation may also
contribute.

**Multiplicative intervention** (AARF, ReDeEP framework):

    Attn_l'(h_l) = γ · Attn_l(h_l)       (γ > 1: amplify context-tracking heads)
    FFN_l'(u_l)   = β · FFN_l(u_l)        (β < 1: suppress parametric overrides)
    h_l+1' = u_l' + FFN_l'(u_l')

where γ and β are scalar multipliers applied to sub-module outputs
before residual addition. Because attention outputs at shallower
layers have norms O(10-50) rather than O(250), multiplicative scaling
can move the residual by magnitudes representable in bf16.

Multiplicative intervention is a distinct mathematical class from
additive orthogonal kicks. Whether γ/β modulation changes continuations
on AVG fixtures is **unmeasured**. Parked until (i) residual→logit or
module-output liveness under multiplicative edits, and (ii) rescue
metrics on held-out degenerate corpora with FP controls.

### 2. Diagnostic Instrument Evaluation (ECS / PKS)

The ReDeEP framework's two core metrics were evaluated across candidate
layers {2, 6, 14, 21, 27} on t2s_degenerate (N=100) vs prose-100
(N=100):

- **ECS (External Context Score):** fraction of attention mass from
  generated tokens to prompt region, per head per layer.
- **PKS (Parametric Knowledge Score):** Jensen-Shannon Divergence
  between pre-FFN and post-FFN LogitLens vocabulary projections,
  per layer.

Both instruments were measured via teacher-forced replay (DS-038) and
live KV-cache generation on detection-gap records (night-010).

#### Empirical findings and disqualifiers

    | Finding | Implication |
    |---|---|
    | Mixed / inverted polarity (late PKS: prose > degenerate) | ReDeEP RAG story does not transfer to t2s macro-loops |
    | Degenerate fire rates ≪ 1.0 | Not a replacement for spectral PR |
    | Prose FP severe (PKS late layers ~0.87-0.93) | Unusable as binary gates |
    | Live ECS systematically lower than teacher-forced; high rank correlation on some heads (L21H3 r=0.98) | Signal is real and boundary-sensitive; still not gate-quality |

**No ECS/PKS thresholds enter diagnose().** Both instruments are
maintained strictly as offline diagnostic tools for inspecting attention
reallocation and FFN update dynamics under experimental conditions.
Live ECS is preserved as a descriptive instrument — not a gate.

### 3. Unified Macro-Loop Dynamics Model (measurement-only)

Working model, derived from night-003, DS-038, and night-010:

    1. Mid-Layer Attention Detachment (Layer 6)
       Live L6H6 ECS collapses to ~0.17 (Δ = -0.30 vs teacher-forced).
       Mid-layer attention heads detach from the prompt context and lock
       into the active generated token loop.

    2. Late-Layer Context Preservation (Layer 21)
       Live L21H3 ECS retains ~0.72. Context tracking persists late in
       the network but lacks sufficient magnitude to break mid-layer
       attractor momentum.

    3. Upper-Layer FFN Update Starvation (DS-038 PKS)
       Late-layer PKS drops on degenerate text (PKS_deg = 0.016 vs
       PKS_prose = 0.072 at Layer 27). FFNs stop writing updates to the
       residual stream, passing un-updated loop states forward.

    4. Logit-Orthogonal Hidden Collapse (night-003)
       Collapse direction c₁ is nearly orthogonal to residual→logit v₁
       (|cos| ≤ 0.017, below random baseline ~0.0204). Surface bigrams
       remain locally valid while internal trajectory collapses onto a
       low-rank attractor manifold.

Spectral PR detects the compression (Stage 2); surface bigrams need not
reflect it. Actuation remains logit-path until a residual class of
intervention shows logit movement and rescue. This is a descriptive
model, not an architectural commitment.

### 4. AARF Disposition

AARF and multiplicative residual intervention remain **PARKED** pending:

1. Residual→logit or module-output liveness under multiplicative edits
   (e.g., γ/β sweep at layers {6, 14} on t2s detection-gap records).
2. Rescue metrics on held-out degenerate corpora with FP controls on
   prose-100.

Neither static nor live ECS/PKS signals may be added as fast-path binary
gating predicates or AARF actuation drivers due to unacceptably high
prose false-positive rates. The gate doctrine's 0-FP standard applies [1].

### 5. Measured Evidence (cited)

- DS-026, DS-028, DS-029, DS-036: additive residual perturbation null
  results (bf16 precision-floor candidate mechanism)
- night-002: four-layer residual→logit singular-vector sweep
- night-003: J-Lens alignment diagnostic (c₁ ⊥ v₁)
- DS-038: ECS/PKS teacher-forced measurement on t2s vs prose-100
  (FROZEN_ECS_PKS_THRESHOLDS.md)
- night-010: live-generation ECS on t2s detection-gap records
