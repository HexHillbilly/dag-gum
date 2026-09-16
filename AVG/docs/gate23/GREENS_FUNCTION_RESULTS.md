# DS-036 — Green's Function SVD probe (residual channel liveness)

> MEASUREMENT REPORT. Single-record diagnostic (MEASUREMENT ONLY).
> No controller edits, no thresholds, no new architecture, no
> green/red gate. The liveness classification below is the task's
> explicit reading rule for the measured KL, offered for the
> human/human review decision (DS-037 queue vs residual deprecation).

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture record | 0 (heldout-degenerate v2) |
| Prompt tokens | 28 |
| Intervention layer | 26 |
| Intervention position | last prompt token (index 27) |
| Force budget | η = 0.5 L2 (production clamp [0.35, 0.75], controller.py:566) |
| Probe magnitude | ε = 0.001 |
| Probe directions | K = 128 (QR-orthonormal) |
| Probe dtype | fp32 (fallback from bf16 dead zone) |
| Power iterations | 10 |
| Vocab dim V | 151936 (lm_head out_features) |
| Hidden dim d | 1536 |
| Per-token KL horizon | first 10 generated tokens |
| bmm Triton override | deregistered |
| Wall clock (s) | 8.8 |

## Determinism smoke (baseline capture, twice)

| prompt tokens | h_t identical | baseline logits identical |
|---|---|---|
| 28 | True | True |

## Phase 1 — v₁ estimation (finite differences)

- K = 128 QR-orthonormal probe directions e_k ∈ R^1536.
- ε = 0.001 (finite-difference probe magnitude).
- Probe dtype: **fp32 (fallback from bf16 dead zone)**.
- Empirical matrix ΔY ∈ R^(128×151936) held in float32 CPU.
- Mean ||Δy_k|| = 4.0077e-03, max ||Δy_k|| = 5.3771e-03.
- **Note:** bf16 spec-literal probe max ||Δy_k|| = 1.733e+00, zero-shift fraction = 99.2%; fp32 fallback used for Jacobian estimation.
- Power iteration on the K×K Gram G = (1/ε²)ΔYΔYᵀ (= M Mᵀ for M = ΔY/ε).
- v₁ = Eᵀc₁/‖Eᵀc₁‖ ∈ R^1536 (Rayleigh–Ritz restriction of the empirical Jacobian's top right-singular direction to the probe subspace; K < d, so the projected eigenproblem is the well-posed version of M^T M power iteration).

### Power-iteration convergence trace

| iter | cos(v_i, v_{i-1}) | ||G·c|| |
|---|---|---|
| 1 | 0.644053 | 24.2517 |
| 2 | 0.827202 | 184.1133 |
| 3 | 0.998230 | 231.3907 |
| 4 | 0.999967 | 231.9093 |
| 5 | 0.999999 | 231.9202 |
| 6 | 1.000000 | 231.9206 |
| 7 | 1.000000 | 231.9206 |
| 8 | 1.000000 | 231.9206 |
| 9 | 1.000000 | 231.9206 |
| 10 | 1.000000 | 231.9206 |

Converged (final cosine ≥ 0.99): True

### Top-5 singular values of ΔY (normalized σ_i/σ_1)

| rank | σ_i(ΔY) | σ_i/σ_1 |
|---|---|---|
| 1 | 1.5229e-02 | 1.000000 |
| 2 | 7.1810e-03 | 0.471534 |
| 3 | 6.0830e-03 | 0.399440 |
| 4 | 5.7189e-03 | 0.375531 |
| 5 | 5.4229e-03 | 0.356091 |

### v₁ diagnostics

| quantity | value |
|---|---|
| cos(v₁, h_t) | 0.043131 |
| cos(v₁, random-orthogonal nullspace direction) | 0.015001 |
| random-nullspace expected |cos| for d=1536 (≈1/√d) | 0.025516 |
| ‖v₁‖ (L2) | 1.000000 |

cos(v₁, h_t) ≈ 0 confirms v₁ is essentially orthogonal to the
residual. cos(v₁, random-orthogonal) ≈ 1/√d ≈ 0.0255 confirms v₁
is *not* a specially-aligned hidden direction by construction — it
is the empirical Jacobian's dominant input direction, which is a
different object from the random nullspace directions that gave
KL ≈ 10⁻⁵ in DS-026/028/029.

## Phase 2 — bounded kick along v₁ (η = 0.50 L2)

- v₁ orthogonalised against h_t (production fallback double Gram-Schmidt); unit-norm direction.
- Applied h_t' = h_t + 0.5·v₁_unit at layer 26, last prompt token position.
- Measured applied L2 delta = 0.5173 (target 0.5); cos(h_t, h_t') = 0.999998.
- Kicked greedy continuation (10 tokens): ` #### 2 An #### 2 An #### `
- Dormant greedy continuation (10 tokens): ` #### 2 An #### 2 An #### `

### Per-token KL (kicked || dormant), first 10 generated tokens

| generated token | KL |
|---|---|
| 1 | 3.5991e-05 |
| 2 | 4.4803e-07 |
| 3 | 2.6435e-05 |
| 4 | 2.7643e-06 |
| 5 | 2.6397e-06 |
| 6 | 1.0036e-06 |
| 7 | 2.7824e-07 |
| 8 | 3.2624e-07 |
| 9 | 2.3704e-05 |
| 10 | 3.9327e-07 |

- Mean over 10 tokens: **9.3983e-06**
- KL at token 1 (first predicted token): **3.5991e-05**

## Liveness classification (task reading rule)

- Measured mean KL = **9.3983e-06**.
- Reading rule: KL > 10⁻² → ALIVE (DS-037 unblocked); KL ≈ 10⁻⁵ → DEAD (deprecate residual, cancel DS-037).
- **Classification: DEAD (deprecate residual; CANCEL DS-037).**

This is a measurement report; the classification is the task's
explicit threshold applied to the measured value. No controller
change, no threshold change, and no DS-037 work is performed here.

## JSONL

Machine-readable results are in `greens_function_results.jsonl`.

## Notes / caveats

- Single-record diagnostic (record 0 of heldout_degenerate_v2).
- v₁ is the dominant right-singular direction of the empirical
  Jacobian restricted to the K=128 probe subspace (K < d). It is
  not guaranteed to be the global top singular direction of the
  full 151,936×1536 Jacobian.
- **bf16 dead-zone finding:** the task's ε=10⁻³ is *not* large enough
  to be representable in bf16 at layer 26. The residual norm there
  is O(250) and bf16's absolute step is O(0.6); an ε=10⁻³ hidden-
  state perturbation is rounded to ~0 for 99.2% of the probe
  directions (127/128 zero shifts). The one nonzero shift is a
  bf16-quantization artifact (max ||Δy_k|| = 1.73), not a Jacobian
  response. The finite-difference Jacobian was therefore estimated
  in fp32 (same bf16-loaded weights, higher numerics) so that ε=10⁻³
  is resolvable for every probe direction. Phase 2 (the 0.50 L2
  kick and KL) runs in the production bf16 model.
- bf16 spec-literal probe max ||Δy_k|| = 1.733e+00, zero-shift probe fraction = 0.9921875 (dead-zone floors: max-shift 1e-06, zero-frac 0.5).
- The Gram matrix and power iteration run in float32 on CPU.
- KL uses the DS-026/028/029 formula (softmax distributions, 
  eps=1e-12, clamp at 0) for direct comparability.
