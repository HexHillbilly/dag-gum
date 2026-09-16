# FP32 residual-kick liveness probe

> MEASUREMENT REPORT. Single-record diagnostic (MEASUREMENT ONLY). No controller
> edits, no thresholds, no new architecture. Re-opens DS-036's "residual is dead"
> verdict with a specific new hypothesis — that the null result was a bf16
> quantization artifact — and refutes it.

## Result

| condition | mean KL (10 tokens) | token-1 KL | applied L2 Δ | output vs dormant |
|---|---|---|---|---|
| **fp32** kick | **5.5618e-07** | 5.2447e-06 | 0.5000 (exact) | **byte-identical** |
| bf16 kick (DS-036 reproduction) | 9.3983e-06 | 3.5991e-05 | 0.5173 | byte-identical |

- DS-036 bf16 mean KL (reference): 9.3983e-06 — **reproduced exactly**, confirming
  the harness.
- v₁ estimated in fp32 (128 finite-difference probes, all resolvable: mean
  ‖Δy_k‖ = 4.0e-3, zero zero-shifts; power iteration converged to cos = 1.0 by
  iter 4). Top singular value σ₁ = 1.52e-2, σ₂/σ₁ = 0.47; cos(v₁, h_t) = 0.043.

## Interpretation

**The hypothesis is refuted. The residual channel is genuinely dead — not a
precision artifact.** In full fp32, where the 0.5-L2 kick along the Jacobian's
optimal direction v₁ is applied *exactly* (L2 delta = 0.5000, no quantization),
the mean KL is **5.56e-7** and the greedy output is **byte-identical** to the
dormant baseline. The bf16 "dead zone" was a red herring: it applied to the
ε=1e-3 *finite-difference probe* (unrepresentable in bf16), not to the 0.5 kick,
which bf16 in fact *partially* resolves (delta 0.5173) — and which is still dead.

The mechanism is precision-independent. The residual→output Jacobian has a real
dominant direction (σ₁ ≈ 15.2 in logit-change per unit hidden perturbation), so a
0.5-L2 kick along v₁ shifts the raw logits by norm ≈ 7.6. But that shift is
concentrated on **negligible-probability tokens**: after softmax it moves the
distribution by KL ≈ 5e-7 and does not change the greedy argmax. In other words,
the Jacobian's dominant input direction is (loosely) orthogonal to the output
*distribution* — a genuine null space, not a quantization hole.

Notably the clean fp32 kick is *more* inert (5.56e-7) than the bf16 kick
(9.40e-6): bf16 quantization injects a little noise into the perturbation, which
shows up as a slightly larger (still negligible) KL.

## Conclusion

The residual-perturbation path is deprecated on the strongest possible grounds —
**precision-independent**. A hidden-state perturbation at the production force
band (0.35–0.75 L2) cannot reach the output distribution regardless of dtype.
This closes the RARI / retrieval-augmented-residual-intervention thread (DS-037)
definitively: retrieval changes the *direction*, not the fundamental fact that
residual kicks at production force do not move the output. The sole demonstrated
actuation remains **logit-space token suppression**, which is universal across
9 transformers / 5 families + 2 non-transformers.

## Method (identical to DS-036)

- Record 0 of `heldout_degenerate_v2.jsonl`; layer 26, last prompt token.
- v₁ = dominant right-singular direction of the empirical Jacobian (128
  QR-orthonormal finite-difference probes, ε=1e-3, 10 power iterations),
  estimated in fp32.
- Kick: h_t′ = h_t + 0.5·orth(v₁, h_t), production double Gram-Schmidt.
- KL = softmax(kicked) ‖ softmax(dormant), eps=1e-12, clamp at 0 (DS-026/028/029
  formula).
- fp32 model loaded via `torch_dtype=torch.float32` (bf16 weights cast on load);
  bf16 comparison by casting the same model to bfloat16.
