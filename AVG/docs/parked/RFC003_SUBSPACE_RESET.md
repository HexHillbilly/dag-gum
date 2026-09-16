# RFC-003 Subspace Reset Engine — PARKED

**Original:** RFC-003 §3.4 (Subspace Orthogonal Projection Engine) and
§5 Phase 3 (Macro-Trajectory Ring Buffer & Subspace Projection Engine)

**Status:** PARKED indefinitely per RFC-004 Amendment A4. The residual
perturbation path — bounded orthogonal kicks, Gram-Schmidt prompt-based
kick vectors, and subspace projection — is formally deprecated for
Qwen2.5-1.5B under greedy decoding. Six measurement probes (DS-026,
DS-028, DS-029, DS-030, DS-034b, DS-036) plus independent corroboration
(SKOP) established that bounded additive residual perturbations produce
KL ≈ 10⁻⁵ — the unembedding never sees the perturbation. The bf16
dead-zone at layer 26 provides the mechanistic explanation.

**Reactivation checklist (Amendment A4):**
If the residual path is to be considered live on a new model:

1. Profile residual norms at candidate intervention layers. If norms
   are significantly lower than O(250) at the target depth, the bf16
   precision floor may not block bounded perturbations.
2. Measure bf16 precision step at the intervention layer. If the
   per-component perturbation at the production force band (0.35-0.75
   L2) exceeds the precision floor, the dead-zone does not apply.
3. Repeat the Green's function SVD probe (DS-036 / night-002) at the
   candidate layers to establish whether the residual→logit channel
   has non-zero singular structure.
4. Apply a bounded kick along v₁ and measure KL. If KL > 10⁻², the
   channel is alive on the new model and residual intervention can be
   re-evaluated.
5. Run component ablation (DS-029/DS-030) to verify that residual adds
   marginal rescue over logit-penalty-only actuation.
6. Run cross-fixture validation (DS-035) with the reactivated residual
   path to confirm no regression on known degenerate fixtures.
7. Human sign-off per the gate doctrine [1].

**Original RFC-003 text is preserved in**
`docs/AVG_RFC_003_DUAL_RESOLUTION_DESIGN.md` for historical reference.
