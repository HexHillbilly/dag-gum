# Actuation Investigation Summary — DS-026 through DS-032

**Status:** COMPLETE (DS-032 merged as FAIL, production kept at (8,-5))
**Period:** 2026-08-08 to 2026-08-10
**Model:** Qwen2.5-1.5B @ 8faed761, bf16, SEED=42
**Fixture:** heldout_degenerate_v2 (ds-027, N=100, 50-record seeded subset)
**Decoding:** Greedy (do_sample=False) unless noted

## Question

How does AVG's intervention mechanism actually rescue degenerate loops?
The controller has three simultaneous mechanisms: residual orthogonal
perturbation, token-suppression logit penalties, and kickstart vocabulary
steering. Which component(s) produce the measured rescue?

## Measurement arc

| Probe | Date | Question | Answer |
|---|---|---|---|
| DS-026 | 08-08 | Single residual kick — any effect? | No. ΔD2 = 0.0000 at all layers. |
| DS-028 | 08-08 | Persistent residual kicks (production ramp)? | No. 5,150 fires/layer, KL ≈ 10⁻⁵. |
| DS-029 | 08-09 | Residual vs logit-penalty vs joint — which rescues? | Logit-penalty alone rescues 50/50. Residual adds zero (joint ≡ logit-only). |
| DS-030 | 08-09 | Suppression vs kickstart — which logit mechanism rescues? | Suppression: 48/50. Kickstart: 15/50. Suppression is the primary actuator. |
| DS-031 | 08-09 | Optimal cooldown × penalty for suppression? | (5,-5): 48/50 rescue, EOS 0.20, prose D2 0.90. Best quality at equal rescue. |
| DS-032 | 08-10 | Validate (5,-5): combined path, sampling, cross-fixture? | **FAIL.** Combined EOS 0.50. Cross-fixture: qwen 68%, t2s 15%. |

## Key findings

### 1. The residual path is inert under greedy decoding
DS-026 (single kicks), DS-028 (persistent kicks, 5,150 fires at 0.75 L2),
and DS-029 (joint with logit penalties, token-identical to penalty-only)
collectively prove that bounded orthogonal residual perturbations produce
zero measurable effect on continuations under greedy decoding on this model
and fixture. KL divergence remains at 10⁻⁵ — the unembedding never sees
the perturbation.

### 2. The logit-penalty path is necessary and sufficient
Token suppression alone (cooldown=8, penalty=-5.0) rescues 48/50 records
(DS-030). Adding kickstart vocabulary steering rescues the remaining 2
(records 19/20) but doubles the EOS rate from 0.28 to 0.54 (DS-029).
Kickstart alone rescues only 15/50 and produces poor prose quality (DS-030).

### 3. Cooldown=5 improves quality but doesn't generalize
On the primary fixture, (5,-5) suppression-only matches (8,-5) rescue rate
(48/50) with lower EOS (0.20 vs 0.28) and higher prose Distinct-2 (0.90 vs
0.91) (DS-031). However, DS-032 proved this does not transfer to other
degenerate corpora: qwen_degenerate rescue drops to 68% and t2s_degenerate
to 15% under (5,-5). The production (8,-5) baseline on these fixtures is
unmeasured (DS-033 queued).

### 4. Sampling amplifies suppression effectiveness
Under do_sample=True (temp=0.8, top_p=0.85), suppression-only at (5,-5)
rescues 50/50 records with EOS 0.20, mean 110 tokens generated, and prose
Distinct-2 of 0.97 on continuations reaching the analysis window (DS-032
Part 2). Sampling provides natural entropy that prevents the EOS-death
trap seen in greedy combined mode.

## Current state

| Component | Status | Evidence |
|---|---|---|
| Residual perturbation | Inert under greedy | DS-026, DS-028, DS-029 |
| Residual under sampling | Unmeasured | — |
| Token suppression | Primary actuator | DS-030, DS-031 |
| Kickstart vocabulary steering | Supplementary (4% tail) | DS-030 |
| Production default | (8,-5), kept | DS-032 FAIL |
| Production cross-fixture | Unmeasured | DS-033 queued |
| RARI proposal | Parked | Awaiting residual-under-sampling + cross-fixture baseline |

## Open questions

1. Does production (8,-5) also fail on t2s_degenerate, or is this specific
   to shorter cooldown? (DS-033)
2. Does the residual path show any life under sampling (KL ≫ 10⁻⁵)?
3. Is t2s_degenerate failure a detection-window problem (macro-syntax motifs
   exceeding the 24-token bigram window) rather than a cooldown problem?
4. Should the controller use dual-regime parameters (cooldown=8 for greedy,
   cooldown=5 for sampling) given Part 2's strong result?

## References

- DS-026: docs/gate23/LAYER_EFFECT_RESULTS.md
- DS-028: docs/gate23/LAYER_EFFECT_PERSISTENT_RESULTS.md
- DS-029: docs/gate23/COMPONENT_ABLATION_RESULTS.md
- DS-030: docs/gate23/PENALTY_DECOMPOSITION_RESULTS.md
- DS-031: docs/gate23/COOLDOWN_SWEEP_RESULTS.md
- DS-032: docs/gate23/COOLDOWN_VALIDATION_RESULTS.md
- RFC-004: docs/AVG_RFC_004_TWO_STAGE_DETECTION.md
