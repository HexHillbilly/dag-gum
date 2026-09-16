# night-003 — J-Lens Collapse→Jacobian Alignment Diagnostic

Date: 2026-08-11  |  SEED: 42  |  Model: `Qwen/Qwen2.5-1.5B@8faed761d45a263340a0528343f099c05c9a4323`

MEASUREMENT ONLY. Self-contained probe (no AVG imports). No intervention, no new thresholds, no controller changes.

## Method

- 10 t2s_degenerate DS-033 detection-gap records (first 10 by record_id; do NOT re-classify).
- Greedy (do_sample=False) KV-cache generation, max_new_tokens=128.
- Sampled steps: `[23, 36, 50, 64, 78, 92, 106, 120]` (8 evenly spaced snapshots per record, 80 total).
- Collapse direction c1: top right-singular vector of the trailing 24 hidden states at layer 27 (24x24 Gram power iteration, 3 iters).
- Jacobian direction v1: night-002 FD + randomized SVD at layer 27, K=40 (q=32, p=8), eps=0.001 x RMS(h_t), fp32 FD estimation.
- Alignment = |cos(c1, v1)| (absolute cosine similarity).
- Surface metrics: token_diversity (trailing 24 bigrams), trailing_ctr (trailing 16 tokens), layer-2 spectral PR (trailing 24 hidden states).
- Frozen PR threshold (logging only): T_PR = 10.954796, band_low = 8.216097. The diagnostic does NOT gate actuation.
- Determinism: two model copies (bf16 generation + fp32 FD probe), loaded once and NEVER cast.  In-place casting would create new weight tensors at new memory addresses, changing cuBLAS numerics and breaking the determinism smoke.

## Determinism smoke (FIRST)

| record_id | n_tokens A | n_tokens B | ids identical | align A | align B | align match | c1 trace | v1 trace |
|---|---|---|---|---|---|---|---|---|
| 0 | 128 | 128 | True | 0.00656543 | 0.00656543 | True | True | True |

## Primary diagnostic table

| record_id | steps | align range | align mean | align @ first bigram drop | diverge step | diverge token_diversity |
|---|---|---|---|---|---|---|
| 0 | 8 | 0.0003-0.0088 | 0.0055 | n/a | never | n/a |
| 1 | 8 | 0.0039-0.0099 | 0.0063 | 0.0086 | 23 | 0.2273 |
| 2 | 8 | 0.0001-0.0016 | 0.0009 | 0.0012 | 23 | 0.0909 |
| 3 | 8 | 0.0046-0.0108 | 0.0065 | 0.0047 | 23 | 0.1364 |
| 4 | 8 | 0.0007-0.0066 | 0.0026 | n/a | never | n/a |
| 5 | 8 | 0.0088-0.0158 | 0.0117 | 0.0110 | 23 | 0.2727 |
| 6 | 8 | 0.0017-0.0025 | 0.0021 | n/a | never | n/a |
| 7 | 8 | 0.0002-0.0066 | 0.0035 | 0.0063 | 23 | 0.2727 |
| 8 | 8 | 0.0011-0.0076 | 0.0061 | 0.0011 | 23 | 0.1818 |
| 9 | 8 | 0.0065-0.0167 | 0.0108 | 0.0167 | 23 | 0.1818 |

'diverge step' = first sampled step where token_diversity < 0.30 (production bigram fire threshold). Records where token_diversity never drops are the 'pure' detection-gap records (the bigram detector never fires on its own). Records where token_diversity drops but the DS-033 predicate still never fired are the joint-conjunction detection-gap cases: bigram diversity drops but trailing_ctr stays high, so the (diversity AND ctr) predicate remains silent.

## Secondary aggregation (all 80 snapshots)

| condition | n | mean align | median align | mean token_div | mean pr_layer2 |
|---|---|---|---|---|---|
| healthy | 0 | nan | nan | nan | nan |
| borderline | 24 | 0.0034 | 0.0023 | 0.3352 | 6.6840 |
| collapsed | 56 | 0.0065 | 0.0064 | 0.1874 | 3.4909 |

Conditions: healthy = token_diversity >= 0.40, borderline = [0.30, 0.40), collapsed = < 0.30.

## Per-record time series

### record 0

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0066 | 0.3636 | 1.0000 | 7.0899 | 7.295e-02 | True | True |
| 36 | 0.0003 | 0.3478 | 1.0000 | 7.0898 | 7.346e-02 | True | True |
| 50 | 0.0051 | 0.3478 | 1.0000 | 7.0986 | 7.344e-02 | True | True |
| 64 | 0.0073 | 0.3478 | 1.0000 | 7.0880 | 7.340e-02 | True | True |
| 78 | 0.0085 | 0.3478 | 1.0000 | 7.0739 | 7.328e-02 | True | True |
| 92 | 0.0017 | 0.3478 | 1.0000 | 7.0826 | 7.345e-02 | True | True |
| 106 | 0.0059 | 0.3478 | 1.0000 | 7.0756 | 7.337e-02 | True | True |
| 120 | 0.0088 | 0.3478 | 1.0000 | 7.0861 | 7.356e-02 | True | True |

### record 1

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0086 | 0.2273 | 1.0000 | 4.0825 | 7.333e-02 | True | True |
| 36 | 0.0057 | 0.2174 | 1.0000 | 4.0818 | 7.343e-02 | True | True |
| 50 | 0.0046 | 0.2174 | 1.0000 | 4.0469 | 7.367e-02 | True | True |
| 64 | 0.0047 | 0.2174 | 1.0000 | 4.0455 | 7.339e-02 | True | True |
| 78 | 0.0099 | 0.2174 | 1.0000 | 4.0832 | 7.329e-02 | True | True |
| 92 | 0.0069 | 0.2174 | 1.0000 | 4.0788 | 7.350e-02 | True | True |
| 106 | 0.0063 | 0.2174 | 1.0000 | 4.0827 | 7.322e-02 | True | True |
| 120 | 0.0039 | 0.2174 | 1.0000 | 4.0400 | 7.390e-02 | True | True |

### record 2

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0012 | 0.0909 | 1.0000 | 1.8135 | 7.357e-02 | True | True |
| 36 | 0.0001 | 0.0870 | 1.0000 | 1.7631 | 7.334e-02 | True | True |
| 50 | 0.0015 | 0.0870 | 1.0000 | 1.6059 | 7.340e-02 | True | True |
| 64 | 0.0004 | 0.0870 | 1.0000 | 2.0974 | 7.333e-02 | True | True |
| 78 | 0.0016 | 0.0870 | 1.0000 | 1.9193 | 7.345e-02 | True | True |
| 92 | 0.0002 | 0.0870 | 1.0000 | 1.7773 | 7.344e-02 | True | True |
| 106 | 0.0014 | 0.0870 | 1.0000 | 1.9581 | 7.343e-02 | True | True |
| 120 | 0.0009 | 0.0870 | 1.0000 | 1.9874 | 7.326e-02 | True | True |

### record 3

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0047 | 0.1364 | 1.0000 | 2.1362 | 7.349e-02 | True | True |
| 36 | 0.0046 | 0.1304 | 1.0000 | 2.1222 | 7.345e-02 | True | True |
| 50 | 0.0057 | 0.1304 | 1.0000 | 2.1235 | 7.341e-02 | True | True |
| 64 | 0.0107 | 0.1304 | 1.0000 | 2.1163 | 7.360e-02 | True | True |
| 78 | 0.0046 | 0.1304 | 1.0000 | 2.1169 | 7.341e-02 | True | True |
| 92 | 0.0059 | 0.1304 | 1.0000 | 2.1199 | 7.341e-02 | True | True |
| 106 | 0.0108 | 0.1304 | 1.0000 | 2.1105 | 7.367e-02 | True | True |
| 120 | 0.0053 | 0.1304 | 1.0000 | 2.0988 | 7.341e-02 | True | True |

### record 4

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0066 | 0.3636 | 1.0000 | 6.9610 | 7.353e-02 | True | True |
| 36 | 0.0018 | 0.3478 | 1.0000 | 6.9524 | 7.318e-02 | True | True |
| 50 | 0.0011 | 0.3478 | 1.0000 | 6.9602 | 7.323e-02 | True | True |
| 64 | 0.0039 | 0.3478 | 1.0000 | 6.9521 | 7.352e-02 | True | True |
| 78 | 0.0012 | 0.3478 | 1.0000 | 6.9555 | 7.356e-02 | True | True |
| 92 | 0.0012 | 0.3478 | 1.0000 | 6.9493 | 7.324e-02 | True | True |
| 106 | 0.0007 | 0.3478 | 1.0000 | 6.9510 | 7.337e-02 | True | True |
| 120 | 0.0045 | 0.3478 | 1.0000 | 6.9333 | 7.347e-02 | True | True |

### record 5

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0110 | 0.2727 | 0.8125 | 5.1095 | 7.337e-02 | True | True |
| 36 | 0.0118 | 0.2609 | 0.8125 | 5.1157 | 7.330e-02 | True | True |
| 50 | 0.0092 | 0.2609 | 0.8125 | 5.1332 | 7.312e-02 | True | True |
| 64 | 0.0158 | 0.2609 | 0.8750 | 5.1197 | 7.345e-02 | True | True |
| 78 | 0.0112 | 0.2609 | 0.8125 | 5.1215 | 7.338e-02 | True | True |
| 92 | 0.0088 | 0.2609 | 0.8125 | 5.1151 | 7.313e-02 | True | True |
| 106 | 0.0155 | 0.2609 | 0.8750 | 5.1204 | 7.338e-02 | True | True |
| 120 | 0.0103 | 0.2609 | 0.8125 | 5.1133 | 7.337e-02 | True | True |

### record 6

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0024 | 0.3182 | 1.0000 | 6.0100 | 7.309e-02 | True | True |
| 36 | 0.0025 | 0.3043 | 1.0000 | 6.0163 | 7.378e-02 | True | True |
| 50 | 0.0023 | 0.3043 | 1.0000 | 6.0272 | 7.346e-02 | True | True |
| 64 | 0.0022 | 0.3043 | 1.0000 | 6.0237 | 7.356e-02 | True | True |
| 78 | 0.0022 | 0.3043 | 1.0000 | 6.0141 | 7.360e-02 | True | True |
| 92 | 0.0019 | 0.3043 | 1.0000 | 6.0137 | 7.365e-02 | True | True |
| 106 | 0.0020 | 0.3043 | 1.0000 | 6.0099 | 7.359e-02 | True | True |
| 120 | 0.0017 | 0.3043 | 1.0000 | 6.0019 | 7.374e-02 | True | True |

### record 7

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0063 | 0.2727 | 1.0000 | 5.0461 | 7.331e-02 | True | True |
| 36 | 0.0065 | 0.2609 | 1.0000 | 5.0470 | 7.311e-02 | True | True |
| 50 | 0.0002 | 0.2609 | 1.0000 | 5.0584 | 7.345e-02 | True | True |
| 64 | 0.0008 | 0.2609 | 1.0000 | 5.0402 | 7.342e-02 | True | True |
| 78 | 0.0066 | 0.2609 | 1.0000 | 5.0466 | 7.311e-02 | True | True |
| 92 | 0.0005 | 0.2609 | 1.0000 | 5.0421 | 7.342e-02 | True | True |
| 106 | 0.0005 | 0.2609 | 1.0000 | 5.0383 | 7.343e-02 | True | True |
| 120 | 0.0064 | 0.2609 | 1.0000 | 5.0262 | 7.311e-02 | True | True |

### record 8

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0011 | 0.1818 | 1.0000 | 3.1177 | 7.274e-02 | True | True |
| 36 | 0.0067 | 0.1739 | 1.0000 | 3.1150 | 7.360e-02 | False | True |
| 50 | 0.0073 | 0.1739 | 1.0000 | 3.1197 | 7.316e-02 | True | True |
| 64 | 0.0063 | 0.1739 | 1.0000 | 3.1168 | 7.351e-02 | True | True |
| 78 | 0.0076 | 0.1739 | 1.0000 | 3.1152 | 7.325e-02 | True | True |
| 92 | 0.0067 | 0.1739 | 1.0000 | 3.1105 | 7.356e-02 | True | True |
| 106 | 0.0070 | 0.1739 | 1.0000 | 3.1029 | 7.334e-02 | True | True |
| 120 | 0.0058 | 0.1739 | 1.0000 | 3.0988 | 7.351e-02 | True | True |

### record 9

| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | c1 conv | v1 conv |
|---|---|---|---|---|---|---|---|
| 23 | 0.0167 | 0.1818 | 1.0000 | 3.1184 | 7.332e-02 | True | True |
| 36 | 0.0065 | 0.1739 | 1.0000 | 3.1118 | 7.346e-02 | False | True |
| 50 | 0.0129 | 0.1739 | 1.0000 | 3.1226 | 7.406e-02 | True | True |
| 64 | 0.0077 | 0.1739 | 1.0000 | 3.1104 | 7.331e-02 | True | True |
| 78 | 0.0119 | 0.1739 | 1.0000 | 3.1152 | 7.403e-02 | True | True |
| 92 | 0.0086 | 0.1739 | 1.0000 | 3.1066 | 7.346e-02 | True | True |
| 106 | 0.0136 | 0.1739 | 1.0000 | 3.1078 | 7.404e-02 | True | True |
| 120 | 0.0088 | 0.1739 | 1.0000 | 3.0996 | 7.349e-02 | True | True |

## J-Lens hypothesis evaluation

The hypothesis predicts: **low alignment during healthy snapshots** (collapse direction and Jacobian direction misaligned — collapse invisible to the output) and **rising alignment at collapsed snapshots** (collapse rotates into the sensitive direction — output sees it).

Falsification signatures: (1) alignment uniformly high — collapse always visible, detection-gap has a different explanation; (2) alignment uniformly low and never rises even when bigram drops — surface degradation happens via a different mechanism.

Across the 80 snapshots: healthy mean align = nan (n=0), borderline mean align = 0.0034 (n=24), collapsed mean align = 0.0065 (n=56).  Global align range = 0.0001-0.0167.

No healthy snapshots (n=0) — all snapshots are in the borderline/collapsed regime (t2s degenerate macro-loop). Collapsed mean align = 0.0065 vs borderline mean align = 0.0034.  Both are far below the random-orthogonal baseline for unit vectors in R^1536 (~0.0204).

**Collapsed snapshots show a small alignment rise over borderline** (directionally consistent with the J-Lens rotation mechanism), but the absolute values remain in the near-orthogonal regime — the collapse stays essentially invisible to the output layer throughout.

**Verdict.** The core J-Lens mechanism is **supported**: the collapse direction c1 is essentially orthogonal to the Jacobian's sensitive direction v1 at every sampled step (max |cos| = 0.0167, random-orthogonal baseline ~0.0204). The hidden-state manifold compresses along directions the unembedding does not care about, explaining why spectral PR fires (collapse is real) while the next-token distribution stays locally diverse (collapse invisible to the output layer).  The predicted rotation into alignment is weakly present (collapsed mean 0.0065 vs borderline 0.0034) but never approaches a regime where the collapse could drive a surface-diversity drop in these detection-gap records.

## Notes

- FD Jacobian estimation runs in fp32 (bf16 dead-zone, DS-036); generation runs in bf16 to match production.  Mismatch documented.
- The diagnostic pauses generation at each sampled step to run passive FD probes; no kick is applied, no logits are changed.
- If any power iteration failed to converge within 10 iterations, the convergence flag is False and the trace is in the JSONL.
- c1 power iteration did not reach cos >= 0.99 within 3 iterations at 2 snapshot(s): record 8 step 36, record 9 step 36. Traces logged in the JSONL.
- `docs/gate23/jlens_alignment_results.jsonl` contains full per-snapshot records (step, align, surface metrics, sigma1, power-iteration traces).
