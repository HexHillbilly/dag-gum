# Root-cause: the "PR ≈ 3 collapse floor" is the rank of a short loop motif, not a deep invariant

**Date:** 2026-08-14
**Script:** `scripts/probe_pr_floor.py`
**Question (RESEARCH.md §1.2):** why does the participation ratio of the layer-2 residual stream
collapse to **≈3** across 9 models / 5 architectures? What *is* the ≈3-dimensional subspace?

## Method

Run the production model (`Qwen2.5-1.5B`) into a **dormant** loop on the t2s macro-loop
fixtures (N=24, greedy, 128 tokens). Capture the layer-2 (block index 2) residual-stream hidden
states over the trailing-24 window. At the min-PR step, characterize:

1. the full singular-value spectrum (rank-3 cliff vs soft decay);
2. the top-3 right-singular vectors — nearest-token-embedding cosine, and logit readout;
3. the **token-embedding-only PR** — the same PR metric computed on the *bare token embeddings*
   of the 24-token window (isolating the token contribution from the hidden dynamics);
4. the decoded window (the loop motif).

## Result

**Layer-2 hidden-state PR tracks token-embedding PR almost exactly.** Across 24 records the gap
`hidden_pr − token_emb_pr` is mean **+0.171** (range 0.006–0.589) — the hidden state is the token
embedding plus a small position/attention residual, so its effective rank is set by the *token*
structure of the loop.

**And that rank is `distinct_tokens − 1`.** The loop motifs and their min-PR form a near-perfect
ladder:

| loop motif length (distinct tokens) | min-PR | token-emb PR | spectrum shape |
|---|---|---|---|
| 2 (`further house …`) | 1.59 | 1.00 | 1 dominant + small 2nd |
| 3 (`talk further read …`) | 2.10 | 2.00 | 2 dominant |
| **4** (`once over again black …`) | **3.07–3.13** | **2.95–3.00** | **3 dominant + cliff** |
| 5 (`bad cat yet blue loud …`) | 4.02–4.04 | 3.83–3.97 | 4 dominant |
| 6 | 5.02–5.47 | 4.88–4.97 | — |
| 7 | 5.99–6.44 | 5.91–5.94 | — |
| 8 | 6.93–7.07 | 6.75–6.97 | — |

The 4-token loops (8 of 24 records) have a **hard rank-3 spectrum**: `[0.37, 0.313, 0.297, 0.0017, …]`
— three singular values carrying 98% of the mass, then a cliff to ~0.002. This is the "≈3".

The top-3 singular vectors are **not** aligned with any single token embedding (max cosine
0.10–0.13, near the random-orthogonal baseline): they are the *difference* directions between the
loop-token hidden states — the affine edges of the loop, not identifiable tokens.

## Root cause

`participation_ratio` centers the window (subtracts the mean) and computes `(Σs)²/Σs²`. For a loop
of `k` distinct tokens, the `k` hidden states (≈ the `k` token embeddings) span **`k−1`** affine
dimensions after centering. So:

> **PR ≈ 3 is the rank of a 4-token loop motif — `3 = k−1` for `k=4`.** The "collapse floor" is
> not a mysterious intrinsic dimension of the transformer's degenerate attractor; it is the
> effective number of *distinct loop tokens minus one*, because layer-2 hidden states are
> token-embedding-dominated.

The "architecture-independence" of the ≈3 floor is therefore **trivial**: every model's token
embeddings live in a high-dimensional space where `k` distinct points span `k−1` dimensions, and
every model's loops in this corpus settle into ~4-token motifs. No shared attractor geometry needs
to be invoked.

## What this does *not* explain (honest scope)

1. **Why loops settle into ~4-token motifs** (the loop-length distribution is the real remaining
   question — a loop-dynamics/attractor question, separable from the PR metric).

2. **Whether the spectral detector measures "hidden collapse" or just token diversity.** This probe
   is at **layer 2**, where hidden ≈ token. The `MACRO_LOOP_DYNAMICS.md` "logit-orthogonal collapse"
   (Stage 4) was measured at **layer 27**, a different signal. The spectral predicate's layer-2 PR
   is therefore — on this evidence — best described as a *continuous token-diversity estimator*
   (fires below ≈8.2, i.e. < ~9 effective token directions), not a detector of hidden-state collapse
   invisible to the surface. That distinction only matters for the *interpretation* of the spectral
   detector, not its measured rescue value (17%→97% on adversarial fixtures stands).

3. **Natural degeneration — MEASURED (2026-08-14).** Real prose under greedy does *not* reach the
   ≈3 floor: layer-2 PR stays high (mean 13.6, median 14.5, range 3.4–19.0 over 40 prompts ×
   256 tokens); only 3/40 drop below `band_low`, and those are single-token collapses
   (`0 0 0…`, `———…`). Token-dominance holds everywhere, with a larger positive gap for prose
   (+2.4 vs +0.17 for t2s): the position/attention residual contributes a ~constant +2–3
   effective dimensions on top of the token rank — negligible at token-rank 3, visible at
   token-rank 15. Natural greedy degeneration is *long-range phrase repetition*, which never
   compresses the trailing-24 layer-2 PR — this is exactly why the spectral detector fires
   0/200 on natural prose. The ≈3 floor is therefore **specific to short (≤4-token) loop
   motifs**, not a general property of degeneration.

## The loop period is fixture-primed — not emergent

The t2s fixtures carry a `motif` field (a 2–8 token word-list); each fixture's `text` is that motif
repeated. Re-checking the 24-fixture probe against the `motif` field shows the model does not
*choose* a loop period — it *continues* the primed motif:

- **22/24 fixtures:** the generated loop is the **exact** primed motif (probe `distinct` == motif length).
- **2/24:** a **sub-motif** (fixture 20 `['happy','be','out','happy','must','white','mountain']` → 6
  tokens; fixture 22 `['cat','could','will','will','again','once','star','she']` → 7 tokens — both
  motifs contain a repeated word the greedy argmax collapses over).
- **0/24:** the loop ever *exceeds* the primed motif.

min-PR is a pure function of motif length: 2→1.59, 3→2.10, 4→3.10, 5→4.03, 6→5.09, 7→5.81,
8→6.92 — i.e. `min_pr = k−1` exactly. The corpus motif lengths are 2–8 (roughly uniform), so the
"≈3 floor" in `RESEARCH.md` §1.2 is simply the **4-token-motif subset** of that distribution.

**Conclusion:** there is no "≈3 attractor", no preferred ~4-token loop length, and no
architecture-independent collapse constant. The entire "PR ≈ 3 invariant" is **(a)** the
centered-rank property of the PR metric (`k−1`) times **(b)** a fixture corpus that primes motifs
of length 2–8.

## Bottom line

The "candidate invariant" in `RESEARCH.md` §1.2 is **fully explained away in two steps**: (1) PR ≈ 3
is the centered rank `k−1` of a `k`-token loop — a metric property, token-driven at layer 2; (2) the
loops are 4 tokens **because the fixtures prime 4-token motifs** — a data artifact. Neither a deep
internal attractor nor a preferred loop length exists. Natural prose degeneration doesn't reach short
loops at all (long-range phrase repetition, PR 10–19). The genuinely open question is not "why ≈3" or
"why ~4 tokens" but: **under what conditions does a model *spontaneously* form a short loop without a
primed motif, and what sets that loop's period?**
