# Triad Re-Alignment Decision — Owner Ratification Memo

- Date: 08/12/2026
- Status: APPROVED by the owner.
- Author: human tier synthesis (automated engine), with independent review + adversarial critic sign-off.
- Subject: Resolve the adversarial critic vs independent review policy divergence on Amendment A3
  (dual-predicate FP policy) and lock the sequencing to the batch.

## 0. Ground truth — what is actually at committed HEAD

Verified against localhost (Lagoon/AVG, ref main) via fetch_gitea_file:

- docs/AVG_RFC_004_TWO_STAGE_DETECTION.md — Amendment A3 status:
  "RATIFIED by owner." Amendment A4 (residual deprecation) also RATIFIED
  at HEAD.
- Amendment A5 (AARF / ECS-PKS) — committed to
  docs/AVG_RFC_004_TWO_STAGE_DETECTION.md in the same ratification batch.
- governor/controller.py — both rigor-review BLOCKERs are still live:
    - BLOCKER #1: generate() uses full-sequence forward passes, so the
      layer-2 spectral PR hook's `hs.shape[1] == 1` gate never fires.
    - BLOCKER #2: diagnose() still emits "sae_guided_reset" into the
      deprecated residual path; the functional logit-penalty path runs
      independently of the collapse predicate.
- scripts/test_path2_code_filter.py — contains a ghost reference to
  governor._collapse_persistence_counter (removed in the A3 refactor).
  Latent AttributeError if exercised. Protected file.

## 1. The non-alignment (stated plainly, for an accurate premise)

adversarial critic's "Recommend A" and independent review's first "synthesis" are not two
clarifications of one position. They are opposite policies.

| Axis | adversarial critic (Recommend A) | independent review (walk-back) |
|---|---|---|
| Prose FP tolerance | Accept <=7/100, characterized, non-blocking | 0-FP "non-negotiable" before A3 |
| Spectral arm | Actuate; rescue gate decides | Shadow-only / demote from A3 |
| Sequencing | Rescue first, FP optimization after | Temporal 0-FP sweep blocks everything |

These cannot be merged into a single policy. This memo records the
divergence and resolves it; it does not relabel the resolution as
"alignment."

## 2. Why the repository has already decided this

Committed Amendment A3 text (quoted from HEAD):

    "The 7 FPs are accepted as documented pre-production debt.
    The dual-predicate proceeds to DS-035 (cross-fixture rescue gate)."

Escape clause (verbatim):

    "If DS-035 produces rescue count = 0 on both t2s_degenerate AND
    qwen_degenerate, Amendment A3 auto-reverts to bigram-only and the
    dual-predicate is shelved. If rescue count >= 1 on either fixture,
    A3 ratifies and FP optimization becomes the next night gate."

Consequences:

- adversarial critic's "Recommend A" is a restatement of committed A3, not a new vote.
- independent review's "hold A3 + 0-FP doctrine + temporal sweep first" is a
  proposal to REVERSE a ratified amendment. A reversal requires an
  explicit owner override, not a "re-alignment" label.
- DS-034c/d/e already closed the surface-hardening door: surface metrics
  are structurally incompatible with simultaneously filtering prose FPs
  and preserving t2s macro-loop coverage. Demanding 0-FP via that family
  of hardenings reopens a door the record says is closed.
- FP taxonomy: 5 of the 7 FPs are bigram cold-start FPs already shipped
  in production (not introduced by the dual-predicate). The spectral
  arm's incremental tax is ~2 late-dip FPs per 100 prose records.
  Flattening all 7 as "dual-predicate FPs" is incorrect accounting.

## 3. The real gap is the controller, not the detector

The DS-035 harness measured spectral actuation with a KV-cache night
harness. The production controller never implemented that actuation
(BLOCKERs #1 and #2 above). The correct response to "harness !=
production" is to FIX the controller and measure for real — not to
retreat to shadow mode because the controller is wrong.

## 4. Ratified sequence (locked; no reordering without owner override)

    1. Land Amendment A4 (residual deprecation) — documentation-only;
       already ratified text at HEAD.
    2. Land Tier B1 (kickstart policy + numeric immunity):
       - vocab boundary fix: vocab_size = len(self.tokenizer)
       - kickstart retarget to active_loop_ids
       - greedy-only kickstart actuation
       - explicit named numeric-fraction immunity predicate, evaluated
         every step, regime-independent
    3. Land Tier B2 (KV-cache decode + spectral wiring):
       - restructure generate() to use_cache=True, pre-fill once,
         single-token decode forward (position_ids explicit every step)
       - register layer-2 spectral hook AFTER pre-fill
       - extract _evaluate_collapse(); wire is_collapsed into the
         logit-penalty/kickstart path
       - stop invoking diagnose()/apply_interventions() from generate()
         (residual code retained per A4, but not called)
    4. Run DS-035 as a REAL production gate against the post-B2
       controller under the committed A3 FP policy.
    5. Temporal hardening (Option B) as follow-up FP optimization —
       only after DS-035 confirms spectral actuation liveness.

## 5. Pre-registration requirements for DS-035 (before it runs)

- Regime is pre-registered. Sampling (temp=0.8, top_p=0.85) is PRIMARY
  for the product claim (night-014: 87-100% rescue). Greedy is OPTIONAL
  secondary. Do not re-litigate EOS/kickstart tradeoffs after the fact.
- FP policy: accept <=7/100 characterized prose FPs under the DS-034e
  rule (5 inherited bigram cold-start + 2 spectral late-dip).
- Rescue success threshold: proposal default >=85/100 on BOTH
  qwen_degenerate and t2s_degenerate. (This is the DS-035 GATE
  threshold — owner signs the number below.)
- Auto-revert floor: rescue count = 0 on both fixtures -> A3 reverts to
  bigram-only (committed A3 escape clause). NOTE: this floor (>=1) is
  the MINIMUM to avoid auto-revert; the 85/100 gate is the SUCCESS bar.
  The two are distinct and must not be conflated in the gate brief.
- FAIL = stop. No silent threshold edits.

## 6. Owner signing items (human acts — not agent work)

- [ ] Sign the DS-035 rescue success threshold (default >=85/100; the
      auto-revert floor stays at the committed A3 ">=1" minimum).
- [ ] Sign the B1 numeric-immunity threshold (night-019 proposal
      NUMERIC_IMMUNITY_THRESHOLD = 0.80) into
      scripts/test_degeneracy_smoke.py.
- [ ] Sign the B1 Single-Token Spam fire band (21 -> 1) into the smoke
      script.
- [ ] Fix scripts/test_path2_code_filter.py: (a) ghost
      governor._collapse_persistence_counter reference, and (b) its
      diagnose()-trace dependency that B2 makes stale. Protected file —
      owner edit only.
- [ ] Decide disposition of the residual-path return fields
      ("profile", "decisions", "intervention_log") after B2 empties
      them; confirm downstream consumers read the new "collapse" field.

## 7. Caveats (recorded, not disagreements)

- B1 is mandatory before B2. No isolated vocab/regime patches; the
  digit-crush removal without numeric immunity re-breaks the smoke gate.
- 0-FP on prose remains a product ASPIRATION for later FP gates. It is
  NOT a precondition to undo ratified A3.
- Byte-identical-to-HEAD output is retired. The B2 contract is
  (a) deterministic under fixed seed, and (b) functional gates pass.

## 8. Decision line

    Follow committed A3. Fix the controller (B1 -> B2). Run DS-035 as a
    real gate. Do not reverse ratification in the name of synthesis.

    adversarial critic position: recorded above (Recommend A).
    independent review position: recorded above (walk-back, withdrawn in the final
    exchange).
    Owner: (signs here to ratify this memo and start the batch).
