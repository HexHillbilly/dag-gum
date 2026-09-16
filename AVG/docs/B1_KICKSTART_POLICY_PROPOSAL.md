# B1 — Kickstart Policy Rebaseline (Controller Patch Proposal)

- Task: B1-KICKSTART-POLICY
- Status: PROPOSAL — awaiting owner signature. Not yet ratified.
- Engine: code agent (human tier controller diff)
- Reference model: Qwen2.5-1.5B (28L, d_model 1536, bf16)
- Review: independent review (architecture), adversarial critic (attack pass), owner (ratify)

## Protected files — DO NOT TOUCH

    scripts/test_degeneracy_smoke.py
    scripts/test_path2_code_filter.py
    scripts/test_factual_safety.py
    scripts/benchmark_latency.py
    Makefile

## 0. Premise — why there is no "minimal" kickstart patch

night-017 (penalty sweep) and night-018 (suppression-only) closed the
actuation question: kickstart is a cliff, not a slope. No penalty
magnitude in {1e4, 5.0, 2.0, 1.0} escapes the rescue-vs-EOS tradeoff;
-1e4 and -5.0 are statistically identical. Under greedy, suppression
alone collapses t2s rescue to 0.13 — kickstart is the only t2s
macro-loop breaker, and it breaks via EOS bailout, not prose recovery.

Two "minimal" patches (retarget+vocab+regime, then regime-only) each
produced the identical red fingerprint:

    - Single-Token Spam fires 21 -> 1
    - Number Loop fires 0 -> 1
    - numeric fraction 0.053 -> 0.417

Root cause: the legacy _non_prose_token_ids set classifies digit tokens
as non-prose (digits have no vowels) and crushes them with -1e4. That
accidental digit-crush is load-bearing for Number Loop immunity under
sampling — the degeneracy smoke gate runs do_sample=True. Any retarget,
narrow, or disable of kickstart removes the digit-crush.

Therefore B1 replaces the accidental digit-crush with an explicit,
named numeric-immunity predicate, and fixes the kickstart policy in the
same patch.

## 1. Scope — B1 ONLY

Implement exactly four changes:

1. Vocab-range fix: use len(tokenizer) in _cache_vocabulary_subsets.
2. Kickstart retarget: penalize active_loop_ids, not _non_prose_token_ids.
3. Greedy-only kickstart: kickstart fires only under do_sample=False.
4. Explicit numeric-immunity rule: named predicate + threshold,
   evaluated every step in generate(), regime-independent.

DO NOT implement (B2, out of scope for this patch):

    - KV-cache restructure of generate() (rigor-review BLOCKER #1).
    - Residual detach / diagnose() actuation rewiring (BLOCKER #2).
    - Anything touching the residual perturbation path (Amendment A4).

## 2. Red lines

- Do NOT change the diagnose() signature. No new positional or keyword
  arguments. The prior B1 patch failed by threading is_numeric_context
  through diagnose(); that broke two monkey-patched unit tests with a
  TypeError. The numeric guard lives in generate(), NOT diagnose().

- Do NOT modify any protected file (listed above).

- Do NOT sign thresholds into the smoke script. NUMERIC_IMMUNITY_THRESHOLD
  and the new Single-Token Spam fire band are owner-signed night-019
  proposals. The code agent implements the predicate; the owner signs
  the numbers.

- The numeric predicate MUST be a separate named function. Do NOT fold
  it into is_code_syntax_context (independent review/adversarial critic clarification).

## 3. Changes

### 3.1 Vocab-range fix

File: governor/controller.py, _cache_vocabulary_subsets()

Current (fallback returns a bound method, not an int):

    vocab_size = getattr(self.tokenizer, "vocab_size", getattr(self.tokenizer, "len", 151646))

Replace with:

    try:
        vocab_size = len(self.tokenizer)
    except TypeError:
        vocab_size = getattr(self.tokenizer, "vocab_size", 151646)

### 3.2 Kickstart retarget + greedy-only

File: governor/controller.py, generate(), kickstart block

Current:

    if self._kickstart_counter > 0:
        if self._non_prose_token_ids:
            non_prose_tensor = torch.tensor(list(self._non_prose_token_ids), device=self.device)
            if self._kickstart_counter == 3:
                penalty = -1e4
            elif self._kickstart_counter == 2:
                penalty = -5.0
            else:
                penalty = -2.0
            next_logits[:, non_prose_tensor] += penalty
        self._kickstart_counter -= 1

Replace with (retarget to active_loop_ids; greedy-only):

    if not do_sample and self._kickstart_counter > 0 and active_loop_ids:
        loop_tensor = torch.tensor(list(active_loop_ids), device=self.device)
        if self._kickstart_counter == 3:
            penalty = -1e4
        elif self._kickstart_counter == 2:
            penalty = -5.0
        else:
            penalty = -2.0
        next_logits[:, loop_tensor] += penalty
        self._kickstart_counter -= 1
    elif self._kickstart_counter > 0:
        self._kickstart_counter -= 1

Notes:

    - active_loop_ids is the repeated-token list returned by
      compute_token_distinct_2_fast earlier in the same step. Reuse it;
      do not recompute.
    - The magnitude schedule (-1e4 / -5.0 / -2.0) is unchanged.
    - _non_prose_token_ids is retained for diagnostics only (per its
      docstring) and is no longer an actuation target.

### 3.3 Numeric-immunity rule

File: core/metrics.py — new named predicate:

    def numeric_token_fraction(text: str) -> float:
        # Fraction of whitespace-separated tokens containing >= 1 digit.
        # Mirrors the smoke script's local _numeric_token_fraction.
        tokens = text.strip().split()
        if not tokens:
            return 0.0
        numeric = sum(1 for tok in tokens if any(ch.isdigit() for ch in tok))
        return numeric / float(len(tokens))

File: governor/controller.py — module-level constant (PROPOSAL value,
owner signs):

    NUMERIC_IMMUNITY_THRESHOLD = 0.80

File: governor/controller.py, generate(), inside the step loop —
compute numeric context every step, regime-independent:

    is_numeric_context = False
    if self.tokenizer is not None:
        recent_gen = (
            input_ids[0, prompt_len:]
            if input_ids.shape[-1] > prompt_len
            else input_ids[0]
        )
        recent_tokens = (
            recent_gen[-16:] if recent_gen.numel() > 0 else input_ids[0, -16:]
        )
        trailing_text = self.tokenizer.decode(
            recent_tokens, skip_special_tokens=True
        )
        is_numeric_context = (
            numeric_token_fraction(trailing_text) >= NUMERIC_IMMUNITY_THRESHOLD
        )

File: governor/controller.py, generate() — gate the diagnose() call:

    if is_profile_step:
        if is_numeric_context:
            # Number-loop immunity: governor dormant on numeric context.
            self._kickstart_counter = 0
            # Do NOT call diagnose() or apply_interventions() this step.
        elif intervene and (token_diversity < 0.35 or trailing_ctr < 0.35):
            ... existing profile + diagnose + apply_interventions ...

Constraints:

    - The numeric check runs EVERY step. It is NOT gated on the
      is_code_context / CTR subsampling cadence (every_k). A stale-step
      fire = red gate.
    - Regime-independent: the check runs under do_sample=True and
      do_sample=False alike. The accidental digit-crush it replaces fired
      under sampling (the smoke gate regime).
    - decode() of <=16 tokens is cheap. If latency budget is a concern,
      the code agent MAY substitute an integer-token-set fast path:
      precompute _numeric_token_ids at init (token IDs whose decode
      contains a digit) and test recent generated IDs against it,
      avoiding per-step decode. If you choose this, report the latency
      delta in the final report.

## 4. Verification

Run and quote verbatim:

    python -m pytest tests/ -q
        -> 128 pass. No signature change, so monkey-patched unit tests
           stay green.

    python scripts/test_degeneracy_smoke.py
        -> PASS expected:
           - Single-Token Spam fires: >= 1 (band collapse 21 -> 1 is the
             night-019 proposal; the current script only asserts >= 1
             total, so it still passes as-written).
           - Number Loop fires: 0 (numeric immunity holds).
           - numeric fraction: >= 0.04 (the script's vacuous-pass guard).

Do NOT run or claim scripts/test_path2_code_filter.py as green. It has a
separate latent ghost reference (governor._collapse_persistence_counter)
that is an owner fix in a protected file. Note it; do not edit it.

## 5. Definition of done

- All four changes implemented; zero B2 drift.
- diagnose() signature unchanged.
- No protected file modified.
- pytest tests/ -> 128 pass.
- Degeneracy smoke -> PASS per section 4.
- Report: summary; files changed with one-line rationale; gate outputs
  verbatim; any red gate + diagnosis; latency note if the integer fast
  path was used.

## 6. Owner signing items (NOT the code agent's job)

- Sign NUMERIC_IMMUNITY_THRESHOLD = 0.80 into the smoke script.
- Sign the Single-Token Spam fire band 21 -> 1 into the smoke script.
- Fix scripts/test_path2_code_filter.py ghost
  governor._collapse_persistence_counter reference (protected file).
- Sign docs/TIER_B_REBASELINE_SPEC.md (B1 section); proceed to B2
  (KV-cache + residual detach) only after B1 lands green.
