# B2 — KV-Cache Restructure + Residual Detach (Controller Patch Proposal)

- Task: B2-DECODE-RESIDUAL-DETACH
- Status: PROPOSAL — awaiting owner approval, then code-agent handoff
- Engine: code agent (human tier controller diff)
- Reference model: Qwen2.5-1.5B (28L, d_model 1536, bf16)
- Review: independent review (architecture), adversarial critic (attack pass), owner (ratify)
- PREREQUISITE: B1 (kickstart policy) landed green. Do not start B2
  against a tree that does not already contain the B1 diff.

## Protected files — DO NOT TOUCH

    scripts/test_degeneracy_smoke.py
    scripts/test_path2_code_filter.py
    scripts/test_factual_safety.py
    scripts/benchmark_latency.py
    Makefile

## 0. What B2 closes (traceability)

Rigor-review BLOCKER #1: the spectral PR hook is dead in generate()
because generate() runs full-sequence forward passes, so the hook's
hs.shape[1] == 1 gate never fires. Fix: KV-cache restructure.

Rigor-review BLOCKER #2: diagnose() feeds the deprecated residual path
(apply_interventions -> _make_sae_guided_reset_fn), while the functional
logit-penalty path runs independently of the collapse predicate. Fix:
detach residual actuation and wire the collapse predicate to the
logit-penalty path.

Review Concern A (regime change): DS-026..DS-032 measured the
logit-penalty path under full-sequence forward passes. Switching to
KV-cache changes attention masking, positional handling, and float
accumulation order. Fix: re-measurement gate, not byte-identical-to-HEAD.

Review Concern B (t2s ceiling): DS-033 measured production (8,-5) at
t2s rescue 0.15. Wiring spectral detection to the same actuator is only
valuable if the wired path actually rescues macro-loops. Fix:
pre-registered rescue gate on the detection-gap subset.

Review Concern C (profiling dead weight): once diagnose() no longer
gates actuation, the HookManager profile step becomes pure overhead.
Fix: remove the profile step from the generate() hot loop; keep shadow
telemetry as the canonical instrumentation.

## 1. Scope — B2 ONLY

Implement exactly these changes:

1. Restructure generate() to KV-cache incremental decoding
   (use_cache=True, pre-fill once, single-token forward per step).
2. Register the layer-2 spectral PR hook AFTER pre-fill, so it only
   sees single-token decode steps (hs.shape[1] == 1 by construction).
3. Add _evaluate_collapse(): the A3 dual-predicate logic extracted to
   a new method returning is_collapsed (bigram_fire OR spectral_fire).
4. Wire is_collapsed into the logit-penalty path: spectral/bigram fire
   triggers kickstart under B1 policy (greedy-only, active_loop_ids).
5. Remove the HookManager profile step and apply_interventions() call
   from generate(). Residual path code stays in the file (Amendment A4)
   but is no longer invoked by generate().

DO NOT implement (out of scope):

    - Any B1 change (kickstart policy, vocab fix, numeric immunity).
    - Any threshold change (band_low = 8.216097 is frozen).
    - Any edit to diagnose()'s signature or body.
    - Retiring/editing test_path2_code_filter.py (protected; owner act).

## 2. Red lines

- Do NOT change the diagnose() signature. Do NOT edit its body. It must
  remain callable by the 128 pytest unit tests exactly as today. B2 adds
  a NEW method (_evaluate_collapse); it does not modify diagnose().

- Do NOT require byte-identical-to-HEAD output. That contract is retired.
  The new contract is: (a) deterministic across repeated runs under a
  fixed seed, and (b) passes the functional gates. KV-cache is expected
  to change continuations relative to full-sequence forward passes.

- Do NOT modify any protected file. This includes test_path2_code_filter.py,
  which has a pre-existing ghost reference (governor._collapse_persistence_counter)
  and a diagnose()-trace dependency. Both are owner fixes, not code-agent
  edits. Flag them; do not touch them.

- Do NOT change band_low (8.216097), T_PR, or any frozen threshold.

- Do NOT remove or alter the residual path code (_make_sae_guided_reset_fn,
  apply_interventions). Amendment A4 retains it for cross-model transfer.
  B2 only stops generate() from invoking it.

- position_ids MUST be passed explicitly on every forward. Qwen2.5 uses
  rotary embeddings; an incorrect position_ids under KV-cache silently
  corrupts generation. This is non-negotiable.

## 3. Changes

### 3.1 KV-cache restructure

File: governor/controller.py, generate()

Replace the per-step full-sequence forward:

    outputs = self.model(input_ids)          # OLD: grows every step

with the standard pre-fill + incremental decode pattern:

    # --- PRE-FILL (once, before the step loop) ---
    attention_mask = torch.ones_like(input_ids, device=self.device)
    position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :].clone()

    # --- per decode step ---
    # detection/actuation on input_ids bookkeeping (unchanged logic),
    # sample next_token, append to input_ids, then:
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), device=self.device)], dim=-1
    )
    position_ids = torch.tensor(
        [[prompt_len + step]], device=self.device
    )
    outputs = self.model(
        input_ids=next_token,               # single token, NOT the full seq
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :].clone()

Notes:

    - Keep input_ids growing by concatenation for BOOKKEEPING ONLY
      (diversity, CTR, decode, numeric context, active_loop_ids). The
      model forward consumes next_token + past_key_values, not input_ids.
      This isolates the regime change to the forward call and leaves the
      detection/actuation logic untouched.
    - batch == 1 is assumed (existing code already indexes input_ids[0, :]).
      If batch > 1 arrives, raise a clear error rather than mis-index.
    - next_logits MUST be .clone()'d before any in-place mutation
      (sample_top_p, penalty writes).
    - use_cache=True explicitly; do not rely on the model default.

### 3.2 Spectral hook registration timing

File: governor/controller.py, generate()

Move the layer-2 spectral PR hook registration to AFTER the pre-fill
forward and BEFORE the decode loop. Remove it in the finally block
(existing removal path stays).

Rationale: with KV-cache, decode steps always have hs.shape[1] == 1.
Registering the hook before pre-fill would also fire it during pre-fill
(shape[1] == prompt_len), which is excluded by the == 1 gate except for
the degenerate prompt_len == 1 case. Registering after pre-fill makes
the == 1 gate correct by construction and avoids prompt-token
contamination of the generated-token PR buffer.

Keep the hook body as-is:

    if hs.shape[1] == 1:
        self._layer2_pr_buffer.append(hs[:, -1:, :].detach().to(torch.float32))
        if len(self._layer2_pr_buffer) >= 24:
            win = torch.cat(self._layer2_pr_buffer[-24:], dim=1)
            self._layer2_pr = float(participation_ratio(win).item())

Under KV-cache this now fires on every decode step. PR is computed from
the 24th generated hidden state onward, matching DS-034b.

### 3.3 _evaluate_collapse() extraction

File: governor/controller.py

Add a new method. It mirrors EXACTLY the A3 predicate section currently
inside diagnose(), but returns a bool and does not touch residual
actuation:

    def _evaluate_collapse(
        self,
        token_diversity: float,
        trailing_ctr: float,
        is_code_context: bool,
    ) -> bool:
        # 1. dual-gate immunity & state reset
        if trailing_ctr >= 0.75 and token_diversity >= 0.35:
            self._consecutive_interventions = {}
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            return False

        # 2. code exclusion
        if self.use_code_filter and is_code_context:
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            self._consecutive_interventions = {}
            return False

        # 3. bigram predicate
        if (token_diversity < 0.30) and (trailing_ctr < 0.30):
            self._bigram_ctr += 1
        else:
            self._bigram_ctr = 0
        bigram_fire = self._bigram_ctr >= 2

        # 4. spectral predicate (band_low frozen at 8.216097)
        spectral_collapse = (
            self._layer2_pr is not None
            and self._layer2_pr < 8.216097
        )
        if spectral_collapse:
            self._spectral_ctr += 1
        else:
            self._spectral_ctr = 0
        if self._bigram_ctr < 2:
            spectral_fire = (
                self._spectral_ctr >= 2
                and token_diversity < 0.40
            )
        else:
            spectral_fire = self._spectral_ctr >= 2

        if is_code_context:
            spectral_fire = False

        return bigram_fire or spectral_fire

Notes:

    - Do NOT edit diagnose() to call _evaluate_collapse. diagnose() keeps
      its own body verbatim (deprecated path). Flag the resulting
      predicate duplication as known debt; reconcile post-B2 only.
    - _evaluate_collapse is called EVERY decode step (cheap; no SVD).
      It replaces the is_profile_step gate for collapse detection.

### 3.4 Wire is_collapsed into the logit-penalty path

File: governor/controller.py, generate()

Inside the decode loop, after computing token_diversity / trailing_ctr /
is_code_context (and the B1 numeric guard), call:

    is_collapsed = self._evaluate_collapse(
        token_diversity, trailing_ctr, is_code_context
    )

Kickstart trigger becomes (B1 policy: greedy-only, active_loop_ids):

    if (
        not do_sample
        and active_loop_ids
        and (trailing_ctr < 0.50 or is_collapsed)
    ):
        self._kickstart_counter = max(self._kickstart_counter, 3)

Notes:

    - This is the DS-035 wiring: spectral_fire (macro-loop detection,
      where bigram diversity is healthy) now reaches kickstart, with
      active_loop_ids (unigram repeated tokens) as the penalty target.
    - Suppression (_active_suppress_tokens) continues to be driven by
      active_loop_ids exactly as today; B2 does not change suppression.
    - Do NOT gate kickstart on spectral_fire alone when active_loop_ids
      is empty; the kickstart target requires a token set. Log the
      spectral-only fire (telemetry) and skip actuation in that case.

### 3.5 Remove profile step + residual actuation

File: governor/controller.py, generate()

Delete from the decode loop:

    - self.hook_manager.register_residual_hooks(...)  (profiling hooks)
    - self.profiler.profile(input_ids)
    - self.diagnose(...)
    - self.apply_interventions(step_decisions)
    - the hooks_registered / set_dormant machinery for profiling

Retain in the file (deprecated, not called by generate()):

    - diagnose(), apply_interventions(), _make_sae_guided_reset_fn(),
      profiler.profile(). Unit tests that call diagnose() directly stay
      green.

Return-schema consequence (explicit, do not paper over):

    - "profile": None (was self._last_profile)
    - "decisions": [] (was all_decisions)
    - "intervention_log": [] and "intervention_summary": {"n_interventions": 0}

    These residual-path fields become empty under B2. The canonical
    telemetry is now the shadow log (get_shadow_log()) plus a new
    "collapse" record added to the return dict:

        "collapse": {
            "bigram_fires": <count>,
            "spectral_fires": <count>,
            "steps": <list of (step, is_collapsed, bigram_ctr, spectral_ctr)>,
        }

    Any downstream consumer of the old fields must be updated by the
    owner; the code agent only populates the new field.

## 4. Verification — run and quote verbatim

4.1 Unit suite (no signature/body change to diagnose()):

    python -m pytest tests/ -q
        -> 128 pass.

4.2 KV determinism contract (replaces byte-identical-to-HEAD):

    Run generate() twice on the same degenerate prompt under a fixed
    seed. Assert token-identical output AND identical per-step
    _layer2_pr log. STOP on any divergence. Do NOT compare against
    HEAD's full-sequence output; that comparison is retired.

4.3 Spectral liveness (proves BLOCKER #1 is actually fixed):

    Run a t2s_degenerate record. Assert self._layer2_pr becomes
    non-None at or after the 24th decode step. If _layer2_pr is still
    None at end-of-generation, B2 did not fix the BLOCKER; STOP.

4.4 Functional re-measurement (Concern A):

    python scripts/test_degeneracy_smoke.py
        -> PASS under the new KV-cache regime. If Single-Token Spam or
           Number Loop behavior shifts outside the owner-signed band,
           STOP and report; do not tune thresholds.

    python scripts/test_factual_safety.py
        -> PASS (governor dormant, 0.00 L2 force on factual prompts).

4.5 t2s macro-loop rescue gate (Concern B, pre-registered):

    On the DS-033 detection-gap subset (93 records the bigram detector
    missed), measure rescue rate of the fully wired path
    (spectral_fire -> kickstart, B1 policy) under KV-cache greedy.
    Acceptance (owner signs the number; proposal default below):

        rescue >= 0.50

    If below, the actuator architecture — not just detection — needs
    redesign. Report the exact rate; do not negotiate the threshold.

4.6 Path 2 is NOT claimed green:

    scripts/test_path2_code_filter.py has a pre-existing ghost reference
    (governor._collapse_persistence_counter) and a diagnose()-trace
    dependency that B2's generate() no longer satisfies. Do not run it,
    do not edit it. List it in the final report as owner-blocked.

## 5. Definition of done

- KV-cache restructure in place; spectral hook fires on decode steps.
- _evaluate_collapse added; diagnose() signature AND body unchanged.
- is_collapsed wired to kickstart (B1 policy); suppression unchanged.
- Profile step + apply_interventions removed from generate().
- Residual path code retained in file, not invoked by generate().
- No protected file modified. No threshold changed.
- 128 unit tests pass; determinism contract holds; smoke + factual
  safety pass under KV-cache; t2s rescue gate reported with the exact
  rate against the pre-registered threshold.
- Report: summary; files changed with one-line rationale; gate outputs
  verbatim; the t2s rescue rate; any red gate + diagnosis; the
  return-schema change and the collapse-field addition.

## 6. Owner signing items (NOT the code agent's job)

- Sign the t2s rescue acceptance threshold (proposal default >= 0.50).
- Retire or fix scripts/test_path2_code_filter.py: the ghost
  governor._collapse_persistence_counter reference AND its
  diagnose()-trace dependency (B2 stops generate() from calling
  diagnose()). Protected file; owner edit.
- Sign docs/TIER_B_REBASELINE_SPEC.md (B2 section) after B2 lands green.
- Decide disposition of the empty residual-path return fields
  ("profile", "decisions", "intervention_log"); confirm downstream
  consumers are updated for the new "collapse" field.
