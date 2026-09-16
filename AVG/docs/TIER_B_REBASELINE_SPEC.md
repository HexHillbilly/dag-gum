# TIER_B_REBASELINE_SPEC.md

**Status:** PROPOSAL — NOT ratified. Pending owner signature.
No controller edit, no smoke rewrite, and no threshold change may
proceed beyond what this document authorizes.

**Non-negotiable constraints (restated):** thresholds remain frozen;
protected gate scripts are read-only for agents and are updated only by
the human [1]. A re-baseline doc is signed *before* the controller diff,
never after. A red gate remains a signal, not a negotiation [1].

---

## 1. The load-bearing finding (recorded fact)

Two consecutive protected-gate failures produced an identical regression
fingerprint and mapped a coupling that static review missed:

> The degeneracy smoke gate runs under sampling (the controller defaults
> to `do_sample=True`). The legacy `non_prose_token_ids` set crushes digit
> tokens because digits have no vowels. That accidental digit-crushing is
> the mechanism holding Number Loop at 0 fires and numeric-token fraction
> at ~0.053. Any kickstart change — retarget, narrow, or disable — removes
> that digit suppression and yields the same regression: Single-Token Spam
> 21→1 fires, Number Loop 0→1 fires, numeric fraction 0.053→0.417.

**Consequence:** there is no safe additive kickstart edit. The accidental
vowel-classification policy is load-bearing and may only be replaced
deliberately, never patched around.

---

## 2. Package split (two signed packages, not one mega-diff)

| Package | Contents | Invariants it touches |
|---|---|---|
| **B1 — Kickstart policy** | Retarget to `active_loop_ids`; vocab-range fix (`len(tokenizer)`); greedy-only kickstart; **explicit Number Loop rule** | Smoke fire counts, numeric fraction, EOS behavior |
| **B2 — Decode + residual path** | KV-cache incremental decoding; detach logit actuation from `diagnose()`/residual path | FP accumulation path, residual engagement, byte-identical-to-HEAD |

B1 is self-contained and must be provably green on its own before B2 is
applied. B2 follows behind its own signed baseline.

---

## 3. B1 — Kickstart policy

### 3.1 Scope

1. Kickstart penalties retargeted from `non_prose_token_ids` to
   `active_loop_ids` (the actually-repeated tokens).
2. Vocab boundary fix: `vocab_size = len(self.tokenizer)` so EOS
   (id 151643) and added special tokens are classified.
3. Kickstart **armed only if** `do_sample=False` (greedy). Under
   sampling, the controller runs suppression-only.
4. **Number Loop rule** — the load-bearing replacement for accidental
   digit-crushing (§3.2).

### 3.2 Number Loop rule (Option b — explicit, deliberate, named)

The accidental digit suppression in `non_prose_token_ids` is replaced by
a **separate, named numeric/structural predicate** in the Stage-1
lexical filter — NOT folded into `is_code_syntax_context`. Code syntax
and digit-heavy runs are distinct modalities and must stay auditable
separately in smoke telemetry.

**Proposed predicate name:** `is_numeric_syntax_context` (or an
equivalent `numeric_fraction` guard), emitting
`suppressed_by=numeric_immunity` in the telemetry, distinct from
`suppressed_by=code_syntax`.

**Implementation sketch (subject to measurement):**

    numeric_fraction = count_numeric_tokens(trailing_tokens) / max(1, len(trailing_tokens))
    if numeric_fraction >= NUMERIC_IMMUNITY_THRESHOLD:
        is_collapsed = False   # dormant, like code-context immunity

**`NUMERIC_IMMUNITY_THRESHOLD` is unmeasured.** The threshold and the new
spam fire band are filled by a human-signed measurement pass on the
existing Number Loop and Single-Token fixtures. Agents do not propose the
number "to make green" [1].

### 3.3 New B1 smoke expectations

The old `21 fires / Number Loop 0 / numeric 0.053` is retired for B1.
The new contract is:

| Fixture | New expectation | Basis |
|---|---|---|
| Single-Token Spam | TBD — measured under retargeted kickstart (greedy), asserted as a frozen band | Night-016/night-017 measured the retargeted behavior inline |
| Number Loop | 0 fires — immunity preserved by the **explicit numeric rule**, not the vowel accident | §3.2 predicate |
| Numeric-token fraction | TBD — measured, frozen with the rule's threshold | Night-018 baseline |
| EOS under sampling | Kickstart disabled → EOS not elevated by kickstart | Night-014 product finding |

**Explicit:** the B1 smoke still measures residual fires until B2
detaches the residual path from logit actuation. Nobody should expect
spam fires to drop to zero just because kickstart retargets.

**Explicit:** suppression still runs under sampling in B1. Number Loop
immunity under sampling therefore rests on **the numeric rule plus
suppression**, not on kickstart.

The `TBD` values are filled by a human-signed measurement pass **before**
the controller diff lands, and the protected smoke script's assertions
are rewritten to match by the human, not an agent [1].

---

## 4. B2 — Decode + residual path

### 4.1 Scope

1. KV-cache incremental decoding (pre-fill once, single-token forwards
   with `past_key_values`) so the layer-2 PR hook sees `hs.shape[1] == 1`
   every step.
2. Detach logit actuation from `diagnose()`/`apply_interventions()` —
   the residual `sae_guided_reset` path is no longer the collapse
   predicate's actuation sink.

### 4.2 `byte-identical-to-HEAD` policy

Strict byte-identity against the full-sequence HEAD is **retired as a
hard gate** for the KV path — impossible by construction, since FP
accumulation differs between S=1 and S=N forward passes.

Replaced by a **KV determinism + engagement contract**, implemented as
`test_kv_determinism.py`:

- Same seed, same dtype, same device → same token ids AND same per-step
  PR log.
- Force bounds still validated when the residual path is exercised.
- Healthy prose still shows no spurious intervention / no quality
  regression vs dormant.

A legacy full-sequence path may be retained **only as an offline
archaeology script** (`scripts/archaeology_full_sequence.py`). It is not
a merge blocker and does not stay in the production `generate()` path.

### 4.3 New B2 engagement expectations

The residual-fire counts the smoke currently measures (`21` on spam) are
re-derived under the detached logit path and frozen with the human's
signature.

**Explicit:** residual remains **deprecated** per RFC-004 Amendment A4.
B2 only *decouples* it from logit actuation and re-freezes engagement
numbers — it does **not** revive residual as a product path.

---

## 5. Sequencing (fixed, human-owned)

1. **Write and sign §3** — Number Loop rule, new fire/numeric
   expectations, EOS-in-vocab behavior.
2. **Run B1 measurement pass** (night-019) to fill the TBD bands from
   real data on the Number Loop + Single-Token fixtures.
3. **Land B1 controller diff** — retarget + vocab + greedy-only kickstart
   + Number Loop rule. Run the full protected suite against the **new**
   contract.
4. **Write and sign §4** — KV determinism contract, residual engagement
   under the detached path, byte-identical policy.
5. **Land B2 controller diff** — KV-cache + residual detach. Run the full
   protected suite against the new contract.
6. **Human merge** on 100% green; the merge train runs the full suite in
   the repo venv with one push at the end [1].

No agent may re-baseline a protected gate or freeze a threshold alone [1].

---

## 6. Resolved triad questions (locked, not re-litigated at merge time)

1. **Number Loop rule location:** a separate, named numeric/structural
   predicate next to code-context immunity — not buried inside
   `is_code_syntax_context`.
2. **Greedy-only kickstart:** kept inside B1, bundled with the explicit
   numeric rule and the new smoke contract — never landed in isolation.
3. **`byte-identical-to-HEAD`:** retired as a hard gate for the KV path;
   replaced with `test_kv_determinism.py`; legacy full-sequence retained
   only as an offline archaeology script.

---
**Status:** PROPOSAL — awaiting owner signature. Not yet ratified.
