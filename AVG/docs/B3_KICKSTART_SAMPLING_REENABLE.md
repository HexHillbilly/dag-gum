# B3 — Re-enable kickstart under sampling (supersedes B1 greedy-only)

- Task: B3-KICKSTART-SAMPLING-REENABLE
- Status: RATIFIED (owner approval, 2026-08-13)
- Reference model: Qwen2.5-1.5B (28L, d_model 1536, bf16)

## 0. What this changes

B1 made the kickstart **greedy-only** (`not do_sample` guard) to kill the
legacy digit-crush bug. This decision re-enables the kickstart under the
sampling regime (removes the `not do_sample` guard in two places), because
the protections that motivated the guard are now provided by explicit
mechanisms instead.

## 1. Why B1 made kickstart greedy-only

The legacy kickstart penalized `_non_prose_token_ids`, a vocab subset that
*accidentally* classified digit tokens as non-prose and crushed them with
`-1e4`. That accidental digit-crush was load-bearing for **Number Loop
immunity under sampling** (the degeneracy smoke gate runs `do_sample=True`).
B1 removed the risk two ways:

1. Retargeted the kickstart to `active_loop_ids` (no more vocab-set crush).
2. Added an explicit, named `numeric_token_fraction` predicate +
   `NUMERIC_IMMUNITY_THRESHOLD = 0.80`, evaluated every step,
   regime-independent.

The `not do_sample` guard was the *third* safety valve — belt-and-suspenders
from the digit-crush era. With (1) and (2) in place, it is redundant and,
worse, leaves sampling **under-actuated**.

## 2. The evidence

DS-035 (post-B2) measured the production controller under sampling and found
t2s_degenerate rescue at **79/100** — below the 85/100 success bar. Root
cause (verified): under sampling, the only actuation was `-5.0` token
suppression, which is too weak to break a perfect deterministic loop. The
spectral predicate fired correctly on every non-rescue, but there was no
actuation strong enough to act on it. The same 20 non-rescued records are
**20/20 rescued under greedy** (kickstart ON), ΔD2 +0.52…+0.91.

| metric | before (suppression-only) | after (kickstart-under-sampling) |
|---|---|---|
| t2s_degenerate rescue | 79/100 | **97/100** |
| qwen_degenerate rescue | 97/100 | 97/100 |
| EOS bailout | — | **0/100 (both fixtures)** |
| mean ΔD2 | — | +0.71 (t2s), +0.80 (qwen) |

## 3. Why EOS-death does not recur

night-017 warned that kickstart-under-sampling drives ~64% EOS bailout. That
was measured against the legacy `_non_prose_token_ids` kickstart, which
crushed every non-prose token *except* EOS. B1's retarget to `active_loop_ids`
penalizes only the loop tokens, so the model falls into **diverse prose**
(act_d2 0.91–1.00), not termination. Measured EOS = 0 on both fixtures.

## 4. The change (2 lines)

`governor/controller.py`, `generate()`:

- Kickstart trigger (was `not is_numeric_context and not do_sample and
  active_loop_ids and (trailing_ctr < 0.50 or is_collapsed)`): remove
  `and not do_sample`.
- Kickstart penalty application (was `if not do_sample and
  self._kickstart_counter > 0 and active_loop_ids:`): remove
  `not do_sample and`.

The magnitude schedule (`-1e4 / -5.0 / -2.0`), the `active_loop_ids` target,
the `not is_numeric_context` guard, and the `trailing_ctr < 0.50 or
is_collapsed` trigger are all unchanged.

## 5. Supersession

B1 §3.2 ("Greedy-only kickstart: kickstart fires only under do_sample=False")
is **superseded**. Number Loop immunity is now carried by the explicit
`numeric_token_fraction` predicate, not by the regime gate.

## 6. Gates re-verified (all green)

- Degeneracy smoke: PASS (Single-Token Spam 1 fire; Number Loop 0 fires,
  3 immunity steps).
- Factual safety: PASS (0 fires, 0.00 L2).
- `pytest tests/`: 128 passed.
- Latency budget: PASS (-0.19 ms/token <= 7.0 ms/token).
- DS-035 success bar (sampling >= 85/100 both fixtures): **MET**
  (97/100 t2s, 97/100 qwen).
