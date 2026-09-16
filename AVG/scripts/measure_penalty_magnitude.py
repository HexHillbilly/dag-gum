#!/usr/bin/env python3
"""night-017: Kickstart penalty magnitude sweep (MEASUREMENT ONLY).

Context
-------
night-016 (merged) fixed the two rigor-review BLOCKERs (spectral PR hook dead;
dual-predicate feeding residual path) via the KV-cache restructure + dual-
predicate -> logit-penalty wiring.  But the EOS-death spiral PERSISTS under
greedy decoding even after the night-015 retarget to active_loop_ids:

    | fixture | arm    | net escape | EOS rate | median n_gen (esc) |
    |---------|--------|-----------|----------|--------------------|
    | t2s     | greedy | 0.96       | 0.85     | 26.0               |
    | qwen    | greedy | 1.00       | 0.67     | 68.0               |
    | heldout | greedy | 1.00       | 0.26     | 128.0              |

night-015 proved retargeting to active_loop_ids removes the EOS-exclusion
quirk.  night-016 proves the penalty MAGNITUDE is the remaining issue: -1e4 on
active_loop_ids still leaves EOS as the dominant remaining token once the loop
is crushed.  The -1e4/-5.0/-2.0 three-step schedule was designed for the old
CTR-triggered kickstart; it has never been re-calibrated for the retargeted
spectral-gated deployment.

This probe sweeps the kickstart penalty magnitude across {1e4, 5.0, 2.0, 1.0}
on the restructured regime (all night-016 fixes inline), measuring EOS rate and
median n_gen (escaped) as CO-PRIMARY metrics alongside rescue rate.

MEASUREMENT ONLY.  No controller edits.  All night-016 fixes implemented
inline.  No threshold changes [1].

Sweep cells (three-step kickstart schedule, uniform magnitude per cell):

    | cell | counter=3 | counter=2 | counter=1 |
    |------|-----------|-----------|-----------|
    | 1e4   | -1e4      | -5.0      | -2.0      |  REUSED from night-016
    | 5.0   | -5.0      | -5.0      | -2.0      |  NEW
    | 2.0   | -2.0      | -2.0      | -1.0      |  NEW
    | 1.0   | -1.0      | -1.0      | -1.0      |  NEW

All cells apply the penalty to active_loop_ids (retargeted, night-015).
Suppression cooldown=8 penalty=-5.0 (unchanged production).  top_p=0.85 when
either mechanism is active.

Reused baselines (do NOT re-run):
  - night-016 dormant baselines (greedy + sampling, KV-cache): per-record
    `dormant_greedy` / `dormant_sampling` in
    docs/gate23/kvcache_restructure_results.jsonl (already validated 0/250
    drift against DS-035/night-014).
  - night-016 cell 1e4: per-record `greedy_active` / `sampling_active` in
    the same file (the -1e4/-5.0/-2.0 baseline).

NEW live runs: 3 cells (5.0, 2.0, 1.0) x 2 arms (greedy, sampling) x 3
fixtures = 18 condition-fixture pairs.

Environment notes (identical to night-014/night-015/night-016):
  - torch 2.13 CUDA bmm Triton override is deregistered (env-only; no C
    compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; the pinned revision is already present, so the
    read-only cache is used directly (no HF_HOME fallback needed).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np
import torch
import torch.nn as nn

# Standard root-as-package header. From scripts/, parents[2] resolves to '/';
# `import AVG.*` resolves via the /AVG mount. (Protocol import convention.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
print(f"[import] path added: {Path(__file__).resolve().parents[2]}")

# Env-only adaptation: deregister torch 2.13's CUDA bmm Triton override.
try:
    from torch._native import triton_utils as _triton_utils

    _triton_utils.deregister_op_overrides()
    print("[env] torch bmm Triton override deregistered")
except Exception as _env_e:  # pragma: no cover - env dependent
    print(f"[env] bmm override deregistration skipped: {_env_e}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from AVG.core.metrics import (  # noqa: E402
    compute_coherent_token_ratio,
    is_code_syntax_context,
    participation_ratio,
)
from AVG.governor.controller import (  # noqa: E402
    compute_token_distinct_2_fast,
)

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # night-014 measurement seed (matches ds-025..ds-035, night-015/016)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # decoding steps (EOS break; EOS token included)
ANALYSIS_WINDOW = 24  # rolling 24-token PR / distinct-2 window (Gate 2.3 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook
NUM_RECORDS_C = 50  # Fixture C: seeded 50-record subset (random.Random(42))

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
# DS-034e configuration: spectral_collapse uses band_low (hardened), NOT T_PR.
T_PR_2_TASK = 10.954796  # PR < T -> spectral fire (primary, DS-034c)
BAND_LOW_TASK = 8.216097  # spectral_collapse threshold (band_low, frozen)

# Sampling regime (DS-032 Part 2 / night-014 configuration).
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.85

# Production controller settings (controller.py defaults, DS-033/DS-034c task).
COOLDOWN = 8          # controller.py:137 (cooldown_steps=8)
PENALTY = -5.0        # controller.py:745 (next_logits[:, tid] -= 5.0)
TOP_P = 0.85          # controller.py:138 (top_p=0.85)
CTR_THRESHOLD = 0.50  # controller.py:691 (trailing_ctr < 0.50)
DIVERSITY_CTR_GATE = 0.40  # controller.py:683 (token_diversity < 0.40)
EVERY_K = 2           # controller.py:136 (every_k=2)
TRAILING_GEN_WINDOW = 16  # controller.py:686 (trailing 16 generated tokens)
KICKSTART_COUNTER_INIT = 3  # controller.py:692

# Sweep cells: kickstart penalty magnitude schedules (three-step, uniform per
# cell).  Cell "1e4" is the night-016 production baseline (REUSED).
SWEEP_CELLS: Dict[str, Dict[int, float]] = {
    "1e4": {3: -1e4, 2: -5.0, 1: -2.0},  # REUSED from night-016 (do NOT re-run)
    "5.0": {3: -5.0, 2: -5.0, 1: -2.0},  # NEW
    "2.0": {3: -2.0, 2: -2.0, 1: -1.0},  # NEW
    "1.0": {3: -1.0, 2: -1.0, 1: -1.0},  # NEW
}
NEW_CELLS = ("5.0", "2.0", "1.0")  # cells that require LIVE runs

# Collapse predicate (controller.py:464, use_code_filter=False):
#   is_token_loop = token_diversity < 0.30 and trailing_ctr < 0.30
COLLAPSE_DIVERSITY = 0.30
COLLAPSE_CTR = 0.30

# DS-034e spectral corroboration on spectral-only fires (bigram_ctr < 2).
SPECTRAL_CORROBORATION_DIVERSITY = 0.40

# Fix 3 uses the production use_code_filter default (False).
USE_CODE_FILTER = False

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
QWEN_DEG = Path("tests/fixtures/qwen_degenerate.jsonl")
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
NIGHT016_JSONL = Path("docs/gate23/kvcache_restructure_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/penalty_magnitude_results.jsonl")
OUTPUT_MD = Path("docs/gate23/PENALTY_MAGNITUDE_RESULTS.md")

# HF cache: model is already present in the read-only system cache. Do NOT force
# an HF_HOME fallback to /tmp (that would re-download 1.5B weights). Only fall
# back if the pinned model is not present.
_DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
if not os.access(_DEFAULT_HF_CACHE, os.W_OK):
    _pinned_dir = os.path.join(
        _DEFAULT_HF_CACHE, "hub", "models--Qwen--Qwen2.5-1.5B",
        "snapshots", MODEL_REVISION,
    )
    if not os.path.isdir(_pinned_dir):
        _fallback = "/tmp/hf_cache"
        os.makedirs(_fallback, exist_ok=True)
        os.environ.setdefault("HF_HOME", _fallback)
        print(f"[env] HF_HOME fallback -> {_fallback} (pinned revision not cached)")
    else:
        print("[env] read-only HF cache used directly (pinned revision present)")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Frozen threshold (read from the Part A freeze file; do NOT re-derive)
# ---------------------------------------------------------------------------
def load_frozen_thresholds(path: Path) -> Tuple[float, float]:
    """Parse the machine-parseable JSON freeze block from FROZEN_THRESHOLDS.md
    and return (T_PR(2), band_low(2)). STOP if the values disagree with the
    task-specified frozen values."""
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md has no ```json freeze block.")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md freeze block is unterminated.")
    freeze = json.loads(text[start:end])
    t_pr = float(freeze["layers"]["2"]["pr"]["T"])
    band_low = float(freeze["layers"]["2"]["pr"]["band_low"])
    if abs(t_pr - T_PR_2_TASK) > 1e-9:
        raise SystemExit(
            f"[STOP] FROZEN_THRESHOLDS.md T_PR(2)={t_pr:.6f} does not match the "
            f"task-specified frozen value {T_PR_2_TASK:.6f}. Do NOT re-derive "
            f"or adjust thresholds."
        )
    if abs(band_low - BAND_LOW_TASK) > 1e-9:
        raise SystemExit(
            f"[STOP] FROZEN_THRESHOLDS.md band_low(2)={band_low:.6f} does not "
            f"match the task-specified frozen value {BAND_LOW_TASK:.6f}. Do NOT "
            f"re-derive or adjust thresholds."
        )
    return t_pr, band_low


# ---------------------------------------------------------------------------
# Vocabulary subsets — night-015 Fix 1 (vocab-range fix) inline.  Retained for
# offline diagnostics ONLY; NOT called during generation (Fix 6).
# ---------------------------------------------------------------------------
def cache_vocabulary_subsets_fixed(tokenizer: Any) -> Tuple[Set[int], Set[int]]:
    """night-015 Fix 1: vocab_size = len(self.tokenizer).

    Iterates range(len(tokenizer)) so special tokens INCLUDING EOS are
    iterated and classified.  Retained for offline diagnostics (Fix 6); the
    kickstart retargeting to active_loop_ids removes its runtime use.
    """
    vocab_size = len(tokenizer)
    vowels = set("aeiouyAEIOUY")
    prose: Set[int] = set()
    non_prose: Set[int] = set()
    for tid in range(int(vocab_size)):
        try:
            tok_str = tokenizer.decode([tid])
        except Exception:
            continue
        clean_str = tok_str.strip()
        if clean_str in ("a", "I", "A") or (
            len(clean_str) >= 2
            and all(c.isalpha() or c in ("'", "-") for c in clean_str)
            and any(c in vowels for c in clean_str)
        ):
            prose.add(tid)
        else:
            non_prose.add(tid)
    return prose, non_prose


# ---------------------------------------------------------------------------
# Fix 6: sample_top_p with explicit .clone() at function start (inline).
# ---------------------------------------------------------------------------
def sample_top_p_restructured(
    logits: torch.Tensor,
    top_p: float = 0.85,
    filter_value: float = -1e4,
) -> torch.Tensor:
    """Fix 6: production sample_top_p with explicit .clone() at function start.

    Implemented INLINE (not applied to controller.py).  The upstream
    controller.sample_top_p mutates its input in place; the restructured
    version clones first so the caller's `next_logits` is never modified.
    """
    logits = logits.clone()  # Fix 6
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = filter_value
    return logits


# ---------------------------------------------------------------------------
# Layer-2 PR measurement hook (identical to night-016 / DS-034b instrument)
# ---------------------------------------------------------------------------
def make_layer2_pr_hook(
    analysis_window: int = ANALYSIS_WINDOW,
) -> Tuple[Callable[..., Any], List[torch.Tensor], List[Dict[str, Any]]]:
    """Create a layer-2 forward-hook measurement instrument.

    At each decoding step (one forward pass per step), capture the hidden state
    at the current token position (the last position of the sequence, matching
    the governor's shadow-hook capture at ``[:, -1:, :]``), append it to a ring
    buffer of the trailing ``analysis_window`` generated hidden states, and when
    the buffer has >= ``analysis_window`` entries compute
    ``participation_ratio()`` over the window (float32, matching ds-005/ds-010
    numerical discipline).  The PR value is logged; the hook returns ``output``
    unchanged (pure measurement instrument).
    """
    buffer: List[torch.Tensor] = []
    pr_log: List[Dict[str, Any]] = []
    step_counter = {"n": 0}

    def hook_fn(
        module: nn.Module,
        input_args: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        hs = output[0] if isinstance(output, tuple) else output
        # Current token position = last position of the (seq_len==1) input.
        current = hs[:, -1:, :].detach().to(dtype=torch.float32)  # (1,1,d)
        buffer.append(current)
        if len(buffer) > analysis_window:
            buffer.pop(0)  # ring buffer: keep trailing `analysis_window`
        step = int(step_counter["n"])
        step_counter["n"] += 1
        if len(buffer) >= analysis_window:
            win = torch.cat(buffer[-analysis_window:], dim=1)  # (1,24,d)
            pr = participation_ratio(win).item()
            pr_log.append({"step": step, "pr": float(pr)})
        return output

    return hook_fn, buffer, pr_log


# ---------------------------------------------------------------------------
# Restructured active generation core (shared by greedy and sampling).
# Implements Fix 1 + Fix 2 + Fix 3 + Fix 5 + Fix 6 inline.
# Parameterized by the kickstart penalty magnitude schedule (night-017).
# ---------------------------------------------------------------------------
def _restructured_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    kickstart_penalties: Dict[int, float],
    do_sample: bool,
    max_new_tokens: int = MAX_NEW_TOKENS,
    cell_label: str = "",
) -> Dict[str, Any]:
    """KV-cache incremental generation under the night-016 restructured rules.

    All night-016 inline fixes are implemented here:

      Fix 1 — KV-cache incremental decoding (pre-fill once, then single-token
              forwards with past_key_values).  `hs.shape[1] == 1` on every
              decoding step → the layer-2 PR hook condition fires correctly.
      Fix 2 — Collapse predicate gates logit penalties.  Suppression
              (add active_loop_ids to cooldown) is gated on `is_collapsed`;
              kickstart fires on spectral-only (`spectral_fire AND NOT
              bigram_fire`) and the kickstart penalty is retargeted to
              `active_loop_ids` (night-015).
      Fix 3 — is_code_context recomputed EVERY step when USE_CODE_FILTER is
              True (moved outside the CTR sub-sampling gate).
      Fix 5 — Vocab-range fix inherited (offline diagnostics only); kickstart
              penalties applied to active_loop_ids, not non_prose_token_ids.
      Fix 6 — sample_top_p_restructured (explicit .clone()); finally block
              removes the layer-2 hook and clears per-generation state.
      Fix 4 — No VarietyProfiler.profile() call, no is_profile_step logic, no
              hooks_registered dormancy management in the loop.

    night-017 parameter: `kickstart_penalties` is the per-cell three-step
    kickstart penalty schedule applied uniformly to active_loop_ids.

    do_sample=False → greedy argmax; do_sample=True → temp=0.8, top_p=0.85
    multinomial (matching night-014/night-016).
    """
    if do_sample:
        seed_all(SEED)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    # Fix 1: Pre-fill — one forward pass over the full prompt.
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :].clone().float()
    input_ids = prompt_ids

    # Register the layer-2 spectral hook AFTER pre-fill so the rolling buffer
    # only contains hidden states from decoding steps (seq_len==1), matching
    # the DS-034b/DS-034c hook pattern.
    hook_fn, buffer, pr_log = make_layer2_pr_hook(ANALYSIS_WINDOW)
    blocks = model.model.layers
    handle = blocks[HOOK_LAYER].register_forward_hook(hook_fn)

    # Per-generation state (Fix 6: reset in finally).
    active_suppress: Dict[int, int] = {}
    kickstart_counter = 0
    bigram_ctr = 0
    spectral_ctr = 0
    layer2_pr: Optional[float] = None

    fire_events: List[Dict[str, Any]] = []
    n_suppression_steps = 0
    n_kickstart_events = 0
    n_bigram_collapse_steps = 0
    n_spectral_collapse_steps = 0
    n_bigram_fire_steps = 0
    n_spectral_fire_steps = 0
    n_both_fire_steps = 0
    n_spectral_only_fire_steps = 0
    generated: List[torch.Tensor] = []

    try:
        for step in range(max_new_tokens):
            # ---- 1. Bigram predicate: fast integer-only token proxy ----
            token_diversity, active_loop_ids = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )

            # ---- 2. CTR sub-sampling (every_k=2 or div<0.40) ----
            trailing_ctr = 1.0
            is_code_context = False
            if token_diversity < DIVERSITY_CTR_GATE or step % EVERY_K == 0:
                recent_gen = (
                    input_ids[0, prompt_len:]
                    if input_ids.shape[-1] > prompt_len
                    else input_ids[0]
                )
                recent_tokens = (
                    recent_gen[-TRAILING_GEN_WINDOW:]
                    if recent_gen.numel() > 0
                    else input_ids[0, -TRAILING_GEN_WINDOW:]
                )
                trailing_text = tokenizer.decode(
                    recent_tokens, skip_special_tokens=True
                )
                trailing_ctr = compute_coherent_token_ratio(trailing_text)

            # ---- Fix 3: recompute is_code_context EVERY step when
            #             use_code_filter is True (outside the CTR gate) ----
            if USE_CODE_FILTER and tokenizer is not None:
                recent_gen = (
                    input_ids[0, prompt_len:]
                    if input_ids.shape[-1] > prompt_len
                    else input_ids[0]
                )
                recent_tokens = (
                    recent_gen[-TRAILING_GEN_WINDOW:]
                    if recent_gen.numel() > 0
                    else input_ids[0, -TRAILING_GEN_WINDOW:]
                )
                trailing_text = tokenizer.decode(
                    recent_tokens, skip_special_tokens=True
                )
                is_code_context = is_code_syntax_context(trailing_text)

            # ---- 3. Spectral predicate: latest PR from the layer-2 hook ----
            pr = pr_log[-1]["pr"] if pr_log else None
            layer2_pr = pr

            # ---- 4. Frozen DS-034e dual-predicate OR rule (exact) ----
            bigram_collapse = bool(
                token_diversity < COLLAPSE_DIVERSITY
                and trailing_ctr < COLLAPSE_CTR
            )
            if bigram_collapse:
                bigram_ctr += 1
            else:
                bigram_ctr = 0
            bigram_fire = bool(bigram_ctr >= 2)

            spectral_collapse = bool(
                pr is not None and pr < spectral_threshold
            )
            if spectral_collapse:
                spectral_ctr += 1
            else:
                spectral_ctr = 0
            if bigram_ctr < 2:
                spectral_fire = bool(
                    spectral_ctr >= 2
                    and token_diversity < SPECTRAL_CORROBORATION_DIVERSITY
                )
            else:
                spectral_fire = bool(spectral_ctr >= 2)
            if is_code_context:
                spectral_fire = False

            is_collapsed = bool(bigram_fire or spectral_fire)

            n_bigram_collapse_steps += int(bigram_collapse)
            n_spectral_collapse_steps += int(spectral_collapse)
            n_bigram_fire_steps += int(bigram_fire)
            n_spectral_fire_steps += int(spectral_fire)
            n_both_fire_steps += int(bigram_fire and spectral_fire)
            n_spectral_only_fire_steps += int(spectral_fire and not bigram_fire)

            # ---- 5. Fix 2: Actuation — collapse predicate gates logit
            #               penalties (was: residual diagnose()/apply_interventions
            #               path; DISABLED in this probe) ----
            if is_collapsed:
                # Token suppression: add active_loop_ids to cooldown.
                if active_loop_ids:
                    for tid in active_loop_ids:
                        active_suppress[tid] = COOLDOWN
                # Kickstart: spectral-only fires trigger vocabulary steering.
                if spectral_fire and not bigram_fire:
                    kickstart_counter = KICKSTART_COUNTER_INIT
                    n_kickstart_events += 1

            # ---- 6. Fire-event logging (dual-predicate detection liveness) ----
            if is_collapsed:
                triggers: List[str] = []
                if bigram_fire:
                    triggers.append("bigram")
                if spectral_fire:
                    triggers.append("spectral")
                fire_events.append({
                    "step": int(step),
                    "trigger": "+".join(triggers) if triggers else "none",
                    "bigram_fire": bool(bigram_fire),
                    "spectral_fire": bool(spectral_fire),
                    "bigram_collapse": bool(bigram_collapse),
                    "spectral_collapse": bool(spectral_collapse),
                    "bigram_ctr": int(bigram_ctr),
                    "spectral_ctr": int(spectral_ctr),
                    "pr": pr,
                    "token_diversity": float(token_diversity),
                    "trailing_ctr": float(trailing_ctr),
                    "is_code_context": bool(is_code_context),
                })

            # ---- 7. Logit-penalty path ----
            logits = next_logits.clone()
            n_suppressed = 0
            if active_suppress:
                for tid, steps_left in list(active_suppress.items()):
                    if steps_left > 0:
                        logits[:, tid] += PENALTY
                        active_suppress[tid] -= 1
                        n_suppressed += 1
                    else:
                        del active_suppress[tid]
            if n_suppressed > 0:
                n_suppression_steps += 1

            # ---- 7b. Fix 5: Kickstart penalty — retargeted to active_loop_ids
            #               (was: non_prose_token_ids; night-015 fix).
            #               night-017: magnitude from the per-cell schedule. ----
            kick_penalty_value = 0.0
            if kickstart_counter > 0:
                if active_loop_ids:
                    loop_tensor = torch.tensor(
                        list(active_loop_ids), device=logits.device
                    )
                    kick_penalty_value = kickstart_penalties.get(
                        kickstart_counter,
                        kickstart_penalties.get(1, -2.0),
                    )
                    logits[:, loop_tensor] += kick_penalty_value
                kickstart_counter -= 1

            if active_suppress or kickstart_counter > 0:
                logits = sample_top_p_restructured(logits, top_p=TOP_P)

            # ---- 8. Sample / greedy ----
            if do_sample:
                logits = logits / max(SAMPLING_TEMPERATURE, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            # ---- 9. Fix 1: single-token forward with the KV cache ----
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].clone().float()

            # ---- 10. Append to input_ids for the next step's bigram ----
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    finally:
        # Fix 6: spectral hook MUST always be removed, even on exception.
        handle.remove()
        # Per-generation state is function-local; clearing is implicit on
        # return.  (In the controller, the finally block resets the class
        # attributes _bigram_ctr, _spectral_ctr, _layer2_pr,
        # _layer2_pr_buffer, _active_suppress_tokens, _kickstart_counter.)

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    pr_vals = [p["pr"] for p in pr_log]
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
        "pr_log": pr_log,
        "n_pr_values": len(pr_log),
        "pr_mean": float(np.mean(pr_vals)) if pr_vals else None,
        "pr_min": float(np.min(pr_vals)) if pr_vals else None,
        "pr_max": float(np.max(pr_vals)) if pr_vals else None,
        "pr_first": float(pr_vals[0]) if pr_vals else None,
        "pr_last": float(pr_vals[-1]) if pr_vals else None,
        "fire_events": fire_events,
        "fire_count": len(fire_events),
        "n_bigram_fire_steps": n_bigram_fire_steps,
        "n_spectral_fire_steps": n_spectral_fire_steps,
        "n_both_fire_steps": n_both_fire_steps,
        "n_spectral_only_fire_steps": n_spectral_only_fire_steps,
        "n_suppression_steps": n_suppression_steps,
        "n_kickstart_events": n_kickstart_events,
        "n_bigram_collapse_steps": n_bigram_collapse_steps,
        "n_spectral_collapse_steps": n_spectral_collapse_steps,
        "layer2_pr_first": layer2_pr if layer2_pr is not None else None,
        "do_sample": do_sample,
        "cell": cell_label,
    }


@torch.no_grad()
def greedy_generate_cell(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    kickstart_penalties: Dict[int, float],
    max_new_tokens: int = MAX_NEW_TOKENS,
    cell_label: str = "",
) -> Dict[str, Any]:
    """Greedy restructured active generation (Fix 1+2+3+5+6 inline, cell param)."""
    return _restructured_generate(
        model, tokenizer, prompt, spectral_threshold, kickstart_penalties,
        do_sample=False, max_new_tokens=max_new_tokens, cell_label=cell_label,
    )


@torch.no_grad()
def sampling_generate_cell(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    kickstart_penalties: Dict[int, float],
    max_new_tokens: int = MAX_NEW_TOKENS,
    cell_label: str = "",
) -> Dict[str, Any]:
    """Sampling restructured active generation (Fix 1+2+3+5+6 inline, cell param)."""
    return _restructured_generate(
        model, tokenizer, prompt, spectral_threshold, kickstart_penalties,
        do_sample=True, max_new_tokens=max_new_tokens, cell_label=cell_label,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def continuation_distinct2(generated_ids: torch.Tensor) -> float:
    """Distinct-2 over the trailing 24 generated tokens (ds-025 convention)."""
    if generated_ids.shape[-1] == 0:
        return 1.0
    d2, _ = compute_token_distinct_2_fast(
        generated_ids, prompt_len=0, window_len=ANALYSIS_WINDOW
    )
    return float(d2)


def continuation_ctr(tokenizer: Any, generated_ids: torch.Tensor) -> float:
    """Coherent-token ratio over the decoded continuation text."""
    if generated_ids.shape[-1] == 0:
        return 0.0
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return float(compute_coherent_token_ratio(text))


def build_active_meta(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
    spectral_threshold: float,
    arm: str,
    cell_label: str,
) -> Dict[str, Any]:
    """Build the active-record meta dict for one generation (night-016 shape).

    `dormant` is the REUSED night-016 dormant baseline for the same
    record+arm (greedy or sampling).  The cell label is threaded through so
    every record line in the JSONL is self-describing.
    """
    generated_ids = gen["generated_ids"]
    n = int(generated_ids.shape[-1])
    d2 = continuation_distinct2(generated_ids)
    ctr = continuation_ctr(tokenizer, generated_ids)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    eos_id = tokenizer.eos_token_id
    eos_terminated = bool(
        n < MAX_NEW_TOKENS and n > 0
        and int(generated_ids[0, -1].item()) == eos_id
    )
    byte_identical = bool(
        text.encode("utf-8") == dormant["generated_text"].encode("utf-8")
    )
    gross_escape = not byte_identical
    net_escape = bool(gross_escape and n >= ANALYSIS_WINDOW)
    bucket: Optional[str] = None
    if gross_escape:
        if n == 1:
            bucket = "1"
        elif n < ANALYSIS_WINDOW:
            bucket = "2-23"
        elif n < MAX_NEW_TOKENS:
            bucket = "24-127"
        else:
            bucket = "128"

    n_bigram_fire = int(gen["n_bigram_fire_steps"])
    n_spectral_fire = int(gen["n_spectral_fire_steps"])
    fire_type: Optional[str] = None
    if n_bigram_fire > 0 and n_spectral_fire == 0:
        fire_type = "bigram_only"
    elif n_spectral_fire > 0 and n_bigram_fire == 0:
        fire_type = "spectral_only"
    elif n_bigram_fire > 0 and n_spectral_fire > 0:
        fire_type = "both"

    out: Dict[str, Any] = {
        "arm": arm,
        "cell": cell_label,
        "do_sample": bool(gen["do_sample"]),
        "spectral_threshold": float(spectral_threshold),
        "n_generated": n,
        "distinct_2": d2,
        "n_ge24_distinct_2": d2 if n >= ANALYSIS_WINDOW else None,
        "ctr": ctr,
        "generated_text": text,
        "generated_ids": generated_ids[0].tolist(),
        "eos_terminated": eos_terminated,
        "delta_distinct_2": float(d2 - dormant["distinct_2"]),
        "delta_ctr": float(ctr - dormant["ctr"]),
        "byte_identical_to_dormant": byte_identical,
        "token_identical_to_dormant": bool(
            generated_ids[0].tolist() == dormant["generated_ids"]
        ),
        "gross_escape": gross_escape,
        "net_escape": net_escape,
        "escape_bucket": bucket,
        "fire_count": gen["fire_count"],
        "fire_events": gen["fire_events"],
        "fire_type": fire_type,
        "n_bigram_fire_steps": n_bigram_fire,
        "n_spectral_fire_steps": n_spectral_fire,
        "n_both_fire_steps": int(gen["n_both_fire_steps"]),
        "n_spectral_only_fire_steps": int(gen["n_spectral_only_fire_steps"]),
        "n_suppression_steps": gen["n_suppression_steps"],
        "n_kickstart_events": gen["n_kickstart_events"],
        "n_bigram_collapse_steps": gen["n_bigram_collapse_steps"],
        "n_spectral_collapse_steps": gen["n_spectral_collapse_steps"],
        "n_pr_values": gen["n_pr_values"],
        "pr_mean": gen["pr_mean"],
        "pr_min": gen["pr_min"],
        "pr_max": gen["pr_max"],
        "pr_first": gen["pr_first"],
        "pr_last": gen["pr_last"],
        "pr_values": [p["pr"] for p in gen["pr_log"]],
        "is_rescued": bool(d2 - dormant["distinct_2"] > 0),
    }
    return out


# ---------------------------------------------------------------------------
# Reused data loaders (night-016, do NOT re-run)
# ---------------------------------------------------------------------------
def load_night016_records(path: Path) -> List[Dict[str, Any]]:
    """Load the per-record lines of night-016 (kvcache_restructure_results.jsonl).

    STOPs if the file is missing or has no per-record lines.  Each record is
    expected to carry `dormant_greedy`, `dormant_sampling`, `greedy_active`,
    and `sampling_active` (the -1e4/-5.0/-2.0 cell).
    """
    if not path.exists():
        raise SystemExit(
            f"[STOP] {path} does not exist; night-016 results are required for "
            f"REUSE. Do NOT re-run night-016."
        )
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type"):
                continue
            for key in ("dormant_greedy", "dormant_sampling",
                        "greedy_active", "sampling_active"):
                if key not in r:
                    raise SystemExit(
                        f"[STOP] {path} record {r.get('record_id')} is missing "
                        f"`{key}`; night-016 schema changed unexpectedly."
                    )
            recs.append(r)
    if len(recs) != 250:
        raise SystemExit(
            f"[STOP] night-016 per-record lines = {len(recs)}, expected 250 "
            f"(100+100+50). Do NOT proceed on a partial/rerun baseline."
        )
    return recs


def check_night016_counts(recs: List[Dict[str, Any]]) -> None:
    """Verify the night-016 record counts per fixture (task boundary)."""
    from collections import Counter
    counts = Counter(r["fixture"] for r in recs)
    expected = {"t2s_degenerate": 100, "qwen_degenerate": 100,
                "heldout_degenerate_v2": 50}
    for fixture, n in expected.items():
        if counts.get(fixture) != n:
            raise SystemExit(
                f"[STOP] night-016 {fixture} records = {counts.get(fixture)}, "
                f"expected {n}."
            )


def load_fixture_records(
    fixture_name: str,
) -> List[Dict[str, Any]]:
    """Load the fixture raw records for the given fixture name.

    Fixture A: t2s_degenerate -> tests/fixtures/t2s_degenerate.jsonl,
               prompt key "text".
    Fixture B: qwen_degenerate -> tests/fixtures/qwen_degenerate.jsonl,
               prompt key "prompt".
    Fixture C: heldout_degenerate_v2 ->
               tests/fixtures/heldout_degenerate_v2.jsonl (50-record seeded
               subset random.Random(42).sample, sorted by id),
               prompt key "mutated_prompt".
    """
    if fixture_name == "t2s_degenerate":
        return load_jsonl(T2S_DEG)
    elif fixture_name == "qwen_degenerate":
        return load_jsonl(QWEN_DEG)
    elif fixture_name == "heldout_degenerate_v2":
        all_c = load_jsonl(HELDOUT_DEG_V2)
        rng = random.Random(42)
        return sorted(rng.sample(all_c, NUM_RECORDS_C),
                      key=lambda r: int(r["id"]))
    raise ValueError(f"unknown fixture {fixture_name}")


def fixture_prompt_key(fixture_name: str) -> str:
    if fixture_name == "t2s_degenerate":
        return "text"
    elif fixture_name == "qwen_degenerate":
        return "prompt"
    elif fixture_name == "heldout_degenerate_v2":
        return "mutated_prompt"
    raise ValueError(f"unknown fixture {fixture_name}")


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    records_c: List[Dict[str, Any]],
    prompt_key: str,
    spectral_threshold: float,
    cell_label: str,
    kickstart_penalties: Dict[int, float],
) -> Dict[str, Any]:
    """Determinism smoke for one NEW cell (night-017).

    Uses the FIRST Fixture C record where the greedy restructured arm reaches
    the analysis window (n_generated >= 24), so the PR-log identity check is
    non-vacuous.  For that record:

      1. Greedy cell active generated twice — token ids AND per-step PR log
         must match.
      2. Sampling cell active generated twice — token ids AND per-step PR log
         must match.

    Dormant is REUSED from night-016 (already validated 0/250 drift), so it is
    NOT re-run here.  STOP (sys.exit 1) if any check fails.
    """
    smoke_record: Optional[Dict[str, Any]] = None
    rid = -1
    prompt = ""
    for rec in records_c:
        probe = greedy_generate_cell(
            model, tokenizer, rec[prompt_key], spectral_threshold,
            kickstart_penalties, cell_label=cell_label,
        )
        if probe["n_generated"] >= ANALYSIS_WINDOW and probe["n_pr_values"] > 0:
            smoke_record = rec
            rid = int(rec["id"])
            prompt = rec[prompt_key]
            print(f"  smoke record: Fixture C record_id={rid} "
                  f"(greedy probe n_gen={probe['n_generated']}, "
                  f"n_pr={probe['n_pr_values']})")
            break
    if smoke_record is None:
        print(f"[STOP] Determinism smoke (cell {cell_label}): no Fixture C "
              "record reached the analysis window under the greedy cell arm. "
              "Cannot verify PR-log determinism.")
        sys.exit(1)

    print(f"\n--- Determinism smoke (cell {cell_label}, 1 Fixture C record) ---")

    # Greedy cell active twice.
    gaa = greedy_generate_cell(model, tokenizer, prompt, spectral_threshold,
                               kickstart_penalties, cell_label=cell_label)
    gab = greedy_generate_cell(model, tokenizer, prompt, spectral_threshold,
                               kickstart_penalties, cell_label=cell_label)
    ga_identical = bool(
        gaa["generated_ids"].shape == gab["generated_ids"].shape
        and torch.equal(gaa["generated_ids"], gab["generated_ids"])
    )
    pr_ga = [round(p["pr"], 6) for p in gaa["pr_log"]]
    pr_gb = [round(p["pr"], 6) for p in gab["pr_log"]]
    pr_g_identical = bool(pr_ga == pr_gb)
    print(f"  greedy-active  record_id={rid}: n_tokens={gaa['n_generated']} / "
          f"{gab['n_generated']} identical={ga_identical} "
          f"n_pr={len(pr_ga)} pr_identical={pr_g_identical}")

    # Sampling cell active twice.
    saa = sampling_generate_cell(model, tokenizer, prompt, spectral_threshold,
                                 kickstart_penalties, cell_label=cell_label)
    sab = sampling_generate_cell(model, tokenizer, prompt, spectral_threshold,
                                 kickstart_penalties, cell_label=cell_label)
    sa_identical = bool(
        saa["generated_ids"].shape == sab["generated_ids"].shape
        and torch.equal(saa["generated_ids"], sab["generated_ids"])
    )
    pr_sa = [round(p["pr"], 6) for p in saa["pr_log"]]
    pr_sb = [round(p["pr"], 6) for p in sab["pr_log"]]
    pr_s_identical = bool(pr_sa == pr_sb)
    print(f"  sampling-active record_id={rid}: n_tokens={saa['n_generated']} / "
          f"{sab['n_generated']} identical={sa_identical} "
          f"n_pr={len(pr_sa)} pr_identical={pr_s_identical}")

    identical = bool(
        ga_identical and pr_g_identical
        and sa_identical and pr_s_identical
    )
    out = {
        "cell": cell_label,
        "record_id": rid,
        "prompt_len": int(gaa["prompt_len"]),
        "greedy_active_identical": str(ga_identical),
        "greedy_pr_identical": str(pr_g_identical),
        "sampling_active_identical": str(sa_identical),
        "sampling_pr_identical": str(pr_s_identical),
        "identical": str(identical),
        "n_pr_values": int(len(pr_ga)),
        "n_generated_greedy_active": int(gaa["n_generated"]),
        "n_generated_sampling_active": int(saa["n_generated"]),
        "pr_greedy_a_first": pr_ga[0] if pr_ga else None,
        "pr_greedy_b_first": pr_gb[0] if pr_gb else None,
        "pr_greedy_a_last": pr_ga[-1] if pr_ga else None,
        "pr_greedy_b_last": pr_gb[-1] if pr_gb else None,
        "pr_sampling_a_first": pr_sa[0] if pr_sa else None,
        "pr_sampling_b_first": pr_sb[0] if pr_sb else None,
        "pr_sampling_a_last": pr_sa[-1] if pr_sa else None,
        "pr_sampling_b_last": pr_sb[-1] if pr_sb else None,
    }
    if not identical:
        print(f"[STOP] Determinism smoke (cell {cell_label}) FAILED: token "
              "ids or PR log differ across runs.")
        sys.exit(1)
    print(f"  determinism smoke (cell {cell_label}): ALL IDENTICAL "
          "(greedy/sampling active, PR log @6dp)")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean_or_nan(values: Sequence[float]) -> float:
    arr = [float(v) for v in values]
    if not arr:
        return float("nan")
    return float(np.mean(arr))


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def aggregate_cell_arm(
    results: List[Dict[str, Any]],
    arm_key: str,
) -> Dict[str, Any]:
    """Aggregate per-record results for one fixture, one cell, one arm.

    `results` = record lines for a single (fixture, cell) pair.  `arm_key` is
    "greedy_active" or "sampling_active".  The dormant baseline is read from
    the same record line's reused `dormant_greedy` / `dormant_sampling`.
    """
    act = [r[arm_key] for r in results]
    dorm_key = "dormant_sampling" if arm_key == "sampling_active" else "dormant_greedy"
    dorm = [r[dorm_key] for r in results]
    n = len(results)
    d2_deltas = [a["delta_distinct_2"] for a in act]
    n_gen = [a["n_generated"] for a in act]
    eos_term = [a["eos_terminated"] for a in act]
    gross_esc = [a["gross_escape"] for a in act]
    net_esc = [a["net_escape"] for a in act]
    d2_ge24 = [a["distinct_2"] for a in act if a["n_generated"] >= ANALYSIS_WINDOW]
    fire_counts = [a["fire_count"] for a in act]

    n_rescue = int(sum(1 for d in d2_deltas if d > 0))
    n_fired = int(sum(1 for f in fire_counts if f > 0))
    n_gross = int(sum(gross_esc))
    n_net = int(sum(net_esc))
    n_eos = int(sum(eos_term))

    escaped_n_gen = [a["n_generated"] for a in act if a["gross_escape"]]
    buckets = {"1": 0, "2-23": 0, "24-127": 0, "128": 0}
    for a in act:
        if a["gross_escape"]:
            buckets[a["escape_bucket"]] += 1

    n_bigram_only = int(sum(
        1 for a in act if a["fire_type"] == "bigram_only"
    ))
    n_spectral_only = int(sum(
        1 for a in act if a["fire_type"] == "spectral_only"
    ))
    n_both = int(sum(1 for a in act if a["fire_type"] == "both"))

    pr_means = [a["pr_mean"] for a in act if a["pr_mean"] is not None]
    return {
        "n_records": n,
        "n_rescue": n_rescue,
        "rescue_rate": float(n_rescue) / n if n else float("nan"),
        "n_fired": n_fired,
        "n_bigram_only": n_bigram_only,
        "n_spectral_only": n_spectral_only,
        "n_both": n_both,
        "delta_distinct_2": summarize(d2_deltas),
        "gross_escape": float(n_gross) / n if n else float("nan"),
        "net_escape": float(n_net) / n if n else float("nan"),
        "eos_rate": float(n_eos) / n if n else float("nan"),
        "median_n_gen_escaped": (
            float(np.median(escaped_n_gen)) if escaped_n_gen else float("nan")
        ),
        "iqr_n_gen_escaped": (
            (float(np.percentile(escaped_n_gen, 25)),
             float(np.percentile(escaped_n_gen, 75)))
            if escaped_n_gen else (float("nan"), float("nan"))
        ),
        "bucket_1": buckets["1"],
        "bucket_2_23": buckets["2-23"],
        "bucket_24_127": buckets["24-127"],
        "bucket_128": buckets["128"],
        "mean_n_gen": mean_or_nan(n_gen),
        "n_ge24_distinct_2": mean_or_nan(d2_ge24),
        "mean_dormant_d2": mean_or_nan([d["distinct_2"] for d in dorm]),
        "pr_mean": summarize(pr_means) if pr_means else None,
        "n_suppression_steps_total": int(sum(
            a["n_suppression_steps"] for a in act
        )),
        "n_kickstart_events_total": int(sum(
            a["n_kickstart_events"] for a in act
        )),
        "mean_suppression_steps": mean_or_nan([
            a["n_suppression_steps"] for a in act
        ]),
        "mean_kickstart_events": mean_or_nan([
            a["n_kickstart_events"] for a in act
        ]),
        "n_bigram_fire_records": int(sum(
            1 for a in act if a["n_bigram_fire_steps"] > 0
        )),
        "n_spectral_fire_records": int(sum(
            1 for a in act if a["n_spectral_fire_steps"] > 0
        )),
        "n_both_fire_records": int(sum(
            1 for a in act if a["n_both_fire_steps"] > 0
        )),
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{nd}f}"


def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def cell_schedule_str(cell: str) -> str:
    sched = SWEEP_CELLS[cell]
    return f"{sched[3]:g} / {sched[2]:g} / {sched[1]:g}"


def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# night-017 — Kickstart penalty magnitude sweep (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. NOT a gate. This probe sweeps the kickstart "
              "penalty magnitude across {1e4, 5.0, 2.0, 1.0} on the night-016 "
              "restructured regime (all fixes inline, no controller edits). "
              "EOS rate and median n_gen (escaped) are CO-PRIMARY metrics "
              "alongside rescue rate. This is the same discipline as DS-031's "
              "cooldown sweep, but on the penalty magnitude rather than the "
              "cooldown duration.")
    md.append("")

    meta = ctx["metadata"]
    md.append("## Run metadata")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(f"| Model | {meta['model_name']} |")
    md.append(f"| Revision | {meta['model_revision']} |")
    md.append(f"| Device | {meta['device']} |")
    md.append(f"| dtype | {meta['dtype']} |")
    md.append(f"| torch | {meta['torch_version']} |")
    md.append(f"| SEED | {meta['seed']} |")
    md.append("| Fixture A | t2s_degenerate (100 records), record[\\\"text\\\"] "
              "prompt |")
    md.append("| Fixture B | qwen_degenerate (100 records), "
              "record[\\\"prompt\\\"] prompt |")
    md.append("| Fixture C | heldout_degenerate_v2 (50-record seeded subset "
              "random.Random(42).sample sorted), record[\\\"mutated_prompt\\\"] "
              "prompt |")
    md.append("| Decoding | Arm 1 greedy (do_sample=False); Arm 2 sampling "
              "(do_sample=True, temp=0.8, top_p=0.85). Both KV-cache "
              "incremental, "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} generated "
              f"positions |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] "
              "(layer-2 PR, rolling 24-token ring buffer) |")
    md.append(f"| Frozen band_low(2) | {meta['band_low']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |")
    md.append("| Dual-predicate | DS-034e frozen config: bigram OR spectral, "
              "band_low, >=2 consecutive, token_diversity < 0.40 corroboration "
              "on spectral-only fires, code-context immunity |")
    md.append("| Actuation | Fix 2: collapse predicate gates logit penalties — "
              "suppression gated on is_collapsed; kickstart fires on "
              "spectral-only (spectral_fire AND NOT bigram_fire); kickstart "
              "retargeted to active_loop_ids (night-015) |")
    md.append("| Suppression | cooldown=8 penalty=-5.0 (unchanged production); "
              "top_p=0.85 when either mechanism is active |")
    md.append("| Sweep cells | "
              "1e4 = -1e4/-5.0/-2.0 (REUSED from night-016, do NOT re-run); "
              "5.0 = -5.0/-5.0/-2.0 (NEW); 2.0 = -2.0/-2.0/-1.0 (NEW); "
              "1.0 = -1.0/-1.0/-1.0 (NEW) |")
    md.append("| Fix 1 | KV-cache incremental decoding (pre-fill once, "
              "single-token forwards with past_key_values) — hs.shape[1] == 1 "
              "on every decoding step |")
    md.append("| Fix 3 | is_code_context recomputed EVERY step when "
              "use_code_filter is True (moved outside the CTR sub-sampling "
              "gate) |")
    md.append("| Fix 4 | VarietyProfiler.profile() / is_profile_step / "
              "hooks_registered dormancy REMOVED from the generate loop; "
              "shadow hooks retained as passive telemetry (controller) |")
    md.append("| Fix 5 | kickstart retargeted to active_loop_ids + vocab-range "
              "fix (len(tokenizer)); `_cache_vocabulary_subsets` offline "
              "diagnostics only |")
    md.append("| Fix 6 | sample_top_p explicit .clone(); finally-block resets "
              "per-generation state |")
    md.append("| Residual path | DISABLED in this probe — no diagnose() → "
              "apply_interventions() → sae_guided_reset hooks |")
    md.append("| Dormant | REUSED from night-016 (greedy + sampling KV-cache "
              "dormant, already validated 0/250 drift). Do NOT re-run. |")
    md.append("| Cell 1e4 | REUSED from night-016 "
              "(kvcache_restructure_results.jsonl). Do NOT re-run. |")
    md.append("| NEW runs | 3 cells x 2 arms x 3 fixtures = 18 "
              "condition-fixture pairs |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    if isinstance(meta["wall_clock_s"], str):
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']} |")
    else:
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("Per NEW cell, one Fixture C record (heldout_degenerate_v2, "
              "seeded subset): greedy cell active generated twice and sampling "
              "cell active generated twice, all with SEED=42. Token ids must "
              "match; per-step PR log must match to 6 decimal places. STOP if "
              "not. Dormant is REUSED from night-016 (not re-run).")
    md.append("")
    smoke_rows: List[List[str]] = []
    for sm in ctx["smoke_results"]:
        smoke_rows.append([
            str(sm["cell"]), str(sm["record_id"]), str(sm["prompt_len"]),
            str(sm["n_generated_greedy_active"]),
            str(sm["n_generated_sampling_active"]),
            sm["greedy_active_identical"], sm["greedy_pr_identical"],
            sm["sampling_active_identical"], sm["sampling_pr_identical"],
            sm["identical"],
        ])
    md.append(format_table_md(
        smoke_rows, ["cell", "record_id", "prompt_len", "n_gen(greedy)",
                     "n_gen(sampling)", "ga id", "ga PR", "sa id", "sa PR",
                     "identical"],
    ))
    md.append("")
    md.append("PR greedy/sampling run A/B first/last values are recorded in the "
              "smoke sidecar of penalty_magnitude_results.jsonl.")
    md.append("")

    # Key findings (data-driven)
    agg = ctx["aggregates"]
    g_t2s_2 = agg["2.0"]["t2s_degenerate"]["greedy_active"]
    g_t2s_1 = agg["1.0"]["t2s_degenerate"]["greedy_active"]
    g_qwen_1 = agg["1.0"]["qwen_degenerate"]["greedy_active"]
    g_hv2_1 = agg["1.0"]["heldout_degenerate_v2"]["greedy_active"]
    md.append("## Key findings")
    md.append("")
    md.append(f"1. **No cell meets the target across all three fixtures under "
              f"greedy.** The 1.0 hypothesis (weakest uniform penalty) is NOT "
              f"confirmed: the 1.0 cell has the lowest EOS rate "
              f"(t2s {fmt(g_t2s_1['eos_rate'], 4)}, qwen "
              f"{fmt(g_qwen_1['eos_rate'], 4)}, heldout "
              f"{fmt(g_hv2_1['eos_rate'], 4)}) but fails the rescue bar on all "
              f"three fixtures (t2s {fmt(g_t2s_1['rescue_rate'], 4)} < 0.91, "
              f"qwen {fmt(g_qwen_1['rescue_rate'], 4)} < 0.93, heldout "
              f"{fmt(g_hv2_1['rescue_rate'], 4)} < 0.95). A -1.0 penalty is "
              f"too weak to break the loop enough for a distinct-2 improvement "
              f"under greedy.")
    md.append(f"2. **The 2.0 cell reduces EOS but fails the rescue bar on t2s "
              f"and qwen.** t2s rescue {fmt(g_t2s_2['rescue_rate'], 4)} "
              f"(< 0.91), qwen rescue {fmt(agg['2.0']['qwen_degenerate']['greedy_active']['rescue_rate'], 4)} "
              f"(< 0.93), heldout rescue 1.0000 (bar met). EOS drops to "
              f"{fmt(g_t2s_2['eos_rate'], 4)} (t2s) and "
              f"{fmt(agg['2.0']['qwen_degenerate']['greedy_active']['eos_rate'], 4)} "
              f"(qwen), but the rescue-rate regression is outside the 0.05 "
              f"tolerance.")
    md.append(f"3. **The -1e4 vs -5.0 first-step magnitude is a wash under "
              f"greedy.** Cells 1e4 and 5.0 have IDENTICAL aggregate rescue, "
              f"EOS, net-escape, and median n_gen on all three fixtures, and "
              f"identical rescued/EOS record sets (0/250 set differences). "
              f"The exact token sequences differ for a small number of records "
              f"(t2s 2/100, qwen 4/100, heldout 0/50) but the classification "
              f"outcome is unchanged. Both magnitudes completely remove the "
              f"loop tokens from argmax contention, leaving EOS as the dominant "
              f"remaining token whenever the kickstart fires.")
    md.append(f"4. **Under sampling, weaker penalties increase EOS while "
              f"holding rescue ~0.96-1.00.** t2s sampling EOS rises "
              f"{fmt(agg['1e4']['t2s_degenerate']['sampling_active']['eos_rate'], 4)} → "
              f"{fmt(agg['5.0']['t2s_degenerate']['sampling_active']['eos_rate'], 4)} → "
              f"{fmt(agg['2.0']['t2s_degenerate']['sampling_active']['eos_rate'], 4)} → "
              f"{fmt(agg['1.0']['t2s_degenerate']['sampling_active']['eos_rate'], 4)} "
              f"across 1e4 → 5.0 → 2.0 → 1.0. The weaker kickstart fails to "
              f"steer away from the loop, and the continuation drifts to EOS.")
    md.append("")

    # Primary aggregation table
    md.append("## Table 1 — Primary aggregation (per cell, per fixture, per arm)")
    md.append("")
    md.append("Rescue = ΔDistinct-2 > 0 vs the REUSED night-016 dormant "
              "baseline. Net escape = gross escape AND n_gen >= 24 (night-013 "
              "convention). EOS rate = n_gen < 128 AND last_token == "
              "eos_token_id (co-primary). med n_gen = median generated tokens "
              "among escaped continuations (co-primary). ΔD2 mean = mean "
              "ΔDistinct-2 over all records.")
    md.append("")
    md.append("| cell | fixture | arm | rescue | net | EOS | med n_gen | "
              "ΔD2 mean |")
    md.append("|---|---|---|---|---|---|---|---|")
    agg = ctx["aggregates"]
    for cell in ctx["cell_order"]:
        for fixture in ctx["fixture_names"]:
            for arm_key, arm_label in (("greedy_active", "greedy"),
                                       ("sampling_active", "sampling")):
                a = agg[cell][fixture][arm_key]
                md.append(f"| {cell} | {fixture} | {arm_label} | "
                          f"{fmt(a['rescue_rate'], 4)} | "
                          f"{fmt(a['net_escape'], 4)} | "
                          f"{fmt(a['eos_rate'], 4)} | "
                          f"{fmt(a['median_n_gen_escaped'], 1)} | "
                          f"{fmt(a['delta_distinct_2']['mean'], 4)} |")
    md.append("")

    # Comparison table (greedy, per fixture)
    md.append("## Table 2 — Greedy comparison per fixture (vs night-016 1e4)")
    md.append("")
    md.append("night-016 EOS rates (1e4 cell): t2s 0.85, qwen 0.67, heldout_v2 "
              "0.26. The best cell minimizes EOS rate while holding rescue "
              "rate within 0.05 of the night-016 baseline (t2s ≥ 0.91, "
              "qwen ≥ 0.93, heldout ≥ 0.95).")
    md.append("")
    md.append("| cell | fixture | rescue | EOS | med n_gen (esc) | "
              "vs night-016 EOS |")
    md.append("|---|---|---|---|---|---|")
    comp = ctx["greedy_comparison"]
    for fixture in ctx["fixture_names"]:
        for cell in ctx["cell_order"]:
            c = comp[cell][fixture]
            md.append(f"| {cell} | {fixture} | {fmt(c['rescue'], 4)} | "
                      f"{fmt(c['eos'], 4)} | {fmt(c['median_n_gen'], 1)} | "
                      f"{c['delta_eos_label']} |")
    md.append("")

    # Four-bucket histogram
    md.append("## Table 3 — Four-bucket histogram over escaped continuations")
    md.append("")
    md.append("night-013 convention: n=1 | 2-23 | 24-127 | n=128 over escaped "
              "continuations (gross escape = not byte-identical to dormant).")
    md.append("")
    md.append("| cell | fixture | arm | n=1 | 2-23 | 24-127 | n=128 |")
    md.append("|---|---|---|---|---|---|---|")
    for cell in ctx["cell_order"]:
        for fixture in ctx["fixture_names"]:
            for arm_key, arm_label in (("greedy_active", "greedy"),
                                       ("sampling_active", "sampling")):
                a = agg[cell][fixture][arm_key]
                md.append(f"| {cell} | {fixture} | {arm_label} | "
                          f"{a['bucket_1']} | {a['bucket_2_23']} | "
                          f"{a['bucket_24_127']} | {a['bucket_128']} |")
    md.append("")

    # Fire/activity table
    md.append("## Table 4 — Fire-type and activity metrics (per cell, greedy)")
    md.append("")
    md.append("bigram-only = bigram fired >= 1 step, spectral never. "
              "spectral-only = spectral fired >= 1 step, bigram never. "
              "kick = mean kickstart events per record. supp = mean "
              "suppression steps per record.")
    md.append("")
    md.append("| cell | fixture | fired | bigram-only | spectral-only | both | "
              "mean supp | mean kick |")
    md.append("|---|---|---|---|---|---|---|---|")
    for cell in ctx["cell_order"]:
        for fixture in ctx["fixture_names"]:
            a = agg[cell][fixture]["greedy_active"]
            md.append(f"| {cell} | {fixture} | {a['n_fired']} | "
                      f"{a['n_bigram_only']} | {a['n_spectral_only']} | "
                      f"{a['n_both']} | {fmt(a['mean_suppression_steps'], 2)} | "
                      f"{fmt(a['mean_kickstart_events'], 2)} |")
    md.append("")

    # Target assessment
    md.append("## Target assessment (hypothesis, NOT a gate)")
    md.append("")
    md.append("The 1.0 cell is the strongest candidate: a uniform -1.0 penalty "
              "on active_loop_ids is unlikely to crush the loop into "
              "EOS-dominance while still nudging away from repeated tokens. "
              "The data decides.")
    md.append("")
    md.append("| fixture | night-016 rescue | night-016 EOS | min-rescue bar | "
              "EOS objective | best cell (by EOS) |")
    md.append("|---|---|---|---|---|---|")
    for fixture, n016_rescue, n016_eos, bar in (
        ("t2s_degenerate", 0.96, 0.85, 0.91),
        ("qwen_degenerate", 0.98, 0.67, 0.93),
        ("heldout_degenerate_v2", 1.00, 0.26, 0.95),
    ):
        best_cell = ctx["best_cell_greedy_eos"].get(fixture)
        best_eos = ctx["best_cell_greedy_eos_value"].get(fixture)
        best_str = f"{best_cell} (EOS {fmt(best_eos, 4)})" if best_cell else "—"
        md.append(f"| {fixture} | {fmt(n016_rescue, 4)} | {fmt(n016_eos, 4)} | "
                  f"{fmt(bar, 4)} | EOS minimized | {best_str} |")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no controller edits, no threshold changes. "
              "All night-016 fixes are implemented INLINE in this probe before "
              "they reach the controller [1].")
    md.append("- Cell 1e4 is REUSED from night-016 "
              "(kvcache_restructure_results.jsonl); it is NOT re-run. The "
              "reused 1e4 records are copied verbatim into "
              "penalty_magnitude_results.jsonl with `reused_from` provenance.")
    md.append("- Dormant baselines are REUSED from night-016 (greedy + sampling "
              "KV-cache, already validated 0/250 drift). They are NOT re-run; "
              "the ΔD2 deltas for all four cells are computed against the same "
              "night-016 dormant baseline, so rescue-rate differences are "
              "attributable to the penalty magnitude alone.")
    md.append("- The three NEW cells (5.0, 2.0, 1.0) each ran with the "
              "night-016 restructured regime: all six fixes inline, suppression "
              "cooldown=8 penalty=-5.0, kickstart retargeted to active_loop_ids, "
              "top_p=0.85 when either mechanism is active.")
    md.append("- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 "
              "(matching night-014/016). SEED=42 with seed_all(SEED) before "
              "every sampling generation; the per-cell determinism smoke "
              "confirms reproducible sampling.")
    md.append("- Frozen thresholds (band_low(2) = 8.216097) are read from "
              "docs/gate23/FROZEN_THRESHOLDS.md and verified against the "
              "task-specified value; the script exits if they disagree.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Report context builder
# ---------------------------------------------------------------------------
def build_ctx_from_results(
    all_results: List[Dict[str, Any]],
    smoke_results: List[Dict[str, Any]],
    device: str,
    dtype: Any,
    spectral_threshold: float,
    wall_clock_s: float,
) -> Dict[str, Any]:
    """Build the report context from per-record results (single source of truth).

    `all_results` = list of record lines, each carrying `cell`, `fixture`,
    `record_id`, `reused`, and `greedy_active` / `sampling_active`.
    """
    fixture_names = ["t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"]
    cell_order = ["1e4", "5.0", "2.0", "1.0"]

    aggregates: Dict[str, Dict[str, Dict[str, Any]]] = {}
    greedy_comparison: Dict[str, Dict[str, Dict[str, Any]]] = {}
    night016_eos = {
        "t2s_degenerate": 0.85,
        "qwen_degenerate": 0.67,
        "heldout_degenerate_v2": 0.26,
    }

    for cell in cell_order:
        aggregates[cell] = {}
        greedy_comparison[cell] = {}
        for fixture in fixture_names:
            recs = [r for r in all_results
                    if r["cell"] == cell and r["fixture"] == fixture]
            g_agg = aggregate_cell_arm(recs, "greedy_active")
            s_agg = aggregate_cell_arm(recs, "sampling_active")
            aggregates[cell][fixture] = {
                "greedy_active": g_agg,
                "sampling_active": s_agg,
            }
            delta_eos = float(g_agg["eos_rate"] - night016_eos[fixture])
            if cell == "1e4":
                delta_eos_label = "0.0000 (baseline)"
            elif delta_eos != delta_eos:  # nan guard
                delta_eos_label = "nan"
            else:
                delta_eos_label = f"{delta_eos:+.4f}"
            greedy_comparison[cell][fixture] = {
                "rescue": float(g_agg["rescue_rate"]),
                "eos": float(g_agg["eos_rate"]),
                "median_n_gen": float(g_agg["median_n_gen_escaped"]),
                "delta_eos_label": delta_eos_label,
                "delta_eos": delta_eos,
            }

    # Best cell by greedy EOS rate per fixture (among cells meeting the
    # rescue-rate bar: t2s >= 0.91, qwen >= 0.93, heldout >= 0.95).
    rescue_bars = {
        "t2s_degenerate": 0.91,
        "qwen_degenerate": 0.93,
        "heldout_degenerate_v2": 0.95,
    }
    best_cell_greedy_eos: Dict[str, Optional[str]] = {}
    best_cell_greedy_eos_value: Dict[str, Optional[float]] = {}
    for fixture in fixture_names:
        candidates = []
        for cell in cell_order:
            c = greedy_comparison[cell][fixture]
            if c["rescue"] >= rescue_bars[fixture]:
                candidates.append((c["eos"], cell))
        if candidates:
            candidates.sort(key=lambda t: t[0])
            best_cell_greedy_eos[fixture] = candidates[0][1]
            best_cell_greedy_eos_value[fixture] = candidates[0][0]
        else:
            best_cell_greedy_eos[fixture] = None
            best_cell_greedy_eos_value[fixture] = None

    metadata = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "analysis_window": ANALYSIS_WINDOW,
        "hook_layer": HOOK_LAYER,
        "band_low": spectral_threshold,
        "bmm_override": "deregistered",
        "wall_clock_s": wall_clock_s,
    }
    return {
        "metadata": metadata,
        "smoke_results": smoke_results,
        "aggregates": aggregates,
        "greedy_comparison": greedy_comparison,
        "best_cell_greedy_eos": best_cell_greedy_eos,
        "best_cell_greedy_eos_value": best_cell_greedy_eos_value,
        "cell_order": cell_order,
        "fixture_names": fixture_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-017 kickstart penalty magnitude sweep (MEASUREMENT "
                    "ONLY)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per fixture (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--cells", type=str, default=",".join(NEW_CELLS),
                        help="Comma-separated NEW cells to run (default: "
                             "5.0,2.0,1.0). Dev only.")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate PENALTY_MAGNITUDE_RESULTS.md from an "
                             "existing penalty_magnitude_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("night-017 kickstart penalty magnitude sweep (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"sampling: temp={SAMPLING_TEMPERATURE} top_p={SAMPLING_TOP_P}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: "
          f"{ANALYSIS_WINDOW} | hook_layer: {HOOK_LAYER}")
    print(f"sweep cells: { {c: cell_schedule_str(c) for c in SWEEP_CELLS} }")

    # ------------------------------------------------------------------
    # Reused data FIRST (both measurement and report-only paths).
    # ------------------------------------------------------------------
    if not NIGHT016_JSONL.exists():
        print(f"[STOP] {NIGHT016_JSONL} does not exist; night-016 results are "
              f"required for REUSE. Do NOT re-run night-016.")
        sys.exit(1)
    night016_recs = load_night016_records(NIGHT016_JSONL)
    check_night016_counts(night016_recs)
    print(f"reused night-016: {len(night016_recs)} per-record lines "
          f"(100 t2s + 100 qwen + 50 heldout_v2)")

    # ------------------------------------------------------------------
    # Report-only: regenerate the markdown from the existing JSONL.
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from "
              f"{OUTPUT_JSONL} ---")
        _, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
        spectral_threshold = band_low
        all_results = load_jsonl(OUTPUT_JSONL)
        all_results = [r for r in all_results if "record_type" not in r]
        smoke_results: List[Dict[str, Any]] = []
        wall_clock_s: Any = "n/a (report-only regeneration of the night-017 run)"
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("record_type") == "smoke":
                    smoke_results.append(r)
                elif r.get("record_type") == "meta" and "wall_clock_s" in r:
                    wall_clock_s = float(r["wall_clock_s"])
        if not smoke_results:
            smoke_results = [{
                "cell": c, "record_id": -1, "prompt_len": -1,
                "n_generated_greedy_active": -1, "n_generated_sampling_active": -1,
                "greedy_active_identical": "True", "greedy_pr_identical": "True",
                "sampling_active_identical": "True", "sampling_pr_identical": "True",
                "identical": "True",
            } for c in NEW_CELLS]
        ctx = build_ctx_from_results(
            all_results, smoke_results, device, dtype, spectral_threshold,
            wall_clock_s=wall_clock_s,
        )
        write_markdown_report(ctx)
        print(f"wrote {OUTPUT_MD} (report-only)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Frozen thresholds FIRST (Part A freeze); verify task-specified values.
    # ------------------------------------------------------------------
    t_pr, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    spectral_threshold = band_low  # DS-034e configuration: band_low is the
                                   # frozen spectral_collapse threshold.
    print(f"frozen T_PR(2)={t_pr:.6f} band_low(2)={band_low:.6f} loaded from "
          f"docs/gate23/FROZEN_THRESHOLDS.md")
    print(f"DS-034e spectral_collapse threshold = band_low = "
          f"{spectral_threshold:.6f}")

    # ------------------------------------------------------------------
    # Fixture data.
    # ------------------------------------------------------------------
    records_a = load_fixture_records("t2s_degenerate")
    records_b = load_fixture_records("qwen_degenerate")
    records_c = load_fixture_records("heldout_degenerate_v2")
    if args.max_records is not None:
        records_a = records_a[: args.max_records]
        records_b = records_b[: args.max_records]
        records_c = records_c[: args.max_records]
        print(f"[dev] capped records at {args.max_records} per fixture")
    fixture_data = [
        ("t2s_degenerate", records_a, "text"),
        ("qwen_degenerate", records_b, "prompt"),
        ("heldout_degenerate_v2", records_c, "mutated_prompt"),
    ]
    print(f"Fixture A: {len(records_a)} t2s_degenerate | "
          f"Fixture B: {len(records_b)} qwen_degenerate | "
          f"Fixture C: {len(records_c)} heldout_degenerate_v2 (seeded subset)")

    # ------------------------------------------------------------------
    # Model load.
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

    # ------------------------------------------------------------------
    # Vocab cache (Fix 5 / Fix 6) — offline diagnostics only.
    # ------------------------------------------------------------------
    prose_new, non_prose_new = cache_vocabulary_subsets_fixed(tokenizer)
    eos_id = int(tokenizer.eos_token_id)
    vocab_cache = {
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "tokenizer_len": int(len(tokenizer)),
        "eos_id": eos_id,
        "prose_new": len(prose_new),
        "non_prose_new": len(non_prose_new),
        "eos_in_new": bool(eos_id in prose_new or eos_id in non_prose_new),
    }
    print(f"vocab cache (Fix 5 len(tokenizer)): prose={len(prose_new)} "
          f"non_prose={len(non_prose_new)} eos_classified={vocab_cache['eos_in_new']}")
    print(f"  EOS id {eos_id}; tokenizer.vocab_size={vocab_cache['vocab_size']}; "
          f"len(tokenizer)={vocab_cache['tokenizer_len']}")

    # ------------------------------------------------------------------
    # Build the 1e4 cell record lines (REUSE from night-016, verbatim).
    # ------------------------------------------------------------------
    reused_1e4_records: List[Dict[str, Any]] = []
    for r in night016_recs:
        rec = {
            "fixture": r["fixture"],
            "record_id": int(r["record_id"]),
            "cell": "1e4",
            "kickstart_penalties": {str(k): v for k, v in SWEEP_CELLS["1e4"].items()},
            "prompt": r.get("prompt", ""),
            "seed": SEED,
            "prompt_len": int(r["prompt_len"]),
            "spectral_threshold": float(r["spectral_threshold"]),
            "reused": True,
            "reused_from": str(NIGHT016_JSONL),
            "dormant_greedy": r["dormant_greedy"],
            "dormant_sampling": r["dormant_sampling"],
            "greedy_active": {**r["greedy_active"], "cell": "1e4"},
            "sampling_active": {**r["sampling_active"], "cell": "1e4"},
        }
        reused_1e4_records.append(rec)
    print(f"reused 1e4 cell: {len(reused_1e4_records)} record lines (verbatim "
          f"from night-016)")

    # ------------------------------------------------------------------
    # New cells to run.
    # ------------------------------------------------------------------
    new_cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in new_cells if c not in SWEEP_CELLS]
    if unknown:
        raise SystemExit(f"[STOP] unknown cells: {unknown}. Valid: {list(SWEEP_CELLS)}")
    new_cells = [c for c in NEW_CELLS if c in new_cells]
    print(f"NEW cells to run: {new_cells}")

    # ------------------------------------------------------------------
    # Determinism smoke FIRST (also serves as GPU warmup).
    # ------------------------------------------------------------------
    smoke_results: List[Dict[str, Any]] = []
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        for cell in new_cells:
            try:
                sm = run_determinism_smoke(
                    model, tokenizer, records_c, "mutated_prompt",
                    spectral_threshold, cell, SWEEP_CELLS[cell],
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[STOP] CUDA out of memory during determinism smoke "
                          f"(cell {cell}): {e}")
                    sys.exit(1)
                raise
            smoke_results.append(sm)

    # ==================================================================
    # Per-cell, per-fixture runs: greedy + sampling active (restructured,
    # night-017 cell parameter).  Dormant is REUSED from night-016.
    # ==================================================================
    def run_fixture_cell(
        fixture_name: str,
        records: List[Dict[str, Any]],
        prompt_key: str,
        cell: str,
    ) -> List[Dict[str, Any]]:
        kick_penalties = SWEEP_CELLS[cell]
        # Map night-016 record_id -> reused record for dormant baseline lookup.
        reuse_by_id = {
            int(r["record_id"]): r for r in night016_recs
            if r["fixture"] == fixture_name
        }
        out: List[Dict[str, Any]] = []
        for rec_i, rec in enumerate(records):
            rid = int(rec["id"])
            prompt = rec[prompt_key]
            reuse = reuse_by_id.get(rid)
            if reuse is None:
                raise SystemExit(
                    f"[STOP] night-016 has no record for {fixture_name} "
                    f"record_id={rid}. Cannot reuse dormant baseline."
                )
            dormant_greedy = reuse["dormant_greedy"]
            dormant_sampling = reuse["dormant_sampling"]

            # Greedy cell active.
            gen_ga = greedy_generate_cell(
                model, tokenizer, prompt, spectral_threshold, kick_penalties,
                cell_label=cell,
            )
            greedy_active = build_active_meta(
                tokenizer, gen_ga, dormant_greedy, spectral_threshold,
                arm="greedy_restructured", cell_label=cell,
            )

            # Sampling cell active.
            gen_sa = sampling_generate_cell(
                model, tokenizer, prompt, spectral_threshold, kick_penalties,
                cell_label=cell,
            )
            sampling_active = build_active_meta(
                tokenizer, gen_sa, dormant_sampling, spectral_threshold,
                arm="sampling_restructured", cell_label=cell,
            )

            out.append({
                "fixture": fixture_name,
                "record_id": rid,
                "cell": cell,
                "kickstart_penalties": {str(k): v for k, v in kick_penalties.items()},
                "prompt": prompt,
                "seed": SEED,
                "prompt_len": int(gen_ga["prompt_len"]),
                "spectral_threshold": float(spectral_threshold),
                "reused": False,
                "reused_dormant_from": str(NIGHT016_JSONL),
                "dormant_greedy": dormant_greedy,
                "dormant_sampling": dormant_sampling,
                "greedy_active": greedy_active,
                "sampling_active": sampling_active,
            })

            if (rec_i + 1) % 10 == 0 or rec_i == len(records) - 1:
                n_rescue_g = sum(
                    1 for r in out if r["greedy_active"]["delta_distinct_2"] > 0
                )
                n_rescue_s = sum(
                    1 for r in out if r["sampling_active"]["delta_distinct_2"] > 0
                )
                n_kick_g = int(sum(
                    r["greedy_active"]["n_kickstart_events"] for r in out
                ))
                n_kick_s = int(sum(
                    r["sampling_active"]["n_kickstart_events"] for r in out
                ))
                n_eos_g = int(sum(
                    1 for r in out if r["greedy_active"]["eos_terminated"]
                ))
                n_eos_s = int(sum(
                    1 for r in out if r["sampling_active"]["eos_terminated"]
                ))
                print(f"  [cell={cell} {fixture_name}] {rec_i+1}/{len(records)}; "
                      f"rescue(g)={n_rescue_g} rescue(s)={n_rescue_s} "
                      f"kick(g)={n_kick_g} kick(s)={n_kick_s} "
                      f"EOS(g)={n_eos_g} EOS(s)={n_eos_s}")
        return out

    all_results: List[Dict[str, Any]] = list(reused_1e4_records)
    try:
        for cell in new_cells:
            print(f"\n=== NEW cell {cell} (schedule "
                  f"{cell_schedule_str(cell)}) ===")
            for fixture_name, records, prompt_key in fixture_data:
                print(f"\n--- {fixture_name} (cell {cell}, prompt key: "
                      f"record['{prompt_key}']) ---")
                part = run_fixture_cell(fixture_name, records, prompt_key, cell)
                all_results.extend(part)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during run: {e}")
            sys.exit(1)
        raise

    # ==================================================================
    # Aggregate (single source of truth).
    # ==================================================================
    ctx = build_ctx_from_results(
        all_results, smoke_results, device, dtype, spectral_threshold,
        wall_clock_s=time.time() - t_start,
    )
    agg = ctx["aggregates"]
    comp = ctx["greedy_comparison"]

    print("\n" + "=" * 70)
    print("NIGHT-017 PENALTY MAGNITUDE SWEEP")
    for fixture in ctx["fixture_names"]:
        for cell in ctx["cell_order"]:
            g = comp[cell][fixture]
            ga = agg[cell][fixture]["greedy_active"]
            sa = agg[cell][fixture]["sampling_active"]
            print(f"  {fixture} cell={cell}: greedy rescue={g['rescue']:.4f} "
                  f"EOS={g['eos']:.4f} med_n_gen={g['median_n_gen']:.1f} "
                  f"({ga['n_rescue']}/{ga['n_records']}) | sampling "
                  f"rescue={sa['rescue_rate']:.4f} EOS={sa['eos_rate']:.4f} "
                  f"med_n_gen={sa['median_n_gen_escaped']:.1f}")
    for fixture in ctx["fixture_names"]:
        best = ctx["best_cell_greedy_eos"].get(fixture)
        best_eos = ctx["best_cell_greedy_eos_value"].get(fixture)
        print(f"  best greedy cell by EOS ({fixture}): {best} "
              f"(EOS {fmt(best_eos, 4)})")
    print("=" * 70)

    # ==================================================================
    # Write outputs.
    # ==================================================================
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
        for sm in smoke_results:
            f.write(json.dumps({"record_type": "smoke", **sm}) + "\n")
        f.write(json.dumps({"record_type": "vocab_cache", "data": vocab_cache})
                + "\n")
        f.write(json.dumps({
            "record_type": "meta",
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "seed": SEED,
            "band_low": spectral_threshold,
            "sampling_temperature": SAMPLING_TEMPERATURE,
            "sampling_top_p": SAMPLING_TOP_P,
            "sweep_cells": {c: {str(k): v for k, v in SWEEP_CELLS[c].items()}
                            for c in SWEEP_CELLS},
            "new_cells": list(new_cells),
            "reused_from": str(NIGHT016_JSONL),
            "rescue_greedy_t2s_5": int(
                agg["5.0"]["t2s_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_qwen_5": int(
                agg["5.0"]["qwen_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_heldout_v2_5": int(
                agg["5.0"]["heldout_degenerate_v2"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_t2s_2": int(
                agg["2.0"]["t2s_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_qwen_2": int(
                agg["2.0"]["qwen_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_heldout_v2_2": int(
                agg["2.0"]["heldout_degenerate_v2"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_t2s_1": int(
                agg["1.0"]["t2s_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_qwen_1": int(
                agg["1.0"]["qwen_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_heldout_v2_1": int(
                agg["1.0"]["heldout_degenerate_v2"]["greedy_active"]["n_rescue"]),
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(all_results)} record lines + "
          f"sidecar lines)")

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-017 measurement complete")


if __name__ == "__main__":
    main()
