#!/usr/bin/env python3
"""night-018: Suppression-only, kickstart fully disabled (MEASUREMENT ONLY).

Context
-------
night-014 Arm 2 measured suppression-only behavior (kickstart never fired
under sampling): heldout_v2 net escape 0.96, EOS 0.34. night-016 (merged)
restructured to KV-cache with the collapse predicate gating logit penalties,
but the spectral-gated kickstart (-1e4/-5.0/-2.0) drove EOS rates of
0.67-0.85 under greedy. night-017 (merged) swept the kickstart penalty
magnitude and found a cliff: 1e4 and 5.0 are identical (both crush the loop
and leave EOS dominant), while 1.0 cuts EOS but collapses rescue. No static
magnitude escapes the rescue-vs-EOS tradeoff.

This probe tests the cleanest remaining configuration: kickstart FULLY
DISABLED (no _kickstart_counter, no kickstart penalty path at all) with
token suppression as the sole actuator. If suppression-only holds rescue
while dropping EOS to the floor, kickstart can be removed from the
production controller entirely.

MEASUREMENT ONLY.  No controller edits — fixes implemented inline.  No
threshold changes [1].

Configuration
-------------
- Kickstart FULLY DISABLED: `_kickstart_counter` is never set; the kickstart
  penalty block is removed from the generation loop.  The
  `_cache_vocabulary_subsets` call is not needed at runtime (offline
  diagnostics only per night-015 Fix 5).
- Suppression: unconditional token suppression at EVERY step where
  `active_loop_ids` is non-empty (cooldown=8, penalty=-5.0).  This is the
  production behavior — NOT gated on is_collapsed.  This matches night-014's
  suppression path, which produced net escape 0.96 on heldout_v2.
- Fix 1 (KV-cache incremental decoding) retained: hs.shape[1] == 1 every
  step, spectral PR hook fires correctly.
- Fix 3 (is_code_context every step) retained.
- Fix 4 (profiler removed, shadow hooks retained) retained.
- Fix 6 (sample_top_p clone, finally-block resets) retained.
- Residual path: disabled (no diagnose -> apply_interventions hooks).

Dual-predicate detection (unchanged, detection-only)
----------------------------------------------------
The collapse predicate runs but does NOT gate actuation in this probe —
suppression is unconditional.  The predicate is logged for fire-type
breakdown only.

Reused baselines (do NOT re-run):
  - night-014 Arm 2: docs/gate23/sampling_dual_predicate_results.jsonl
  - night-016:       docs/gate23/kvcache_restructure_results.jsonl
  - Dormant:         REUSE from night-016 (greedy + sampling KV-cache
                     dormant, validated 0/250 drift).  Do NOT re-run.

NEW live runs: 2 arms (greedy, sampling) x 3 fixtures = 6
condition-fixture pairs.

Environment notes (identical to night-014..night-017):
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
SEED = 42  # night-014 measurement seed (matches ds-025..ds-035, night-015/016/017)
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
NIGHT014_JSONL = Path("docs/gate23/sampling_dual_predicate_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/suppression_only_results.jsonl")
OUTPUT_MD = Path("docs/gate23/SUPPRESSION_ONLY_RESULTS.md")

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
# Suppression-only active generation core (kickstart FULLY DISABLED).
# ---------------------------------------------------------------------------
def _suppression_only_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    do_sample: bool,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """KV-cache incremental generation under the night-018 suppression-only rules.

    All retained night-016 inline fixes are implemented here:

      Fix 1 — KV-cache incremental decoding (pre-fill once, then single-token
              forwards with past_key_values).  `hs.shape[1] == 1` on every
              decoding step → the layer-2 PR hook condition fires correctly.
      Fix 3 — is_code_context recomputed EVERY step when USE_CODE_FILTER is
              True (moved outside the CTR sub-sampling gate).
      Fix 4 — No VarietyProfiler.profile() call, no is_profile_step logic, no
              hooks_registered dormancy management in the loop.
      Fix 6 — sample_top_p_restructured (explicit .clone()); finally block
              removes the layer-2 hook and clears per-generation state.

    night-018 changes:
      - Kickstart FULLY DISABLED: `_kickstart_counter` is never set; the
        kickstart penalty block is REMOVED from the loop.
      - Suppression is UNCONDITIONAL: at EVERY step where `active_loop_ids` is
        non-empty, add ALL active_loop_ids to the cooldown dict (cooldown=8,
        penalty=-5.0).  NOT gated on is_collapsed.  This matches night-014's
        suppression path (net escape 0.96 on heldout_v2).
      - The dual-predicate collapse predicate runs DETECTION-ONLY: it is
        logged for fire-type breakdown but does NOT gate actuation.

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
    bigram_ctr = 0
    spectral_ctr = 0
    layer2_pr: Optional[float] = None

    fire_events: List[Dict[str, Any]] = []
    n_suppression_steps = 0
    n_kickstart_events = 0  # always 0: kickstart fully disabled
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

            # ---- 5. Actuation: UNCONDITIONAL token suppression ----
            # Production combined path (controller.py:677-679): add ALL
            # active_loop_ids to the cooldown dict at EVERY step where they are
            # detected, regardless of predicate state.  night-018: kickstart is
            # FULLY DISABLED (no _kickstart_counter, no kickstart penalty).
            if active_loop_ids:
                for tid in active_loop_ids:
                    active_suppress[tid] = COOLDOWN

            # ---- 6. Fire-event logging (dual-predicate detection liveness;
            #             does NOT gate actuation in this probe) ----
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

            # ---- 7. Logit-penalty path (suppression only; no kickstart) ----
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

            if active_suppress:
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
    }


@torch.no_grad()
def greedy_generate_suppression_only(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy suppression-only active generation (kickstart disabled)."""
    return _suppression_only_generate(
        model, tokenizer, prompt, spectral_threshold,
        do_sample=False, max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def sampling_generate_suppression_only(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Sampling suppression-only active generation (kickstart disabled)."""
    return _suppression_only_generate(
        model, tokenizer, prompt, spectral_threshold,
        do_sample=True, max_new_tokens=max_new_tokens,
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
) -> Dict[str, Any]:
    """Build the active-record meta dict for one suppression-only generation.

    `dormant` is the REUSED night-016 dormant baseline for the same
    record+arm (greedy or sampling).
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
# Reused data loaders (night-016 + night-014, do NOT re-run)
# ---------------------------------------------------------------------------
def load_night016_records(path: Path) -> List[Dict[str, Any]]:
    """Load the per-record lines of night-016 (kvcache_restructure_results.jsonl).

    STOPs if the file is missing or has no per-record lines.  Each record is
    expected to carry `dormant_greedy`, `dormant_sampling`, `greedy_active`,
    and `sampling_active` (the -1e4/-5.0/-2.0 kickstart cell).
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


def load_night014_arm2_rates(path: Path) -> Dict[str, Dict[str, float]]:
    """Load night-014 Arm 2 (spectral-gated kickstart) per-fixture rates.

    Returns {fixture: {"rescue": float, "eos": float}}.  STOPs if any fixture
    is missing Arm 2 records (night-014 schema `arm2_active`).
    """
    if not path.exists():
        raise SystemExit(
            f"[STOP] {path} does not exist; night-014 results are required for "
            f"REUSE. Do NOT re-run night-014."
        )
    fixtures = ("t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2")
    counts: Dict[str, Dict[str, int]] = {}
    n_eos: Dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type"):
                continue
            fixture = r.get("fixture")
            if fixture not in fixtures:
                continue
            a2 = r.get("arm2_active")
            if a2 is None:
                raise SystemExit(
                    f"[STOP] {path} record {r.get('record_id')} has no "
                    f"arm2_active; night-014 schema changed unexpectedly."
                )
            rec = counts.setdefault(fixture, {"n": 0, "n_rescue": 0})
            rec["n"] += 1
            if a2.get("delta_distinct_2", 0.0) > 0:
                rec["n_rescue"] += 1
            if a2.get("eos_terminated", False):
                n_eos[fixture] = n_eos.get(fixture, 0) + 1
    rates: Dict[str, Dict[str, float]] = {}
    for fixture in fixtures:
        rec = counts.get(fixture)
        if rec is None or rec["n"] == 0:
            raise SystemExit(
                f"[STOP] night-014 has no Arm 2 records for fixture {fixture}."
            )
        rates[fixture] = {
            "rescue": float(rec["n_rescue"]) / float(rec["n"]),
            "eos": float(n_eos.get(fixture, 0)) / float(rec["n"]),
        }
    return rates


def load_fixture_records(fixture_name: str) -> List[Dict[str, Any]]:
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
) -> Dict[str, Any]:
    """Determinism smoke (night-018): one Fixture C record.

    Uses the FIRST Fixture C record where the greedy suppression-only arm
    reaches the analysis window (n_generated >= 24), so the PR-log identity
    check is non-vacuous.  For that record:

      1. Greedy suppression-only generated twice — token ids AND per-step PR
         log must match.
      2. Sampling suppression-only generated twice — token ids AND per-step PR
         log must match.

    Dormant is REUSED from night-016 (already validated 0/250 drift), so it is
    NOT re-run here.  STOP (sys.exit 1) if any check fails.
    """
    smoke_record: Optional[Dict[str, Any]] = None
    rid = -1
    prompt = ""
    for rec in records_c:
        probe = greedy_generate_suppression_only(
            model, tokenizer, rec[prompt_key], spectral_threshold,
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
        print("[STOP] Determinism smoke: no Fixture C record reached the "
              "analysis window under the greedy suppression-only arm. Cannot "
              "verify PR-log determinism.")
        sys.exit(1)

    print("\n--- Determinism smoke (1 Fixture C record) ---")

    # Greedy suppression-only twice.
    gaa = greedy_generate_suppression_only(model, tokenizer, prompt, spectral_threshold)
    gab = greedy_generate_suppression_only(model, tokenizer, prompt, spectral_threshold)
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

    # Sampling suppression-only twice.
    saa = sampling_generate_suppression_only(model, tokenizer, prompt, spectral_threshold)
    sab = sampling_generate_suppression_only(model, tokenizer, prompt, spectral_threshold)
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
        print("[STOP] Determinism smoke FAILED: token ids or PR log differ "
              "across runs.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (greedy/sampling active, "
          "PR log @6dp)")
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


def aggregate_arm(
    results: List[Dict[str, Any]],
    arm_key: str,
) -> Dict[str, Any]:
    """Aggregate per-record results for one fixture and one arm.

    `results` = record lines for a single fixture.  `arm_key` is
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


def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# night-018 — Suppression-only, kickstart fully disabled "
              "(MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. NOT a gate. This probe tests the cleanest "
              "remaining configuration: kickstart FULLY DISABLED (no "
              "_kickstart_counter, no kickstart penalty path at all) with token "
              "suppression as the sole actuator. If suppression-only holds "
              "rescue while dropping EOS to the floor, kickstart can be removed "
              "from the production controller entirely. All fixes are "
              "implemented INLINE; no controller edits [1].")
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
              "on spectral-only fires, code-context immunity. Runs "
              "DETECTION-ONLY — logged for fire-type breakdown, does NOT gate "
              "actuation. |")
    md.append("| Actuation | UNCONDITIONAL token suppression at EVERY step "
              "where active_loop_ids is non-empty (cooldown=8, penalty=-5.0). "
              "Kickstart FULLY DISABLED. |")
    md.append("| Suppression | cooldown=8 penalty=-5.0 (unchanged production); "
              "top_p=0.85 when suppression is active |")
    md.append("| Fix 1 | KV-cache incremental decoding (pre-fill once, "
              "single-token forwards with past_key_values) — hs.shape[1] == 1 "
              "on every decoding step |")
    md.append("| Fix 3 | is_code_context recomputed EVERY step when "
              "use_code_filter is True (moved outside the CTR sub-sampling "
              "gate) |")
    md.append("| Fix 4 | VarietyProfiler.profile() / is_profile_step / "
              "hooks_registered dormancy REMOVED from the generate loop; "
              "shadow hooks retained as passive telemetry (controller) |")
    md.append("| Fix 6 | sample_top_p explicit .clone(); finally-block resets "
              "per-generation state |")
    md.append("| Residual path | DISABLED in this probe — no diagnose() → "
              "apply_interventions() → sae_guided_reset hooks |")
    md.append("| Kickstart | FULLY DISABLED — no _kickstart_counter, no "
              "kickstart penalty path at all |")
    md.append("| Dormant | REUSED from night-016 (greedy + sampling KV-cache "
              "dormant, already validated 0/250 drift). Do NOT re-run. |")
    md.append("| Reused baselines | night-016 "
              "(kvcache_restructure_results.jsonl, kickstart 1e4 cell); "
              "night-014 Arm 2 (sampling_dual_predicate_results.jsonl). "
              "Do NOT re-run. |")
    md.append("| NEW runs | 2 arms x 3 fixtures = 6 condition-fixture pairs |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    if isinstance(meta["wall_clock_s"], str):
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']} |")
    else:
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("One Fixture C record (heldout_degenerate_v2, seeded subset): "
              "greedy suppression-only generated twice and sampling "
              "suppression-only generated twice, all with SEED=42. Token ids "
              "must match; per-step PR log must match to 6 decimal places. "
              "STOP if not. Dormant is REUSED from night-016 (not re-run).")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = [[
        str(sm["record_id"]), str(sm["prompt_len"]),
        str(sm["n_generated_greedy_active"]),
        str(sm["n_generated_sampling_active"]),
        sm["greedy_active_identical"], sm["greedy_pr_identical"],
        sm["sampling_active_identical"], sm["sampling_pr_identical"],
        sm["identical"],
    ]]
    md.append(format_table_md(
        smoke_rows, ["record_id", "prompt_len", "n_gen(greedy)",
                     "n_gen(sampling)", "ga id", "ga PR", "sa id", "sa PR",
                     "identical"],
    ))
    md.append("")
    md.append(f"PR greedy run A/B first: {fmt(sm['pr_greedy_a_first'])} / "
              f"{fmt(sm['pr_greedy_b_first'])}; last: "
              f"{fmt(sm['pr_greedy_a_last'])} / {fmt(sm['pr_greedy_b_last'])}. "
              f"PR sampling run A/B first: {fmt(sm['pr_sampling_a_first'])} / "
              f"{fmt(sm['pr_sampling_b_first'])}; last: "
              f"{fmt(sm['pr_sampling_a_last'])} / {fmt(sm['pr_sampling_b_last'])}.")
    md.append("")

    tabs = ctx["tables"]
    agg = tabs["aggregate"]
    comp = tabs["comparison"]
    fixture_labels = [("t2s_degenerate", "A t2s_degenerate"),
                      ("qwen_degenerate", "B qwen_degenerate"),
                      ("heldout_degenerate_v2", "C heldout_degenerate_v2")]

    # Key findings (data-driven, written after aggregation)
    md.append("## Key findings")
    md.append("")
    findings = ctx["key_findings"]
    for i, finding in enumerate(findings, 1):
        md.append(f"{i}. {finding}")
    md.append("")

    # Table 1 — Primary aggregation
    md.append("## Table 1 — Primary aggregation (per fixture, per arm)")
    md.append("")
    md.append("Rescue = ΔDistinct-2 > 0 vs the REUSED night-016 dormant "
              "baseline. Net escape = gross escape AND n_gen >= 24 (night-013 "
              "convention). EOS rate = n_gen < 128 AND last_token == "
              "eos_token_id (co-primary). med n_gen = median generated tokens "
              "among escaped continuations (co-primary). ΔD2 mean = mean "
              "ΔDistinct-2 over all records.")
    md.append("")
    md.append("| fixture | arm | rescue | net | EOS | med n_gen (esc) | "
              "ΔD2 mean |")
    md.append("|---|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in (("greedy", "greedy_active"),
                                   ("sampling", "sampling_active")):
            a = agg[fixture][arm_key]
            md.append(f"| {label} | {arm_label} | {fmt(a['rescue_rate'], 4)} | "
                      f"{fmt(a['net_escape'], 4)} | "
                      f"{fmt(a['eos_rate'], 4)} | "
                      f"{fmt(a['median_n_gen_escaped'], 1)} | "
                      f"{fmt(a['delta_distinct_2']['mean'], 4)} |")
    md.append("")

    # Table 2 — Comparison vs night-016 and night-014 Arm 2
    md.append("## Table 2 — Comparison (vs night-016 kickstart, night-014 Arm 2)")
    md.append("")
    md.append("night-016 (kickstart) = REUSED kvcache_restructure_results.jsonl "
              "1e4 cell (greedy_active / sampling_active). night-014 Arm 2 = "
              "REUSED sampling_dual_predicate_results.jsonl arm2_active "
              "(spectral-gated kickstart; greedy arm not run in night-014). "
              "night-018 (no kickstart) = this probe.")
    md.append("")
    md.append("| fixture | arm | night-016 (kickstart) | night-014 Arm2 | "
              "night-018 (no kickstart) |")
    md.append("|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in (("greedy", "greedy_active"),
                                   ("sampling", "sampling_active")):
            c = comp[fixture][arm_key]
            md.append(f"| {label} | {arm_label} | "
                      f"{fmt(c['n016_rescue'], 4)} / EOS {fmt(c['n016_eos'], 4)} | "
                      f"{c['n014_arm2']} | "
                      f"{fmt(c['n018_rescue'], 4)} / EOS {fmt(c['n018_eos'], 4)} |")
    md.append("")

    # Table 3 — Four-bucket histogram
    md.append("## Table 3 — Four-bucket histogram over escaped continuations")
    md.append("")
    md.append("night-013 convention: n=1 | 2-23 | 24-127 | n=128 over escaped "
              "continuations (gross escape = not byte-identical to dormant).")
    md.append("")
    md.append("| fixture | arm | n=1 | 2-23 | 24-127 | n=128 |")
    md.append("|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in (("greedy", "greedy_active"),
                                   ("sampling", "sampling_active")):
            a = agg[fixture][arm_key]
            md.append(f"| {label} | {arm_label} | "
                      f"{a['bucket_1']} | {a['bucket_2_23']} | "
                      f"{a['bucket_24_127']} | {a['bucket_128']} |")
    md.append("")

    # Table 4 — Fire-type breakdown (dual-predicate detection-only)
    md.append("## Table 4 — Fire-type breakdown (dual-predicate detection-only)")
    md.append("")
    md.append("The collapse predicate runs DETECTION-ONLY in this probe — it "
              "does NOT gate actuation (suppression is unconditional). "
              "bigram-only records = bigram fired >= 1 step, spectral never. "
              "spectral-only = spectral fired >= 1 step, bigram never. both = "
              "both fired >= 1 step.")
    md.append("")
    md.append("| fixture | arm | fired | bigram-only | spectral-only | both | "
              "mean supp | mean kick |")
    md.append("|---|---|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in (("greedy", "greedy_active"),
                                   ("sampling", "sampling_active")):
            a = agg[fixture][arm_key]
            md.append(f"| {label} | {arm_label} | {a['n_fired']} | "
                      f"{a['n_bigram_only']} | {a['n_spectral_only']} | "
                      f"{a['n_both']} | {fmt(a['mean_suppression_steps'], 2)} | "
                      f"{fmt(a['mean_kickstart_events'], 2)} |")
    md.append("")

    # Target assessment
    md.append("## Target assessment (hypothesis, NOT a gate)")
    md.append("")
    md.append("Suppression-only must hold rescue within 0.05 of the best prior "
              "while dropping EOS:")
    md.append("")
    md.append("| fixture | arm | target | night-018 | met? |")
    md.append("|---|---|---|---|---|")
    for row in ctx["target_assessment"]:
        md.append(f"| {row['fixture']} | {row['arm']} | {row['target']} | "
                  f"{row['actual']} | {row['met']} |")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no controller edits, no threshold changes. "
              "All retained night-016 fixes are implemented INLINE in this "
              "probe before they reach the controller [1].")
    md.append("- Kickstart is FULLY DISABLED: `_kickstart_counter` is never "
              "set; the kickstart penalty block is removed from the generation "
              "loop. The `_cache_vocabulary_subsets` call is not needed at "
              "runtime (offline diagnostics only per night-015 Fix 5).")
    md.append("- Suppression is UNCONDITIONAL: at EVERY step where "
              "active_loop_ids is non-empty, ALL active_loop_ids are added to "
              "the cooldown dict (cooldown=8, penalty=-5.0). NOT gated on "
              "is_collapsed. This matches night-014's suppression path (net "
              "escape 0.96 on heldout_v2).")
    md.append("- The dual-predicate collapse predicate runs DETECTION-ONLY: "
              "the predicate is logged for fire-type breakdown but does NOT "
              "gate actuation.")
    md.append("- Dormant baselines are REUSED from night-016 (greedy + sampling "
              "KV-cache, already validated 0/250 drift). They are NOT re-run; "
              "the ΔD2 deltas are computed against the same night-016 dormant "
              "baseline, so rescue-rate differences are attributable to the "
              "actuation change (kickstart removal) alone.")
    md.append("- night-016 (kickstart) values are REUSED from "
              "kvcache_restructure_results.jsonl (1e4 cell). night-014 Arm 2 "
              "values are REUSED from sampling_dual_predicate_results.jsonl "
              "(arm2_active, spectral-gated kickstart). Do NOT re-run.")
    md.append("- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 "
              "(matching night-014/016). SEED=42 with seed_all(SEED) before "
              "every sampling generation; the determinism smoke confirms "
              "reproducible sampling.")
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
    night016_recs: List[Dict[str, Any]],
    night014_rates: Dict[str, Dict[str, float]],
    smoke: Dict[str, Any],
    device: str,
    dtype: Any,
    spectral_threshold: float,
    wall_clock_s: float,
) -> Dict[str, Any]:
    """Build the report context from per-record results (single source of truth)."""
    fixture_names = ["t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"]

    # Aggregate night-018 (no kickstart) per fixture per arm.
    aggregates: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fixture in fixture_names:
        recs = [r for r in all_results if r["fixture"] == fixture]
        aggregates[fixture] = {
            "greedy_active": aggregate_arm(recs, "greedy_active"),
            "sampling_active": aggregate_arm(recs, "sampling_active"),
        }

    # night-016 (kickstart 1e4) per fixture per arm — reuse from night-016 recs.
    n016_agg: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fixture in fixture_names:
        recs = [r for r in night016_recs if r["fixture"] == fixture]
        n016_agg[fixture] = {
            "greedy_active": aggregate_arm(recs, "greedy_active"),
            "sampling_active": aggregate_arm(recs, "sampling_active"),
        }

    # Comparison table cells.
    comparison: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for fixture in fixture_names:
        comparison[fixture] = {}
        for arm_key in ("greedy_active", "sampling_active"):
            n18 = aggregates[fixture][arm_key]
            n16 = n016_agg[fixture][arm_key]
            n14_rescue = night014_rates[fixture]["rescue"]
            n14_eos = night014_rates[fixture]["eos"]
            comparison[fixture][arm_key] = {
                "n016_rescue": float(n16["rescue_rate"]),
                "n016_eos": float(n16["eos_rate"]),
                "n014_arm2": (
                    f"{fmt(n14_rescue, 4)} / EOS {fmt(n14_eos, 4)}"
                    if arm_key == "sampling_active" else "—"
                ),
                "n018_rescue": float(n18["rescue_rate"]),
                "n018_eos": float(n18["eos_rate"]),
            }

    # Target assessment (hypothesis, NOT a gate).
    target_assessment: List[Dict[str, str]] = []
    targets = [
        # (fixture, arm_key, arm_label, target_str, actual_fn)
        ("t2s_degenerate", "greedy_active", "greedy",
         "rescue ≥ 0.91, EOS < 0.85"),
        ("qwen_degenerate", "greedy_active", "greedy",
         "rescue ≥ 0.93, EOS < 0.67"),
        ("heldout_degenerate_v2", "greedy_active", "greedy",
         "rescue ≥ 0.95, EOS ≤ 0.26"),
        ("heldout_degenerate_v2", "sampling_active", "sampling",
         "net escape ≥ 0.90, EOS ≤ 0.34"),
    ]
    for fixture, arm_key, arm_label, target_str in targets:
        a = aggregates[fixture][arm_key]
        if arm_key == "sampling_active" and fixture == "heldout_degenerate_v2":
            actual = f"rescue {fmt(a['rescue_rate'], 4)}, net {fmt(a['net_escape'], 4)}, EOS {fmt(a['eos_rate'], 4)}"
            met = bool(a["net_escape"] >= 0.90 and a["eos_rate"] <= 0.34)
        elif arm_key == "greedy_active":
            actual = f"rescue {fmt(a['rescue_rate'], 4)}, EOS {fmt(a['eos_rate'], 4)}"
            if fixture == "t2s_degenerate":
                met = bool(a["rescue_rate"] >= 0.91 and a["eos_rate"] < 0.85)
            elif fixture == "qwen_degenerate":
                met = bool(a["rescue_rate"] >= 0.93 and a["eos_rate"] < 0.67)
            else:  # heldout
                met = bool(a["rescue_rate"] >= 0.95 and a["eos_rate"] <= 0.26)
        else:  # pragma: no cover
            met = False
        target_assessment.append({
            "fixture": fixture,
            "arm": arm_label,
            "target": target_str,
            "actual": actual,
            "met": "YES" if met else "NO",
        })

    # Key findings — data-driven.
    findings: List[str] = []
    g_t2s = aggregates["t2s_degenerate"]["greedy_active"]
    g_qwen = aggregates["qwen_degenerate"]["greedy_active"]
    g_hv2 = aggregates["heldout_degenerate_v2"]["greedy_active"]
    s_hv2 = aggregates["heldout_degenerate_v2"]["sampling_active"]
    s_t2s = aggregates["t2s_degenerate"]["sampling_active"]
    s_qwen = aggregates["qwen_degenerate"]["sampling_active"]
    n16_t2s_g = n016_agg["t2s_degenerate"]["greedy_active"]
    n16_qwen_g = n016_agg["qwen_degenerate"]["greedy_active"]
    n16_hv2_g = n016_agg["heldout_degenerate_v2"]["greedy_active"]
    findings.append(
        f"**Greedy suppression-only rescue/EOS:** t2s {fmt(g_t2s['rescue_rate'], 4)} "
        f"(bar ≥ 0.91) / EOS {fmt(g_t2s['eos_rate'], 4)} (night-016 0.85); "
        f"qwen {fmt(g_qwen['rescue_rate'], 4)} (bar ≥ 0.93) / EOS "
        f"{fmt(g_qwen['eos_rate'], 4)} (night-016 0.67); heldout "
        f"{fmt(g_hv2['rescue_rate'], 4)} (bar ≥ 0.95) / EOS "
        f"{fmt(g_hv2['eos_rate'], 4)} (night-016 0.26)."
    )
    findings.append(
        f"**Sampling heldout net escape:** {fmt(s_hv2['net_escape'], 4)} "
        f"(target ≥ 0.90), EOS {fmt(s_hv2['eos_rate'], 4)} (target ≤ 0.34, "
        f"night-014 Arm 2 0.34)."
    )
    findings.append(
        f"**Sampling t2s/qwen:** t2s rescue {fmt(s_t2s['rescue_rate'], 4)} / "
        f"EOS {fmt(s_t2s['eos_rate'], 4)} (night-016 0.96/0.64, night-014 Arm 2 "
        f"0.87/0.68); qwen rescue {fmt(s_qwen['rescue_rate'], 4)} / "
        f"EOS {fmt(s_qwen['eos_rate'], 4)} (night-016 0.95/0.43, night-014 "
        f"Arm 2 0.98/0.39)."
    )
    findings.append(
        f"**Kickstart events are all zero** (n_kickstart_events == 0 on every "
        f"record, both arms, all fixtures) — kickstart is fully disabled; any "
        f"actuation is suppression-only."
    )

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
        "determinism_smoke": smoke,
        "tables": {
            "aggregate": aggregates,
            "comparison": comparison,
        },
        "key_findings": findings,
        "target_assessment": target_assessment,
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-018 suppression-only, kickstart fully disabled "
                    "measurement (MEASUREMENT ONLY)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per fixture (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate SUPPRESSION_ONLY_RESULTS.md from an "
                             "existing suppression_only_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("night-018 suppression-only, kickstart fully disabled (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"sampling: temp={SAMPLING_TEMPERATURE} top_p={SAMPLING_TOP_P}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: "
          f"{ANALYSIS_WINDOW} | hook_layer: {HOOK_LAYER}")
    print("kickstart: FULLY DISABLED | suppression: unconditional "
          f"(cooldown={COOLDOWN}, penalty={PENALTY})")

    # ------------------------------------------------------------------
    # Reused data FIRST (both measurement and report-only paths).
    # ------------------------------------------------------------------
    if not NIGHT016_JSONL.exists() or not NIGHT014_JSONL.exists():
        print(f"[STOP] {NIGHT016_JSONL} or {NIGHT014_JSONL} does not exist; "
              f"reused baselines are required. Do NOT re-run the prior "
              f"measurements.")
        sys.exit(1)
    night016_recs = load_night016_records(NIGHT016_JSONL)
    check_night016_counts(night016_recs)
    night014_rates = load_night014_arm2_rates(NIGHT014_JSONL)
    print(f"reused night-016: {len(night016_recs)} per-record lines "
          f"(100 t2s + 100 qwen + 50 heldout_v2)")
    print(f"reused night-014 Arm 2 rates: "
          f"t2s={night014_rates['t2s_degenerate']['rescue']:.4f} "
          f"(EOS {night014_rates['t2s_degenerate']['eos']:.4f}), "
          f"qwen={night014_rates['qwen_degenerate']['rescue']:.4f} "
          f"(EOS {night014_rates['qwen_degenerate']['eos']:.4f}), "
          f"heldout_v2={night014_rates['heldout_degenerate_v2']['rescue']:.4f} "
          f"(EOS {night014_rates['heldout_degenerate_v2']['eos']:.4f})")

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
        smoke: Dict[str, Any] = {
            "record_id": -1, "prompt_len": -1, "n_generated_greedy_active": -1,
            "n_generated_sampling_active": -1, "greedy_active_identical": "True",
            "greedy_pr_identical": "True", "sampling_active_identical": "True",
            "sampling_pr_identical": "True", "identical": "True",
            "pr_greedy_a_first": None, "pr_greedy_b_first": None,
            "pr_greedy_a_last": None, "pr_greedy_b_last": None,
            "pr_sampling_a_first": None, "pr_sampling_b_first": None,
            "pr_sampling_a_last": None, "pr_sampling_b_last": None,
        }
        wall_clock_s: Any = "n/a (report-only regeneration of the night-018 run)"
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("record_type") == "smoke":
                    smoke = {**smoke, **{k: v for k, v in r.items()
                                         if k != "record_type"}}
                elif r.get("record_type") == "meta" and "wall_clock_s" in r:
                    wall_clock_s = float(r["wall_clock_s"])
        ctx = build_ctx_from_results(
            all_results, night016_recs, night014_rates, smoke,
            device, dtype, spectral_threshold, wall_clock_s=wall_clock_s,
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
    # Determinism smoke FIRST (also serves as GPU warmup).
    # ------------------------------------------------------------------
    smoke: Dict[str, Any] = {
        "record_id": -1, "prompt_len": -1, "n_generated_greedy_active": -1,
        "n_generated_sampling_active": -1, "greedy_active_identical": "True",
        "greedy_pr_identical": "True", "sampling_active_identical": "True",
        "sampling_pr_identical": "True", "identical": "True",
        "pr_greedy_a_first": None, "pr_greedy_b_first": None,
        "pr_greedy_a_last": None, "pr_greedy_b_last": None,
        "pr_sampling_a_first": None, "pr_sampling_b_first": None,
        "pr_sampling_a_last": None, "pr_sampling_b_last": None,
    }
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        try:
            smoke = run_determinism_smoke(
                model, tokenizer, records_c, "mutated_prompt",
                spectral_threshold,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during determinism smoke: {e}")
                sys.exit(1)
            raise

    # ==================================================================
    # Per-fixture runs: greedy + sampling active (suppression-only).
    # Dormant is REUSED from night-016.
    # ==================================================================
    def run_fixture(
        fixture_name: str,
        records: List[Dict[str, Any]],
        prompt_key: str,
    ) -> List[Dict[str, Any]]:
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

            # Greedy suppression-only active.
            gen_ga = greedy_generate_suppression_only(
                model, tokenizer, prompt, spectral_threshold,
            )
            greedy_active = build_active_meta(
                tokenizer, gen_ga, dormant_greedy, spectral_threshold,
                arm="greedy_suppression_only",
            )

            # Sampling suppression-only active.
            gen_sa = sampling_generate_suppression_only(
                model, tokenizer, prompt, spectral_threshold,
            )
            sampling_active = build_active_meta(
                tokenizer, gen_sa, dormant_sampling, spectral_threshold,
                arm="sampling_suppression_only",
            )

            out.append({
                "fixture": fixture_name,
                "record_id": rid,
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
                print(f"  [{fixture_name}] {rec_i+1}/{len(records)}; "
                      f"rescue(g)={n_rescue_g} rescue(s)={n_rescue_s} "
                      f"kick(g)={n_kick_g} kick(s)={n_kick_s} "
                      f"EOS(g)={n_eos_g} EOS(s)={n_eos_s}")
        return out

    all_results: List[Dict[str, Any]] = []
    try:
        print("\n--- Fixture A: t2s_degenerate (prompt key: record['text']) ---")
        part_a = run_fixture("t2s_degenerate", records_a, "text")
        all_results.extend(part_a)

        print("\n--- Fixture B: qwen_degenerate (prompt key: record['prompt']) ---")
        part_b = run_fixture("qwen_degenerate", records_b, "prompt")
        all_results.extend(part_b)

        print("\n--- Fixture C: heldout_degenerate_v2 (prompt key: "
              "record['mutated_prompt']) ---")
        part_c = run_fixture("heldout_degenerate_v2", records_c, "mutated_prompt")
        all_results.extend(part_c)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during run: {e}")
            sys.exit(1)
        raise

    # ==================================================================
    # Aggregate (single source of truth).
    # ==================================================================
    ctx = build_ctx_from_results(
        all_results, night016_recs, night014_rates, smoke,
        device, dtype, spectral_threshold,
        wall_clock_s=time.time() - t_start,
    )
    agg = ctx["tables"]["aggregate"]
    comp = ctx["tables"]["comparison"]

    print("\n" + "=" * 70)
    print("NIGHT-018 SUPPRESSION-ONLY (NO KICKSTART)")
    for fixture in ("t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"):
        for arm_key, arm_label in (("greedy_active", "greedy"),
                                   ("sampling_active", "sampling")):
            a = agg[fixture][arm_key]
            c = comp[fixture][arm_key]
            print(f"  {fixture} {arm_label}: rescue={a['rescue_rate']:.4f} "
                  f"net={a['net_escape']:.4f} EOS={a['eos_rate']:.4f} "
                  f"med_n_gen={a['median_n_gen_escaped']:.1f} | "
                  f"n016={c['n016_rescue']:.4f}/{c['n016_eos']:.4f} | "
                  f"n014_A2={c['n014_arm2']}")
    print("=" * 70)

    # ==================================================================
    # Write outputs.
    # ==================================================================
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"record_type": "smoke", **smoke}) + "\n")
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
            "kickstart": "fully_disabled",
            "suppression": "unconditional",
            "cooldown": COOLDOWN,
            "penalty": PENALTY,
            "reused_dormant_from": str(NIGHT016_JSONL),
            "reused_night014_arm2_from": str(NIGHT014_JSONL),
            "rescue_greedy_t2s": int(
                agg["t2s_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_qwen": int(
                agg["qwen_degenerate"]["greedy_active"]["n_rescue"]),
            "rescue_greedy_heldout_v2": int(
                agg["heldout_degenerate_v2"]["greedy_active"]["n_rescue"]),
            "rescue_sampling_t2s": int(
                agg["t2s_degenerate"]["sampling_active"]["n_rescue"]),
            "rescue_sampling_qwen": int(
                agg["qwen_degenerate"]["sampling_active"]["n_rescue"]),
            "rescue_sampling_heldout_v2": int(
                agg["heldout_degenerate_v2"]["sampling_active"]["n_rescue"]),
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(all_results)} record lines + "
          f"sidecar lines)")

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-018 measurement complete")


if __name__ == "__main__":
    main()
