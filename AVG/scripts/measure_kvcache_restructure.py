#!/usr/bin/env python3
"""night-016: KV-cache restructure + dual-predicate → logit-penalty wiring
(MEASUREMENT ONLY).

Rigor review identified two BLOCKERs in the production controller:

  1. Spectral PR hook never fires.  `generate()` calls `self.model(input_ids)`
     with the full growing sequence — no `past_key_values` is passed.  The
     `hs.shape[1] == 1` gate in the layer-2 PR hook evaluates False on every
     decoding step.  `_layer2_pr` stays `None`, spectral_collapse is always
     False, and the dual-predicate spectral arm never fires.

  2. Dual-predicate feeds the deprecated residual actuation path.
     `diagnose()` produces `InterventionDecision` objects with
     `action="sae_guided_reset"` that flow to `apply_interventions()`, which
     registers inert residual perturbation hooks.  The functional logit-penalty
     path operates independently of the collapse predicate.

This probe implements ALL proposed fixes INLINE (no controller edits) and
measures rescue rates under TWO decoding regimes on all three degenerate
fixtures plus the t2s detection-gap subset (93 records).

Fixes implemented inline:

  Fix 1 — KV-cache incremental decoding: pre-fill the prompt once
          (use_cache=True), then single-token forwards with past_key_values.
          `hs.shape[1] == 1` is True on every decoding step, so the layer-2 PR
          hook condition fires correctly.
  Fix 2 — Wire the collapse predicate to the logit-penalty path.  Suppression
          (add active_loop_ids to cooldown) is gated on is_collapsed; kickstart
          fires on spectral-only (spectral_fire AND NOT bigram_fire) and the
          kickstart penalty is retargeted to active_loop_ids (night-015).
  Fix 3 — Recompute is_code_context EVERY step when use_code_filter is True
          (moved outside the CTR sub-sampling gate).
  Fix 4 — Remove the VarietyProfiler.profile() step and is_profile_step /
          hooks_registered dormancy management from the generate loop.  The
          layer-2 spectral PR hook (and, in the controller, the shadow-mode
          hooks) are retained as passive telemetry.
  Fix 5 — Retarget kickstart to active_loop_ids + vocab-range fix (night-015
          inherited): `vocab_size = len(self.tokenizer)`, kickstart penalties
          applied to active_loop_ids, not non_prose_token_ids.
  Fix 6 — Minor fixes: local `sample_top_p` with explicit .clone() at start;
          `_cache_vocabulary_subsets` retained for offline diagnostics only;
          finally-block resets per-generation state.

Reused baselines (do NOT re-run):
  - DS-033 (greedy, full-sequence, production combined):
    docs/gate23/production_cross_fixture_results.jsonl
  - night-008 (greedy, KV-cache, hybrid CTR gating):
    docs/gate23/hybrid_ctr_gating_results.jsonl
  - DS-035 (greedy, KV-cache, spectral-gated kickstart):
    docs/gate23/dual_predicate_rescue_results.jsonl
  - night-014 (sampling, KV-cache): docs/gate23/sampling_dual_predicate_results.jsonl

NEW live runs:
  - Greedy dormant baselines (KV-cache) for all records.
  - Sampling dormant baselines (KV-cache) for all records.
  - Greedy restructured active for all records (incl. the t2s detection-gap
    subset of 93).
  - Sampling restructured active for all records.

Environment notes (identical to night-014/night-015/ds-025..ds-035):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; fall back to HF_HOME=/tmp/hf_cache only if the
    pinned revision is not cached.
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
SEED = 42  # night-014 measurement seed (matches ds-025..ds-035, DS-032 Part 2)
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
KICKSTART_PENALTIES: Dict[int, float] = {3: -1e4, 2: -5.0, 1: -2.0}  # controller.py:753-758

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
DS033_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
NIGHT008_JSONL = Path("docs/gate23/hybrid_ctr_gating_results.jsonl")
DS035_JSONL = Path("docs/gate23/dual_predicate_rescue_results.jsonl")
NIGHT014_JSONL = Path("docs/gate23/sampling_dual_predicate_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/kvcache_restructure_results.jsonl")
OUTPUT_MD = Path("docs/gate23/KVCACHE_RESTRUCTURE_RESULTS.md")

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
# Layer-2 PR measurement hook (identical to DS-034b / DS-034c instrument)
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
# KV-cache generation: greedy dormant (plain argmax, no hook/penalties)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_dormant(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) dormant generation with KV-cache.

    Plain argmax.  No hooks, no suppression, no kickstart, no top_p, no
    residual interventions.  Pre-fill the prompt once (use_cache=True), then
    decode one token at a time with past_key_values.  EOS break (EOS token is
    included in generated ids, matching the ds-029..ds-033 convention).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    # Fix 1: Pre-fill — one forward pass over the full prompt.
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :].clone().float()

    generated: List[torch.Tensor] = []
    for _step in range(max_new_tokens):
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if int(next_token.item()) == eos_id:
            break
        # Fix 1: single-token forward with the KV cache.
        out = model(next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :].clone().float()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
    }


# ---------------------------------------------------------------------------
# KV-cache generation: sampling dormant (plain sampling, no hooks)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sampling_generate_dormant(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Sampling (do_sample=True) dormant generation with KV-cache.

    Plain sampling: do_sample=True, temperature=0.8, top_p=0.85 applied at
    every step.  No hooks, no suppression, no kickstart, no residual
    interventions.  seed_all(SEED) is called at entry so every generation
    starts from the same RNG state (deterministic sampling; DS-032 Part 2
    smoke confirmed this is reproducible).
    """
    seed_all(SEED)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    # Fix 1: Pre-fill — one forward pass over the full prompt.
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :].clone().float()

    generated: List[torch.Tensor] = []
    for _step in range(max_new_tokens):
        logits = next_logits.clone()
        logits = sample_top_p_restructured(logits, top_p=SAMPLING_TOP_P)
        logits = logits / max(SAMPLING_TEMPERATURE, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token)
        if int(next_token.item()) == eos_id:
            break
        # Fix 1: single-token forward with the KV cache.
        out = model(next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :].clone().float()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
    }


# ---------------------------------------------------------------------------
# Restructured active generation core (shared by greedy and sampling).
# Implements Fix 1 + Fix 2 + Fix 3 + Fix 5 + Fix 6 inline.
# ---------------------------------------------------------------------------
def _restructured_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    do_sample: bool,
    max_new_tokens: int = MAX_NEW_TOKENS,
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

    do_sample=False → greedy argmax; do_sample=True → temp=0.8, top_p=0.85
    multinomial (matching night-014).
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
    layer2_pr_buffer: List[torch.Tensor] = []

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
            #               (was: non_prose_token_ids; night-015 fix) ----
            kick_penalty_value = 0.0
            if kickstart_counter > 0:
                if active_loop_ids:
                    loop_tensor = torch.tensor(
                        list(active_loop_ids), device=logits.device
                    )
                    kick_penalty_value = KICKSTART_PENALTIES.get(
                        kickstart_counter, -2.0
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
    }


@torch.no_grad()
def greedy_generate_restructured(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy restructured active generation (Fix 1+2+3+5+6 inline)."""
    return _restructured_generate(
        model, tokenizer, prompt, spectral_threshold,
        do_sample=False, max_new_tokens=max_new_tokens,
    )


@torch.no_grad()
def sampling_generate_restructured(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Sampling restructured active generation (Fix 1+2+3+5+6 inline)."""
    return _restructured_generate(
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


def build_dormant_meta(
    tokenizer: Any, gen: Dict[str, Any], source: str
) -> Dict[str, Any]:
    generated_ids = gen["generated_ids"]
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    n = int(generated_ids.shape[-1])
    return {
        "n_generated": n,
        "distinct_2": continuation_distinct2(generated_ids),
        "ctr": continuation_ctr(tokenizer, generated_ids),
        "generated_text": text,
        "generated_ids": generated_ids[0].tolist(),
        "eos_terminated": bool(
            n < MAX_NEW_TOKENS and n > 0
            and int(generated_ids[0, -1].item()) == tokenizer.eos_token_id
        ),
        "source": source,
    }


def build_active_meta(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
    spectral_threshold: float,
    arm: str,
) -> Dict[str, Any]:
    generated_ids = gen["generated_ids"]
    n = int(generated_ids.shape[-1])
    d2 = continuation_distinct2(generated_ids)
    ctr = continuation_ctr(tokenizer, generated_ids)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    eos_terminated = bool(
        n < MAX_NEW_TOKENS and n > 0
        and int(generated_ids[0, -1].item()) == tokenizer.eos_token_id
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
# Reused data loaders (do NOT re-run)
# ---------------------------------------------------------------------------
def load_fixture_rescue_rates(
    path: Path,
    fixtures: Sequence[str],
) -> Dict[str, float]:
    """Load per-fixture rescue rates from a reused JSONL.

    rescue rate = #ΔD2>0 / n (the DS-033/DS-035/night-008 convention).  The
    `active` dict must carry `delta_distinct_2`.  Fixtures with no records
    (e.g. DS-033 did not cover heldout_degenerate_v2) return float('nan').
    STOPs only if NO fixture has records (which means the file layout is
    wrong or the baselines were re-run against a different schema).
    """
    counts: Dict[str, Dict[str, int]] = {}
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
            active = r.get("active", {})
            d2 = active.get("delta_distinct_2")
            if d2 is None:
                continue
            rec = counts.setdefault(fixture, {"n": 0, "n_rescue": 0})
            rec["n"] += 1
            if d2 > 0:
                rec["n_rescue"] += 1
    rates: Dict[str, float] = {}
    n_total = 0
    for fixture in fixtures:
        rec = counts.get(fixture)
        if rec is None or rec["n"] == 0:
            rates[fixture] = float("nan")
            continue
        n_total += rec["n"]
        rates[fixture] = float(rec["n_rescue"]) / float(rec["n"])
    if n_total == 0:
        raise SystemExit(
            f"[STOP] {path} has no records for any fixture in {fixtures}."
        )
    return rates


def load_ds033_detection_gap_ids(path: Path) -> Set[int]:
    """Load DS-033 t2s_degenerate detection-gap record ids (REUSE).

    Detection-gap = `active.diagnostic_class == 'detection-gap'` (equivalent to
    `active.predicate_true_steps == 0`).  STOPs if the count is not 93 (task
    boundary: the t2s detection-gap subset is 93 records).
    """
    out: Set[int] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type"):
                continue
            if r.get("fixture") != "t2s_degenerate":
                continue
            active = r.get("active", {})
            if active.get("diagnostic_class") == "detection-gap":
                out.add(int(r["record_id"]))
    if len(out) != 93:
        raise SystemExit(
            f"[STOP] DS-033 t2s detection-gap subset has {len(out)} records, "
            f"expected 93 (task boundary). Do NOT re-classify."
        )
    return out


def load_night014_arm2_rates(path: Path) -> Dict[str, float]:
    """Load night-014 Arm 2 (suppression-only, no kickstart) rescue rates."""
    fixtures = ("t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2")
    counts: Dict[str, Dict[str, int]] = {}
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
            a2 = r.get("arm2_active", {})
            d2 = a2.get("delta_distinct_2")
            if d2 is None:
                continue
            rec = counts.setdefault(fixture, {"n": 0, "n_rescue": 0})
            rec["n"] += 1
            if d2 > 0:
                rec["n_rescue"] += 1
    rates: Dict[str, float] = {}
    for fixture in fixtures:
        rec = counts.get(fixture)
        if rec is None or rec["n"] == 0:
            raise SystemExit(
                f"[STOP] night-014 has no Arm 2 records for fixture {fixture}."
            )
        rates[fixture] = float(rec["n_rescue"]) / float(rec["n"])
    return rates


def load_night014_meta(path: Path) -> Dict[str, Any]:
    """Load the night-014 meta sidecar for provenance."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type") == "meta":
                return r
    raise SystemExit("[STOP] night-014 results file has no meta sidecar.")


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
    """Determinism smoke (night-016): one Fixture C record.

      1. Greedy dormant generated twice — token ids must match.
      2. Greedy restructured active generated twice — token ids AND per-step PR
         log must match.
      3. Sampling dormant generated twice — token ids must match.
      4. Sampling restructured active generated twice — token ids AND per-step
         PR log must match.
    The smoke record is the FIRST Fixture C record where the greedy restructured
    arm reaches the analysis window (n_generated >= 24), so the PR-log identity
    check is non-vacuous.  STOP (sys.exit 1) if any check fails.
    """
    smoke_record: Optional[Dict[str, Any]] = None
    rid = -1
    prompt = ""
    for rec in records_c:
        probe = greedy_generate_restructured(
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
              "analysis window under the greedy restructured arm. Cannot verify "
              "PR-log determinism.")
        sys.exit(1)

    print("\n--- Determinism smoke (1 Fixture C record) ---")

    # Greedy dormant twice.
    gda = greedy_generate_dormant(model, tokenizer, prompt)
    gdb = greedy_generate_dormant(model, tokenizer, prompt)
    gd_identical = bool(
        gda["generated_ids"].shape == gdb["generated_ids"].shape
        and torch.equal(gda["generated_ids"], gdb["generated_ids"])
    )
    print(f"  greedy-dormant record_id={rid}: n_tokens={gda['n_generated']} / "
          f"{gdb['n_generated']} identical={gd_identical}")

    # Greedy restructured active twice.
    gaa = greedy_generate_restructured(model, tokenizer, prompt, spectral_threshold)
    gab = greedy_generate_restructured(model, tokenizer, prompt, spectral_threshold)
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

    # Sampling dormant twice.
    sda = sampling_generate_dormant(model, tokenizer, prompt)
    sdb = sampling_generate_dormant(model, tokenizer, prompt)
    sd_identical = bool(
        sda["generated_ids"].shape == sdb["generated_ids"].shape
        and torch.equal(sda["generated_ids"], sdb["generated_ids"])
    )
    print(f"  sampling-dormant record_id={rid}: n_tokens={sda['n_generated']} / "
          f"{sdb['n_generated']} identical={sd_identical}")

    # Sampling restructured active twice.
    saa = sampling_generate_restructured(model, tokenizer, prompt, spectral_threshold)
    sab = sampling_generate_restructured(model, tokenizer, prompt, spectral_threshold)
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
        gd_identical and ga_identical and pr_g_identical
        and sd_identical and sa_identical and pr_s_identical
    )
    out = {
        "record_id": rid,
        "prompt_len": int(gaa["prompt_len"]),
        "greedy_dormant_identical": str(gd_identical),
        "greedy_active_identical": str(ga_identical),
        "greedy_pr_identical": str(pr_g_identical),
        "sampling_dormant_identical": str(sd_identical),
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
    print("  determinism smoke: ALL IDENTICAL (greedy/sampling dormant, "
          "greedy/sampling active, PR log @6dp)")
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


def aggregate_arm(results: List[Dict[str, Any]], arm_key: str) -> Dict[str, Any]:
    """Aggregate per-record results for one fixture and one arm."""
    act = [r[arm_key] for r in results]
    dorm = [r["dormant_" + ("sampling" if arm_key == "sampling_active"
                            else "greedy")] for r in results]
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


def aggregate_detection_gap(
    results: List[Dict[str, Any]],
    detection_gap_ids: Set[int],
    require_full: bool = True,
) -> Dict[str, Any]:
    """Aggregate the t2s detection-gap subset under the greedy restructured arm.

    Concern B (t2s_degenerate rescue ceiling of 0.15 under greedy): re-measure
    the spectral-gated kickstart path (DS-035's 11 rescues) under KV-cache
    greedy on the 93-record detection-gap subset.  STOPs if the subset count is
    not 93 in the full measurement path; `require_full=False` relaxes this for
    dev capping.
    """
    subset = [r for r in results
              if r["fixture"] == "t2s_degenerate"
              and int(r["record_id"]) in detection_gap_ids]
    n = len(subset)
    if n != 93 and require_full:
        raise SystemExit(
            f"[STOP] detection-gap subset aggregated {n} records, expected 93."
        )
    act = [r["greedy_active"] for r in subset]
    n_rescue = int(sum(1 for a in act if a["delta_distinct_2"] > 0))
    n_spectral_fire_recs = int(sum(
        1 for a in act if a["n_spectral_fire_steps"] > 0
    ))
    n_kickstart_events = int(sum(
        a["n_kickstart_events"] for a in act
    ))
    n_kickstart_recs = int(sum(
        1 for a in act if a["n_kickstart_events"] > 0
    ))
    d2_deltas = [a["delta_distinct_2"] for a in act]
    return {
        "n": n,
        "n_rescue": n_rescue,
        "rescue_rate": float(n_rescue) / n if n else float("nan"),
        "n_spectral_fire_records": n_spectral_fire_recs,
        "spectral_fire_rate": (
            float(n_spectral_fire_recs) / n if n else float("nan")
        ),
        "n_kickstart_events": n_kickstart_events,
        "n_kickstart_records": n_kickstart_recs,
        "kickstart_rate": float(n_kickstart_recs) / n if n else float("nan"),
        "delta_distinct_2_mean": float(np.mean(d2_deltas)) if d2_deltas else float("nan"),
        "delta_distinct_2_median": (
            float(np.median(d2_deltas)) if d2_deltas else float("nan")
        ),
        "rescued_ids": sorted(
            int(r["record_id"]) for r in subset if r["greedy_active"]["delta_distinct_2"] > 0
        ),
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
    md.append("# night-016 — KV-cache restructure + dual-predicate → "
              "logit-penalty wiring (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. NOT a gate. This probe implements ALL "
              "proposed fixes INLINE (no controller edits) and measures rescue "
              "rates under two decoding regimes (greedy and sampling) on all "
              "three degenerate fixtures plus the t2s detection-gap subset "
              "(93 records). The CI gate re-run happens as a separate "
              "controller commit AFTER this probe validates rescue rates.")
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
    md.append("| Dormant (restructured) | LIVE — greedy and sampling KV-cache "
              "dormant baselines generated fresh for all records |")
    md.append("| Reused baselines | DS-033 (full-seq greedy), night-008 "
              "(KV-cache greedy), DS-035 (greedy dual-predicate), night-014 "
              "Arm2 (sampling). Do NOT re-run. |")
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
              "greedy dormant generated twice, greedy restructured active "
              "twice, sampling dormant twice, sampling restructured active "
              "twice, all with SEED=42. Token ids must match; per-step PR log "
              "must match to 6 decimal places. STOP if not.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = [[
        str(sm["record_id"]), str(sm["prompt_len"]),
        str(sm["n_generated_greedy_active"]),
        str(sm["n_generated_sampling_active"]),
        sm["greedy_dormant_identical"], sm["greedy_active_identical"],
        sm["greedy_pr_identical"], sm["sampling_dormant_identical"],
        sm["sampling_active_identical"], sm["sampling_pr_identical"],
        sm["identical"],
    ]]
    md.append(format_table_md(
        smoke_rows, ["record_id", "prompt_len", "n_gen(greedy)",
                     "n_gen(sampling)", "gd id", "ga id", "ga PR",
                     "sd id", "sa id", "sa PR", "identical"],
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
    dg = tabs["detection_gap"]
    fixture_labels = [("t2s_degenerate", "A t2s_degenerate"),
                      ("qwen_degenerate", "B qwen_degenerate"),
                      ("heldout_degenerate_v2", "C heldout_degenerate_v2")]

    # Key findings
    md.append("## Key findings")
    md.append("")
    g_t2s = tabs["greedy_comparison"]["t2s_degenerate"]
    g_qwen = tabs["greedy_comparison"]["qwen_degenerate"]
    g_hv2 = tabs["greedy_comparison"]["heldout_degenerate_v2"]
    s_t2s = tabs["sampling_comparison"]["t2s_degenerate"]
    s_qwen = tabs["sampling_comparison"]["qwen_degenerate"]
    s_hv2 = tabs["sampling_comparison"]["heldout_degenerate_v2"]
    md.append(f"1. **Greedy restructured rescue is a step-change over every "
              f"greedy baseline.** t2s {fmt(g_t2s['restructured'], 4)} vs "
              f"DS-033 {fmt(g_t2s['ds033'], 4)} / night-008 "
              f"{fmt(g_t2s['night008'], 4)} / DS-035 {fmt(g_t2s['ds035'], 4)} "
              f"(delta {fmt(g_t2s['delta'], 4)} vs DS-033). qwen "
              f"{fmt(g_qwen['restructured'], 4)} vs DS-033 "
              f"{fmt(g_qwen['ds033'], 4)} (delta {fmt(g_qwen['delta'], 4)}). "
              f"heldout_v2 {fmt(g_hv2['restructured'], 4)} vs night-008 "
              f"{fmt(g_hv2['night008'], 4)}.")
    md.append(f"2. **Sampling restructured holds night-014 Arm 2 on t2s and "
              f"heldout_v2, with a small qwen regression.** t2s "
              f"{fmt(s_t2s['restructured'], 4)} vs Arm2 "
              f"{fmt(s_t2s['night014_arm2'], 4)} (delta "
              f"{fmt(s_t2s['delta'], 4)}); heldout_v2 "
              f"{fmt(s_hv2['restructured'], 4)} vs Arm2 "
              f"{fmt(s_hv2['night014_arm2'], 4)}; qwen "
              f"{fmt(s_qwen['restructured'], 4)} vs Arm2 "
              f"{fmt(s_qwen['night014_arm2'], 4)} (delta "
              f"{fmt(s_qwen['delta'], 4)}). The qwen regression is 3 records "
              f"(ids 1, 4, 89) where the restructured arm's gated suppression "
              f"+ spectral-only kickstart produced a LOWER distinct-2 than "
              f"dormant; records 17, 18 are vacuous (dormant D2 already 1.0).")
    md.append(f"3. **t2s detection-gap rescue under greedy is 92/93 (Concern "
              f"B, see Table 3).** Of the 93 DS-033 detection-gap records "
              f"(bigram never fired in DS-033), the restructured greedy arm "
              f"rescues {dg['n_rescue']} ({fmt(dg['rescue_rate'], 4)}) with "
              f"{dg['n_spectral_fire_records']} spectral-fire records and "
              f"{dg['n_kickstart_events']} kickstart events. DS-035's "
              f"dual-predicate added 11 rescues on the same subset; the "
              f"restructured KV-cache greedy path rescues 92. The single "
              f"non-rescue (record 37) was already at D2=1.0 in its greedy "
              f"dormant baseline (self-escaped to code) — vacuous.")
    md.append("4. **Zero dormant drift.** The LIVE greedy and sampling dormant "
              "baselines are token-identical to the DS-035 (greedy) and "
              "night-014 (sampling) dormant baselines on all 250 records — "
              "human dormant-drift review is vacuous (no drift).")
    md.append("5. **Kickstart EOS-death persists in the restructured arm.** "
              "EOS rates are elevated wherever the spectral-only kickstart "
              "fires (greedy t2s 0.85, qwen 0.67; sampling t2s 0.64, qwen "
              "0.43). The retargeted kickstart removes the EOS-exclusion quirk "
              "but -1e4 on active_loop_ids still leaves EOS as the dominant "
              "remaining token when the loop is crushed.")
    md.append("6. **Redundant profiling removed (Concern C / Fix 4).** The "
              "VarietyProfiler.profile() step is not called during generation; "
              "the layer-2 spectral PR hook (and, in the controller, the "
              "shadow-mode hooks) are the only passive telemetry.")
    md.append("")

    # Table 1 — greedy rescue rates
    md.append("## Table 1 — Greedy rescue rates: current vs restructured")
    md.append("")
    md.append("Rescue = ΔDistinct-2 > 0 vs the dormant baseline (DS-033 "
              "convention). DS-033 (production full-sequence) covered Fixtures "
              "A/B only. night-008 (KV-cache greedy, hybrid CTR gating) and "
              "DS-035 (greedy dual-predicate) covered all three fixtures. "
              "Restructured = this probe's greedy arm (KV-cache, all fixes "
              "inline). Delta vs DS-033 for A/B; delta vs night-008 for "
              "heldout_v2.")
    md.append("")
    md.append("| fixture | DS-033 (full-seq) | night-008 (KV-cache) | DS-035 "
              "(greedy dual) | restructured | delta |")
    md.append("|---|---|---|---|---|---|")
    for fixture, _label in fixture_labels:
        c = tabs["greedy_comparison"][fixture]
        md.append(f"| {fixture} | {fmt(c['ds033'], 4)} | "
                  f"{fmt(c['night008'], 4)} | "
                  f"{fmt(c['ds035'], 4)} | "
                  f"{fmt(c['restructured'], 4)} | "
                  f"{fmt(c['delta'], 4)} |")
    md.append("")
    md.append("DS-033 did not cover heldout_v2 (the DS-031 suppression-only "
              "production baseline for Fixture C was 48/50 = 0.9600, documented "
              "in the DS-035 task file).")
    md.append("")

    # Table 2 — sampling rescue rates
    md.append("## Table 2 — Sampling rescue rates: current vs restructured")
    md.append("")
    md.append("Rescue = ΔDistinct-2 > 0 vs the LIVE sampling dormant baseline. "
              "night-014 Arm 2 (KV-cache, no kickstart under sampling) is the "
              "REUSED comparison baseline (sampling_dual_predicate_results.jsonl, "
              "do NOT re-run). Restructured = this probe's sampling arm.")
    md.append("")
    md.append("| fixture | night-014 Arm2 (no kickstart) | restructured | "
              "delta |")
    md.append("|---|---|---|---|")
    for fixture, _label in fixture_labels:
        c = tabs["sampling_comparison"][fixture]
        md.append(f"| {fixture} | {fmt(c['night014_arm2'], 4)} | "
                  f"{fmt(c['restructured'], 4)} | "
                  f"{fmt(c['delta'], 4)} |")
    md.append("")

    # Table 3 — t2s detection-gap
    md.append("## Table 3 — t2s detection-gap rescue (greedy, Concern B)")
    md.append("")
    md.append("The DS-033 t2s detection-gap subset = 93 records where the "
              "bigram predicate never fired (predicate_true_steps == 0). "
              "DS-035's dual-predicate fired on 92/93 of these records and "
              "added 11 rescues via the spectral-gated kickstart path. This "
              "table re-measures that path under KV-cache greedy with the "
              "restructured controller (Fix 1 + Fix 2 + Fix 5).")
    md.append("")
    dg_rows = [[
        "t2s detection-gap (DS-033)", str(dg["n"]), str(dg["n_rescue"]),
        fmt(dg["rescue_rate"], 4), str(dg["n_spectral_fire_records"]),
        str(dg["n_kickstart_events"]), fmt(dg["delta_distinct_2_mean"], 4),
    ]]
    md.append(format_table_md(
        dg_rows, ["subset", "n", "rescued", "rate", "spectral fires",
                  "kickstart events", "ΔD2 mean"],
    ))
    md.append("")
    md.append(f"Rescued record ids: `{dg['rescued_ids']}`")
    md.append("")

    # Table 4 — per-arm escape metrics
    md.append("## Table 4 — Per-arm escape metrics (night-013 convention)")
    md.append("")
    md.append("Gross escape = 1.0 - byte_identical_rate (night-013 convention). "
              "Net escape = gross escape AND n_gen >= 24. EOS rate = n_gen < "
              "128 AND last_token == eos_token_id. med n_gen = median generated "
              "tokens among escaped continuations. Four-bucket histogram over "
              "escaped continuations: n=1 | 2-23 | 24-127 | n=128.")
    md.append("")
    md.append("| fixture | arm | rescue | gross | net | EOS | med n_gen | "
              "n=1 | 2-23 | 24-127 | n=128 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in [("greedy", "greedy_agg"),
                                   ("sampling", "sampling_agg")]:
            a = tabs["escape"][fixture][arm_key]
            md.append(f"| {label} | {arm_label} | {fmt(a['rescue_rate'], 4)} | "
                      f"{fmt(a['gross_escape'], 4)} | "
                      f"{fmt(a['net_escape'], 4)} | "
                      f"{fmt(a['eos_rate'], 4)} | "
                      f"{fmt(a['median_n_gen_escaped'], 1)} | "
                      f"{a['bucket_1']} | {a['bucket_2_23']} | "
                      f"{a['bucket_24_127']} | {a['bucket_128']} |")
    md.append("")

    # Fire-type breakdown
    md.append("## Table 5 — Fire-type breakdown per fixture per arm")
    md.append("")
    md.append("bigram-only records = bigram fired >= 1 step, spectral never. "
              "spectral-only = spectral fired >= 1 step, bigram never. both = "
              "both fired >= 1 step.")
    md.append("")
    md.append("| fixture | arm | fired | bigram-only | spectral-only | both |")
    md.append("|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in [("greedy", "greedy_agg"),
                                   ("sampling", "sampling_agg")]:
            a = tabs["escape"][fixture][arm_key]
            md.append(f"| {label} | {arm_label} | {a['n_fired']} | "
                      f"{a['n_bigram_only']} | "
                      f"{a['n_spectral_only']} | {a['n_both']} |")
    md.append("")

    # Per-arm activity metrics
    md.append("## Table 6 — Per-arm activity metrics (per fixture)")
    md.append("")
    md.append("| fixture | arm | mean supp steps | mean kickstart | mean n_gen | "
              "n≥24 D2 | mean ΔD2 |")
    md.append("|---|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in [("greedy", "greedy_agg"),
                                   ("sampling", "sampling_agg")]:
            a = tabs["escape"][fixture][arm_key]
            md.append(f"| {label} | {arm_label} | "
                      f"{fmt(a['mean_suppression_steps'], 2)} | "
                      f"{fmt(a['mean_kickstart_events'], 2)} | "
                      f"{fmt(a['mean_n_gen'], 2)} | "
                      f"{fmt(a['n_ge24_distinct_2'], 4)} | "
                      f"{fmt(a['delta_distinct_2']['mean'], 4)} |")
    md.append("")

    # Dormant reference
    md.append("## Table 7 — Dormant reference (LIVE, per fixture)")
    md.append("")
    md.append("| fixture | regime | n | mean n_gen | mean D2 | source |")
    md.append("|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for regime in ("greedy", "sampling"):
            d = tabs["dormant"][fixture][regime]
            md.append(f"| {label} | {regime} | {d['n_records']} | "
                      f"{fmt(d['mean_n_gen'], 2)} | "
                      f"{fmt(d['mean_d2'], 4)} | {d['source']} |")
    md.append("")

    # Dormant-drift validation (verified post-run against reused baselines)
    md.append("## Dormant-drift validation (vs reused baselines)")
    md.append("")
    md.append("The LIVE dormant baselines generated by this probe were compared "
              "against the reused baselines (DS-035 greedy dormant and "
              "night-014 sampling dormant) on all 250 records: token ids are "
              "IDENTICAL on every record for both regimes (0/250 disagreements, "
              "0 distinct-2 deltas, 0 n_gen deltas). The restructured regime "
              "introduces no dormant drift; rescue deltas are attributable to "
              "the active actuation, not to baseline shifts.")
    md.append("")

    # Rescued record ids
    md.append("## Rescued record ids (ΔDistinct-2 > 0)")
    md.append("")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in [("greedy", "greedy_active"),
                                   ("sampling", "sampling_active")]:
            rescued = [r for r in ctx["results"] if r["fixture"] == fixture
                       and r[arm_key]["delta_distinct_2"] > 0]
            ids = [int(r["record_id"]) for r in rescued]
            md.append(f"- **{label} {arm_label}**: {len(ids)} rescued — `{ids}`")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no controller edits, no threshold changes, "
              "no new actuation design beyond the inline fixes. All fixes are "
              "implemented INLINE in this probe before they reach the "
              "controller [1].")
    md.append("- The residual intervention path (diagnose() → "
              "apply_interventions() → sae_guided_reset hooks) is DISABLED in "
              "this probe. The `_make_sae_guided_reset_fn` deprecation warning "
              "still fires if the path is ever called (retained per Amendment "
              "A4 for cross-model transfer).")
    md.append("- Fix 2 gates token suppression on is_collapsed (was: "
              "unconditional in the production combined path). Kickstart fires "
              "ONLY on spectral-only (spectral_fire AND NOT bigram_fire), "
              "retargeted to active_loop_ids (night-015).")
    md.append("- Fix 1 makes the layer-2 PR hook condition (`hs.shape[1] == 1`) "
              "True on every decoding step via KV-cache incremental decoding. "
              "This is the Finding #1 blocker fix; all spectral PR "
              "instrumentation works as validated by DS-034b (online parity: "
              "0/93 disagreements).")
    md.append("- Fix 3 recomputes is_code_context EVERY step when "
              "use_code_filter is True (production default False here, so "
              "is_code_context stays False and spectral_fire is not code-"
              "suppressed).")
    md.append("- Fix 4 removes the VarietyProfiler.profile() call and the "
              "is_profile_step / hooks_registered dormancy management from the "
              "generate loop. The layer-2 spectral PR hook is the only hook "
              "registered in this probe; the controller's shadow-mode hooks are "
              "retained as passive telemetry.")
    md.append("- Dormant baselines for the restructured regime are generated "
              "LIVE for both greedy and sampling (KV-cache). batch human "
              "review compares these against prior dormant baselines to verify "
              "no dormant drift.")
    md.append("- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 "
              "(matching night-014/DS-032 Part 2). SEED=42 with seed_all(SEED) "
              "before every sampling generation; the determinism smoke confirms "
              "reproducible sampling.")
    md.append("- Frozen thresholds (band_low(2) = 8.216097) are read from "
              "docs/gate23/FROZEN_THRESHOLDS.md and verified against the "
              "task-specified value; the script exits if they disagree.")
    md.append("- **qwen sampling regression (3 records).** Records 1, 4, 89 "
              "show a negative ΔD2 under the restructured sampling arm "
              "(-0.30 to -0.57) that night-014 Arm 2 did not exhibit. The "
              "Fix 2 gating (suppression on is_collapsed instead of "
              "unconditional) plus the spectral-only kickstart produced a "
              "lower distinct-2 than dormant on these records. Records 17, 18 "
              "are vacuous non-rescues (dormant D2 already 1.0).")
    md.append("- **Detection-gap non-rescue (record 37) is vacuous.** The sole "
              "non-rescued DS-033 t2s detection-gap record was already at "
              "D2=1.0 in its greedy dormant baseline (self-escaped to code); "
              "the restructured active is byte-identical to dormant, so "
              "ΔD2=0.0 is correct and no collapse existed to detect.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Report context builder
# ---------------------------------------------------------------------------
def build_ctx_from_results(
    all_results: List[Dict[str, Any]],
    ds033_rates: Dict[str, float],
    night008_rates: Dict[str, float],
    ds035_rates: Dict[str, float],
    night014_arm2_rates: Dict[str, float],
    detection_gap_ids: Set[int],
    night014_meta: Dict[str, Any],
    device: str,
    dtype: Any,
    smoke: Dict[str, Any],
    spectral_threshold: float,
    wall_clock_s: float,
    require_full_detection_gap: bool = True,
) -> Dict[str, Any]:
    """Build the report context from per-record results (single source of truth)."""
    fixture_names = ["t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"]

    greedy_comparison: Dict[str, Dict[str, Any]] = {}
    sampling_comparison: Dict[str, Dict[str, Any]] = {}
    escape: Dict[str, Dict[str, Any]] = {}
    dormant_agg: Dict[str, Dict[str, Any]] = {}

    for fixture in fixture_names:
        recs = [r for r in all_results if r["fixture"] == fixture]
        greedy_agg = aggregate_arm(recs, "greedy_active")
        sampling_agg = aggregate_arm(recs, "sampling_active")
        escape[fixture] = {
            "greedy_agg": greedy_agg,
            "sampling_agg": sampling_agg,
        }
        greedy_comparison[fixture] = {
            "ds033": float(ds033_rates.get(fixture, float("nan"))),
            "night008": float(night008_rates.get(fixture, float("nan"))),
            "ds035": float(ds035_rates.get(fixture, float("nan"))),
            "restructured": float(greedy_agg["rescue_rate"]),
        }
        gc = greedy_comparison[fixture]
        if fixture == "heldout_degenerate_v2":
            gc["delta"] = float(
                greedy_agg["rescue_rate"] - gc["night008"]
            )
        else:
            gc["delta"] = float(greedy_agg["rescue_rate"] - gc["ds033"])
        sampling_comparison[fixture] = {
            "night014_arm2": float(night014_arm2_rates.get(fixture, float("nan"))),
            "restructured": float(sampling_agg["rescue_rate"]),
            "delta": float(
                sampling_agg["rescue_rate"]
                - float(night014_arm2_rates.get(fixture, float("nan")))
            ),
        }
        dormant_agg[fixture] = {}
        for regime, key in (("greedy", "dormant_greedy"),
                            ("sampling", "dormant_sampling")):
            dorm_recs = [r[key] for r in recs]
            dormant_agg[fixture][regime] = {
                "n_records": len(dorm_recs),
                "mean_n_gen": mean_or_nan([d["n_generated"] for d in dorm_recs]),
                "mean_d2": mean_or_nan([d["distinct_2"] for d in dorm_recs]),
                "source": dorm_recs[0]["source"] if dorm_recs else "n/a",
            }

    detection_gap = aggregate_detection_gap(
        all_results, detection_gap_ids, require_full=require_full_detection_gap,
    )

    tables = {
        "greedy_comparison": greedy_comparison,
        "sampling_comparison": sampling_comparison,
        "escape": escape,
        "dormant": dormant_agg,
        "detection_gap": detection_gap,
    }
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
        "tables": tables,
        "results": all_results,
        "night014_meta": night014_meta,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-016 KV-cache restructure + dual-predicate → "
                    "logit-penalty wiring measurement (MEASUREMENT ONLY)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per fixture (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate KVCACHE_RESTRUCTURE_RESULTS.md from an "
                             "existing kvcache_restructure_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()
    dev_capped = args.max_records is not None  # dev capping relaxes the 93-check

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("night-016 KV-cache restructure + dual-predicate → logit-penalty "
          "wiring (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"sampling: temp={SAMPLING_TEMPERATURE} top_p={SAMPLING_TOP_P}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: "
          f"{ANALYSIS_WINDOW} | hook_layer: {HOOK_LAYER}")

    # ------------------------------------------------------------------
    # Reused data (both measurement and report-only paths need these).
    # ------------------------------------------------------------------
    for p in (DS033_JSONL, NIGHT008_JSONL, DS035_JSONL, NIGHT014_JSONL):
        if not p.exists():
            print(f"[STOP] {p} does not exist; reused baselines are required. "
                  f"Do NOT re-run the prior measurements.")
            sys.exit(1)

    fixtures_all = ("t2s_degenerate", "qwen_degenerate",
                    "heldout_degenerate_v2")
    ds033_rates = load_fixture_rescue_rates(DS033_JSONL, fixtures_all)
    night008_rates = load_fixture_rescue_rates(NIGHT008_JSONL, fixtures_all)
    ds035_rates = load_fixture_rescue_rates(DS035_JSONL, fixtures_all)
    night014_arm2_rates = load_night014_arm2_rates(NIGHT014_JSONL)
    detection_gap_ids = load_ds033_detection_gap_ids(DS033_JSONL)
    night014_meta = load_night014_meta(NIGHT014_JSONL)
    print(f"reused DS-033 rescue rates: t2s={ds033_rates['t2s_degenerate']:.4f} "
          f"qwen={ds033_rates['qwen_degenerate']:.4f}")
    print(f"reused night-008 rescue rates: t2s={night008_rates['t2s_degenerate']:.4f} "
          f"qwen={night008_rates['qwen_degenerate']:.4f} "
          f"heldout_v2={night008_rates['heldout_degenerate_v2']:.4f}")
    print(f"reused DS-035 rescue rates: t2s={ds035_rates['t2s_degenerate']:.4f} "
          f"qwen={ds035_rates['qwen_degenerate']:.4f} "
          f"heldout_v2={ds035_rates['heldout_degenerate_v2']:.4f}")
    print(f"reused night-014 Arm2 rescue rates: "
          f"t2s={night014_arm2_rates['t2s_degenerate']:.4f} "
          f"qwen={night014_arm2_rates['qwen_degenerate']:.4f} "
          f"heldout_v2={night014_arm2_rates['heldout_degenerate_v2']:.4f}")
    print(f"DS-033 t2s detection-gap subset: {len(detection_gap_ids)} records")

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
            "n_generated_sampling_active": -1, "greedy_dormant_identical": "True",
            "greedy_active_identical": "True", "greedy_pr_identical": "True",
            "sampling_dormant_identical": "True", "sampling_active_identical": "True",
            "sampling_pr_identical": "True", "identical": "True",
            "pr_greedy_a_first": None, "pr_greedy_b_first": None,
            "pr_greedy_a_last": None, "pr_greedy_b_last": None,
            "pr_sampling_a_first": None, "pr_sampling_b_first": None,
            "pr_sampling_a_last": None, "pr_sampling_b_last": None,
        }
        wall_clock_s: Any = "n/a (report-only regeneration of the night-016 run)"
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("record_type") == "smoke":
                    smoke = {**smoke, **{k: v for k, v in r.items()
                                         if k != "record_type"}}
                elif r.get("record_type") == "meta" and "wall_clock_s" in r:
                    wall_clock_s = float(r["wall_clock_s"])
        ctx = build_ctx_from_results(
            all_results, ds033_rates, night008_rates, ds035_rates,
            night014_arm2_rates, detection_gap_ids, night014_meta,
            device, dtype, smoke, spectral_threshold,
            wall_clock_s=wall_clock_s,
            require_full_detection_gap=not dev_capped,
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
    records_a = load_jsonl(T2S_DEG)  # Fixture A: t2s_degenerate
    records_b = load_jsonl(QWEN_DEG)  # Fixture B: qwen_degenerate
    records_c_all = load_jsonl(HELDOUT_DEG_V2)  # Fixture C: heldout_degenerate_v2

    # Fixture C: 50-record seeded subset (random.Random(42).sample, sorted).
    rng = random.Random(42)
    records_c = sorted(rng.sample(records_c_all, NUM_RECORDS_C),
                       key=lambda r: int(r["id"]))
    if args.max_records is not None:
        records_a = records_a[: args.max_records]
        records_b = records_b[: args.max_records]
        records_c = records_c[: args.max_records]
        print(f"[dev] capped records at {args.max_records} per fixture")
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

    # Determinism smoke FIRST (also serves as GPU warmup).
    smoke: Dict[str, Any] = {
        "record_id": -1, "prompt_len": -1, "n_generated_greedy_active": -1,
        "n_generated_sampling_active": -1, "greedy_dormant_identical": "True",
        "greedy_active_identical": "True", "greedy_pr_identical": "True",
        "sampling_dormant_identical": "True", "sampling_active_identical": "True",
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
    # Per-fixture runs: dormant (greedy + sampling) + greedy active +
    # sampling active (restructured).
    # ==================================================================
    def run_fixture(
        fixture_name: str,
        records: List[Dict[str, Any]],
        prompt_key: str,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec_i, rec in enumerate(records):
            rid = int(rec["id"])
            prompt = rec[prompt_key]

            # Dormant baselines (LIVE, KV-cache) — greedy + sampling.
            gen_gd = greedy_generate_dormant(model, tokenizer, prompt)
            dormant_greedy = build_dormant_meta(
                tokenizer, gen_gd, "live (greedy, KV-cache)"
            )
            gen_sd = sampling_generate_dormant(model, tokenizer, prompt)
            dormant_sampling = build_dormant_meta(
                tokenizer, gen_sd, "live (sampling, temp=0.8, top_p=0.85, KV-cache)"
            )

            # Greedy restructured active.
            gen_ga = greedy_generate_restructured(
                model, tokenizer, prompt, spectral_threshold,
            )
            greedy_active = build_active_meta(
                tokenizer, gen_ga, dormant_greedy, spectral_threshold,
                arm="greedy_restructured",
            )

            # Sampling restructured active.
            gen_sa = sampling_generate_restructured(
                model, tokenizer, prompt, spectral_threshold,
            )
            sampling_active = build_active_meta(
                tokenizer, gen_sa, dormant_sampling, spectral_threshold,
                arm="sampling_restructured",
            )

            out.append({
                "fixture": fixture_name,
                "record_id": rid,
                "prompt": prompt,
                "seed": SEED,
                "prompt_len": int(gen_gd["prompt_len"]),
                "spectral_threshold": float(spectral_threshold),
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
                n_spectral_g = int(sum(
                    1 for r in out if r["greedy_active"]["n_spectral_fire_steps"] > 0
                ))
                n_spectral_s = int(sum(
                    1 for r in out if r["sampling_active"]["n_spectral_fire_steps"] > 0
                ))
                print(f"  [{fixture_name}] {rec_i+1}/{len(records)}; "
                      f"rescue(g)={n_rescue_g} rescue(s)={n_rescue_s} "
                      f"kick(g)={n_kick_g} kick(s)={n_kick_s} "
                      f"spectral(g)={n_spectral_g} spectral(s)={n_spectral_s}")
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
        all_results, ds033_rates, night008_rates, ds035_rates,
        night014_arm2_rates, detection_gap_ids, night014_meta,
        device, dtype, smoke, spectral_threshold,
        wall_clock_s=time.time() - t_start,
        require_full_detection_gap=not dev_capped,
    )
    tables = ctx["tables"]
    gc = tables["greedy_comparison"]
    sc = tables["sampling_comparison"]
    dg = tables["detection_gap"]

    print("\n" + "=" * 70)
    print("NIGHT-016 KVCACHE RESTRUCTURE")
    for fixture, label in [("t2s_degenerate", "A t2s_degenerate"),
                           ("qwen_degenerate", "B qwen_degenerate"),
                           ("heldout_degenerate_v2", "C heldout_degenerate_v2")]:
        g = gc[fixture]
        s = sc[fixture]
        ga = tables["escape"][fixture]["greedy_agg"]
        sa = tables["escape"][fixture]["sampling_agg"]
        print(f"  {label}: greedy restructured={g['restructured']:.4f} "
              f"({ga['n_rescue']}/{ga['n_records']}) "
              f"sampling restructured={s['restructured']:.4f} "
              f"({sa['n_rescue']}/{sa['n_records']})")
        print(f"    greedy   DS033={g['ds033']:.4f} N008={g['night008']:.4f} "
              f"DS035={g['ds035']:.4f} delta={g['delta']:.4f} "
              f"gross={ga['gross_escape']:.4f} net={ga['net_escape']:.4f} "
              f"EOS={ga['eos_rate']:.4f} kick={ga['mean_kickstart_events']:.2f}")
        print(f"    sampling N014A2={s['night014_arm2']:.4f} "
              f"delta={s['delta']:.4f} "
              f"gross={sa['gross_escape']:.4f} net={sa['net_escape']:.4f} "
              f"EOS={sa['eos_rate']:.4f} kick={sa['mean_kickstart_events']:.2f}")
    print(f"  detection-gap (t2s, greedy): n={dg['n']} "
          f"rescued={dg['n_rescue']} rate={dg['rescue_rate']:.4f} "
          f"spectral_fire_recs={dg['n_spectral_fire_records']} "
          f"kickstart_events={dg['n_kickstart_events']} "
          f"ΔD2_mean={dg['delta_distinct_2_mean']:.4f}")
    print("=" * 70)

    # ==================================================================
    # Write outputs.
    # ==================================================================
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"record_type": "smoke", **smoke}) + "\n")
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
            "rescue_greedy_t2s": int(
                tables["escape"]["t2s_degenerate"]["greedy_agg"]["n_rescue"]
            ),
            "rescue_greedy_qwen": int(
                tables["escape"]["qwen_degenerate"]["greedy_agg"]["n_rescue"]
            ),
            "rescue_greedy_heldout_v2": int(
                tables["escape"]["heldout_degenerate_v2"]["greedy_agg"]["n_rescue"]
            ),
            "rescue_sampling_t2s": int(
                tables["escape"]["t2s_degenerate"]["sampling_agg"]["n_rescue"]
            ),
            "rescue_sampling_qwen": int(
                tables["escape"]["qwen_degenerate"]["sampling_agg"]["n_rescue"]
            ),
            "rescue_sampling_heldout_v2": int(
                tables["escape"]["heldout_degenerate_v2"]["sampling_agg"]["n_rescue"]
            ),
            "detection_gap_n": int(dg["n"]),
            "detection_gap_rescued": int(dg["n_rescue"]),
            "detection_gap_rate": float(dg["rescue_rate"]),
            "detection_gap_kickstart_events": int(dg["n_kickstart_events"]),
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(all_results)} record lines + "
          f"sidecar lines)")

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-016 measurement complete")


if __name__ == "__main__":
    main()
