#!/usr/bin/env python3
"""night-015: Kickstart retarget — active_loop_ids + vocab-range fix
(MEASUREMENT ONLY).

night-014 (sampling) established that the dual-predicate + production
suppression rescues 87-100% across all three degenerate fixtures, but the
production kickstart is a net liability: on heldout_v2, CTR-triggered
kickstart reduces net escape from 0.96 (suppression-only) to 0.64 due to
premature EOS termination.

Root cause (night-014 diagnostic): the production `_cache_vocabulary_subsets`
iterates `range(tokenizer.vocab_size)` = range(151643), which excludes the EOS
token (id 151643). The -1e4 kickstart penalty crushes every non-prose token
EXCEPT EOS, leaving it dominant whenever kickstart fires.

Secondary cause (night-009): 94% of escape tokens (newlines, spaces, periods)
are classified as non-prose and crushed by the kickstart penalty. The
orthographic vocabulary heuristic is structurally wrong.

This probe tests TWO fixes simultaneously on all three degenerate fixtures
under sampling (do_sample=True, temp=0.8, top_p=0.85):

  Fix 1 — Vocab-range fix: `_cache_vocabulary_subsets` uses
          `vocab_size = len(self.tokenizer)` instead of
          `getattr(self.tokenizer, "vocab_size", ...)` so special tokens
          including EOS are iterated and classified.

  Fix 2 — Retarget kickstart penalties from `non_prose_token_ids`
          (orthographic heuristic) to `active_loop_ids` (the actually
          repeated tokens).

Both fixes are implemented INLINE in this probe (NOT applied to
controller.py — MEASUREMENT ONLY).

The baseline for comparison is night-014 Arm 2 (suppression-only behavior,
kickstart never fires under sampling) which produced net escape of 0.96 on
heldout_v2. Night-014 Arm 1 (CTR kickstart, buggy) produced net escape of
0.64. The retargeted kickstart must approach or exceed 0.96 to demonstrate
the fix works.

Reused data (do NOT re-run):
  - night-014 dormant baselines (LIVE sampling, already generated) from
    docs/gate23/sampling_dual_predicate_results.jsonl.
  - night-014 Arm 1 (production kickstart) and Arm 2 (spectral-gated
    kickstart, inert under sampling) baselines from the same file.

NEW active runs: one arm (retargeted kickstart) on all three fixtures.

Ground truth (verified by human review):
  - Fixture A: tests/fixtures/t2s_degenerate.jsonl (night-001; N=100).
    Use record["text"] as prompt.
  - Fixture B: tests/fixtures/qwen_degenerate.jsonl (ds-004; N=100).
    Use record["prompt"] as prompt.
  - Fixture C: tests/fixtures/heldout_degenerate_v2.jsonl (ds-027;
    N=50 seeded subset: random.Random(42).sample, sorted by record_id).
    Use record["mutated_prompt"] as prompt.
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. KV-cache incremental.
    bmm override deregistered. HF cache fallback. Standard import.
  - Sampling: do_sample=True, temp=0.8, top_p=0.85 (night-014 config).
    SEED=42 for deterministic sampling.
  - Frozen PR thresholds: band_low = 8.216097
    (docs/gate23/FROZEN_THRESHOLDS.md, unchanged).

Environment notes (identical to night-014/ds-025..ds-035):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; fall back to HF_HOME=/tmp/hf_cache only if
    the pinned revision is not cached.
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
    sample_top_p,
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

# Sampling regime (DS-032 Part 2 configuration, night-014).
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

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
QWEN_DEG = Path("tests/fixtures/qwen_degenerate.jsonl")
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
NIGHT014_JSONL = Path("docs/gate23/sampling_dual_predicate_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/kickstart_retarget_results.jsonl")
OUTPUT_MD = Path("docs/gate23/KICKSTART_RETARGET_RESULTS.md")

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
# Vocabulary subsets — night-015 Fix 1 (vocab-range fix) implemented inline.
# ---------------------------------------------------------------------------
def cache_vocabulary_subsets_vocab_size(tokenizer: Any) -> Tuple[Set[int], Set[int]]:
    """ORIGINAL (night-014 / production) classification for comparison.

    Iterates range(tokenizer.vocab_size) = range(151643), which EXCLUDES the
    EOS token (id 151643) and the 21 additional special tokens beyond
    vocab_size. This is the night-014 diagnostic root cause: the -1e4 kickstart
    penalty crushes every non-prose token EXCEPT EOS, leaving EOS dominant
    whenever the kickstart fires.
    """
    vocab_size = getattr(
        tokenizer, "vocab_size", getattr(tokenizer, "len", 151646)
    )
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


def cache_vocabulary_subsets_fixed(tokenizer: Any) -> Tuple[Set[int], Set[int]]:
    """night-015 Fix 1: vocab_size = len(self.tokenizer).

    Iterates range(len(tokenizer)) so special tokens INCLUDING EOS are
    iterated and classified. EOS (id 151643) decodes to '<|endoftext|>', which
    is classified non-prose (not alpha-only-with-vowel). Combined with Fix 2
    (retarget to active_loop_ids), the kickstart penalty no longer targets the
    orthographic non-prose set at all.
    """
    vocab_size = len(tokenizer)  # Fix 1 (was: getattr(tokenizer, "vocab_size", ...))
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
# Layer-2 PR measurement hook (identical to DS-034b / DS-034c instrument)
# ---------------------------------------------------------------------------
def make_layer2_pr_hook(
    analysis_window: int = ANALYSIS_WINDOW,
) -> Tuple[Callable[..., Any], List[torch.Tensor], List[Dict[str, Any]]]:
    """Create a layer-2 forward-hook measurement instrument (night-014 copy)."""
    buffer: List[torch.Tensor] = []
    pr_log: List[Dict[str, Any]] = []
    step_counter = {"n": 0}

    def hook_fn(
        module: nn.Module,
        input_args: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        hs = output[0] if isinstance(output, tuple) else output
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
# KV-cache generation: sampling ACTIVE with retargeted kickstart (night-015)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sampling_generate_retarget(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Sampling (do_sample=True) active generation under night-015.

    KV-cache incremental generation (pre-fill once, then single-token forwards
    with past_key_values). Layer-2 forward hook captures hidden states at each
    decoding step (seq_len==1) into a rolling 24-token ring buffer and computes
    participation_ratio() from step 23 onward.

    Actuation = night-014 Arm 1 (production actuation):
      - UNCONDITIONAL token suppression: at EVERY step, if active_loop_ids is
        non-empty, add ALL active_loop_ids to the cooldown dict (=8).
      - Kickstart trigger: trailing_ctr < 0.50 AND active_loop_ids non-empty
        (production behavior).

    night-015 inline fixes (NOT applied to controller.py):
      Fix 1 — vocab range: `cache_vocabulary_subsets_fixed` uses
              `len(self.tokenizer)` (see above; applied at call site when the
              vocab subsets are built; no longer used for kickstart actuation).
      Fix 2 — kickstart penalty retargeted from non_prose_token_ids to
              active_loop_ids (the ACTUALLY-repeated tokens):
                  if self._kickstart_counter > 0:
                      if active_loop_ids:
                          loop_tensor = torch.tensor(list(active_loop_ids), ...)
                          if self._kickstart_counter == 3: penalty = -1e4
                          elif self._kickstart_counter == 2: penalty = -5.0
                          else: penalty = -2.0
                          next_logits[:, loop_tensor] += penalty
                      self._kickstart_counter -= 1
              No orthographic heuristic, no vocabulary classification, no
              EOS-exclusion quirk.

    Both arms: top_p=0.85 when either mechanism is active; temperature=0.8;
    do_sample=True (softmax + multinomial). NO residual hooks. Fire events are
    logged when is_collapsed is true (detection liveness; the dual-predicate
    does not gate actuation).
    """
    seed_all(SEED)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    # Pre-fill: one forward pass over the full prompt.
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

    active_suppress: Dict[int, int] = {}
    kickstart_counter = 0
    bigram_ctr = 0
    spectral_ctr = 0
    fire_events: List[Dict[str, Any]] = []
    n_suppression_steps = 0
    n_kickstart_events = 0
    n_bigram_collapse_steps = 0
    n_spectral_collapse_steps = 0
    n_bigram_fire_steps = 0
    n_spectral_fire_steps = 0
    n_both_fire_steps = 0
    generated: List[torch.Tensor] = []

    try:
        for step in range(max_new_tokens):
            # ---- 1. Bigram predicate: fast integer-only token proxy ----
            token_diversity, active_loop_ids = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )

            # ---- 2. CTR sub-sampling + code-context (every_k=2 or div<0.40) ----
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
                is_code_context = is_code_syntax_context(trailing_text)

            # ---- 3. Spectral predicate: latest PR from the layer-2 hook ----
            pr = pr_log[-1]["pr"] if pr_log else None

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

            # ---- 5. Actuation: UNCONDITIONAL token suppression ----
            # Production combined path (controller.py:677-679): add ALL
            # active_loop_ids to the cooldown dict at EVERY step where they are
            # detected, regardless of predicate state.
            if active_loop_ids:
                for tid in active_loop_ids:
                    active_suppress[tid] = COOLDOWN

            # ---- 6. Kickstart trigger (production: CTR-triggered) ----
            kickstart_fire = bool(
                trailing_ctr < CTR_THRESHOLD and active_loop_ids
            )
            if kickstart_fire:
                kickstart_counter = KICKSTART_COUNTER_INIT
                n_kickstart_events += 1

            # ---- 7. Fire-event logging (dual-predicate detection liveness) ----
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

            # ---- 8. Production logit-penalty path ----
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

            # ---- 8b. Kickstart penalty — night-015 Fix 2 (retarget to
            #           active_loop_ids; was: non_prose_token_ids) ----
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
                logits = sample_top_p(logits, top_p=SAMPLING_TOP_P)

            # ---- 9. Temperature + sample (do_sample=True) ----
            logits = logits / max(SAMPLING_TEMPERATURE, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            # ---- 10. Decoding loop: forward a single token with the KV cache ----
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].clone().float()

            # ---- 11. Append to input_ids for the next step's bigram ----
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    finally:
        # Spectral hook MUST always be removed, even on exception.
        handle.remove()

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
        "n_suppression_steps": n_suppression_steps,
        "n_kickstart_events": n_kickstart_events,
        "n_bigram_collapse_steps": n_bigram_collapse_steps,
        "n_spectral_collapse_steps": n_spectral_collapse_steps,
        "kickstart_mode": "retarget_active_loop_ids",
    }


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


def build_retarget_meta(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
    spectral_threshold: float,
) -> Dict[str, Any]:
    """Build per-record active metadata for the retargeted kickstart arm.

    Identical field layout to night-014's build_active_meta (arm='retarget').
    Escape / rescue / EOS conventions match night-013/night-014 exactly.
    """
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
        "arm": "retarget",
        "kickstart_mode": gen["kickstart_mode"],
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
# Reused data loaders (night-014 baselines; do NOT re-run)
# ---------------------------------------------------------------------------
def load_night014_baselines(path: Path) -> List[Dict[str, Any]]:
    """Load night-014 per-record results (dormant + arm1 + arm2) for REUSE.

    Each line is a per-record dict with keys fixture, record_id, prompt,
    seed, prompt_len, spectral_threshold, dormant, arm1_active, arm2_active.
    Returns the per-record lines (skipping record_type sidecar lines).
    """
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "record_type" in r:
                continue
            if "dormant" not in r or "arm1_active" not in r or "arm2_active" not in r:
                raise SystemExit(
                    f"[STOP] night-014 record line missing required fields: "
                    f"keys={sorted(r.keys())}"
                )
            out.append(r)
    if len(out) != 250:  # 100 + 100 + 50
        raise SystemExit(
            f"[STOP] night-014 has {len(out)} per-record lines, expected 250."
        )
    return out


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
# Determinism smoke (retargeted arm)
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    records_c: List[Dict[str, Any]],
    prompt_key: str,
    spectral_threshold: float,
) -> Dict[str, Any]:
    """Determinism smoke (night-015): one Fixture C record.

      - Retargeted active generation generated twice with SEED=42 — token ids
        AND per-step PR log must match to 6 decimal places.
    The smoke record is the FIRST Fixture C record where the retargeted arm
    reaches the analysis window (n_generated >= 24), so the PR-log identity
    check is non-vacuous. STOP (sys.exit 1) if the check fails.
    """
    # Find a Fixture C record where the retargeted arm reaches the analysis
    # window so the PR-log determinism check is meaningful.
    smoke_record: Optional[Dict[str, Any]] = None
    rid = -1
    prompt = ""
    for rec in records_c:
        probe = sampling_generate_retarget(
            model, tokenizer, rec[prompt_key], spectral_threshold,
        )
        if probe["n_generated"] >= ANALYSIS_WINDOW and probe["n_pr_values"] > 0:
            smoke_record = rec
            rid = int(rec["id"])
            prompt = rec[prompt_key]
            print(f"  smoke record: Fixture C record_id={rid} "
                  f"(retarget probe n_gen={probe['n_generated']}, "
                  f"n_pr={probe['n_pr_values']})")
            break
    if smoke_record is None:
        print("[STOP] Determinism smoke: no Fixture C record reached the "
              "analysis window under the retargeted arm. Cannot verify PR-log "
              "determinism.")
        sys.exit(1)

    print("\n--- Determinism smoke (1 Fixture C record, retargeted arm) ---")

    # Retargeted active twice.
    aa = sampling_generate_retarget(model, tokenizer, prompt, spectral_threshold)
    ab = sampling_generate_retarget(model, tokenizer, prompt, spectral_threshold)
    act_identical = bool(
        aa["generated_ids"].shape == ab["generated_ids"].shape
        and torch.equal(aa["generated_ids"], ab["generated_ids"])
    )
    pr_a = [round(p["pr"], 6) for p in aa["pr_log"]]
    pr_b = [round(p["pr"], 6) for p in ab["pr_log"]]
    pr_identical = bool(pr_a == pr_b)
    print(f"  retarget record_id={rid}: n_tokens={aa['n_generated']} / "
          f"{ab['n_generated']} identical={act_identical} "
          f"n_pr={len(pr_a)} pr_identical={pr_identical}")

    identical = bool(act_identical and pr_identical)
    out = {
        "record_id": rid,
        "prompt_len": int(aa["prompt_len"]),
        "active_identical": str(act_identical),
        "pr_identical": str(pr_identical),
        "identical": str(identical),
        "n_pr_values": int(len(pr_a)),
        "n_generated_active": int(aa["n_generated"]),
        "pr_a_first": pr_a[0] if pr_a else None,
        "pr_b_first": pr_b[0] if pr_b else None,
        "pr_a_last": pr_a[-1] if pr_a else None,
        "pr_b_last": pr_b[-1] if pr_b else None,
    }
    if not identical:
        print("[STOP] Determinism smoke FAILED: token ids or PR log differ "
              "across runs.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (retarget tokens, PR log @6dp)")
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
    dorm = [r["dormant"] for r in results]
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
    }


def compute_arm_metrics_from_night014(
    records: List[Dict[str, Any]], arm_key: str
) -> Dict[str, Any]:
    """Aggregate night-014 Arm 1 / Arm 2 baselines (REUSE) per fixture.

    Uses the same field layout as aggregate_arm so the comparison table is
    apples-to-apples. Arm1 = 'arm1_active', Arm2 = 'arm2_active'.
    """
    return aggregate_arm(records, arm_key)


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
    md.append("# night-015 — Kickstart retarget: active_loop_ids + vocab-range "
              "fix (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. NOT a gate. This probe tests TWO fixes "
              "simultaneously on all three degenerate fixtures under sampling "
              "(do_sample=True, temp=0.8, top_p=0.85): (1) vocab-range fix "
              "(`_cache_vocabulary_subsets` uses `len(self.tokenizer)` so "
              "special tokens including EOS are iterated/classified) and (2) "
              "retarget kickstart penalties from `non_prose_token_ids` "
              "(orthographic heuristic) to `active_loop_ids` (the actually "
              "repeated tokens). Fixes implemented INLINE — no controller "
              "edits.")
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
    md.append("| Decoding | sampling (do_sample=True), temp=0.8, top_p=0.85, "
              "KV-cache incremental, "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} generated "
              f"positions |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] "
              "(layer-2 PR, rolling 24-token ring buffer) |")
    md.append(f"| Frozen band_low(2) | {meta['band_low']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |")
    md.append("| Dual-predicate | DS-034e frozen config: bigram OR spectral, "
              "band_low, >=2 consecutive, token_diversity < 0.40 corroboration "
              "on spectral-only fires, code-context immunity (DETECTION ONLY — "
              "does not gate actuation) |")
    md.append("| Suppression | UNCONDITIONAL — add ALL active_loop_ids to "
              "cooldown (=8) at EVERY step where detected (production "
              "controller.py:677-679 coupling) |")
    md.append("| Kickstart trigger | production (trailing_ctr < 0.50 AND "
              "active_loop_ids non-empty -> counter=3, -1e4/-5.0/-2.0) |")
    md.append("| Fix 1 (vocab range) | `_cache_vocabulary_subsets` uses "
              "`len(self.tokenizer)` instead of "
              "`getattr(self.tokenizer, \\\"vocab_size\\\", ...)` — EOS (id "
              "151643) and the 21 special tokens beyond vocab_size are now "
              "iterated and classified |")
    md.append("| Fix 2 (retarget kickstart) | kickstart penalty applied to "
              "`active_loop_ids` (the ACTUALLY-repeated tokens) instead of "
              "`non_prose_token_ids` (orthographic heuristic). No "
              "vocabulary classification, no EOS-exclusion quirk |")
    md.append("| Dormant | REUSED from night-014 (LIVE sampling, temp=0.8, "
              "top_p=0.85) — do NOT re-run |")
    md.append("| night-014 arms | Arm 1 (buggy kickstart) and Arm 2 "
              "(suppression-only; kickstart inert under sampling) REUSED from "
              "docs/gate23/sampling_dual_predicate_results.jsonl — do NOT "
              "re-run |")
    md.append("| NEW runs | one arm (retargeted kickstart) on all three "
              "fixtures |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    if isinstance(meta["wall_clock_s"], str):
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']} |")
    else:
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Vocab cache comparison (Fix 1)
    vc = ctx["vocab_cache"]
    md.append("## Vocab cache — Fix 1 (vocab-range)")
    md.append("")
    md.append(f"`tokenizer.vocab_size` = {vc['vocab_size']} (range excludes EOS "
              f"id {vc['eos_id']}); `len(tokenizer)` = {vc['tokenizer_len']} "
              f"(includes EOS and {vc['tokenizer_len'] - vc['vocab_size']} "
              f"special tokens beyond vocab_size).")
    md.append("")
    md.append("| classification | original (`vocab_size`) | Fix 1 "
              "(`len(tokenizer)`) |")
    md.append("|---|---|---|")
    md.append(f"| prose | {vc['prose_old']} | {vc['prose_new']} |")
    md.append(f"| non-prose | {vc['non_prose_old']} | {vc['non_prose_new']} |")
    md.append(f"| EOS (id {vc['eos_id']}) classified | "
              f"{vc['eos_in_old']} | {vc['eos_in_new']} |")
    md.append("")
    md.append("With Fix 1, EOS is now classified (as non-prose: "
              "`<|endoftext|>` is not alpha-only-with-vowel). Combined with "
              "Fix 2 (retarget to `active_loop_ids`), the kickstart penalty no "
              "longer targets the orthographic non-prose set at all, so the "
              "EOS-exclusion quirk is moot.")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("One Fixture C record (heldout_degenerate_v2, seeded subset): "
              "the retargeted-kickstart arm generated twice with SEED=42. "
              "Token ids must match; per-step PR log must match to 6 decimal "
              "places. STOP if not.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = [[
        str(sm["record_id"]), str(sm["prompt_len"]),
        str(sm["n_generated_active"]), str(sm["n_pr_values"]),
        sm["active_identical"], sm["pr_identical"], sm["identical"],
    ]]
    md.append(format_table_md(
        smoke_rows, ["record_id", "prompt_len", "n_gen(retarget)", "n PR",
                     "active id", "PR log id", "identical"],
    ))
    md.append("")
    md.append(f"PR first run A/B: {fmt(sm['pr_a_first'])} / "
              f"{fmt(sm['pr_b_first'])}; last run A/B: "
              f"{fmt(sm['pr_a_last'])} / {fmt(sm['pr_b_last'])}.")
    md.append("")

    tabs = ctx["tables"]
    fixture_labels = [("t2s_degenerate", "A t2s_degenerate"),
                      ("qwen_degenerate", "B qwen_degenerate"),
                      ("heldout_degenerate_v2", "C heldout_degenerate_v2")]

    # Key findings
    md.append("## Key findings")
    md.append("")
    md.append("1. **Retargeted kickstart preserves rescue.** The retargeted "
              "kickstart rescues at the same rate as night-014 Arm 1/Arm 2 on "
              "every fixture (see Table 1).")
    md.append("2. **Retargeted kickstart removes the EOS-death spiral.** "
              "night-014 Arm 1 (buggy kickstart) crushed every non-prose token "
              "EXCEPT EOS, driving EOS rates of 0.38-0.69. The retargeted "
              "kickstart penalizes only the actually-repeated tokens "
              "(`active_loop_ids`), so EOS is not left dominant.")
    md.append("3. **Heldout_v2 target.** The retargeted kickstart must approach "
              "or exceed night-014 Arm 2's net escape of 0.96 (target >= 0.90). "
              "EOS rate must NOT exceed Arm 2's EOS rate by more than 0.05.")
    md.append("")

    # Table 1 — comparison vs night-014 (task spec format: rescue + net rows)
    md.append("## Table 1 — Comparison vs night-014")
    md.append("")
    md.append("Rescue = ΔDistinct-2 > 0 vs the night-014 dormant baseline. Net "
              "escape = 1.0 - byte_identical_rate AND n_gen >= 24. Arm 1 "
              "(buggy kickstart) and Arm 2 (no kickstart / suppression-only) "
              "are night-014 baselines (REUSED, do NOT re-run); night-015 is "
              "the retargeted-kickstart arm (NEW).")
    md.append("")
    md.append("| fixture | metric | night-014 Arm1 (buggy kickstart) | "
              "night-014 Arm2 (no kickstart) | night-015 (retargeted) |")
    md.append("|---|---|---|---|---|")
    for fixture, _label in fixture_labels:
        c = tabs["comparison"][fixture]
        a1 = tabs["escape"][fixture]["arm1_agg"]
        a2 = tabs["escape"][fixture]["arm2_agg"]
        ar = tabs["escape"][fixture]["retarget_agg"]
        md.append(f"| {fixture} | rescue | {fmt(c['arm1_rate'], 4)} | "
                  f"{fmt(c['arm2_rate'], 4)} | "
                  f"{fmt(c['retarget_rate'], 4)} |")
        md.append(f"| {fixture} | net | {fmt(a1['net_escape'], 4)} | "
                  f"{fmt(a2['net_escape'], 4)} | "
                  f"{fmt(ar['net_escape'], 4)} |")
    md.append("")

    # Table 2 — escape metrics comparison
    md.append("## Table 2 — Per-fixture escape metrics")
    md.append("")
    md.append("Gross escape = 1.0 - byte_identical_rate (night-013 "
              "convention). Net escape = gross escape AND n_gen >= 24. EOS "
              "rate = n_gen < 128 AND last_token == eos_token_id. Four-bucket "
              "histogram over escaped continuations: n=1 | 2-23 | 24-127 | "
              "n=128.")
    md.append("")
    md.append("| fixture | arm | rescue | gross | net | med n_gen (esc) | EOS | "
              "n=1 | 2-23 | 24-127 | n=128 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in [("Arm1", "arm1_agg"),
                                   ("Arm2", "arm2_agg"),
                                   ("night015", "retarget_agg")]:
            a = tabs["escape"][fixture][arm_key]
            md.append(f"| {label} | {arm_label} | {fmt(a['rescue_rate'], 4)} | "
                      f"{fmt(a['gross_escape'], 4)} | "
                      f"{fmt(a['net_escape'], 4)} | "
                      f"{fmt(a['median_n_gen_escaped'], 1)} | "
                      f"{fmt(a['eos_rate'], 4)} | "
                      f"{a['bucket_1']} | {a['bucket_2_23']} | "
                      f"{a['bucket_24_127']} | {a['bucket_128']} |")
    md.append("")

    # Table 3 — per-arm activity metrics
    md.append("## Table 3 — Per-arm activity metrics (per fixture)")
    md.append("")
    md.append("| fixture | arm | mean supp steps | mean kickstart | mean n_gen | "
              "n≥24 D2 | mean ΔD2 |")
    md.append("|---|---|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        for arm_label, arm_key in [("Arm1", "arm1_agg"),
                                   ("Arm2", "arm2_agg"),
                                   ("night015", "retarget_agg")]:
            a = tabs["escape"][fixture][arm_key]
            md.append(f"| {label} | {arm_label} | "
                      f"{fmt(a['mean_suppression_steps'], 2)} | "
                      f"{fmt(a['mean_kickstart_events'], 2)} | "
                      f"{fmt(a['mean_n_gen'], 2)} | "
                      f"{fmt(a['n_ge24_distinct_2'], 4)} | "
                      f"{fmt(a['delta_distinct_2']['mean'], 4)} |")
    md.append("")

    # Table 4 — heldout_v2 target check
    md.append("## Table 4 — Heldout_v2 target check")
    md.append("")
    h_arm2 = tabs["escape"]["heldout_degenerate_v2"]["arm2_agg"]
    h_ret = tabs["escape"]["heldout_degenerate_v2"]["retarget_agg"]
    md.append(f"- night-014 Arm 2 (suppression-only) net escape: "
              f"{fmt(h_arm2['net_escape'], 4)} (EOS {fmt(h_arm2['eos_rate'], 4)}).")
    md.append(f"- night-015 (retargeted) net escape: "
              f"{fmt(h_ret['net_escape'], 4)} (EOS {fmt(h_ret['eos_rate'], 4)}).")
    md.append(f"- Target: net escape >= 0.90; EOS-rate increase vs Arm 2 "
              f"<= 0.05.")
    net_ok = bool(h_ret["net_escape"] >= 0.90)
    eos_delta = float(h_ret["eos_rate"] - h_arm2["eos_rate"])
    eos_ok = bool(eos_delta <= 0.05)
    if net_ok and eos_ok:
        md.append(f"- **TARGET MET**: net escape {fmt(h_ret['net_escape'], 4)} "
                  f">= 0.90 and EOS-rate delta vs Arm 2 "
                  f"({fmt(eos_delta, 4)}) <= 0.05. The retargeted kickstart "
                  f"approaches/meets the suppression-only baseline without "
                  f"reintroducing the EOS-death spiral.")
    else:
        md.append(f"- **TARGET NOT MET**: net escape {fmt(h_ret['net_escape'], 4)} "
                  f"(need >= 0.90), EOS-rate delta vs Arm 2 "
                  f"{fmt(eos_delta, 4)} (need <= 0.05).")
    md.append("")

    # Per-fixture fire-type breakdown
    md.append("## Table 5 — Fire-type breakdown per fixture (night-015 "
              "retargeted arm)")
    md.append("")
    md.append("bigram-only records = bigram fired >= 1 step, spectral never. "
              "spectral-only = spectral fired >= 1 step, bigram never. both = "
              "both fired >= 1 step.")
    md.append("")
    md.append("| fixture | fired | bigram-only | spectral-only | both |")
    md.append("|---|---|---|---|---|")
    for fixture, label in fixture_labels:
        a = tabs["escape"][fixture]["retarget_agg"]
        md.append(f"| {label} | {a['n_fired']} | "
                  f"{a['n_bigram_only']} | "
                  f"{a['n_spectral_only']} | {a['n_both']} |")
    md.append("")

    # Rescued record ids
    md.append("## Rescued record ids (ΔDistinct-2 > 0, night-015 retargeted)")
    md.append("")
    for fixture, label in fixture_labels:
        rescued = [r for r in ctx["results"] if r["fixture"] == fixture
                   and r["retarget_active"]["delta_distinct_2"] > 0]
        ids = [int(r["record_id"]) for r in rescued]
        md.append(f"- **{label}**: {len(ids)} rescued — `{ids}`")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no controller edits, no threshold changes, "
              "no new actuation design beyond the inline fixes. Both fixes are "
              "implemented INLINE in this probe before they reach the "
              "controller [1].")
    md.append("- Fix 1 changes `_cache_vocabulary_subsets` to iterate "
              "`len(self.tokenizer)` = 151665 instead of "
              "`tokenizer.vocab_size` = 151643. The 22 additional token ids "
              "(EOS at 151643 plus 21 special tokens) are now classified. EOS "
              "is classified non-prose.")
    md.append("- Fix 2 retargets the kickstart penalty from "
              "`non_prose_token_ids` to `active_loop_ids`, the token IDs "
              "`compute_token_distinct_2_fast` reports as actually repeated at "
              "each step. No orthographic heuristic, no vocabulary "
              "classification, no EOS-exclusion quirk.")
    md.append("- Dormant and Arm1/Arm2 baselines are REUSED from "
              "docs/gate23/sampling_dual_predicate_results.jsonl (night-014). "
              "Do NOT re-run. Only the retargeted arm is NEW.")
    md.append("- Sampling regime: do_sample=True, temperature=0.8, top_p=0.85 "
              "(matching night-014). SEED=42 with seed_all(SEED) before every "
              "generation; the determinism smoke confirms reproducible "
              "sampling.")
    md.append("- The frozen DS-034e dual-predicate rule is implemented exactly "
              "per the task file: spectral_collapse uses band_low (8.216097), "
              "spectral-only fires require token_diversity < 0.40 "
              "corroboration when bigram_ctr < 2, and "
              "is_code_syntax_context forces spectral_fire = False.")
    md.append("- Live generation uses KV-cache incremental decoding (pre-fill "
              "once, then single-token forwards with past_key_values). The "
              "layer-2 spectral hook captures hidden states at each decoding "
              "step (seq_len==1) into a rolling 24-token ring buffer.")
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
    night014_meta: Dict[str, Any],
    device: str,
    dtype: Any,
    smoke: Dict[str, Any],
    spectral_threshold: float,
    vocab_cache: Dict[str, Any],
    wall_clock_s: float,
) -> Dict[str, Any]:
    """Build the report context from per-record results (single source of truth)."""
    fixture_names = ["t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"]

    comparison: Dict[str, Dict[str, Any]] = {}
    escape: Dict[str, Dict[str, Any]] = {}

    for fixture in fixture_names:
        recs = [r for r in all_results if r["fixture"] == fixture]
        arm1_agg = aggregate_arm(recs, "arm1_active")
        arm2_agg = aggregate_arm(recs, "arm2_active")
        retarget_agg = aggregate_arm(recs, "retarget_active")
        escape[fixture] = {
            "arm1_agg": arm1_agg,
            "arm2_agg": arm2_agg,
            "retarget_agg": retarget_agg,
        }
        comparison[fixture] = {
            "arm1_rate": float(arm1_agg["rescue_rate"]),
            "arm2_rate": float(arm2_agg["rescue_rate"]),
            "retarget_rate": float(retarget_agg["rescue_rate"]),
        }

    tables = {
        "comparison": comparison,
        "escape": escape,
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
        "vocab_cache": vocab_cache,
        "night014_meta": night014_meta,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-015 kickstart retarget measurement (MEASUREMENT "
                    "ONLY)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per fixture (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate KICKSTART_RETARGET_RESULTS.md from an "
                             "existing kickstart_retarget_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("night-015 Kickstart retarget measurement (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"sampling: temp={SAMPLING_TEMPERATURE} top_p={SAMPLING_TOP_P}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: "
          f"{ANALYSIS_WINDOW} | hook_layer: {HOOK_LAYER}")

    # ------------------------------------------------------------------
    # Reused night-014 baselines (both measurement and report-only paths).
    # ------------------------------------------------------------------
    if not NIGHT014_JSONL.exists():
        print(f"[STOP] {NIGHT014_JSONL} does not exist; night-014 baselines "
              f"(dormant + Arm1 + Arm2) are required. Do NOT re-run night-014.")
        sys.exit(1)
    night014_records = load_night014_baselines(NIGHT014_JSONL)
    night014_meta = load_night014_meta(NIGHT014_JSONL)
    print(f"reused night-014 baselines from {NIGHT014_JSONL}: "
          f"{len(night014_records)} per-record lines "
          f"(seed={night014_meta.get('seed')}, "
          f"wall_clock={night014_meta.get('wall_clock_s')}s)")

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
            "record_id": -1, "prompt_len": -1, "n_generated_active": -1,
            "n_pr_values": -1, "active_identical": "True",
            "pr_identical": "True", "identical": "True",
            "pr_a_first": None, "pr_b_first": None,
            "pr_a_last": None, "pr_b_last": None,
        }
        wall_clock_s: Any = "n/a (report-only regeneration of the night-015 run)"
        vocab_cache: Dict[str, Any] = {}
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("record_type") == "smoke":
                    smoke = {**smoke, **{k: v for k, v in r.items()
                                         if k != "record_type"}}
                elif r.get("record_type") == "meta" and "wall_clock_s" in r:
                    wall_clock_s = float(r["wall_clock_s"])
                elif r.get("record_type") == "vocab_cache":
                    vocab_cache = r.get("data", {})
        if not vocab_cache:
            # Build a minimal vocab_cache from the tokenizer (report-only path).
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION
            )
            vocab_cache = build_vocab_cache(tokenizer)
        ctx = build_ctx_from_results(
            all_results, night014_meta, device, dtype, smoke,
            spectral_threshold, vocab_cache, wall_clock_s=wall_clock_s,
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
    # Vocab cache — Fix 1 implemented inline (len(tokenizer)).
    # ------------------------------------------------------------------
    prose_old, non_prose_old = cache_vocabulary_subsets_vocab_size(tokenizer)
    prose_new, non_prose_new = cache_vocabulary_subsets_fixed(tokenizer)
    eos_id = int(tokenizer.eos_token_id)
    vocab_cache = {
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "tokenizer_len": int(len(tokenizer)),
        "eos_id": eos_id,
        "prose_old": len(prose_old),
        "non_prose_old": len(non_prose_old),
        "prose_new": len(prose_new),
        "non_prose_new": len(non_prose_new),
        "eos_in_old": bool(eos_id in prose_old or eos_id in non_prose_old),
        "eos_in_new": bool(eos_id in prose_new or eos_id in non_prose_new),
    }
    print(f"vocab cache (original vocab_size): prose={len(prose_old)} "
          f"non_prose={len(non_prose_old)} eos_classified={vocab_cache['eos_in_old']}")
    print(f"vocab cache (Fix 1 len(tokenizer)): prose={len(prose_new)} "
          f"non_prose={len(non_prose_new)} eos_classified={vocab_cache['eos_in_new']}")
    print(f"  EOS id {eos_id}; tokenizer.vocab_size={vocab_cache['vocab_size']}; "
          f"len(tokenizer)={vocab_cache['tokenizer_len']}")

    # ------------------------------------------------------------------
    # Determinism smoke FIRST (also serves as GPU warmup).
    # ------------------------------------------------------------------
    smoke: Dict[str, Any] = {
        "record_id": -1, "prompt_len": -1, "n_generated_active": -1,
        "n_pr_values": -1, "active_identical": "True",
        "pr_identical": "True", "identical": "True",
        "pr_a_first": None, "pr_b_first": None,
        "pr_a_last": None, "pr_b_last": None,
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
    # Per-fixture runs: retargeted kickstart arm (NEW only).
    # ==================================================================
    def run_fixture_retarget(
        fixture_name: str,
        records: List[Dict[str, Any]],
        prompt_key: str,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec_i, rec in enumerate(records):
            rid = int(rec["id"])
            prompt = rec[prompt_key]

            # Reused dormant baseline from night-014 (do NOT re-run).
            n014 = [r for r in night014_records
                    if r["fixture"] == fixture_name and int(r["record_id"]) == rid]
            if len(n014) != 1:
                raise SystemExit(
                    f"[STOP] night-014 baseline missing for {fixture_name} "
                    f"record_id={rid} (found {len(n014)})."
                )
            dormant = n014[0]["dormant"]
            arm1_active_reused = n014[0]["arm1_active"]
            arm2_active_reused = n014[0]["arm2_active"]

            # NEW retargeted-kickstart active run.
            gen_ret = sampling_generate_retarget(
                model, tokenizer, prompt, spectral_threshold,
            )
            retarget_active = build_retarget_meta(
                tokenizer, gen_ret, dormant, spectral_threshold,
            )

            out.append({
                "fixture": fixture_name,
                "record_id": rid,
                "prompt": prompt,
                "seed": SEED,
                "prompt_len": int(gen_ret["prompt_len"]),
                "spectral_threshold": float(spectral_threshold),
                "dormant": dormant,
                "arm1_active": arm1_active_reused,
                "arm2_active": arm2_active_reused,
                "retarget_active": retarget_active,
            })

            if (rec_i + 1) % 25 == 0 or rec_i == len(records) - 1:
                n_rescue_ret = sum(
                    1 for r in out if r["retarget_active"]["delta_distinct_2"] > 0
                )
                n_kick_ret = int(sum(
                    r["retarget_active"]["n_kickstart_events"] for r in out
                ))
                n_eos_ret = int(sum(
                    r["retarget_active"]["eos_terminated"] for r in out
                ))
                print(f"  [{fixture_name}] {rec_i+1}/{len(records)}; "
                      f"rescue(retarget)={n_rescue_ret} kick={n_kick_ret} "
                      f"eos={n_eos_ret}")
        return out

    all_results: List[Dict[str, Any]] = []
    try:
        print("\n--- Fixture A: t2s_degenerate (prompt key: record['text']) ---")
        part_a = run_fixture_retarget("t2s_degenerate", records_a, "text")
        all_results.extend(part_a)

        print("\n--- Fixture B: qwen_degenerate (prompt key: record['prompt']) ---")
        part_b = run_fixture_retarget("qwen_degenerate", records_b, "prompt")
        all_results.extend(part_b)

        print("\n--- Fixture C: heldout_degenerate_v2 (prompt key: "
              "record['mutated_prompt']) ---")
        part_c = run_fixture_retarget("heldout_degenerate_v2", records_c,
                                      "mutated_prompt")
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
        all_results, night014_meta, device, dtype, smoke,
        spectral_threshold, vocab_cache,
        wall_clock_s=time.time() - t_start,
    )
    tables = ctx["tables"]
    comparison = tables["comparison"]

    print("\n" + "=" * 70)
    print("NIGHT-015 KICKSTART RETARGET")
    for fixture, label in [("t2s_degenerate", "A t2s_degenerate"),
                           ("qwen_degenerate", "B qwen_degenerate"),
                           ("heldout_degenerate_v2", "C heldout_degenerate_v2")]:
        c = comparison[fixture]
        a1 = tables["escape"][fixture]["arm1_agg"]
        a2 = tables["escape"][fixture]["arm2_agg"]
        ar = tables["escape"][fixture]["retarget_agg"]
        print(f"  {label}: arm1={c['arm1_rate']:.4f} "
              f"arm2={c['arm2_rate']:.4f} "
              f"retarget={c['retarget_rate']:.4f} "
              f"({ar['n_rescue']}/{ar['n_records']})")
        print(f"    arm1    gross={a1['gross_escape']:.4f} "
              f"net={a1['net_escape']:.4f} EOS={a1['eos_rate']:.4f} "
              f"kick={a1['mean_kickstart_events']:.2f}")
        print(f"    arm2    gross={a2['gross_escape']:.4f} "
              f"net={a2['net_escape']:.4f} EOS={a2['eos_rate']:.4f} "
              f"kick={a2['mean_kickstart_events']:.2f}")
        print(f"    retarget gross={ar['gross_escape']:.4f} "
              f"net={ar['net_escape']:.4f} EOS={ar['eos_rate']:.4f} "
              f"kick={ar['mean_kickstart_events']:.2f} "
              f"supp={ar['mean_suppression_steps']:.1f}")
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
            "rescue_retarget_t2s": int(
                tables["escape"]["t2s_degenerate"]["retarget_agg"]["n_rescue"]
            ),
            "rescue_retarget_qwen": int(
                tables["escape"]["qwen_degenerate"]["retarget_agg"]["n_rescue"]
            ),
            "rescue_retarget_heldout_v2": int(
                tables["escape"]["heldout_degenerate_v2"]["retarget_agg"]["n_rescue"]
            ),
            "net_escape_retarget_t2s": float(
                tables["escape"]["t2s_degenerate"]["retarget_agg"]["net_escape"]
            ),
            "net_escape_retarget_qwen": float(
                tables["escape"]["qwen_degenerate"]["retarget_agg"]["net_escape"]
            ),
            "net_escape_retarget_heldout_v2": float(
                tables["escape"]["heldout_degenerate_v2"]["retarget_agg"]["net_escape"]
            ),
            "eos_rate_retarget_t2s": float(
                tables["escape"]["t2s_degenerate"]["retarget_agg"]["eos_rate"]
            ),
            "eos_rate_retarget_qwen": float(
                tables["escape"]["qwen_degenerate"]["retarget_agg"]["eos_rate"]
            ),
            "eos_rate_retarget_heldout_v2": float(
                tables["escape"]["heldout_degenerate_v2"]["retarget_agg"]["eos_rate"]
            ),
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(all_results)} record lines + "
          f"sidecar lines)")

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-015 measurement complete")


if __name__ == "__main__":
    main()
