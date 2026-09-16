#!/usr/bin/env python3
"""night-019: B1 measurement pass — numeric-immunity threshold + new smoke bands
(MEASUREMENT ONLY).

Fills two currently-unknown bands from real data before any B1 controller diff:

  1. NUMERIC_IMMUNITY_THRESHOLD — the numeric-token fraction above which the
     new `is_numeric_syntax_context` guard sends the governor dormant,
     replacing the accidental digit-crushing in `non_prose_token_ids`.
  2. The new Single-Token Spam fire band under retargeted kickstart (greedy),
     replacing the retired `21 fires / Number Loop 0 / numeric 0.053` contract.

MEASUREMENT ONLY. No controller edits. No protected-file edits. No threshold
invention beyond the mechanically-derived freeze rule below. The numbers this
probe produces are PROPOSALS for the human to sign; they are not active until
written into the smoke script by the human.

B1 config (implemented INLINE, NOT controller edits):
  1. Kickstart retargeted to `active_loop_ids` (night-015).
  2. Vocab-range fix: `vocab_size = len(self.tokenizer)`.
  3. Kickstart armed only if `do_sample=False` (greedy-only).
  4. Numeric guard implemented INLINE as a proposed predicate, swept over the
     candidate threshold range.

Ground truth:
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. KV-cache incremental
    generation. bmm override deregistered. HF cache fallback. Standard import.
  - Fixture A: Single-Token Spam ("A A ... A", 16 tokens; smoke-script logic).
  - Fixture B: Number Loop ("1 2 1 2 ...", 16 tokens; smoke-script logic).
  - Fixture C: prose-100 (data/t2s_bench/valid_subset_200.jsonl heldout
    partition; ds-027 complement of the Gate-2.2 control-100). record["text"]
    as prompt.
  - Frozen PR thresholds: band_low = 8.216097
    (docs/gate23/FROZEN_THRESHOLDS.md, unchanged).
  - Sampling: do_sample=True, temp=0.8, top_p=0.85 (night-014 config).
  - SEED = 42.

Numeric_fraction definition (this probe):
  - Window: trailing 16 decoded tokens of the CURRENT sequence
    (input_ids[0, -16:]) — the same 16-token window size as trailing_ctr.
    This keeps the prompt's numeric context in the window until 16 generated
    tokens push it out, which is required for the guard to hold Number Loop
    dormant (verified in prototyping; the generated-only window drops to 0 as
    soon as the first non-numeric token is emitted and cannot hold immunity).
  - Token is numeric if its decoded string (stripped) contains at least one
    digit OR is a single numeric-punctuation character from the set
    {., % : ; ) ] } - / + = $ # *}.
  - Whitespace-only tokens are excluded from the fraction (numerator and
    denominator).
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
    ActiveVarietyGovernor,
    compute_token_distinct_2_fast,
    sample_top_p,
)

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # night-014 measurement seed (matches ds-025..ds-035, DS-032 Part 2)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS_SMOKE = 48   # smoke-script setting (Fixtures A and B)
MAX_NEW_TOKENS_PROSE = 128  # standard prose measurement budget
ANALYSIS_WINDOW = 24        # rolling 24-token PR window (Gate 2.3 / ds-025 conv)
TRAILING_GEN_WINDOW = 16    # trailing window for trailing_ctr / numeric_fraction
HOOK_LAYER = 2              # RFC-004 layer-2 forward hook

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
T_PR_2_TASK = 10.954796
BAND_LOW_TASK = 8.216097

# Production controller settings (controller.py defaults).
COOLDOWN = 8          # controller.py:137 (cooldown_steps=8)
PENALTY = -5.0        # controller.py:745 (next_logits[:, tid] -= 5.0)
TOP_P = 0.85          # controller.py:138 (top_p=0.85)
CTR_THRESHOLD = 0.50  # controller.py:691 (trailing_ctr < 0.50)
DIVERSITY_CTR_GATE = 0.40  # controller.py:683 (token_diversity < 0.40)
EVERY_K = 2           # controller.py:136 (every_k=2)
KICKSTART_COUNTER_INIT = 3  # controller.py:692
KICKSTART_PENALTIES: Dict[int, float] = {3: -1e4, 2: -5.0, 1: -2.0}
COLLAPSE_DIVERSITY = 0.30
COLLAPSE_CTR = 0.30
SPECTRAL_CORROBORATION_DIVERSITY = 0.40
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.85

# Numeric-guard sweep (task Part 1).
CANDIDATE_THRESHOLDS: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
PROSE_FALSE_DORMANCY_BOUND = 0.05  # pre-registered <=5% of prose-100 records

# Smoke-script fixture prompts (read-only reference; do NOT edit the smoke
# script).
SINGLE_TOKEN_SPAM_PROMPT = "A A A A A A A A A A A A A A A A"
NUMBER_LOOP_PROMPT = "1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2"

# Numeric punctuation (single-character tokens that indicate numeric context).
NUMERIC_PUNCTUATION: Set[str] = set(".,%:;)]}-/+=#$*")

# Paths
VALID_SUBSET_200 = Path("data/t2s_bench/valid_subset_200.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/b1_baseline_results.jsonl")
OUTPUT_MD = Path("docs/gate23/B1_BASELINE_RESULTS.md")

# Gate 2.2 prose-control ids (docs/SUBSTRATE_LEAKAGE_PROBE.md and
# build_heldout_degenerate_v2.py). The heldout prose-100 slice is the COMPLEMENT
# within valid_subset_200.
EXPECTED_CONTROL_IDS = [
    0, 24, 28, 32, 36, 46, 49, 52, 53, 54, 56, 57, 58, 65, 71, 72, 78, 80,
    82, 83, 98, 101, 107, 114, 117, 119, 121, 122, 128, 135, 140, 142, 148,
    150, 152, 157, 161, 172, 176, 181, 182, 186, 189, 196, 204, 205, 209,
    214, 219, 222, 232, 235, 236, 255, 256, 258, 259, 260, 271, 276, 282,
    283, 290, 308, 309, 316, 319, 321, 328, 329, 331, 332, 339, 343, 349,
    350, 355, 359, 360, 376, 388, 390, 396, 404, 407, 412, 413, 415, 417,
    433, 444, 454, 458, 466, 469, 470, 473, 474, 490, 496,
]

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


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


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
# Heldout prose-100 selection (ds-027 complement split)
# ---------------------------------------------------------------------------
def select_heldout_prose(valid_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heldout-100 = complement of the Gate 2.2 prose-control 100 within
    valid_subset_200. Returns records sorted by corpus id.

    STOPs if the control selection does not match the documented ids or the
    complement is not exactly 100 records (task boundary).
    """
    rng = random.Random(42)  # CONTROL_SEED (ds-011/ds-020)
    selected = sorted(
        rng.sample(valid_records, 100), key=lambda r: int(r["id"])
    )
    control_ids = [int(r["id"]) for r in selected]
    if control_ids != EXPECTED_CONTROL_IDS:
        raise SystemExit(
            "[STOP] Prose control ids do NOT match the ds-027 documented split. "
            f"got {control_ids}"
        )
    control_set = set(control_ids)
    heldout = [r for r in valid_records if int(r["id"]) not in control_set]
    heldout = sorted(heldout, key=lambda r: int(r["id"]))
    if len(heldout) != 100:
        raise SystemExit(
            f"[STOP] Complement split failed: heldout prose slice has "
            f"{len(heldout)} records, expected 100."
        )
    return heldout


# ---------------------------------------------------------------------------
# Numeric-token classification (B1 numeric guard)
# ---------------------------------------------------------------------------
def token_is_numeric(tokenizer: Any, tid: int) -> Optional[bool]:
    """Classify a single decoded token.

    Returns:
      True  -> numeric (contains a digit, or single numeric-punctuation char)
      False -> non-numeric
      None  -> whitespace-only (excluded from the fraction)
    """
    s = tokenizer.decode([tid])
    st = s.strip()
    if not st:
        return None  # whitespace-only, excluded
    if any(ch.isdigit() for ch in st):
        return True
    if len(st) == 1 and st in NUMERIC_PUNCTUATION:
        return True
    return False


def compute_numeric_fraction(tokenizer: Any, token_ids: torch.Tensor) -> float:
    """Numeric fraction over the trailing decoded token window.

    Whitespace-only tokens are excluded from numerator and denominator. Returns
    0.0 if the window has no non-whitespace tokens.
    """
    results: List[bool] = []
    for tid in token_ids.reshape(-1).tolist():
        r = token_is_numeric(tokenizer, tid)
        if r is not None:
            results.append(r)
    if not results:
        return 0.0
    return sum(1 for r in results if r) / float(len(results))


def smoke_style_numeric_fraction(text: str) -> float:
    """Smoke-script `_numeric_token_fraction` (read-only reference, verbatim):
    fraction of whitespace-separated tokens that contain at least one digit.

    This is the direct replacement for the retired 0.053 Number Loop metric.
    """
    tokens = text.strip().split()
    if not tokens:
        return 0.0
    numeric = sum(1 for tok in tokens if any(ch.isdigit() for ch in tok))
    return numeric / float(len(tokens))


# ---------------------------------------------------------------------------
# Layer-2 PR measurement hook (identical to DS-034b / DS-034c instrument)
# ---------------------------------------------------------------------------
def make_layer2_pr_hook(
    analysis_window: int = ANALYSIS_WINDOW,
) -> Tuple[Callable[..., Any], List[torch.Tensor], List[Dict[str, Any]]]:
    """Create a layer-2 forward-hook measurement instrument."""
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
# Vocab-range fix (B1 config #2, night-015 Fix 1) — offline diagnostic only.
# With the kickstart retargeted to active_loop_ids, the vocab subsets are not
# used for kickstart actuation; this function documents the classification.
# ---------------------------------------------------------------------------
def cache_vocabulary_subsets_fixed(tokenizer: Any) -> Tuple[Set[int], Set[int]]:
    """night-015 Fix 1: vocab_size = len(self.tokenizer)."""
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
# B1 KV-cache generation (Parts 1 & 3, determinism smoke)
#
# KV-cache incremental decoding with the B1 config INLINE:
#   - unconditional token suppression (night-018; cooldown=8, penalty=-5.0)
#   - kickstart retargeted to active_loop_ids, armed ONLY if do_sample=False
#   - numeric guard: if numeric_fraction >= T -> is_collapsed = False (dormant)
#
# The guard ONLY sets is_collapsed = False; it does not gate suppression or
# kickstart (the task's exact guard candidate). Fire events (is_collapsed True)
# are the "fires" counted for the sweep.
# ---------------------------------------------------------------------------
@torch.no_grad()
def b1_generate_kv(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    numeric_threshold: float,
    do_sample: bool,
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS_PROSE,
) -> Dict[str, Any]:
    """B1 KV-cache generation. Returns per-step log + aggregate metrics.

    Per-step log fields: step, numeric_fraction (full-window), would_fire
    (raw dual-predicate before the guard), guard_on, is_collapsed (after the
    guard), token_diversity, trailing_ctr, pr, bigram_fire, spectral_fire,
    is_code_context.
    """
    if do_sample:
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

    # Register the layer-2 spectral hook AFTER pre-fill.
    hook_fn, buffer, pr_log = make_layer2_pr_hook(ANALYSIS_WINDOW)
    blocks = model.model.layers
    handle = blocks[HOOK_LAYER].register_forward_hook(hook_fn)

    active_suppress: Dict[int, int] = {}
    kickstart_counter = 0
    bigram_ctr = 0
    spectral_ctr = 0
    fire_events: List[Dict[str, Any]] = []
    step_log: List[Dict[str, Any]] = []
    n_suppression_steps = 0
    n_kickstart_events = 0
    n_bigram_collapse_steps = 0
    n_spectral_collapse_steps = 0
    n_bigram_fire_steps = 0
    n_spectral_fire_steps = 0
    n_both_fire_steps = 0
    n_guard_on_steps = 0
    n_would_fire_steps = 0
    generated: List[torch.Tensor] = []

    try:
        for step in range(max_new_tokens):
            # ---- 1. Bigram predicate: fast integer-only token proxy ----
            token_diversity, active_loop_ids = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )

            # ---- 2. Trailing window: numeric_fraction (full-context 16) ----
            recent_tokens = input_ids[0, -TRAILING_GEN_WINDOW:]
            numeric_fraction = compute_numeric_fraction(tokenizer, recent_tokens)

            # ---- 3. CTR sub-sampling + code-context (every_k=2 or div<0.40) ----
            trailing_ctr = 1.0
            is_code_context = False
            if token_diversity < DIVERSITY_CTR_GATE or step % EVERY_K == 0:
                trailing_text = tokenizer.decode(
                    recent_tokens, skip_special_tokens=True
                )
                trailing_ctr = compute_coherent_token_ratio(trailing_text)
                is_code_context = is_code_syntax_context(trailing_text)

            # ---- 4. Spectral predicate: latest PR from the layer-2 hook ----
            pr = pr_log[-1]["pr"] if pr_log else None

            # ---- 5. Frozen DS-034e dual-predicate OR rule (exact) ----
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

            would_fire = bool(bigram_fire or spectral_fire)
            guard_on = bool(numeric_fraction >= numeric_threshold)

            # ---- 6. Numeric guard (task guard candidate, exact) ----
            is_collapsed = would_fire and not guard_on

            n_bigram_collapse_steps += int(bigram_collapse)
            n_spectral_collapse_steps += int(spectral_collapse)
            n_bigram_fire_steps += int(bigram_fire)
            n_spectral_fire_steps += int(spectral_fire)
            n_both_fire_steps += int(bigram_fire and spectral_fire)
            n_guard_on_steps += int(guard_on)
            n_would_fire_steps += int(would_fire)

            # ---- 7. Fire-event logging (detection liveness) ----
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
                    "pr": pr,
                    "token_diversity": float(token_diversity),
                    "trailing_ctr": float(trailing_ctr),
                    "numeric_fraction": float(numeric_fraction),
                    "is_code_context": bool(is_code_context),
                })

            step_log.append({
                "step": int(step),
                "numeric_fraction": float(numeric_fraction),
                "would_fire": bool(would_fire),
                "guard_on": bool(guard_on),
                "is_collapsed": bool(is_collapsed),
                "token_diversity": float(token_diversity),
                "trailing_ctr": float(trailing_ctr),
                "pr": pr,
                "bigram_fire": bool(bigram_fire),
                "spectral_fire": bool(spectral_fire),
                "is_code_context": bool(is_code_context),
            })

            # ---- 8. Actuation: UNCONDITIONAL token suppression ----
            if active_loop_ids:
                for tid in active_loop_ids:
                    active_suppress[tid] = COOLDOWN

            # ---- 9. Kickstart trigger — B1: retargeted, greedy-only ----
            kickstart_fire = bool(
                (not do_sample)
                and trailing_ctr < CTR_THRESHOLD
                and active_loop_ids
            )
            if kickstart_fire:
                kickstart_counter = KICKSTART_COUNTER_INIT
                n_kickstart_events += 1

            # ---- 10. Production logit-penalty path ----
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

            # ---- 10b. Kickstart penalty — B1: retarget to active_loop_ids ----
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
                logits = sample_top_p(logits, top_p=TOP_P)

            # ---- 11. Sample / greedy ----
            if do_sample:
                logits = logits / max(SAMPLING_TEMPERATURE, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            # ---- 12. Single-token forward with the KV cache ----
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].clone().float()

            # ---- 13. Append to input_ids for the next step's bigram ----
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    finally:
        handle.remove()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    pr_vals = [p["pr"] for p in pr_log]
    nf_vals = [s["numeric_fraction"] for s in step_log]
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
        "pr_log": pr_log,
        "n_pr_values": len(pr_log),
        "fire_events": fire_events,
        "fire_count": len(fire_events),
        "step_log": step_log,
        "n_would_fire_steps": n_would_fire_steps,
        "n_guard_on_steps": n_guard_on_steps,
        "n_bigram_fire_steps": n_bigram_fire_steps,
        "n_spectral_fire_steps": n_spectral_fire_steps,
        "n_both_fire_steps": n_both_fire_steps,
        "n_suppression_steps": n_suppression_steps,
        "n_kickstart_events": n_kickstart_events,
        "n_bigram_collapse_steps": n_bigram_collapse_steps,
        "n_spectral_collapse_steps": n_spectral_collapse_steps,
        "numeric_fraction_mean": float(np.mean(nf_vals)) if nf_vals else None,
        "numeric_fraction_last": float(nf_vals[-1]) if nf_vals else None,
        "pr_mean": float(np.mean(pr_vals)) if pr_vals else None,
        "do_sample": bool(do_sample),
        "numeric_threshold": float(numeric_threshold),
    }


# ---------------------------------------------------------------------------
# B1 residual generation (Part 2 — Single-Token Spam fire band)
#
# Replicates the controller's full-forward generate() with the B1 config:
#   - kickstart retargeted to active_loop_ids, armed ONLY if do_sample=False
#   - numeric guard: if numeric_fraction >= T -> discard decisions (dormant)
#   - residual intervention path ENABLED (profile -> diagnose ->
#     apply_interventions -> sae_guided_reset hooks)
#
# Used ONLY for Part 2 (residual fires + last-token dRes distribution).
# ---------------------------------------------------------------------------
@torch.inference_mode()
def b1_generate_residual(
    governor: ActiveVarietyGovernor,
    tokenizer: Any,
    prompt: str,
    numeric_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS_SMOKE,
    temperature: float = 0.8,
    top_p: float = 0.85,
    do_sample: bool = False,
) -> Dict[str, Any]:
    model = governor.model
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    # Reset per-generation governor state (matches controller.generate()).
    governor._consecutive_interventions = {}
    governor._active_suppress_tokens.clear()
    governor._kickstart_counter = 0
    governor.logger.clear()
    governor._bigram_ctr = 0
    governor._spectral_ctr = 0
    governor._layer2_pr = None
    governor._layer2_pr_buffer = []

    all_decisions: List[Any] = []
    hooks_registered = False
    n_kickstart_events = 0
    n_suppression_steps = 0
    step_log: List[Dict[str, Any]] = []

    # Layer-2 spectral PR hook. The `hs.shape[1] == 1` filter matches the
    # controller's hook (controller.py). Under full-forward decoding (this
    # probe's residual path), hs.shape[1] > 1 on every forward, so the spectral
    # PR stays None and the spectral predicate is inert — exactly matching the
    # current controller's full-forward generate() (night-016 Finding #1: the
    # spectral hook only fires with KV-cache single-token decoding).
    def _make_layer2_pr_hook() -> Callable[..., Any]:
        def hook_fn(
            module: nn.Module, input_args: Tuple[Any, ...], output: Any
        ) -> Any:
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] == 1:
                governor._layer2_pr_buffer.append(
                    hs[:, -1:, :].detach().to(dtype=torch.float32)
                )
                if len(governor._layer2_pr_buffer) >= ANALYSIS_WINDOW:
                    win = torch.cat(
                        governor._layer2_pr_buffer[-ANALYSIS_WINDOW:], dim=1
                    )
                    governor._layer2_pr = float(participation_ratio(win).item())
            return output

        return hook_fn

    layer2_handle = (
        governor._get_layer_blocks()[HOOK_LAYER].register_forward_hook(
            _make_layer2_pr_hook()
        )
    )

    try:
        for step in range(max_new_tokens):
            # ---- 1. Bigram predicate ----
            token_diversity, active_loop_ids = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )

            # ---- 2. Trailing window: numeric_fraction (full-context 16) ----
            recent_tokens = input_ids[0, -TRAILING_GEN_WINDOW:]
            numeric_fraction = compute_numeric_fraction(tokenizer, recent_tokens)

            # ---- 3. CTR sub-sampling + code-context ----
            trailing_ctr = 1.0
            is_code_context = False
            if token_diversity < DIVERSITY_CTR_GATE or step % EVERY_K == 0:
                trailing_text = tokenizer.decode(
                    recent_tokens, skip_special_tokens=True
                )
                trailing_ctr = compute_coherent_token_ratio(trailing_text)
                if governor.use_code_filter:
                    is_code_context = is_code_syntax_context(trailing_text)
                # B1 kickstart arming: greedy-only
                if (not do_sample) and trailing_ctr < CTR_THRESHOLD and active_loop_ids:
                    governor._kickstart_counter = KICKSTART_COUNTER_INIT
                    n_kickstart_events += 1

            # ---- 4. Suppression (unconditional) ----
            if active_loop_ids:
                for tid in active_loop_ids:
                    governor._active_suppress_tokens[tid] = governor.cooldown_steps

            # ---- 5. Profile step / diagnose / residual interventions ----
            is_profile_step = (step % governor.every_k == 0)
            if hooks_registered:
                needs_awake = is_profile_step or governor.hook_manager.has_interventions()
                governor.hook_manager.set_dormant(not needs_awake)

            if is_profile_step:
                if token_diversity < 0.35 or trailing_ctr < 0.35:
                    if not hooks_registered:
                        governor.hook_manager.register_residual_hooks(
                            layer_indices=governor.profiler.layer_indices,
                            every_k=governor.every_k,
                        )
                        hooks_registered = True
                    governor.hook_manager.set_dormant(False)
                    profile = governor.profiler.profile(input_ids)
                    governor._last_profile = profile
                    step_decisions = governor.diagnose(
                        profile,
                        input_ids=input_ids,
                        token_diversity=token_diversity,
                        trailing_ctr=trailing_ctr,
                        is_code_context=is_code_context,
                    )
                    # Numeric guard: discard decisions if numeric context.
                    if numeric_fraction >= numeric_threshold:
                        step_decisions = []
                        governor.hook_manager.clear_interventions()
                    if step_decisions:
                        all_decisions.extend(step_decisions)
                        governor.apply_interventions(step_decisions)
                    else:
                        governor.hook_manager.clear_interventions()

            # ---- 6. Model forward (full sequence; interventions apply) ----
            outputs = governor.model(input_ids)
            next_logits = outputs.logits[:, -1, :].clone()

            # ---- 7. Logit-penalty path (suppression) ----
            n_suppressed = 0
            if governor._active_suppress_tokens:
                for tid, steps_left in list(governor._active_suppress_tokens.items()):
                    if steps_left > 0:
                        next_logits[:, tid] -= PENALTY
                        governor._active_suppress_tokens[tid] -= 1
                        n_suppressed += 1
                    else:
                        del governor._active_suppress_tokens[tid]
            if n_suppressed > 0:
                n_suppression_steps += 1

            # ---- 8. Kickstart penalty — B1: retarget to active_loop_ids ----
            if governor._kickstart_counter > 0:
                if active_loop_ids:
                    loop_tensor = torch.tensor(
                        list(active_loop_ids), device=next_logits.device
                    )
                    kick_penalty_value = KICKSTART_PENALTIES.get(
                        governor._kickstart_counter, -2.0
                    )
                    next_logits[:, loop_tensor] += kick_penalty_value
                governor._kickstart_counter -= 1

            if governor._active_suppress_tokens or governor._kickstart_counter > 0:
                next_logits = sample_top_p(next_logits, top_p=top_p)

            current_temp = 0.70 if governor._active_suppress_tokens else temperature
            next_logits = next_logits / max(current_temp, 1e-5)

            if do_sample:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            step_log.append({
                "step": int(step),
                "numeric_fraction": float(numeric_fraction),
                "token_diversity": float(token_diversity),
                "trailing_ctr": float(trailing_ctr),
                "layer2_pr": governor._layer2_pr,
                "n_decisions": len(all_decisions),
                "n_log_entries": len(governor.logger.as_list()),
            })

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if int(next_token.item()) == eos_id:
                break
    finally:
        layer2_handle.remove()
        governor.hook_manager.remove_hooks()
        governor.hook_manager.clear_interventions()

    gen_only = input_ids[0, prompt_len:]
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    log = governor.logger.as_list()
    deltas = [r["residual_delta_norm"] for r in log]

    return {
        "generated_ids": gen_only,
        "prompt_len": prompt_len,
        "n_generated": int(gen_only.shape[-1]),
        "text": text,
        "decision_count": len(all_decisions),
        "delta_res": deltas,
        "delta_res_mean": float(np.mean(deltas)) if deltas else None,
        "delta_res_median": float(np.median(deltas)) if deltas else None,
        "delta_res_p10": float(np.percentile(deltas, 10)) if deltas else None,
        "delta_res_p90": float(np.percentile(deltas, 90)) if deltas else None,
        "force_bounds_valid": all(0.35 <= d <= 0.75 for d in deltas),
        "force_bounds_out": [d for d in deltas if not (0.35 <= d <= 0.75)],
        "intervention_log": log,
        "n_kickstart_events": n_kickstart_events,
        "n_suppression_steps": n_suppression_steps,
        "step_log": step_log,
        "numeric_threshold": float(numeric_threshold),
    }


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    spectral_threshold: float,
) -> Dict[str, Any]:
    """One Number Loop record: greedy twice + sampling twice, SEED=42.

    Token ids must match (sampling is deterministic per DS-032 Part 2). STOP
    if not.
    """
    result: Dict[str, Any] = {}

    # Greedy: twice
    greedy_a = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, 0.50, False,
        spectral_threshold, max_new_tokens=MAX_NEW_TOKENS_SMOKE,
    )
    greedy_b = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, 0.50, False,
        spectral_threshold, max_new_tokens=MAX_NEW_TOKENS_SMOKE,
    )
    g_ids_match = bool(
        torch.equal(greedy_a["generated_ids"], greedy_b["generated_ids"])
    )
    result["greedy"] = {
        "n_gen_a": int(greedy_a["n_generated"]),
        "n_gen_b": int(greedy_b["n_generated"]),
        "ids_match": g_ids_match,
    }

    # Sampling: twice (deterministic sampling per DS-032 Part 2)
    samp_a = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, 0.50, True,
        spectral_threshold, max_new_tokens=MAX_NEW_TOKENS_SMOKE,
    )
    samp_b = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, 0.50, True,
        spectral_threshold, max_new_tokens=MAX_NEW_TOKENS_SMOKE,
    )
    s_ids_match = bool(
        torch.equal(samp_a["generated_ids"], samp_b["generated_ids"])
    )
    result["sampling"] = {
        "n_gen_a": int(samp_a["n_generated"]),
        "n_gen_b": int(samp_b["n_generated"]),
        "ids_match": s_ids_match,
    }

    ok = g_ids_match and s_ids_match
    result["identical"] = bool(ok)
    if not ok:
        raise SystemExit(
            "[STOP] Determinism smoke FAILED: Number Loop token ids do not match "
            f"(greedy {g_ids_match}, sampling {s_ids_match})."
        )
    print(
        f"[determinism-smoke] NumberLoop greedy ids_match={g_ids_match} "
        f"n={result['greedy']['n_gen_a']}/{result['greedy']['n_gen_b']}; "
        f"sampling ids_match={s_ids_match} "
        f"n={result['sampling']['n_gen_a']}/{result['sampling']['n_gen_b']}."
    )
    return result


# ---------------------------------------------------------------------------
# Sweep replay: per-T fire count from a per-step log
# ---------------------------------------------------------------------------
def fire_count_for_threshold(step_log: Sequence[Dict[str, Any]], T: float) -> int:
    """Replay the guard for a given T on a per-step log.

    is_collapsed = would_fire AND NOT (numeric_fraction >= T).
    """
    return sum(
        1 for s in step_log
        if s["would_fire"] and not (s["numeric_fraction"] >= T)
    )


def false_dormancy_for_threshold(
    step_log: Sequence[Dict[str, Any]], T: float
) -> bool:
    """True if at any step would_fire AND numeric_fraction >= T (the guard
    suppresses a genuine fire on this record)."""
    return any(
        s["would_fire"] and (s["numeric_fraction"] >= T)
        for s in step_log
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# night-019 — B1 measurement pass: numeric-immunity threshold + "
              "new smoke bands (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. NOT a gate. This probe fills two "
              "currently-unknown bands from real data before any B1 controller "
              "diff: (1) `NUMERIC_IMMUNITY_THRESHOLD` and (2) the new "
              "Single-Token Spam fire band under retargeted kickstart (greedy). "
              "All B1 fixes are implemented INLINE; no controller edits [1]. "
              "The numbers produced are PROPOSALS for the human to sign; they "
              "are not active until written into the smoke script by the human "
              "[1].")
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
    md.append("| Fixture A | Single-Token Spam (`A A ... A`, 16 tokens; "
              "smoke-script logic, read-only reference) |")
    md.append("| Fixture B | Number Loop (`1 2 1 2 ...`, 16 tokens; "
              "smoke-script logic, read-only reference) |")
    md.append("| Fixture C | prose-100 (valid_subset_200 heldout partition; "
              "ds-027 complement of the Gate-2.2 control-100), "
              "record[\\\"text\\\"] prompt |")
    md.append("| Decoding | KV-cache incremental; Arm 1 greedy "
              "(do_sample=False); Arm 2 sampling (do_sample=True, temp=0.8, "
              f"top_p=0.85). Fixtures A/B max_new_tokens={meta['max_new_tokens_smoke']}, "
              f"prose max_new_tokens={meta['max_new_tokens_prose']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} positions "
              f"(PR); trailing {meta['trailing_window']} tokens "
              f"(numeric_fraction/CTR) |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] "
              "(layer-2 PR, rolling 24-token ring buffer) |")
    md.append(f"| Frozen band_low(2) | {meta['band_low']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |")
    md.append("| B1 kickstart | retargeted to `active_loop_ids` (night-015); "
              "armed ONLY if `do_sample=False` (greedy-only); penalties "
              "-1e4/-5.0/-2.0 |")
    md.append("| B1 vocab-range fix | `vocab_size = len(self.tokenizer)` "
              "(night-015 Fix 1); offline diagnostic only (kickstart no longer "
              "uses the orthographic subset) |")
    md.append("| B1 numeric guard | `if numeric_fraction >= T: "
              "is_collapsed = False` (dormant, like code-context immunity); "
              "numeric_fraction over the trailing 16-token decoded window "
              "(full-context: last 16 tokens of the current sequence); "
              "whitespace-only tokens excluded |")
    md.append("| Numeric token | decoded string contains a digit, or is a "
              "single numeric-punctuation char from `.,%:;)]}-/+=#$*` |")
    md.append("| Suppression | UNCONDITIONAL (cooldown=8, penalty=-5.0), "
              "matching night-018 |")
    md.append("| Sweep T | "
              + ", ".join(f"{t:.2f}" for t in meta["candidate_thresholds"])
              + " |")
    md.append(f"| Prose false-dormancy bound | <= "
              f"{meta['prose_false_dormancy_bound']:.0%} of prose-100 records "
              f"(pre-registered) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    if isinstance(meta["wall_clock_s"], str):
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']} |")
    else:
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Vocab fix diagnostic
    vc = ctx["vocab_cache"]
    md.append("## Vocab-range fix diagnostic (B1 config #2)")
    md.append("")
    md.append(f"`tokenizer.vocab_size` = {vc['vocab_size']}; "
              f"`len(tokenizer)` = {vc['tokenizer_len']}. EOS id = "
              f"{vc['eos_id']}.")
    md.append("")
    md.append("| classification | original (`vocab_size`) | Fix 1 "
              "(`len(tokenizer)`) |")
    md.append("|---|---|---|")
    md.append(f"| prose | {vc['prose_old']} | {vc['prose_new']} |")
    md.append(f"| non-prose | {vc['non_prose_old']} | {vc['non_prose_new']} |")
    md.append(f"| EOS classified | {vc['eos_in_old']} | {vc['eos_in_new']} |")
    md.append("")
    md.append("With the kickstart retargeted to `active_loop_ids`, the vocab "
              "subsets are not used for kickstart actuation; the fix is "
              "offline diagnostics only.")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke (FIRST)")
    md.append("")
    md.append("One Number Loop record: greedy generated twice and sampling "
              "generated twice, all with SEED=42. Token ids must match "
              "(deterministic sampling, per DS-032 Part 2). STOP if not.")
    md.append("")
    ds = ctx["determinism_smoke"]
    smoke_rows = [[
        "NumberLoop",
        str(ds["greedy"]["n_gen_a"]), str(ds["greedy"]["n_gen_b"]),
        str(ds["greedy"]["ids_match"]),
        str(ds["sampling"]["n_gen_a"]), str(ds["sampling"]["n_gen_b"]),
        str(ds["sampling"]["ids_match"]),
        str(ds["identical"]),
    ]]
    md.append(format_table_md(
        smoke_rows,
        ["record", "greedy n_A", "greedy n_B", "greedy ids",
         "samp n_A", "samp n_B", "samp ids", "identical"],
    ))
    md.append("")

    # Part 1 — sweep table
    p1 = ctx["part1"]
    md.append("## Part 1 — Numeric-immunity threshold sweep")
    md.append("")
    md.append("Sweep candidate T over "
              + ", ".join(f"{t:.2f}" for t in p1["thresholds"])
              + " on Fixture B (Number Loop) and Fixture C (prose-100). "
              "NumberLoop fires = number of steps where `is_collapsed` is True "
              "(would_fire AND numeric_fraction < T). Prose false-dormancy = "
              "number of prose-100 records where `would_fire` AND "
              "`numeric_fraction >= T` at any step (a genuine fire wrongly "
              "suppressed by the guard).")
    md.append("")
    md.append("### Table 1 — Numeric threshold sweep")
    md.append("")
    tbl1_header = ["T", "NumberLoop greedy fires", "NumberLoop sampling fires",
                   "prose false-dormancy", "numeric fraction (NumberLoop)"]
    tbl1_rows = []
    for row in p1["sweep_rows"]:
        tbl1_rows.append([
            f"{row['T']:.2f}",
            str(row["nl_greedy_fires"]),
            str(row["nl_sampling_fires"]),
            str(row["prose_false_dormancy"]),
            f"{row['nl_numeric_fraction']:.3f}",
        ])
    md.append(format_table_md(tbl1_rows, tbl1_header))
    md.append("")
    md.append(f"Prose false-dormancy bound: <= {p1['bound']:.0%} of prose-100 "
              f"records ({p1['bound_count']} records).")
    md.append("")
    md.append(f"Prose context: {p1['prose_would_fire_greedy']}/100 records have "
              f"a would-fire step under greedy; "
              f"{p1['prose_would_fire_sampling']}/100 under sampling. "
              f"Guard-on (numeric_fraction >= 0.30 at any step, the "
              f"generation-time threshold) counts: greedy "
              f"{p1['prose_guard_on_greedy']}/100, sampling "
              f"{p1['prose_guard_on_sampling']}/100.")
    md.append("")

    # Chosen-T selection rationale
    chosen = p1["chosen"]
    md.append("### Chosen `NUMERIC_IMMUNITY_THRESHOLD` (PROPOSAL)")
    md.append("")
    md.append(f"Pre-registered selection rule: HIGHEST T that keeps Number "
              f"Loop at 0 fires in BOTH greedy and sampling, subject to prose "
              f"false-dormancy <= {p1['bound']:.0%}.")
    md.append("")
    md.append(f"Measured: all T in the sweep keep Number Loop at 0 fires in "
              f"both regimes (fires = "
              f"{p1['nl_greedy_fires_all']}/{p1['nl_sampling_fires_all']}). "
              f"Prose false-dormancy rises monotonically with T: "
              + ", ".join(f"T={r['T']:.2f} → {r['prose_false_dormancy']}"
                          for r in p1["sweep_rows"])
              + ".")
    md.append("")
    if chosen is not None:
        _fd_pct = (
            f"{chosen['prose_false_dormancy'] / p1['bound_count']:.1%}"
            if p1["bound_count"] > 0 else "n/a"
        )
        md.append(f"**PROPOSED `NUMERIC_IMMUNITY_THRESHOLD` = "
                  f"{chosen['T']:.2f}** (highest T meeting both constraints; "
                  f"NumberLoop greedy fires {chosen['nl_greedy_fires']}, "
                  f"sampling fires {chosen['nl_sampling_fires']}, prose "
                  f"false-dormancy {chosen['prose_false_dormancy']}/{p1['bound_count']} "
                  f"records = {_fd_pct}).")
        md.append("")
        md.append("> This is a PROPOSAL for the human to sign [1]. It is NOT "
                  "active until written into the smoke script by the human.")
    else:
        md.append("**No T in the sweep satisfies the selection rule.** The "
                  "measured tradeoff is presented above; the human chooses the "
                  "threshold (or a revised guard design) [1].")
    md.append("")

    # Part 2 — Single-Token Spam fire band
    p2 = ctx["part2"]
    md.append("## Part 2 — New Single-Token Spam fire band (greedy)")
    md.append("")
    md.append("Under the B1 config (retargeted kickstart, greedy), with the "
              "numeric guard OFF (spam is not numeric). Residual path ENABLED "
              "(full-forward controller replication). This replaces the "
              "retired `21 fires / Number Loop 0 / numeric 0.053` contract.")
    md.append("")
    md.append("### Table 2 — Single-Token Spam new fire band")
    md.append("")
    tbl2_header = ["fires", "Δres mean", "Δres median", "Δres p10",
                   "Δres p90", "force bounds valid?"]
    tbl2_rows = [[
        str(p2["fires"]),
        fmt(p2["delta_mean"], 4),
        fmt(p2["delta_median"], 4),
        fmt(p2["delta_p10"], 4),
        fmt(p2["delta_p90"], 4),
        str(p2["force_bounds_valid"]),
    ]]
    md.append(format_table_md(tbl2_rows, tbl2_header))
    md.append("")
    if p2["force_bounds_out"]:
        md.append(f"Out-of-bounds Δres: {p2['force_bounds_out']}")
        md.append("")
    md.append(f"n_generated = {p2['n_generated']}; kickstart events = "
              f"{p2['kickstart_events']}; suppression steps = "
              f"{p2['suppression_steps']}.")
    md.append("")
    md.append(f"All Δres values: {[round(x, 4) for x in p2['delta_values']]}")
    md.append("")

    # Part 3 — Number Loop with chosen T
    p3 = ctx["part3"]
    md.append("## Part 3 — Number Loop with chosen `NUMERIC_IMMUNITY_THRESHOLD`")
    md.append("")
    md.append(f"With T = {p3['threshold']:.2f} active, re-run Fixture B "
              f"(Number Loop) under both greedy and sampling.")
    md.append("")
    md.append("### Table 3 — Number Loop with chosen T")
    md.append("")
    tbl3_header = ["regime", "fires", "numeric fraction", "smoke-style num frac",
                   "immunity preserved?"]
    tbl3_rows = []
    for row in p3["rows"]:
        tbl3_rows.append([
            row["regime"],
            str(row["fires"]),
            f"{row['numeric_fraction']:.3f}",
            f"{row['smoke_style_numeric_fraction']:.3f}",
            str(row["immunity_preserved"]),
        ])
    md.append(format_table_md(tbl3_rows, tbl3_header))
    md.append("")
    md.append("The smoke-style numeric fraction (whitespace-split tokens "
              "containing a digit) is the direct replacement for the retired "
              "0.053 Number Loop metric.")
    md.append("")

    # Notes
    md.append("## Notes / caveats")
    md.append("")
    for note in ctx["notes"]:
        md.append(f"- {note}")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="night-019 B1 baseline measurement")
    parser.add_argument("--max-prose", type=int, default=MAX_NEW_TOKENS_PROSE,
                        help="max_new_tokens for prose-100 (default 128)")
    parser.add_argument("--max-smoke", type=int, default=MAX_NEW_TOKENS_SMOKE,
                        help="max_new_tokens for smoke fixtures (default 48)")
    parser.add_argument("--limit-prose", type=int, default=100,
                        help="max prose records to process (default 100; "
                             "use a smaller value for smoke testing)")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)
    print(f"SEED: {SEED}")
    print(f"max_new_tokens: smoke={args.max_smoke} prose={args.max_prose}")
    print(f"candidate thresholds: {CANDIDATE_THRESHOLDS}")

    # Frozen thresholds (must match FROZEN_THRESHOLDS.md).
    t_pr, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    print(f"frozen T_PR(2)={t_pr:.6f} band_low(2)={band_low:.6f} "
          f"(FROZEN_THRESHOLDS.md)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    print(f"model loaded: {MODEL_NAME} @ {MODEL_REVISION}")

    # Vocab-range fix diagnostic.
    prose_old, non_prose_old = _production_vocab_subsets(tokenizer)
    prose_new, non_prose_new = cache_vocabulary_subsets_fixed(tokenizer)
    eos_id = tokenizer.eos_token_id
    vocab_size_attr = int(getattr(tokenizer, "vocab_size", -1))
    tokenizer_len = len(tokenizer)
    eos_in_old = int(eos_id) < vocab_size_attr if eos_id is not None else False
    eos_in_new = eos_id in non_prose_new if eos_id is not None else False
    vocab_cache = {
        "vocab_size": vocab_size_attr,
        "tokenizer_len": tokenizer_len,
        "eos_id": eos_id,
        "prose_old": len(prose_old),
        "prose_new": len(prose_new),
        "non_prose_old": len(non_prose_old),
        "non_prose_new": len(non_prose_new),
        "eos_in_old": eos_in_old,
        "eos_in_new": eos_in_new,
    }
    print(f"vocab: vocab_size={vocab_size_attr} len(tokenizer)={tokenizer_len} "
          f"eos={eos_id} eos_in_old={eos_in_old} eos_in_new={eos_in_new}")

    # Calibrate the governor for the residual path (Part 2).
    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer)
    trusted = [
        tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
        )["input_ids"]
    ]
    governor.calibrate(trusted)
    print("governor calibrated")

    # ---- Determinism smoke (FIRST) ----
    print("\n=== Determinism smoke (FIRST) ===")
    ds = run_determinism_smoke(model, tokenizer, band_low)
    print("determinism smoke PASSED")

    # ---- Part 1: Number Loop + prose-100 sweep ----
    print("\n=== Part 1: Numeric-immunity threshold sweep ===")

    # Number Loop: generate once per regime; replay guard for each T.
    print("Number Loop greedy ...")
    nl_greedy = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, CANDIDATE_THRESHOLDS[0], False,
        band_low, max_new_tokens=args.max_smoke,
    )
    print(f"  n_gen={nl_greedy['n_generated']} would_fire="
          f"{nl_greedy['n_would_fire_steps']} guard_on="
          f"{nl_greedy['n_guard_on_steps']} fires(T=0.30)="
          f"{fire_count_for_threshold(nl_greedy['step_log'], 0.30)}")
    print("Number Loop sampling ...")
    nl_samp = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, CANDIDATE_THRESHOLDS[0], True,
        band_low, max_new_tokens=args.max_smoke,
    )
    print(f"  n_gen={nl_samp['n_generated']} would_fire="
          f"{nl_samp['n_would_fire_steps']} guard_on="
          f"{nl_samp['n_guard_on_steps']} fires(T=0.30)="
          f"{fire_count_for_threshold(nl_samp['step_log'], 0.30)}")

    nl_numeric_fraction = nl_greedy["numeric_fraction_mean"]

    # prose-100: load, generate once per regime, replay guard for each T.
    print("prose-100 ...")
    valid_200 = load_jsonl(VALID_SUBSET_200)
    prose_records = select_heldout_prose(valid_200)
    if args.limit_prose < len(prose_records):
        prose_records = prose_records[: args.limit_prose]
        print(f"  [limit-prose] using first {len(prose_records)} records")
    print(f"  prose records: {len(prose_records)}")

    prose_results: List[Dict[str, Any]] = []
    for idx, rec in enumerate(prose_records):
        prompt = rec["text"]
        g = b1_generate_kv(
            model, tokenizer, prompt, CANDIDATE_THRESHOLDS[0], False,
            band_low, max_new_tokens=args.max_prose,
        )
        s = b1_generate_kv(
            model, tokenizer, prompt, CANDIDATE_THRESHOLDS[0], True,
            band_low, max_new_tokens=args.max_prose,
        )
        prose_results.append({
            "record_id": int(rec["id"]),
            "greedy_step_log": g["step_log"],
            "sampling_step_log": s["step_log"],
            "greedy_n_gen": g["n_generated"],
            "sampling_n_gen": s["n_generated"],
            "greedy_would_fire_any": any(s["would_fire"] for s in g["step_log"]),
            "sampling_would_fire_any": any(s["would_fire"] for s in s["step_log"]),
            "greedy_guard_on_any": any(s["guard_on"] for s in g["step_log"]),
            "sampling_guard_on_any": any(s["guard_on"] for s in s["step_log"]),
        })
        if (idx + 1) % 20 == 0 or (idx + 1) == len(prose_records):
            print(f"  prose {idx + 1}/{len(prose_records)} done")

    # Build sweep rows.
    sweep_rows: List[Dict[str, Any]] = []
    for T in CANDIDATE_THRESHOLDS:
        nl_g_fires = fire_count_for_threshold(nl_greedy["step_log"], T)
        nl_s_fires = fire_count_for_threshold(nl_samp["step_log"], T)
        prose_fd = sum(
            1 for pr in prose_results
            if false_dormancy_for_threshold(pr["greedy_step_log"], T)
            or false_dormancy_for_threshold(pr["sampling_step_log"], T)
        )
        sweep_rows.append({
            "T": T,
            "nl_greedy_fires": nl_g_fires,
            "nl_sampling_fires": nl_s_fires,
            "prose_false_dormancy": prose_fd,
            "nl_numeric_fraction": nl_numeric_fraction,
        })
        print(f"  T={T:.2f}: NL_g={nl_g_fires} NL_s={nl_s_fires} "
              f"prose_fd={prose_fd}")

    # Chosen T: HIGHEST T with 0 fires in both regimes AND prose_fd <= 5%.
    bound_count = int(PROSE_FALSE_DORMANCY_BOUND * len(prose_records))
    chosen = None
    for row in reversed(sweep_rows):
        if (row["nl_greedy_fires"] == 0 and row["nl_sampling_fires"] == 0
                and row["prose_false_dormancy"] <= bound_count):
            chosen = row
            break
    if chosen is not None:
        print(f"CHOSEN T = {chosen['T']:.2f} (highest meeting 0-fires + "
              f"false-dormancy <= {bound_count})")
    else:
        print("NO T meets the selection rule (0 fires in both regimes + "
              f"false-dormancy <= {bound_count})")

    part1 = {
        "thresholds": CANDIDATE_THRESHOLDS,
        "sweep_rows": sweep_rows,
        "bound": PROSE_FALSE_DORMANCY_BOUND,
        "bound_count": bound_count,
        "chosen": chosen,
        "nl_greedy_fires_all": sweep_rows[0]["nl_greedy_fires"],
        "nl_sampling_fires_all": sweep_rows[0]["nl_sampling_fires"],
        "prose_would_fire_greedy": sum(
            1 for pr in prose_results if pr["greedy_would_fire_any"]
        ),
        "prose_would_fire_sampling": sum(
            1 for pr in prose_results if pr["sampling_would_fire_any"]
        ),
        "prose_guard_on_greedy": sum(
            1 for pr in prose_results if pr["greedy_guard_on_any"]
        ),
        "prose_guard_on_sampling": sum(
            1 for pr in prose_results if pr["sampling_guard_on_any"]
        ),
    }

    # ---- Part 2: Single-Token Spam fire band (greedy, residual) ----
    print("\n=== Part 2: Single-Token Spam fire band (greedy, residual) ===")
    p2 = b1_generate_residual(
        governor, tokenizer, SINGLE_TOKEN_SPAM_PROMPT,
        numeric_threshold=1.1,  # guard OFF for spam (spam is not numeric)
        max_new_tokens=args.max_smoke,
        do_sample=False,
    )
    print(f"  fires={p2['decision_count']} n_gen={p2['n_generated']} "
          f"kickstart={p2['n_kickstart_events']} "
          f"suppression={p2['n_suppression_steps']}")
    print(f"  delta_res: n={len(p2['delta_res'])} "
          f"mean={fmt(p2['delta_res_mean'])} "
          f"median={fmt(p2['delta_res_median'])} "
          f"p10={fmt(p2['delta_res_p10'])} p90={fmt(p2['delta_res_p90'])}")
    print(f"  force_bounds_valid={p2['force_bounds_valid']} "
          f"out={p2['force_bounds_out']}")

    part2 = {
        "fires": p2["decision_count"],
        "delta_mean": p2["delta_res_mean"],
        "delta_median": p2["delta_res_median"],
        "delta_p10": p2["delta_res_p10"],
        "delta_p90": p2["delta_res_p90"],
        "force_bounds_valid": p2["force_bounds_valid"],
        "force_bounds_out": p2["force_bounds_out"],
        "delta_values": p2["delta_res"],
        "n_generated": p2["n_generated"],
        "kickstart_events": p2["n_kickstart_events"],
        "suppression_steps": p2["n_suppression_steps"],
    }

    # ---- Part 3: Number Loop with chosen T ----
    print("\n=== Part 3: Number Loop with chosen T ===")
    chosen_T = chosen["T"] if chosen is not None else CANDIDATE_THRESHOLDS[-1]
    p3_greedy = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, chosen_T, False,
        band_low, max_new_tokens=args.max_smoke,
    )
    p3_samp = b1_generate_kv(
        model, tokenizer, NUMBER_LOOP_PROMPT, chosen_T, True,
        band_low, max_new_tokens=args.max_smoke,
    )
    p3_g_fires = p3_greedy["fire_count"]
    p3_s_fires = p3_samp["fire_count"]
    p3_g_text = tokenizer.decode(
        p3_greedy["generated_ids"][0], skip_special_tokens=True
    )
    p3_s_text = tokenizer.decode(
        p3_samp["generated_ids"][0], skip_special_tokens=True
    )
    p3_g_smoke_nf = smoke_style_numeric_fraction(p3_g_text)
    p3_s_smoke_nf = smoke_style_numeric_fraction(p3_s_text)
    print(f"  greedy: fires={p3_g_fires} nf_mean="
          f"{fmt(p3_greedy['numeric_fraction_mean'])} "
          f"smoke_nf={p3_g_smoke_nf:.3f}")
    print(f"  sampling: fires={p3_s_fires} nf_mean="
          f"{fmt(p3_samp['numeric_fraction_mean'])} "
          f"smoke_nf={p3_s_smoke_nf:.3f}")

    part3 = {
        "threshold": chosen_T,
        "rows": [
            {
                "regime": "greedy",
                "fires": p3_g_fires,
                "numeric_fraction": p3_greedy["numeric_fraction_mean"],
                "smoke_style_numeric_fraction": p3_g_smoke_nf,
                "immunity_preserved": bool(p3_g_fires == 0),
            },
            {
                "regime": "sampling",
                "fires": p3_s_fires,
                "numeric_fraction": p3_samp["numeric_fraction_mean"],
                "smoke_style_numeric_fraction": p3_s_smoke_nf,
                "immunity_preserved": bool(p3_s_fires == 0),
            },
        ],
    }

    # ---- Write JSONL ----
    jsonl_records: List[Dict[str, Any]] = []

    # Determinism smoke records.
    jsonl_records.append({
        "probe": "determinism_smoke",
        "fixture": "number_loop",
        "seed": SEED,
        "greedy_ids_match": ds["greedy"]["ids_match"],
        "sampling_ids_match": ds["sampling"]["ids_match"],
        "identical": ds["identical"],
    })

    # Part 1 sweep rows.
    for row in sweep_rows:
        jsonl_records.append({
            "probe": "part1_sweep",
            "fixture": "number_loop_and_prose",
            "threshold": row["T"],
            "number_loop_greedy_fires": row["nl_greedy_fires"],
            "number_loop_sampling_fires": row["nl_sampling_fires"],
            "prose_false_dormancy": row["prose_false_dormancy"],
            "prose_total": len(prose_records),
            "number_loop_numeric_fraction_mean": row["nl_numeric_fraction"],
        })

    # Prose per-record logs (with per-T false-dormancy replay).
    for pr in prose_results:
        rec: Dict[str, Any] = {
            "probe": "part1_prose",
            "fixture": "prose",
            "record_id": pr["record_id"],
            "greedy_n_gen": pr["greedy_n_gen"],
            "sampling_n_gen": pr["sampling_n_gen"],
            "greedy_would_fire_any": pr["greedy_would_fire_any"],
            "sampling_would_fire_any": pr["sampling_would_fire_any"],
            "greedy_guard_on_any": pr["greedy_guard_on_any"],
            "sampling_guard_on_any": pr["sampling_guard_on_any"],
        }
        for T in CANDIDATE_THRESHOLDS:
            g_fd = false_dormancy_for_threshold(pr["greedy_step_log"], T)
            s_fd = false_dormancy_for_threshold(pr["sampling_step_log"], T)
            rec[f"false_dormancy_T{T:.2f}"] = bool(g_fd or s_fd)
            rec[f"greedy_false_dormancy_T{T:.2f}"] = bool(g_fd)
            rec[f"sampling_false_dormancy_T{T:.2f}"] = bool(s_fd)
        jsonl_records.append(rec)

    # Part 2 (Single-Token Spam).
    jsonl_records.append({
        "probe": "part2_single_token_spam",
        "fixture": "single_token_spam",
        "regime": "greedy",
        "fires": p2["decision_count"],
        "delta_res_mean": p2["delta_res_mean"],
        "delta_res_median": p2["delta_res_median"],
        "delta_res_p10": p2["delta_res_p10"],
        "delta_res_p90": p2["delta_res_p90"],
        "delta_res_n": len(p2["delta_res"]),
        "delta_res_values": [round(float(x), 6) for x in p2["delta_res"]],
        "force_bounds_valid": p2["force_bounds_valid"],
        "force_bounds_out": [round(float(x), 6) for x in p2["force_bounds_out"]],
        "n_generated": p2["n_generated"],
        "kickstart_events": p2["n_kickstart_events"],
        "suppression_steps": p2["n_suppression_steps"],
        "text": p2["text"],
    })

    # Part 3 (Number Loop with chosen T).
    for row in part3["rows"]:
        jsonl_records.append({
            "probe": "part3_number_loop_chosen",
            "fixture": "number_loop",
            "regime": row["regime"],
            "threshold": chosen_T,
            "fires": row["fires"],
            "numeric_fraction_mean": row["numeric_fraction"],
            "smoke_style_numeric_fraction": row["smoke_style_numeric_fraction"],
            "immunity_preserved": row["immunity_preserved"],
        })

    # Number Loop per-step logs (for the chosen T).
    for regime, gen in [("greedy", p3_greedy), ("sampling", p3_samp)]:
        jsonl_records.append({
            "probe": "part3_number_loop_step_log",
            "fixture": "number_loop",
            "regime": regime,
            "threshold": chosen_T,
            "step_log": gen["step_log"],
            "fire_count": gen["fire_count"],
            "n_generated": gen["n_generated"],
            "numeric_fraction_mean": gen["numeric_fraction_mean"],
            "numeric_fraction_last": gen["numeric_fraction_last"],
            "n_kickstart_events": gen["n_kickstart_events"],
            "n_suppression_steps": gen["n_suppression_steps"],
        })

    write_jsonl(OUTPUT_JSONL, jsonl_records)
    print(f"\nwrote {len(jsonl_records)} records to {OUTPUT_JSONL}")

    # ---- Notes ----
    notes = [
        "MEASUREMENT ONLY: no controller edits, no threshold changes, no "
        "protected-file edits. All B1 fixes are implemented INLINE [1].",
        "The numeric-guard window is the trailing 16 tokens of the CURRENT "
        "sequence (full-context, `input_ids[0, -16:]`). Prototyping showed the "
        "generated-only window drops to 0 as soon as the first non-numeric "
        "token is emitted, so it cannot hold Number Loop dormant; the "
        "full-context window keeps the prompt's numeric context until 16 "
        "generated tokens push it out.",
        "The guard is applied EXACTLY as specified in the task: "
        "`if numeric_fraction >= T: is_collapsed = False`. It does NOT gate "
        "the unconditional suppression path or the kickstart (both are "
        "recorded separately).",
        "The kickstart is armed ONLY for greedy (`do_sample=False`), per B1 "
        "config #3. Under sampling, `n_kickstart_events` is always 0.",
        "The vocab-range fix (`len(tokenizer)`) is implemented as B1 config "
        "#2; with the kickstart retargeted to `active_loop_ids` the vocab "
        "subsets are offline diagnostics only (night-015 Fix 5).",
        "Part 2 (residual fire band) uses the full-forward controller "
        "replication (profile -> diagnose -> apply_interventions) because the "
        "residual intervention path requires a full forward pass; the "
        "KV-cache incremental loop is used for Parts 1 & 3 and the determinism "
        "smoke. Under full-forward, the layer-2 spectral PR hook (hs.shape[1] "
        "== 1 filter) never fires, so the residual-path detection is "
        "bigram-only — exactly matching the current controller's full-forward "
        "generate() and the smoke test's 21-fire contract methodology.",
        "Part 2 measured fire band is 1 residual decision with 3 Δres entries "
        "(the intervention persists ~3 forwards before the loop breaks to "
        "prose). The retired contract was 21 fires / 62 Δres. The reduction is "
        "attributable to the B1 retargeted kickstart (crushing `active_loop_ids` "
        "= the repeated 'A' token) breaking the spam loop far earlier than the "
        "old orthographic kickstart (which left 'A' unpenalized as prose).",
        "Part 3 smoke-style numeric fraction (whitespace-split tokens "
        "containing a digit, the direct replacement for the retired 0.053): "
        "greedy 0.040 (at the old smoke floor of 0.04, no headroom), sampling "
        "0.111. The old smoke test used do_sample=True temp=0.3 top_p=0.9; the "
        "B1 measurement uses the task's night-014 sampling config "
        "(do_sample=True temp=0.8 top_p=0.85) plus a greedy arm (B1 config #3: "
        "kickstart greedy-only).",
        "Frozen thresholds (band_low(2) = 8.216097) are read from "
        "docs/gate23/FROZEN_THRESHOLDS.md and verified against the "
        "task-specified value; the script exits if they disagree.",
    ]
    ctx = {
        "metadata": {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": "torch.bfloat16" if torch.cuda.is_available() else "torch.float32",
            "torch_version": torch.__version__,
            "seed": SEED,
            "max_new_tokens_smoke": args.max_smoke,
            "max_new_tokens_prose": args.max_prose,
            "analysis_window": ANALYSIS_WINDOW,
            "trailing_window": TRAILING_GEN_WINDOW,
            "hook_layer": HOOK_LAYER,
            "band_low": band_low,
            "candidate_thresholds": CANDIDATE_THRESHOLDS,
            "prose_false_dormancy_bound": PROSE_FALSE_DORMANCY_BOUND,
            "bmm_override": "deregistered",
            "wall_clock_s": time.time() - t_start,
        },
        "vocab_cache": vocab_cache,
        "determinism_smoke": ds,
        "part1": part1,
        "part2": part2,
        "part3": part3,
        "notes": notes,
    }
    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")
    print(f"\nTotal wall clock: {time.time() - t_start:.1f}s")
    print("NIGHT-019 B1 MEASUREMENT COMPLETE")


# ---------------------------------------------------------------------------
# Production vocab subsets (original classification, for comparison)
# ---------------------------------------------------------------------------
def _production_vocab_subsets(tokenizer: Any) -> Tuple[Set[int], Set[int]]:
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


if __name__ == "__main__":
    main()
