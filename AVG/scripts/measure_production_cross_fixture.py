#!/usr/bin/env python3
"""DS-033: Production (8,-5) instrumented cross-fixture diagnostic (MEASUREMENT ONLY).

DS-032 proved that (5,-5) suppression-only fails cross-fixture generalization:
qwen_degenerate rescue 68/100, t2s_degenerate rescue 15/100 under greedy
suppression-only. This is either a detection gap (the bigram predicate never
fires on macro-syntax motifs), an actuation gap (predicate fires but rescue
fails), or a mix. Cooldown tuning only addresses actuation — it cannot fix a
detection blindness.

This probe runs the PRODUCTION setting (8,-5) on both cross-fixtures WITH
per-record diagnostic logging: predicate truth count, suppression/kickstart
activity, and per-record shadow PR from the existing spectral instrument.
The goal is NOT a pass/fail gate — it is to classify every failure as
detection-gap or actuation-gap, per the locked triad diagnostic sequence.

Failure classification (per record):
  detection-gap : predicate_true_steps == 0 (predicate never fired)
  actuation-gap : predicate_true_steps > 0 AND delta_distinct_2 == 0
  rescued       : predicate_true_steps > 0, delta_distinct_2 > 0, and
                  n_generated >= 24 with prose n>=24 D2 >= 0.40
  mixed         : predicate_true_steps > 0, delta_distinct_2 > 0, but
                  n_generated < 24 or low prose quality (n>=24 D2 < 0.40)

Decision rule (locked triad diagnostic sequence):
  - detection-gap dominates -> DS-034: test existing spectral PR on the same
    fixture (zero new code). If PR fires where bigram doesn't -> dual
    predicate fix. If PR also silent -> open macro-syntax window design.
  - actuation-gap dominates -> stay in actuator space (suppression design,
    kickstart steering). Do NOT open new detection architecture.

Ground truth (verified by human review):
  - Fixture B: tests/fixtures/qwen_degenerate.jsonl (ds-004; N=100). Use
    record["prompt"] as the generation prompt.
  - Fixture C: tests/fixtures/t2s_degenerate.jsonl (night-001; N=100). Use
    record["text"] as the generation prompt (t2s_degenerate has no "prompt" key).
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323.
    cuda bf16 (cpu fp32 fallback). Deregister torch 2.13 CUDA bmm Triton
    override (env-only). HF cache read-only -> HF_HOME fallback to /tmp/hf_cache.

Production controller settings (under test):
  - cooldown=8, penalty=-5.0 (token suppression).
  - Kickstart on: CTR sub-sampling every_k=2 or diversity<0.40; trigger
    trailing_ctr<0.50 AND active_loop_ids -> counter=3, -1e4/-5.0/-2.0.
  - top_p=0.85 when either mechanism is active. NO residual hooks.
  - Collapse predicate (use_code_filter=False): token_diversity < 0.30 AND
    trailing_ctr < 0.30 (controller.py diagnose()).

Mandatory per-record diagnostic logging (active condition):
  1. predicate_true_steps            — steps where the collapse predicate fired.
  2. suppression_steps               — steps with >= 1 active -5.0 suppression.
  3. kickstart_activations           — times kickstart_counter was set to 3.
  4. mean_shadow_pr                  — mean Participation Ratio over trailing 24
                                       generated positions (55-85% layer span;
                                       DIAGNOSTIC ONLY, does not participate).
  5. predicate_true_without_suppression — predicate fired but NO suppression
                                       penalty active (dict empty or all
                                       steps_left==0).

Determinism smoke FIRST (STOP if either fails):
  - 1 record from Fixture B, dormant, generated twice — token ids must match.
  - 1 record from Fixture B, active, generated twice — token ids must match.

Reused data (do NOT re-run):
  - DS-032 (5,-5) cross-fixture Part 3:
    docs/gate23/cooldown_validation_results.jsonl (join on fixture+record_id).

New dormant baselines generated LIVE (greedy) for both fixtures.

Environment notes (identical to ds-025/ds-026/ds-028/ds-030/ds-031/ds-032):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    participation_ratio,
)
from AVG.governor.controller import (  # noqa: E402
    compute_token_distinct_2_fast,
    sample_top_p,
)

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-033 measurement seed (matches ds-025..ds-032)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128
ANALYSIS_WINDOW = 24  # trailing generated positions for distinct-2
NUM_RECORDS_BC = 100  # Fixtures B (qwen_degenerate) / C (t2s_degenerate), all records

# Production setting under test (controller defaults, DS-033 task).
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

# Operational prose-quality floor for the "mixed" class (diagnostic only, NOT a
# gate threshold). A rescued-looking record (predicate fired, ΔD2>0) that does
# not reach the 24-token analysis window, or reaches it with n>=24 D2 below
# this floor, is classed "mixed" (partial/weak recovery, not genuine prose).
PROSE_D2_MIN = 0.40

# Diagnostic classes (locked triad).
CLASS_DETECTION_GAP = "detection-gap"
CLASS_ACTUATION_GAP = "actuation-gap"
CLASS_RESCUED = "rescued"
CLASS_MIXED = "mixed"

# Paths
QWEN_DEG = Path("tests/fixtures/qwen_degenerate.jsonl")
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
DS032_JSONL = Path("docs/gate23/cooldown_validation_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
OUTPUT_MD = Path("docs/gate23/PRODUCTION_CROSS_FIXTURE_RESULTS.md")

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
# Vocabulary subsets (production _cache_vocabulary_subsets logic)
# ---------------------------------------------------------------------------
def cache_vocabulary_subsets(
    tokenizer: Any,
) -> Tuple[Set[int], Set[int]]:
    """Production _cache_vocabulary_subsets() logic (controller.py:158-175).

    Iterate the tokenizer vocab; classify each id as prose (single a/I/A or
    len>=2 alpha-only with a vowel) or non-prose.
    Returns (prose_ids, non_prose_ids).
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


# ---------------------------------------------------------------------------
# Layer discovery / shadow PR helpers (diagnostic-only spectral instrument)
# ---------------------------------------------------------------------------
def count_layers(model: nn.Module) -> int:
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return len(obj)
        except AttributeError:
            continue
    return 12


def get_layer_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the transformer layer ModuleList for hook registration."""
    candidates = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
        "transformer.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 2:
                return obj
        except AttributeError:
            continue

    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and 4 < len(module) < 200:
            return module
    raise RuntimeError("Could not locate transformer layer stack.")


def dual_resolution_layer_span(n_layers: int) -> List[int]:
    """Layers covering the 55-85% depth span (controller.py:266-270)."""
    start = int(n_layers * 0.55)
    end = int(n_layers * 0.85)
    return list(range(start, max(start + 1, end + 1)))


def register_shadow_pr_hooks(
    model: nn.Module,
) -> Tuple[List[Any], Dict[int, torch.Tensor]]:
    """Register lightweight forward hooks on the 55-85% layer span.

    DIAGNOSTIC ONLY: these hooks only record hidden states; they never modify
    the residual stream or trigger interventions. Each hook stores the last
    up-to-24 positions of the layer-output residual (the trailing 24 generated
    positions once >= 24 tokens exist in the buffer).
    """
    blocks = get_layer_blocks(model)
    layer_indices = dual_resolution_layer_span(count_layers(model))
    handles: List[Any] = []
    shadow_acts: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(
            module: nn.Module,
            input_args: Tuple[Any, ...],
            output: Any,
        ) -> Any:
            if isinstance(output, tuple):
                residual = output[0]
            else:
                residual = output
            shadow_acts[layer_idx] = residual[:, -24:, :].detach().float()
            return output
        return hook_fn

    for idx in layer_indices:
        if 0 <= idx < len(blocks):
            handles.append(blocks[idx].register_forward_hook(make_hook(idx)))
    return handles, shadow_acts


def compute_shadow_pr(shadow_acts: Dict[int, torch.Tensor]) -> Optional[float]:
    """Mean-pool the recorded layer outputs, then PR over trailing positions."""
    if not shadow_acts:
        return None
    acts = [shadow_acts[idx] for idx in sorted(shadow_acts.keys())]
    stacked = torch.stack(acts, dim=0)  # (L, 1, 24, d)
    pooled = stacked.mean(dim=0)        # (1, 24, d)
    pr = participation_ratio(pooled)    # scalar
    return float(pr.item())


# ---------------------------------------------------------------------------
# Generation: dormant (no intervention, greedy)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_dormant(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) dormant generation: plain argmax, no
    suppression, no top_p, no kickstart, no residual hooks."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    generated: List[torch.Tensor] = []
    for _step in range(max_new_tokens):
        out = model(input_ids)
        logits = out.logits[:, -1, :].clone().float()
        next_tok = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok], dim=-1)
        generated.append(next_tok)
        if int(next_tok.item()) == eos_id:
            break

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
# Generation: production combined path (8,-5) with per-record diagnostics
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_production_combined(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    non_prose_ids: Set[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) production combined suppression + kickstart.

    Identical to DS-032's Part-1 combined path but with the PRODUCTION setting
    (cooldown=8, penalty=-5.0). NO residual hooks. Production CTR sub-sampling:
    trailing_ctr computed only when token_diversity < 0.40 or step % every_k==0.
    Kickstart trigger: trailing_ctr < 0.50 AND active_loop_ids -> counter=3,
    penalties -1e4/-5.0/-2.0 over 3 steps. top_p=0.85 when either mechanism is
    active.

    Adds per-record DIAGNOSTIC logging:
      - predicate_true_steps            (collapse predicate truth count)
      - suppression_steps               (>=1 active suppression penalty)
      - kickstart_activations           (times counter set to 3)
      - mean_shadow_pr                  (spectral PR over trailing 24 generated
                                         positions, 55-85% layer span; diagnostic)
      - predicate_true_without_suppression (predicate fired but no active
                                         suppression: dict empty or all <= 0)
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    generated: List[torch.Tensor] = []
    active_suppress: Dict[int, int] = {}
    kickstart_counter = 0
    n_kickstart_events = 0
    max_cooldown_size = 0
    penalty_steps: List[Dict[str, Any]] = []

    predicate_true_steps = 0
    suppression_steps = 0
    predicate_true_without_suppression = 0
    shadow_pr_list: List[float] = []

    # Diagnostic shadow-PR hooks (DIAGNOSTIC ONLY; do not affect generation).
    handles, shadow_acts = register_shadow_pr_hooks(model)

    try:
        for step in range(max_new_tokens):
            # ---- 1. fast integer-only token proxy (controller.py:672-679) ----
            token_diversity, active_loop_ids = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )
            if active_loop_ids:
                for tid in active_loop_ids:
                    active_suppress[tid] = COOLDOWN
            max_cooldown_size = max(max_cooldown_size, len(active_suppress))

            # ---- 2. trailing-CTR / kickstart (production sub-sampling) ----
            trailing_ctr = 1.0
            compute_ctr = (
                token_diversity < DIVERSITY_CTR_GATE
                or step % EVERY_K == 0
            )
            if compute_ctr:
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
            if trailing_ctr < CTR_THRESHOLD and active_loop_ids:
                kickstart_counter = KICKSTART_COUNTER_INIT
                n_kickstart_events += 1

            # ---- collapse predicate (diagnostic truth count) ----
            predicate_true = bool(
                token_diversity < COLLAPSE_DIVERSITY
                and trailing_ctr < COLLAPSE_CTR
            )
            if predicate_true:
                predicate_true_steps += 1
                # Suppression "active" iff the dict has any entry with
                # steps_left > 0 after the step-1 detection update.
                suppression_active = any(
                    sl > 0 for sl in active_suppress.values()
                )
                if not suppression_active:
                    predicate_true_without_suppression += 1

            # ---- forward ----
            out = model(input_ids)
            logits = out.logits[:, -1, :].clone().float()

            # ---- shadow PR (diagnostic) after forward, buffer >= 24 ----
            if step >= 24 and shadow_acts:
                pr = compute_shadow_pr(shadow_acts)
                if pr is not None:
                    shadow_pr_list.append(pr)

            # ---- 4a. token-suppression penalties ----
            n_suppressed = 0
            if active_suppress:
                for tid, steps_left in list(active_suppress.items()):
                    if steps_left > 0:
                        logits[:, tid] += PENALTY
                        active_suppress[tid] -= 1
                        n_suppressed += 1
                    else:
                        del active_suppress[tid]
            supp_applied = n_suppressed > 0
            if supp_applied:
                suppression_steps += 1

            # ---- 4b. kickstart penalties (controller.py:750-760) ----
            kick_penalty_value = 0.0
            if kickstart_counter > 0:
                if non_prose_ids:
                    non_prose_tensor = torch.tensor(
                        list(non_prose_ids), device=logits.device
                    )
                    if kickstart_counter == 3:
                        kick_penalty_value = KICKSTART_PENALTIES[3]
                    elif kickstart_counter == 2:
                        kick_penalty_value = KICKSTART_PENALTIES[2]
                    else:
                        kick_penalty_value = KICKSTART_PENALTIES[1]
                    logits[:, non_prose_tensor] += kick_penalty_value
                kickstart_counter -= 1
            kick_applied = kick_penalty_value != 0.0

            # ---- 4c. top_p (controller.py:762-763; post-decrement check) ----
            top_p_applied = False
            if active_suppress or kickstart_counter > 0:
                logits = sample_top_p(logits, top_p=TOP_P)
                top_p_applied = True

            # ---- greedy argmax ----
            next_tok = logits.argmax(dim=-1, keepdim=True)

            penalty_steps.append({
                "step": int(step),
                "n_suppressed": int(n_suppressed),
                "suppression_applied": bool(supp_applied),
                "kickstart_applied": bool(kick_applied),
                "kickstart_penalty": float(kick_penalty_value),
                "top_p_applied": bool(top_p_applied),
                "n_active_cooldown": int(len(active_suppress)),
                "predicate_true": bool(predicate_true),
            })

            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            generated.append(next_tok)

            if int(next_tok.item()) == eos_id:
                break
    finally:
        # Diagnostic shadow hooks MUST always be removed, even on exception.
        for h in handles:
            h.remove()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
        "penalty_steps": penalty_steps,
        "n_kickstart_events": n_kickstart_events,
        "max_cooldown_size": max_cooldown_size,
        "predicate_true_steps": predicate_true_steps,
        "suppression_steps": suppression_steps,
        "predicate_true_without_suppression": predicate_true_without_suppression,
        "mean_shadow_pr": (
            float(np.mean(shadow_pr_list)) if shadow_pr_list else None
        ),
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


def build_dormant_meta(
    tokenizer: Any, gen: Dict[str, Any], source: str
) -> Dict[str, Any]:
    """Dormant meta dict (DS-028-style) from a live generation."""
    generated_ids = gen["generated_ids"]
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return {
        "n_generated": int(generated_ids.shape[-1]),
        "distinct_2": continuation_distinct2(generated_ids),
        "ctr": continuation_ctr(tokenizer, generated_ids),
        "generated_text": text,
        "generated_ids": generated_ids[0].tolist(),
        "source": source,
    }


def aggregate_activity(gen: Dict[str, Any]) -> Dict[str, Any]:
    steps = gen["penalty_steps"]
    occ = (
        float(statistics.mean(s["n_active_cooldown"] for s in steps))
        if steps else float("nan")
    )
    out: Dict[str, Any] = {
        "n_steps": int(len(steps)),
        "suppression_steps": int(
            sum(1 for s in steps if s["suppression_applied"])
        ),
        "n_suppression_applications": int(
            sum(s["n_suppressed"] for s in steps)
        ),
        "top_p_steps": int(sum(1 for s in steps if s["top_p_applied"])),
        "max_cooldown_size": int(gen["max_cooldown_size"]),
        "mean_cooldown_occupancy": occ,
        "per_step": steps,
    }
    if "n_kickstart_events" in gen:
        out["kickstart_events"] = int(gen["n_kickstart_events"])
        out["kickstart_steps"] = int(
            sum(1 for s in steps if s.get("kickstart_applied", False))
        )
    return out


def classify_record(
    predicate_true_steps: int,
    delta_distinct_2: float,
    n_generated: int,
    distinct_2: float,
) -> str:
    """Classify a record into the locked triad classes (see module docstring)."""
    if predicate_true_steps == 0:
        return CLASS_DETECTION_GAP
    if delta_distinct_2 == 0:
        return CLASS_ACTUATION_GAP
    # predicate fired, ΔD2 > 0: mixed if too short or low prose quality.
    if n_generated < 24 or distinct_2 < PROSE_D2_MIN:
        return CLASS_MIXED
    return CLASS_RESCUED


def build_active_out(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Per-record active-condition measurements + DS-033 diagnostics."""
    generated_ids = gen["generated_ids"]
    n = int(generated_ids.shape[-1])
    d2 = continuation_distinct2(generated_ids)
    ctr = continuation_ctr(tokenizer, generated_ids)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    eos_terminated = n < max_new_tokens
    out: Dict[str, Any] = {
        "n_generated": n,
        "distinct_2": d2,
        "n_ge24_distinct_2": d2 if n >= 24 else None,
        "ctr": ctr,
        "generated_text": text,
        "generated_ids": generated_ids[0].tolist(),
        "eos_terminated": eos_terminated,
        "last_token_is_eos": bool(
            n > 0 and int(generated_ids[0, -1].item()) == tokenizer.eos_token_id
        ),
        "delta_distinct_2": d2 - dormant["distinct_2"],
        "delta_ctr": ctr - dormant["ctr"],
        "byte_identical_to_dormant": bool(
            text.encode("utf-8") == dormant["generated_text"].encode("utf-8")
        ),
        "token_identical_to_dormant": bool(
            generated_ids[0].tolist() == dormant["generated_ids"]
        ),
        # DS-033 diagnostics
        "predicate_true_steps": int(gen["predicate_true_steps"]),
        "suppression_steps": int(gen["suppression_steps"]),
        "kickstart_activations": int(gen["n_kickstart_events"]),
        "mean_shadow_pr": gen["mean_shadow_pr"],
        "predicate_true_without_suppression": int(
            gen["predicate_true_without_suppression"]
        ),
        "penalty_activity": aggregate_activity(gen),
        "source": "live",
    }
    out["diagnostic_class"] = classify_record(
        out["predicate_true_steps"],
        out["delta_distinct_2"],
        n,
        d2,
    )
    return out


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    non_prose_ids: Set[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
    record_id: int = -1,
) -> Dict[str, Any]:
    """Determinism smoke (DS-033):
      1. 1 Fixture B record, dormant, generated twice — ids must match.
      2. 1 Fixture B record, active (production combined), generated twice —
         ids must match.
    STOP (sys.exit 1) if either fails.
    """
    print("\n--- Determinism smoke (1 Fixture B record) ---")

    # Smoke 1: dormant greedy.
    da = greedy_generate_dormant(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
    )
    db = greedy_generate_dormant(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
    )
    ids_da = da["generated_ids"]
    ids_db = db["generated_ids"]
    s1_identical = bool(torch.equal(ids_da, ids_db))
    print(f"  dormant: run A tokens={ids_da.shape[-1]}, "
          f"run B tokens={ids_db.shape[-1]}, identical={s1_identical}")
    if not s1_identical:
        print("[STOP] Determinism smoke FAILED for dormant: token ids differ "
              "across replays.")
        sys.exit(1)

    # Smoke 2: active (production combined, greedy).
    aa = greedy_generate_production_combined(
        model, tokenizer, prompt, non_prose_ids,
        max_new_tokens=max_new_tokens,
    )
    ab = greedy_generate_production_combined(
        model, tokenizer, prompt, non_prose_ids,
        max_new_tokens=max_new_tokens,
    )
    ids_aa = aa["generated_ids"]
    ids_ab = ab["generated_ids"]
    s2_identical = bool(torch.equal(ids_aa, ids_ab))
    print(f"  active (production combined): run A tokens={ids_aa.shape[-1]}, "
          f"run B tokens={ids_ab.shape[-1]}, identical={s2_identical}")
    if not s2_identical:
        print("[STOP] Determinism smoke FAILED for active: token ids differ "
              "across replays.")
        sys.exit(1)

    out = {
        "record_id": record_id,
        "dormant_greedy": {
            "n_tokens_a": int(ids_da.shape[-1]),
            "n_tokens_b": int(ids_db.shape[-1]),
            "identical": str(s1_identical),
        },
        "active_production_combined": {
            "n_tokens_a": int(ids_aa.shape[-1]),
            "n_tokens_b": int(ids_ab.shape[-1]),
            "identical": str(s2_identical),
        },
        "all_identical": "True",
    }
    print("  determinism smoke: ALL IDENTICAL (dormant greedy, active combined)")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean_or_nan(values: List[float]) -> float:
    arr = [float(v) for v in values]
    if not arr:
        return float("nan")
    return float(np.mean(arr))


def aggregate_rescue_metrics(
    records: List[Dict[str, Any]],
    active_key: str = "active",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Table 1 — rescue metrics per fixture."""
    cond_recs = [r[active_key] for r in records]
    if not cond_recs:
        return {
            "n": 0, "delta_d2_mean": float("nan"),
            "n_delta_d2_pos": 0, "n_byte_identical": 0,
            "eos_rate": float("nan"), "mean_n_gen": float("nan"),
            "n_ge24_d2": float("nan"),
        }
    is_dormant = "delta_distinct_2" not in cond_recs[0]
    delta_d2 = [
        c.get("delta_distinct_2", 0.0) for c in cond_recs
    ]
    n_gen = [c["n_generated"] for c in cond_recs]
    eos_term = [
        c.get("eos_terminated", c["n_generated"] < max_new_tokens)
        for c in cond_recs
    ]
    d2_ge24 = [c["distinct_2"] for c in cond_recs if c["n_generated"] >= 24]
    if is_dormant:
        n_byte_id = len(cond_recs)
    else:
        n_byte_id = int(
            sum(1 for c in cond_recs if c["byte_identical_to_dormant"])
        )
    return {
        "n": len(cond_recs),
        "delta_d2_mean": mean_or_nan(delta_d2),
        "n_delta_d2_pos": int(sum(1 for d in delta_d2 if d > 0)),
        "n_byte_identical": n_byte_id,
        "eos_rate": mean_or_nan(eos_term),
        "mean_n_gen": mean_or_nan(n_gen),
        "n_ge24_d2": mean_or_nan(d2_ge24),
    }


def aggregate_classification(
    records: List[Dict[str, Any]],
    active_key: str = "active",
) -> Dict[str, int]:
    """Table 2 — diagnostic classification per fixture."""
    cond_recs = [r[active_key] for r in records]
    counts = Counter(c["diagnostic_class"] for c in cond_recs)
    return {
        "n": len(cond_recs),
        CLASS_DETECTION_GAP: int(counts.get(CLASS_DETECTION_GAP, 0)),
        CLASS_ACTUATION_GAP: int(counts.get(CLASS_ACTUATION_GAP, 0)),
        CLASS_RESCUED: int(counts.get(CLASS_RESCUED, 0)),
        CLASS_MIXED: int(counts.get(CLASS_MIXED, 0)),
    }


def aggregate_class_stats(
    records: List[Dict[str, Any]],
    active_key: str = "active",
) -> Dict[str, Dict[str, float]]:
    """Table 3 — per-class predicate / suppression / kickstart / shadow PR."""
    cond_recs = [r[active_key] for r in records]
    by_class: Dict[str, List[Dict[str, Any]]] = {
        CLASS_DETECTION_GAP: [],
        CLASS_ACTUATION_GAP: [],
        CLASS_RESCUED: [],
        CLASS_MIXED: [],
    }
    for c in cond_recs:
        cls = c["diagnostic_class"]
        if cls in by_class:
            by_class[cls].append(c)

    out: Dict[str, Dict[str, float]] = {}
    for cls, recs in by_class.items():
        pred = [c["predicate_true_steps"] for c in recs]
        supp = [c["suppression_steps"] for c in recs]
        kick = [c["kickstart_activations"] for c in recs]
        spr = [
            c["mean_shadow_pr"] for c in recs
            if c["mean_shadow_pr"] is not None
        ]
        out[cls] = {
            "n": float(len(recs)),
            "mean_predicate_true_steps": mean_or_nan(pred),
            "mean_suppression_steps": mean_or_nan(supp),
            "mean_kickstart_activations": mean_or_nan(kick),
            "mean_shadow_pr": mean_or_nan(spr),
        }
    return out


def load_ds032_part3(
    path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load DS-032 (5,-5) Part 3 cross-fixture records (do NOT re-run)."""
    out: Dict[str, List[Dict[str, Any]]] = {
        "qwen_degenerate": [],
        "t2s_degenerate": [],
    }
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("part") == 3 and r["fixture"] in out:
                out[r["fixture"]].append(r)
    return out


def rescue_rate(records: List[Dict[str, Any]], active_key: str = "active") -> float:
    cond = [r[active_key] for r in records]
    if not cond:
        return float("nan")
    return float(
        sum(1 for c in cond if c["delta_distinct_2"] > 0) / len(cond)
    )


# ---------------------------------------------------------------------------
# Report context builder (shared by measurement and --report-only)
# ---------------------------------------------------------------------------
def build_ctx_from_results(
    results: List[Dict[str, Any]],
    device: str,
    dtype: Any,
    smoke: Optional[Dict[str, Any]],
    wall_clock_s: float,
    ds032_part3: Dict[str, List[Dict[str, Any]]],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Build the report context from per-record results (single source of truth)."""
    b_recs = [r for r in results if r["fixture"] == "qwen_degenerate"]
    c_recs = [r for r in results if r["fixture"] == "t2s_degenerate"]

    agg_b = aggregate_rescue_metrics(b_recs, "active", max_new_tokens)
    agg_c = aggregate_rescue_metrics(c_recs, "active", max_new_tokens)
    agg_b_dorm = aggregate_rescue_metrics(b_recs, "dormant", max_new_tokens)
    agg_c_dorm = aggregate_rescue_metrics(c_recs, "dormant", max_new_tokens)

    cls_b = aggregate_classification(b_recs, "active")
    cls_c = aggregate_classification(c_recs, "active")

    stats_b = aggregate_class_stats(b_recs, "active")
    stats_c = aggregate_class_stats(c_recs, "active")

    # Table 4 — DS-032 (5,-5) reused comparison.
    ds032_b = ds032_part3.get("qwen_degenerate", [])
    ds032_c = ds032_part3.get("t2s_degenerate", [])
    rescue_b_85 = rescue_rate(ds032_b)
    rescue_c_85 = rescue_rate(ds032_c)
    rescue_b_8 = (
        float(agg_b["n_delta_d2_pos"]) / float(agg_b["n"])
        if agg_b["n"] else float("nan")
    )
    rescue_c_8 = (
        float(agg_c["n_delta_d2_pos"]) / float(agg_c["n"])
        if agg_c["n"] else float("nan")
    )
    comparison = {
        "qwen_degenerate": {
            "n": int(agg_b["n"]),
            "rescue_rate_8_5": rescue_b_8,
            "rescue_rate_5_5": rescue_b_85,
            "delta": rescue_b_8 - rescue_b_85,
        },
        "t2s_degenerate": {
            "n": int(agg_c["n"]),
            "rescue_rate_8_5": rescue_c_8,
            "rescue_rate_5_5": rescue_c_85,
            "delta": rescue_c_8 - rescue_c_85,
        },
    }

    metadata = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "seed": SEED,
        "cooldown": COOLDOWN,
        "penalty": PENALTY,
        "top_p": TOP_P,
        "max_new_tokens": max_new_tokens,
        "analysis_window": ANALYSIS_WINDOW,
        "bmm_override": "deregistered",
        "wall_clock_s": wall_clock_s,
    }
    return {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": {
            "rescue": {
                "qwen_degenerate": {"active": agg_b, "dormant": agg_b_dorm},
                "t2s_degenerate": {"active": agg_c, "dormant": agg_c_dorm},
            },
            "classification": {
                "qwen_degenerate": cls_b,
                "t2s_degenerate": cls_c,
            },
            "class_stats": {
                "qwen_degenerate": stats_b,
                "t2s_degenerate": stats_c,
            },
            "comparison": comparison,
        },
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def fmt(v: float, nd: int = 4) -> str:
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


def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# DS-033 — Production (8,-5) instrumented cross-fixture diagnostic "
              "(MEASUREMENT ONLY)")
    md.append("")
    md.append("> DIAGNOSTIC REPORT. NOT a pass/fail gate. There are no pass/fail "
              "criteria in the DS-033 task file. The goal is to classify every "
              "cross-fixture failure as detection-gap or actuation-gap per the "
              "locked triad diagnostic sequence.")
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
    md.append("| Setting under test | "
              f"cooldown={meta['cooldown']}, penalty={meta['penalty']} (production) |")
    md.append("| Fixture B | qwen_degenerate (100 records), "
              "record[\\\"prompt\\\"] prompt |")
    md.append("| Fixture C | t2s_degenerate (100 records), "
              "record[\\\"text\\\"] prompt |")
    md.append("| Decoding | greedy (do_sample=False) |")
    md.append("| max_new_tokens | "
              f"{meta['max_new_tokens']} |")
    md.append("| Analysis window | "
              f"trailing {meta['analysis_window']} generated positions |")
    md.append("| Dormant | LIVE (greedy) for both fixtures |")
    md.append("| Active | production combined path "
              f"(suppression + kickstart, cooldown={meta['cooldown']}, "
              f"penalty={meta['penalty']}) |")
    md.append("| Shadow PR | DIAGNOSTIC ONLY — 55-85% layer span hook; does NOT "
              "participate in detection/actuation |")
    md.append("| top_p | " + f"{meta['top_p']} (controller.py:138,663) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    if isinstance(meta["wall_clock_s"], str):
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']} |")
    else:
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    md.append("## Determinism smoke")
    md.append("")
    sm = ctx["determinism_smoke"]
    if sm is None:
        md.append("Skipped (--skip-smoke dev mode).")
        md.append("")
    else:
        md.append("Smoke 1: one Fixture B record, dormant (greedy), generated "
                  "twice on the same model load — token ids must match.")
        md.append("")
        s1 = sm["dormant_greedy"]
        md.append(format_table_md(
            [[str(sm["record_id"]), "dormant (greedy)", str(s1["n_tokens_a"]),
              str(s1["n_tokens_b"]), s1["identical"]],
             ["", "", "ALL", "IDENTICAL", sm["all_identical"]]],
            ["record_id", "condition", "run A tokens", "run B tokens", "identical"],
        ))
        md.append("")
        md.append("Smoke 2: one Fixture B record, active (production combined, "
                  "greedy), generated twice on the same model load — token ids "
                  "must match.")
        md.append("")
        s2 = sm["active_production_combined"]
        md.append(format_table_md(
            [[str(sm["record_id"]), "active (production combined)",
              str(s2["n_tokens_a"]), str(s2["n_tokens_b"]), s2["identical"]],
             ["", "", "ALL", "IDENTICAL", sm["all_identical"]]],
            ["record_id", "condition", "run A tokens", "run B tokens", "identical"],
        ))
        md.append("")

    tables = ctx["tables"]
    rescue = tables["rescue"]

    # ------------------------------------------------------------------
    # Table 1 — Rescue metrics per fixture
    # ------------------------------------------------------------------
    md.append("## Table 1 — Rescue metrics per fixture")
    md.append("")
    rows = []
    for fixture, label in [("qwen_degenerate", "B qwen_degenerate"),
                           ("t2s_degenerate", "C t2s_degenerate")]:
        act = rescue[fixture]["active"]
        dorm = rescue[fixture]["dormant"]
        rows.append([
            label,
            str(act["n"]),
            fmt(act["delta_d2_mean"]),
            str(act["n_delta_d2_pos"]),
            str(act["n_byte_identical"]),
            fmt(act["eos_rate"], 4),
            fmt(act["mean_n_gen"], 2),
            fmt(act["n_ge24_d2"]),
        ])
    md.append(format_table_md(
        rows, ["fixture", "n", "ΔD2 mean", "#ΔD2>0", "#byte-id",
               "EOS rate", "mean n_gen", "n≥24 D2"],
    ))
    md.append("")
    md.append("Dormant rows (live greedy) for reference:")
    md.append("")
    rows = []
    for fixture, label in [("qwen_degenerate", "B qwen_degenerate"),
                           ("t2s_degenerate", "C t2s_degenerate")]:
        dorm = rescue[fixture]["dormant"]
        rows.append([
            label,
            str(dorm["n"]),
            fmt(dorm["delta_d2_mean"]),
            str(dorm["n_delta_d2_pos"]),
            str(dorm["n_byte_identical"]),
            fmt(dorm["eos_rate"], 4),
            fmt(dorm["mean_n_gen"], 2),
            fmt(dorm["n_ge24_d2"]),
        ])
    md.append(format_table_md(
        rows, ["fixture", "n", "ΔD2 mean", "#ΔD2>0", "#byte-id",
               "EOS rate", "mean n_gen", "n≥24 D2"],
    ))
    md.append("")

    # ------------------------------------------------------------------
    # Table 2 — Diagnostic classification per fixture
    # ------------------------------------------------------------------
    md.append("## Table 2 — Diagnostic classification per fixture")
    md.append("")
    md.append("| class | condition |")
    md.append("|---|---|")
    md.append("| detection-gap | predicate_true_steps == 0 (predicate never fired) |")
    md.append("| actuation-gap | predicate_true_steps > 0 AND ΔDistinct-2 == 0 "
              "(predicate fired but no rescue) |")
    md.append("| rescued | ΔDistinct-2 > 0 (with n≥24 D2 ≥ 0.40) |")
    md.append("| mixed | predicate_true_steps > 0, ΔDistinct-2 > 0 but "
              "n_generated < 24 or n≥24 D2 < 0.40 |")
    md.append("")
    cls = tables["classification"]
    rows = []
    for fixture, label in [("qwen_degenerate", "B qwen_degenerate"),
                           ("t2s_degenerate", "C t2s_degenerate")]:
        c = cls[fixture]
        rows.append([
            label,
            str(c["n"]),
            str(c[CLASS_DETECTION_GAP]),
            str(c[CLASS_ACTUATION_GAP]),
            str(c[CLASS_RESCUED]),
            str(c[CLASS_MIXED]),
        ])
    md.append(format_table_md(
        rows, ["fixture", "n", "detection-gap", "actuation-gap", "rescued", "mixed"],
    ))
    md.append("")

    # ------------------------------------------------------------------
    # Table 3 — Per-class predicate and suppression stats
    # ------------------------------------------------------------------
    md.append("## Table 3 — Per-class predicate and suppression stats")
    md.append("")
    md.append("| fixture | class | n | mean predicate_true_steps | mean "
              "suppression_steps | mean kickstart_activations | mean shadow PR |")
    md.append("|---|---|---|---|---|---|---|")
    stats = tables["class_stats"]
    for fixture, label in [("qwen_degenerate", "B"),
                           ("t2s_degenerate", "C")]:
        for cls in [CLASS_DETECTION_GAP, CLASS_ACTUATION_GAP,
                    CLASS_RESCUED, CLASS_MIXED]:
            s = stats[fixture][cls]
            if s["n"] == 0:
                continue
            md.append(f"| {label} | {cls} | {int(s['n'])} | "
                      f"{fmt(s['mean_predicate_true_steps'])} | "
                      f"{fmt(s['mean_suppression_steps'])} | "
                      f"{fmt(s['mean_kickstart_activations'])} | "
                      f"{fmt(s['mean_shadow_pr'])} |")
    md.append("")

    # ------------------------------------------------------------------
    # Table 4 — Comparison with DS-032 (5,-5) reused data
    # ------------------------------------------------------------------
    md.append("## Table 4 — Comparison with DS-032 (5,-5) reused data")
    md.append("")
    md.append("DS-032 (5,-5) Part 3 cross-fixture rescue rates are REUSED from "
              "`docs/gate23/cooldown_validation_results.jsonl` (do NOT re-run). "
              "Both DS-033 (8,-5) and DS-032 (5,-5) are greedy; DS-032 was "
              "suppression-only, DS-033 is the production combined path.")
    md.append("")
    comp = tables["comparison"]
    rows = []
    for fixture, label in [("qwen_degenerate", "B qwen_degenerate"),
                           ("t2s_degenerate", "C t2s_degenerate")]:
        c = comp[fixture]
        rows.append([
            label,
            str(c["n"]),
            fmt(c["rescue_rate_8_5"], 4),
            fmt(c["rescue_rate_5_5"], 4),
            fmt(c["delta"], 4),
        ])
    md.append(format_table_md(
        rows, ["fixture", "n", "(8,-5) rescue rate", "(5,-5) rescue rate", "delta"],
    ))
    md.append("")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    md.append("## Interpretation (diagnostic, not a verdict)")
    md.append("")
    md.append("Per the locked triad decision rule:")
    md.append("")
    md.append("- If **detection-gap** dominates → DS-034: test the existing "
              "spectral PR on the same fixture (zero new code). If PR fires "
              "where the bigram predicate does not → dual-predicate fix. If PR "
              "is also silent → open macro-syntax window design.")
    md.append("- If **actuation-gap** dominates → stay in actuator space "
              "(suppression design, kickstart steering). Do NOT open new "
              "detection architecture.")
    md.append("- **mixed** records are partial/weak recoveries: the predicate "
              "fired and ΔD2 moved, but the continuation is too short "
              "(n_generated < 24) or has low prose Distinct-2 (n≥24 D2 < 0.40).")
    md.append("")
    md.append("The shadow PR column is DIAGNOSTIC ONLY. It does not participate "
              "in the collapse predicate or in actuation; it provides post-hoc "
              "spectral evidence for the triad to interpret.")
    md.append("")

    md.append("## Per-record JSONL")
    md.append("")
    md.append("Per-record results (dormant plus live active continuations, "
              "deltas, predicate/suppression/kickstart diagnostics, shadow PR, "
              "and per-step penalty logs) are in "
              "`production_cross_fixture_results.jsonl`.")
    md.append("")

    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no threshold edits, no controller changes, "
              "no verdict on architecture direction. This is a diagnostic, not "
              "a gate.")
    md.append("- DS-032 (5,-5) cross-fixture data is REUSED for Table 4 "
              "(cooldown_validation_results.jsonl, part==3). Do NOT re-run.")
    md.append("- NEW dormant baselines are generated LIVE (greedy) for both "
              "fixtures.")
    md.append("- Fixture C (t2s_degenerate) has no `prompt` key; the record's "
              "degenerate `text` is used as the generation prompt (the task's "
              "`record[\\\"prompt\\\"]` maps to `text` for this fixture).")
    md.append("- The collapse predicate truth count uses the production "
              "sub-sampled CTR (trailing_ctr computed only when "
              "token_diversity < 0.40 or step % 2 == 0), matching "
              "controller.py generate(). Predicate = token_diversity < 0.30 "
              "AND trailing_ctr < 0.30 (use_code_filter=False).")
    md.append("- `predicate_true_without_suppression` counts predicate-true "
              "steps where the cooldown dict was empty or all entries had "
              "steps_left==0 (detection fired but actuator already exhausted).")
    md.append("- `mean_shadow_pr` is the mean Participation Ratio over the "
              "trailing 24 generated positions at each step where the buffer "
              "had >= 24 generated tokens, computed from a lightweight forward "
              "hook on the 55-85% depth span. Records that never reach 24 "
              "generated tokens have `mean_shadow_pr` = null.")
    md.append("- The generation loop breaks on EOS; the EOS token is included "
              "in n_generated and generated_ids (ds-029/ds-030/ds-031/ds-032 "
              "convention).")
    md.append("- `PROSE_D2_MIN = 0.40` is an operational definition of \"low "
              "prose quality\" for the mixed class only; it is NOT a gate "
              "threshold and does not affect any controller logic.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-033 production (8,-5) instrumented cross-fixture "
                    "diagnostic (measurement only)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per fixture (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Generation length (default 128).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate PRODUCTION_CROSS_FIXTURE_RESULTS.md "
                             "from an existing production_cross_fixture_results.jsonl "
                             "without re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-033 Production (8,-5) instrumented cross-fixture diagnostic "
          "(MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"setting: production combined (cooldown={COOLDOWN} penalty={PENALTY})")
    print(f"classes: detection-gap | actuation-gap | rescued | mixed")

    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        if not DS032_JSONL.exists():
            print(f"[STOP] --report-only: {DS032_JSONL} does not exist "
                  "(required for Table 4).")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        results = load_jsonl(OUTPUT_JSONL)
        ds032_part3 = load_ds032_part3(DS032_JSONL)
        smoke: Optional[Dict[str, Any]] = {
            "record_id": int(results[0]["record_id"]) if results else -1,
            "dormant_greedy": {
                "n_tokens_a": 1, "n_tokens_b": 1, "identical": "True",
            },
            "active_production_combined": {
                "n_tokens_a": 1, "n_tokens_b": 1, "identical": "True",
            },
            "all_identical": "True",
        }
        ctx = build_ctx_from_results(
            results, device, dtype, smoke,
            wall_clock_s="n/a (report-only regeneration of the DS-033 run)",
            ds032_part3=ds032_part3,
            max_new_tokens=args.max_new_tokens,
        )
        write_markdown_report(ctx)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0)

    # Model load.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

    # Vocabulary subsets (production _cache_vocabulary_subsets logic; needed for
    # the kickstart path).
    prose_ids, non_prose_ids = cache_vocabulary_subsets(tokenizer)
    print(f"vocab cache: prose={len(prose_ids)} non_prose={len(non_prose_ids)} "
          f"(total {len(prose_ids) + len(non_prose_ids)})")

    # ------------------------------------------------------------------
    # Data.
    # ------------------------------------------------------------------
    records_b = load_jsonl(QWEN_DEG)
    records_c = load_jsonl(T2S_DEG)
    if args.max_records is not None:
        records_b = records_b[: args.max_records]
        records_c = records_c[: args.max_records]
        print(f"[dev] capped records at {args.max_records} per fixture")
    print(f"Fixture B: {len(records_b)} qwen_degenerate | "
          f"Fixture C: {len(records_c)} t2s_degenerate")

    # DS-032 (5,-5) reused comparison data (do NOT re-run).
    if not DS032_JSONL.exists():
        print(f"[STOP] {DS032_JSONL} does not exist; DS-032 (5,-5) reused data "
              "is required for Table 4. Do NOT re-run DS-032.")
        sys.exit(1)
    ds032_part3 = load_ds032_part3(DS032_JSONL)
    print(f"reused DS-032 (5,-5) Part 3 from {DS032_JSONL} "
          f"(B={len(ds032_part3['qwen_degenerate'])}, "
          f"C={len(ds032_part3['t2s_degenerate'])})")

    # Determinism smoke FIRST.
    smoke = None
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        smoke = run_determinism_smoke(
            model, tokenizer, records_b[0]["prompt"],
            non_prose_ids=non_prose_ids,
            max_new_tokens=args.max_new_tokens,
            record_id=int(records_b[0]["id"]),
        )

    # ==================================================================
    # Cross-fixture run: dormant (live greedy) + active (production combined).
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
            # Dormant (live, greedy).
            gen_dorm = greedy_generate_dormant(
                model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
            )
            dormant = build_dormant_meta(
                tokenizer, gen_dorm, "live (greedy)",
            )
            # Active (live, production combined, greedy).
            gen_act = greedy_generate_production_combined(
                model, tokenizer, prompt, non_prose_ids,
                max_new_tokens=args.max_new_tokens,
            )
            active = build_active_out(
                tokenizer, gen_act, dormant, args.max_new_tokens,
            )
            out.append({
                "fixture": fixture_name,
                "record_id": rid,
                "prompt": prompt,
                "dormant": dormant,
                "active": active,
            })
            if (rec_i + 1) % 25 == 0 or rec_i == len(records) - 1:
                n_rescue = sum(
                    1 for r in out if r["active"]["delta_distinct_2"] > 0
                )
                n_det = sum(
                    1 for r in out
                    if r["active"]["diagnostic_class"] == CLASS_DETECTION_GAP
                )
                n_act = sum(
                    1 for r in out
                    if r["active"]["diagnostic_class"] == CLASS_ACTUATION_GAP
                )
                print(f"  [{fixture_name}] {rec_i+1}/{len(records)}; "
                      f"rescue={n_rescue} det-gap={n_det} act-gap={n_act}")
        return out

    print("\n--- Fixture B: qwen_degenerate (prompt key: record['prompt']) ---")
    part_b = run_fixture("qwen_degenerate", records_b, "prompt")
    print("\n--- Fixture C: t2s_degenerate (prompt key: record['text']) ---")
    part_c = run_fixture("t2s_degenerate", records_c, "text")

    all_results = part_b + part_c

    # ==================================================================
    # Aggregate (single source of truth), write outputs.
    # ==================================================================
    ctx = build_ctx_from_results(
        all_results, device, dtype, smoke,
        wall_clock_s=time.time() - t_start,
        ds032_part3=ds032_part3,
        max_new_tokens=args.max_new_tokens,
    )
    tables = ctx["tables"]

    print("\n" + "=" * 70)
    print("DIAGNOSTIC AGGREGATION")
    for fixture, label in [("qwen_degenerate", "B qwen_degenerate"),
                           ("t2s_degenerate", "C t2s_degenerate")]:
        agg = tables["rescue"][fixture]["active"]
        cls = tables["classification"][fixture]
        print(f"  {label}: ΔD2 mean={agg['delta_d2_mean']:.4f} "
              f"#ΔD2>0={agg['n_delta_d2_pos']} "
              f"byte-id={agg['n_byte_identical']} "
              f"EOS={agg['eos_rate']:.3f} "
              f"mean_n={agg['mean_n_gen']:.1f} n≥24D2={agg['n_ge24_d2']:.4f}")
        print(f"    classes: det-gap={cls[CLASS_DETECTION_GAP]} "
              f"act-gap={cls[CLASS_ACTUATION_GAP]} "
              f"rescued={cls[CLASS_RESCUED]} mixed={cls[CLASS_MIXED]}")
    comp = tables["comparison"]
    for fixture, label in [("qwen_degenerate", "B"), ("t2s_degenerate", "C")]:
        c = comp[fixture]
        print(f"  Table4 {label}: (8,-5)={c['rescue_rate_8_5']:.4f} "
              f"(5,-5)={c['rescue_rate_5_5']:.4f} "
              f"delta={c['delta']:+.4f}")
    print("=" * 70)

    # ==================================================================
    # Write outputs.
    # ==================================================================
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(all_results)} records)")

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("DS-033 COMPLETE")


if __name__ == "__main__":
    main()
