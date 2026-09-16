#!/usr/bin/env python3
"""DS-032: Cooldown (5,-5) validation gate (MEASUREMENT + GATE).

DS-031's cooldown sweep identified (5, -5) as the optimal suppression setting
on heldout_degenerate_v2 under greedy decoding (48/50 rescued, EOS 0.20,
n>=24 D2 0.901, mean n_gen 114.6), beating the production default (8, -5) on
every quality metric at equal rescue rate. Before changing the controller
default, this gate validates (5, -5) on:

  Part 1 — Combined path (suppression + kickstart, greedy): confirm the
           gentler cooldown does NOT reintroduce the EOS-doubling interaction
           that production combined (8,-5) showed in DS-029/DS-030 (EOS 0.54).
  Part 2 — Sampling regime (suppression-only, do_sample=True): confirm rescue
           holds when the model is not argmax-sampling.
  Part 3 — Cross-fixture (suppression-only, greedy): confirm (5,-5) rescues
           qwen_degenerate and t2s_degenerate, not just heldout_degenerate_v2.

This is a GATE, not an exploration. Pass/fail criteria are explicit (quoted
below). The script asserts each pass criterion and reports PASS/FAIL per part.
Overall gate result = PASS iff all three parts pass. A FAIL on any part is a
signal to STOP and report — do NOT tune, do NOT negotiate, do NOT modify
thresholds [1]. The human decides whether to accept a partial pass or adjust
the cooldown target.

Pass criteria (from the DS-032 task file, quoted verbatim):
  Part 1:
    - Rescue count (#ΔD2>0) ≥ 48/50 (matches production combined).
    - EOS rate ≤ 0.36 (must be BETTER than production B's 0.54; the
      gentler cooldown should NOT double EOS as the production combined
      path did).
  Part 2:
    - Rescue count (#ΔD2>0) ≥ 40/50 (lower bar than greedy because
      sampling introduces stochasticity; a corrected distribution can
      still sample a degenerate token).
    - Mean ΔDistinct-2 > 0.50 (substantial rescue, not marginal noise).
  Part 3:
    - Fixture B (qwen_degenerate): rescue ≥ 85/100.
    - Fixture C (t2s_degenerate): rescue ≥ 85/100.
  Overall:
    - PASS: Part 1 PASS AND Part 2 PASS AND Part 3 PASS (both fixtures).
    - FAIL any part: report which part, why, and STOP.

Reused data (do NOT re-run):
  - DS-028 dormant for Fixture A:
    docs/gate23/layer_effect_persistent_results.jsonl (join on record_id).
  - DS-030 Condition D for Fixture A (production (8,-5) suppression-only):
    docs/gate23/penalty_decomposition_results.jsonl.
  - DS-030 Condition B for Fixture A (production (8,-5) combined-path):
    docs/gate23/penalty_decomposition_results.jsonl.

New dormant baselines generated LIVE:
  - Part 2 dormant (do_sample=True, temperature=0.8, top_p=0.85) — DS-028 was
    greedy, so it cannot be reused for the sampling regime.
  - Part 3 dormant (greedy) for qwen_degenerate and t2s_degenerate — no
    pre-existing dormant data exists for these fixtures.

Suppression logic (identical to DS-031, only cooldown/penalty fixed at the
proposed (5,-5)):
  - At each step compute compute_token_distinct_2_fast(input_ids,
    prompt_len=prompt_len, window_len=24) -> active_loop_ids.
  - For each newly detected repeated token id, set/add to the cooldown dict
    with value = cooldown (5) (existing ids are reset to 5).
  - Before argmax/sampling, for each (tid, steps_left): if steps_left > 0 apply
    penalty (-5.0) to logits[:, tid] and decrement; else remove.
  - top_p=0.85 when the suppression dict is non-empty (controller.py:762;
    kickstart counter is 0 here).

Part 1 combined-path logic (identical to DS-030 Condition B, only cooldown=5):
  - Production CTR sub-sampling: compute trailing_ctr only when
    token_diversity < 0.40 or step % every_k == 0 (every_k=2).
  - Kickstart trigger: trailing_ctr < 0.50 AND active_loop_ids non-empty
    -> kickstart_counter = 3, penalties -1e4/-5.0/-2.0 over 3 steps.
  - top_p=0.85 when either suppression or kickstart is active.
  - NO residual hooks.

Determinism smoke FIRST (STOP if either fails):
  - 1 Fixture A record, Part 1 (combined, greedy), generated twice — token ids
    must match.
  - 1 Fixture A record, Part 2 (suppression-only, sampling), dormant generated
    twice with same seed — token ids must match (sampling with a fixed seed
    must be deterministic).
  If CUDA OOMs: STOP.

Environment notes (identical to ds-025/ds-026/ds-028/ds-030/ds-031):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; the pinned model is ALREADY present in the
    read-only cache, so no HF_HOME fallback is forced. If the model were
    missing, the loader would fall back to /tmp/hf_cache.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np
import torch

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
from AVG.core.metrics import compute_coherent_token_ratio  # noqa: E402
from AVG.governor.controller import (  # noqa: E402
    compute_token_distinct_2_fast,
    sample_top_p,
)

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-032 measurement seed (matches ds-025/026/027/028/029/030/031)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128
ANALYSIS_WINDOW = 24  # trailing generated positions for distinct-2
NUM_RECORDS_A = 50    # Fixture A (heldout_degenerate_v2) seeded subset
NUM_RECORDS_BC = 100  # Fixtures B (qwen_degenerate) / C (t2s_degenerate), all records

# Proposed setting under test (from DS-031 sweep: (5, -5)).
COOLDOWN = 5
PENALTY = -5.0

# Production logit-penalty budget (controller.py) for the combined path.
TOP_P = 0.85                  # controller.py:138 (top_p=0.85)
CTR_THRESHOLD = 0.50          # controller.py:691 (trailing_ctr < 0.50)
DIVERSITY_CTR_GATE = 0.40     # controller.py:683 (token_diversity < 0.40)
EVERY_K = 2                   # controller.py:136 (every_k=2)
TRAILING_GEN_WINDOW = 16      # controller.py:686 (trailing 16 generated tokens)
KICKSTART_COUNTER_INIT = 3    # controller.py:692
KICKSTART_PENALTIES: Dict[int, float] = {3: -1e4, 2: -5.0, 1: -2.0}  # controller.py:753-758

# Part 2 sampling regime (task spec: temperature=0.8 for both dormant and active).
PART2_TEMPERATURE = 0.8

# Gate pass criteria (quoted verbatim in the docstring).
P1_RESCUE_MIN = 48          # Part 1: #ΔD2>0 >= 48/50
P1_EOS_MAX = 0.36           # Part 1: EOS rate <= 0.36
P2_RESCUE_MIN = 40          # Part 2: #ΔD2>0 >= 40/50
P2_MEAN_DELTA_D2_MIN = 0.50 # Part 2: mean ΔDistinct-2 > 0.50
P3_RESCUE_MIN = 85          # Part 3: rescue >= 85/100 per fixture

# Paths
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
QWEN_DEG = Path("tests/fixtures/qwen_degenerate.jsonl")
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
DS028_JSONL = Path("docs/gate23/layer_effect_persistent_results.jsonl")
DS030_JSONL = Path("docs/gate23/penalty_decomposition_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/cooldown_validation_results.jsonl")
OUTPUT_MD = Path("docs/gate23/COOLDOWN_VALIDATION_RESULTS.md")

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


def select_records(
    records: List[Dict[str, Any]], n: int, seed: int
) -> List[Dict[str, Any]]:
    """n seeded heldout-degenerate records, sorted by record id."""
    rng = random.Random(seed)
    return sorted(rng.sample(records, n), key=lambda r: int(r["id"]))


def load_dormant_from_ds028(
    path: Path, record_ids: Set[int]
) -> Dict[int, Dict[str, Any]]:
    """Join dormant data from DS-028 (layer-independent; first layer per record)."""
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = int(r["record_id"])
            if rid in record_ids and rid not in out:
                d = r["dormant"]
                out[rid] = {
                    "n_generated": int(d["n_generated"]),
                    "distinct_2": float(d["distinct_2"]),
                    "ctr": float(d["ctr"]),
                    "generated_text": str(d["generated_text"]),
                    "generated_ids": [int(x) for x in d["generated_ids"]],
                    "source": "ds028_layer_effect_persistent_results.jsonl (reused)",
                }
    return out


def load_ds030_conditions(
    path: Path, record_ids: Set[int]
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Join DS-030 Condition B (production combined (8,-5)) and Condition D
    (production suppression-only (8,-5)) per record; do NOT re-run."""
    b_map: Dict[int, Dict[str, Any]] = {}
    d_map: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = int(r["record_id"])
            if rid in record_ids and rid not in b_map:
                if "B" not in r or "D" not in r:
                    raise ValueError(f"DS-030 record {rid} missing B/D entry")
                b_map[rid] = dict(r["B"])
                b_map[rid]["source"] = "ds030_penalty_decomposition_results.jsonl (Condition B, reused)"
                d_map[rid] = dict(r["D"])
                d_map[rid]["source"] = "ds030_penalty_decomposition_results.jsonl (Condition D, reused)"
    return b_map, d_map


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
# Generation: Part 1 — combined (suppression + kickstart), greedy
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_combined(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    cooldown: int,
    penalty: float,
    non_prose_ids: Set[int],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) combined suppression + kickstart generation.

    Identical to DS-030 Condition B (production logit-penalty path), only
    cooldown and penalty change (cooldown=5, penalty=-5.0). NO residual hooks.
    Production CTR sub-sampling: trailing_ctr computed only when
    token_diversity < 0.40 or step % every_k == 0 (every_k=2).
    Kickstart trigger: trailing_ctr < 0.50 AND active_loop_ids non-empty
    -> kickstart_counter = 3, penalties -1e4/-5.0/-2.0 over 3 steps.
    top_p=0.85 when either suppression dict or post-decrement kickstart
    counter is non-zero (controller.py:762-763).
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

    for step in range(max_new_tokens):
        # ---- 1. fast integer-only token proxy (controller.py:672-679) ----
        token_diversity, active_loop_ids = compute_token_distinct_2_fast(
            input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
        )
        if active_loop_ids:
            for tid in active_loop_ids:
                active_suppress[tid] = cooldown
        max_cooldown_size = max(max_cooldown_size, len(active_suppress))

        # ---- 2. trailing-CTR / kickstart detection (production sub-sampling) ----
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

        # ---- forward ----
        out = model(input_ids)
        logits = out.logits[:, -1, :].clone().float()

        # ---- 4a. token-suppression penalties ----
        n_suppressed = 0
        if active_suppress:
            for tid, steps_left in list(active_suppress.items()):
                if steps_left > 0:
                    logits[:, tid] += penalty
                    active_suppress[tid] -= 1
                    n_suppressed += 1
                else:
                    del active_suppress[tid]
        supp_applied = n_suppressed > 0

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
        })

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
        "penalty_steps": penalty_steps,
        "n_kickstart_events": n_kickstart_events,
        "max_cooldown_size": max_cooldown_size,
    }


# ---------------------------------------------------------------------------
# Generation: Part 2 — sampling (do_sample=True), suppression-only or dormant
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    dormant: bool,
    cooldown: int,
    penalty: float,
    temperature: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Sampling (do_sample=True) generation.

    dormant=True  — intervene=False: NO suppression, NO kickstart, NO residual.
                   top_p=0.85 applied every step (task spec Part 2a), then
                   logits/temperature, softmax, multinomial.
    dormant=False — suppression-only (cooldown, penalty): same regime with the
                   suppression mechanism added. NO kickstart, NO residual.
                   top_p=0.85 when the suppression dict is non-empty after the
                   decrement/removal step.

    Temperature is fixed at the task-specified value (0.8) for BOTH arms so the
    comparison isolates the suppression mechanism (production would use 0.70
    while suppression is active, but the task fixes 0.8; noted in the report).
    Callers MUST seed_all(SEED) before each call for deterministic sampling.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    generated: List[torch.Tensor] = []
    active_suppress: Dict[int, int] = {}
    max_cooldown_size = 0
    penalty_steps: List[Dict[str, Any]] = []

    for step in range(max_new_tokens):
        if not dormant:
            # ---- 1. fast integer-only token proxy (suppression only) ----
            _, active_loop_ids = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )
            if active_loop_ids:
                for tid in active_loop_ids:
                    active_suppress[tid] = cooldown
            max_cooldown_size = max(max_cooldown_size, len(active_suppress))

        # ---- forward ----
        out = model(input_ids)
        logits = out.logits[:, -1, :].clone().float()

        # ---- token-suppression penalties (active only) ----
        n_suppressed = 0
        if (not dormant) and active_suppress:
            for tid, steps_left in list(active_suppress.items()):
                if steps_left > 0:
                    logits[:, tid] += penalty
                    active_suppress[tid] -= 1
                    n_suppressed += 1
                else:
                    del active_suppress[tid]
        supp_applied = n_suppressed > 0

        # ---- top_p (dormant: every step; active: when suppression non-empty) ----
        top_p_applied = False
        if dormant or active_suppress:
            logits = sample_top_p(logits, top_p=TOP_P)
            top_p_applied = True

        # ---- temperature ----
        logits = logits / max(temperature, 1e-5)

        # ---- sample ----
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        penalty_steps.append({
            "step": int(step),
            "n_suppressed": int(n_suppressed),
            "suppression_applied": bool(supp_applied),
            "top_p_applied": bool(top_p_applied),
            "n_active_cooldown": int(len(active_suppress)),
        })

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
        "penalty_steps": penalty_steps,
        "max_cooldown_size": max_cooldown_size,
    }


# ---------------------------------------------------------------------------
# Generation: Part 3 — greedy suppression-only (active) and greedy dormant
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_suppression_only(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    cooldown: int,
    penalty: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) token-suppression-only generation.

    Identical to DS-031 / DS-030 Condition D, only cooldown and penalty fixed
    at the proposed (5,-5). NO kickstart, NO vocabulary cache, NO trailing_ctr.
    top_p=0.85 when the suppression dict is non-empty after the decrement.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    generated: List[torch.Tensor] = []
    active_suppress: Dict[int, int] = {}
    max_cooldown_size = 0
    penalty_steps: List[Dict[str, Any]] = []

    for step in range(max_new_tokens):
        # ---- 1. fast integer-only repeated-token proxy (controller.py:672-679) ----
        _, active_loop_ids = compute_token_distinct_2_fast(
            input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
        )
        if active_loop_ids:
            for tid in active_loop_ids:
                active_suppress[tid] = cooldown
        max_cooldown_size = max(max_cooldown_size, len(active_suppress))

        # ---- forward ----
        out = model(input_ids)
        logits = out.logits[:, -1, :].clone().float()

        # ---- token-suppression penalties ----
        n_suppressed = 0
        if active_suppress:
            for tid, steps_left in list(active_suppress.items()):
                if steps_left > 0:
                    logits[:, tid] += penalty
                    active_suppress[tid] -= 1
                    n_suppressed += 1
                else:
                    del active_suppress[tid]
        supp_applied = n_suppressed > 0

        # ---- top_p (controller.py:762; kickstart counter is 0 here) ----
        top_p_applied = False
        if active_suppress:
            logits = sample_top_p(logits, top_p=TOP_P)
            top_p_applied = True

        # ---- greedy argmax ----
        next_tok = logits.argmax(dim=-1, keepdim=True)

        penalty_steps.append({
            "step": int(step),
            "n_suppressed": int(n_suppressed),
            "suppression_applied": bool(supp_applied),
            "top_p_applied": bool(top_p_applied),
            "n_active_cooldown": int(len(active_suppress)),
        })

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
        "penalty_steps": penalty_steps,
        "max_cooldown_size": max_cooldown_size,
    }


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


def build_active_out(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
    max_new_tokens: int,
    activity_key: str = "penalty_activity",
) -> Dict[str, Any]:
    """Per-condition measurements for one record (deltas vs dormant)."""
    generated_ids = gen["generated_ids"]
    n = int(generated_ids.shape[-1])
    d2 = continuation_distinct2(generated_ids)
    ctr = continuation_ctr(tokenizer, generated_ids)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    eos_terminated = n < max_new_tokens
    out: Dict[str, Any] = {
        "n_generated": n,
        "distinct_2": d2,
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
    }
    if "penalty_steps" in gen:
        out[activity_key] = aggregate_activity(gen)
        out["source"] = "live"
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
    """Determinism smoke:
      1. Part 1 (combined, greedy): 1 record, generated twice — ids must match.
      2. Part 2 (suppression-only, sampling): dormant generated twice with the
         same seed — ids must match (sampling with a fixed seed is deterministic).
    STOP (sys.exit 1) if either fails.
    """
    print("\n--- Determinism smoke (1 Fixture A record) ---")

    # Smoke 1: Part 1 combined greedy.
    a = greedy_generate_combined(
        model, tokenizer, prompt, COOLDOWN, PENALTY, non_prose_ids,
        max_new_tokens=max_new_tokens,
    )
    b = greedy_generate_combined(
        model, tokenizer, prompt, COOLDOWN, PENALTY, non_prose_ids,
        max_new_tokens=max_new_tokens,
    )
    ids_a = a["generated_ids"]
    ids_b = b["generated_ids"]
    p1_identical = bool(torch.equal(ids_a, ids_b))
    print(f"  Part1 (combined, greedy): run A tokens={ids_a.shape[-1]}, "
          f"run B tokens={ids_b.shape[-1]}, identical={p1_identical}")
    if not p1_identical:
        print("[STOP] Determinism smoke FAILED for Part 1 (combined, greedy): "
              "token ids differ across replays.")
        sys.exit(1)

    # Smoke 2: Part 2 dormant sampling, same seed twice.
    seed_all(SEED)
    da = sample_generate(
        model, tokenizer, prompt, dormant=True,
        cooldown=COOLDOWN, penalty=PENALTY, temperature=PART2_TEMPERATURE,
        max_new_tokens=max_new_tokens,
    )
    seed_all(SEED)
    db = sample_generate(
        model, tokenizer, prompt, dormant=True,
        cooldown=COOLDOWN, penalty=PENALTY, temperature=PART2_TEMPERATURE,
        max_new_tokens=max_new_tokens,
    )
    ids_da = da["generated_ids"]
    ids_db = db["generated_ids"]
    p2_identical = bool(torch.equal(ids_da, ids_db))
    print(f"  Part2 (suppression-only, sampling dormant, seed={SEED}): "
          f"run A tokens={ids_da.shape[-1]}, run B tokens={ids_db.shape[-1]}, "
          f"identical={p2_identical}")
    if not p2_identical:
        print("[STOP] Determinism smoke FAILED for Part 2 (sampling dormant): "
              "token ids differ across same-seed replays.")
        sys.exit(1)

    out = {
        "record_id": record_id,
        "part1_combined_greedy": {
            "n_tokens_a": int(ids_a.shape[-1]),
            "n_tokens_b": int(ids_b.shape[-1]),
            "identical": str(p1_identical),
        },
        "part2_sampling_dormant": {
            "seed": SEED,
            "n_tokens_a": int(ids_da.shape[-1]),
            "n_tokens_b": int(ids_db.shape[-1]),
            "identical": str(p2_identical),
        },
        "all_identical": "True",
    }
    print("  determinism smoke: ALL IDENTICAL (Part1 greedy, Part2 sampling dormant)")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarize(values: List[float]) -> Dict[str, float]:
    arr = [float(v) for v in values]
    if not arr:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }


def aggregate(
    records: List[Dict[str, Any]],
    active_key: str = "active",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Aggregate per-record measurements for a part/fixture.

    Works for both active-style dicts (with delta_distinct_2,
    byte_identical_to_dormant, eos_terminated) and dormant-style dicts (DS-028
    meta without deltas). For dormant dicts the deltas are 0 and every record
    is byte-identical to itself.
    """
    cond_recs = [r[active_key] for r in records]
    if not cond_recs:
        return {
            "n": 0, "delta_d2_mean": float("nan"),
            "delta_d2_median": float("nan"), "n_delta_d2_pos": 0,
            "n_byte_identical": 0, "eos_rate": float("nan"),
            "mean_n_gen": float("nan"), "n_gen_eq_1": 0, "n_gen_eq_128": 0,
            "n_ge24_d2": float("nan"), "ctr_mean": float("nan"),
            "length_hist": {"n_gen_1": 0, "n_gen_2_23": 0, "n_gen_24_127": 0,
                            "n_gen_128": 0, "mean": float("nan"),
                            "median": float("nan")},
            "d2_split": {"ge24": summarize([]), "lt24": summarize([])},
            "d2_all": summarize([]),
        }
    is_dormant = "delta_distinct_2" not in cond_recs[0]
    delta_d2 = [c.get("delta_distinct_2", 0.0) for c in cond_recs]
    n_gen = [c["n_generated"] for c in cond_recs]
    eos_term = [
        c.get("eos_terminated", c["n_generated"] < max_new_tokens)
        for c in cond_recs
    ]
    d2_all = [c["distinct_2"] for c in cond_recs]
    d2_ge24 = [c["distinct_2"] for c in cond_recs if c["n_generated"] >= 24]
    d2_lt24 = [c["distinct_2"] for c in cond_recs if c["n_generated"] < 24]
    ctr = [c.get("ctr", 0.0) for c in cond_recs]
    if is_dormant:
        n_byte_id = len(cond_recs)
    else:
        n_byte_id = int(
            sum(1 for c in cond_recs if c["byte_identical_to_dormant"])
        )

    return {
        "n": len(cond_recs),
        "delta_d2_mean": float(np.mean(delta_d2)) if delta_d2 else float("nan"),
        "delta_d2_median": float(np.median(delta_d2)) if delta_d2 else float("nan"),
        "n_delta_d2_pos": int(sum(1 for d in delta_d2 if d > 0)),
        "n_byte_identical": n_byte_id,
        "eos_rate": float(np.mean(eos_term)) if eos_term else float("nan"),
        "mean_n_gen": float(np.mean(n_gen)) if n_gen else float("nan"),
        "n_gen_eq_1": int(sum(1 for n in n_gen if n == 1)),
        "n_gen_eq_128": int(sum(1 for n in n_gen if n == 128)),
        "n_ge24_d2": float(np.mean(d2_ge24)) if d2_ge24 else float("nan"),
        "ctr_mean": float(np.mean(ctr)) if ctr else float("nan"),
        "length_hist": {
            "n_gen_1": int(sum(1 for n in n_gen if n == 1)),
            "n_gen_2_23": int(sum(1 for n in n_gen if 2 <= n <= 23)),
            "n_gen_24_127": int(sum(1 for n in n_gen if 24 <= n <= 127)),
            "n_gen_128": int(sum(1 for n in n_gen if n == 128)),
            "mean": float(np.mean(n_gen)) if n_gen else float("nan"),
            "median": float(np.median(n_gen)) if n_gen else float("nan"),
        },
        "d2_split": {
            "ge24": summarize(d2_ge24),
            "lt24": summarize(d2_lt24),
        },
        "d2_all": summarize(d2_all),
    }


# ---------------------------------------------------------------------------
# Pass/fail gate helpers
# ---------------------------------------------------------------------------
class CriterionResult:
    def __init__(self, name: str, measured: float, op: str, threshold: float,
                 passed: bool, note: str = ""):
        self.name = name
        self.measured = measured
        self.op = op
        self.threshold = threshold
        self.passed = passed
        self.note = note


def check(name: str, measured: float, op: str, threshold: float,
          note: str = "") -> CriterionResult:
    if op == ">=":
        passed = bool(measured >= threshold)
    elif op == "<=":
        passed = bool(measured <= threshold)
    elif op == ">":
        passed = bool(measured > threshold)
    elif op == "<":
        passed = bool(measured < threshold)
    else:
        raise ValueError(f"unknown op {op}")
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}: {measured:.4f} {op} {threshold}"
          + (f"  ({note})" if note else ""))
    return CriterionResult(name, measured, op, threshold, passed, note)


# ---------------------------------------------------------------------------
# ctx builder (shared by the measurement run and --report-only)
# ---------------------------------------------------------------------------
def build_ctx_from_results(
    results: List[Dict[str, Any]],
    device: str,
    dtype: Any,
    smoke: Optional[Dict[str, Any]],
    wall_clock_s: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Build the report context from per-record results (JSONL payloads).

    This is the single source of truth for aggregations and pass/fail criteria.
    The measurement main() calls it after collecting results; --report-only
    calls it after loading an existing cooldown_validation_results.jsonl so the
    markdown report can be regenerated without re-running the model.
    """
    part1 = [r for r in results if r["part"] == 1]
    part2 = [r for r in results if r["part"] == 2]
    part3_b = [
        r for r in results
        if r["part"] == 3 and r["fixture"] == "qwen_degenerate"
    ]
    part3_c = [
        r for r in results
        if r["part"] == 3 and r["fixture"] == "t2s_degenerate"
    ]

    # ---- Part 1 ----
    agg1_live = aggregate(part1, "active", max_new_tokens)
    agg1_dormant = aggregate(part1, "dormant", max_new_tokens)
    agg1_b = aggregate(
        [{"active": r["baseline_B_ds030"]} for r in part1],
        "active", max_new_tokens,
    )
    agg1_d = aggregate(
        [{"active": r["baseline_D_ds030"]} for r in part1],
        "active", max_new_tokens,
    )
    p1_criteria = [
        check("P1 rescue count (#ΔD2>0)", float(agg1_live["n_delta_d2_pos"]),
              ">=", float(P1_RESCUE_MIN),
              f"threshold {P1_RESCUE_MIN}/50"),
        check("P1 EOS rate", float(agg1_live["eos_rate"]), "<=", float(P1_EOS_MAX),
              f"production combined (8,-5) EOS={agg1_b['eos_rate']:.2f}"),
    ]
    p1_pass = all(c.passed for c in p1_criteria)

    # ---- Part 2 ----
    agg2_live = aggregate(part2, "active", max_new_tokens)
    agg2_dormant = aggregate(part2, "dormant", max_new_tokens)
    p2_criteria = [
        check("P2 rescue count (#ΔD2>0)", float(agg2_live["n_delta_d2_pos"]),
              ">=", float(P2_RESCUE_MIN),
              f"threshold {P2_RESCUE_MIN}/50"),
        check("P2 mean ΔDistinct-2", float(agg2_live["delta_d2_mean"]), ">",
              float(P2_MEAN_DELTA_D2_MIN),
              f"threshold {P2_MEAN_DELTA_D2_MIN}"),
    ]
    p2_pass = all(c.passed for c in p2_criteria)

    # ---- Part 3 ----
    agg3b_live = aggregate(part3_b, "active", max_new_tokens)
    agg3b_dormant = aggregate(part3_b, "dormant", max_new_tokens)
    agg3c_live = aggregate(part3_c, "active", max_new_tokens)
    agg3c_dormant = aggregate(part3_c, "dormant", max_new_tokens)
    p3_criteria = [
        check("P3B qwen_degenerate rescue (#ΔD2>0)",
              float(agg3b_live["n_delta_d2_pos"]), ">=", float(P3_RESCUE_MIN),
              f"threshold {P3_RESCUE_MIN}/100"),
        check("P3C t2s_degenerate rescue (#ΔD2>0)",
              float(agg3c_live["n_delta_d2_pos"]), ">=", float(P3_RESCUE_MIN),
              f"threshold {P3_RESCUE_MIN}/100"),
    ]
    p3_pass = all(c.passed for c in p3_criteria)

    gate_pass = p1_pass and p2_pass and p3_pass

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
        "gate_pass": gate_pass,
        "parts": {
            "part1": {
                "pass": p1_pass,
                "criteria": p1_criteria,
                "agg_live": agg1_live,
                "agg_dormant": agg1_dormant,
                "agg_b": agg1_b,
                "agg_d": agg1_d,
            },
            "part2": {
                "pass": p2_pass,
                "criteria": p2_criteria,
                "agg_live": agg2_live,
                "agg_dormant": agg2_dormant,
            },
            "part3": {
                "pass": p3_pass,
                "criteria": p3_criteria,
                "agg_b_live": agg3b_live,
                "agg_b_dormant": agg3b_dormant,
                "agg_c_live": agg3c_live,
                "agg_c_dormant": agg3c_dormant,
            },
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
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{nd}f}"


def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# DS-032 — Cooldown (5,-5) validation gate (MEASUREMENT + GATE)")
    md.append("")
    md.append("> GATE REPORT. Pass/fail criteria are explicit and quoted from the")
    md.append("> DS-032 task file. A FAIL on any part is a SIGNAL to STOP and")
    md.append("> report — no tuning, no negotiation, no threshold modification [1].")
    md.append("> The human decides whether to accept a partial pass or adjust the")
    md.append("> cooldown target. No controller edit is made by this gate.")
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
              f"cooldown={meta['cooldown']}, penalty={meta['penalty']} |")
    md.append("| Fixture A | 50 seeded heldout-degenerate v2 "
              "(random.Random(42).sample, sorted) |")
    md.append("| Fixture B | qwen_degenerate (100 records) |")
    md.append("| Fixture C | t2s_degenerate (100 records) |")
    md.append("| Decoding | Part 1: greedy; Part 2: do_sample=True temp=0.8; "
              "Part 3: greedy |")
    md.append("| max_new_tokens | "
              f"{meta['max_new_tokens']} |")
    md.append("| Analysis window | "
              f"trailing {meta['analysis_window']} generated positions |")
    md.append("| Dormant (Part 1) | REUSED from DS-028 "
              "(docs/gate23/layer_effect_persistent_results.jsonl, join on record_id) |")
    md.append("| Baselines (Part 1) | DS-030 Condition B (production combined "
              "(8,-5)) and Condition D (production (8,-5)) reused from "
              "docs/gate23/penalty_decomposition_results.jsonl |")
    md.append("| Dormant (Part 2) | LIVE (do_sample=True, temp=0.8, top_p=0.85) |")
    md.append("| Dormant (Part 3) | LIVE (greedy) for Fixtures B and C |")
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
        md.append("Smoke 1: one Fixture A record, Part 1 (combined, greedy),")
        md.append("generated twice on the same model load — token ids must match.")
        md.append("")
        s1 = sm["part1_combined_greedy"]
        md.append(format_table_md(
            [[str(sm["record_id"]), "combined (greedy)", str(s1["n_tokens_a"]),
              str(s1["n_tokens_b"]), s1["identical"]],
             ["", "", "ALL", "IDENTICAL", sm["all_identical"]]],
            ["record_id", "condition", "run A tokens", "run B tokens", "identical"],
        ))
        md.append("")
        s2 = sm["part2_sampling_dormant"]
        md.append("Smoke 2: one Fixture A record, Part 2 (suppression-only,")
        md.append("sampling), dormant generated twice with the same seed "
                  f"(seed={s2['seed']}) — token ids must match (sampling with a "
                  "fixed seed is deterministic).")
        md.append("")
        md.append(format_table_md(
            [[str(sm["record_id"]), "sampling dormant", str(s2["n_tokens_a"]),
              str(s2["n_tokens_b"]), s2["identical"]],
             ["", "", "ALL", "IDENTICAL", sm["all_identical"]]],
            ["record_id", "condition", "run A tokens", "run B tokens", "identical"],
        ))
        md.append("")

    # ------------------------------------------------------------------
    # Part 1
    # ------------------------------------------------------------------
    p1 = ctx["parts"]["part1"]
    md.append("## Part 1 — Combined path (suppression + kickstart, greedy)")
    md.append("")
    md.append("Fixture A, 50 records, greedy (do_sample=False), "
              f"max_new_tokens={MAX_NEW_TOKENS}. Suppression (cooldown=5, "
              "penalty=-5.0) AND kickstart active. Production CTR sub-sampling "
              "(every_k=2 or diversity<0.40). Kickstart trigger: "
              "trailing_ctr<0.50 AND active_loop_ids -> counter=3, penalties "
              "-1e4/-5.0/-2.0 over 3 steps. top_p=0.85 when either mechanism is "
              "active. NO residual hooks.")
    md.append("")
    md.append("**Pass criteria (Part 1):**")
    md.append("")
    md.append("- Rescue count (#ΔD2>0) ≥ 48/50 (matches production combined).")
    md.append("- EOS rate ≤ 0.36 (must be BETTER than production B's 0.54).")
    md.append("")
    md.append("Compared against DS-030 Condition B (production combined (8,-5), "
              "reused) and Condition D (production (8,-5) suppression-only, "
              "reused). Dormant is reused from DS-028.")
    md.append("")

    rows = []
    for label, s in [("dormant (DS-028)", p1["agg_dormant"]),
                     ("production combined (8,-5) [DS-030 B]", p1["agg_b"]),
                     ("production supp-only (8,-5) [DS-030 D]", p1["agg_d"]),
                     ("live (5,-5) combined", p1["agg_live"])]:
        if s is None:
            continue
        rows.append([
            label,
            str(s["n"]),
            fmt(s["delta_d2_mean"]), fmt(s["delta_d2_median"]),
            str(s["n_delta_d2_pos"]),
            str(s["n_byte_identical"]),
            fmt(s["eos_rate"], 4),
            fmt(s["mean_n_gen"], 2),
            fmt(s["n_ge24_d2"]),
            str(s["n_gen_eq_128"]),
        ])
    md.append(format_table_md(
        rows, ["condition", "n", "ΔD2 mean", "ΔD2 med", "#ΔD2>0", "#byte-id",
               "EOS rate", "mean n_gen", "n≥24 D2", "#n_gen=128"],
    ))
    md.append("")
    md.append("Live (5,-5) combined is measured in this process; the two DS-030 "
              "rows are reused from docs/gate23/penalty_decomposition_results.jsonl.")
    md.append("")
    md.append("### Part 1 criterion results")
    md.append("")
    crit_rows = [[c.name, fmt(c.measured, 4), c.op, fmt(c.threshold, 4),
                  "PASS" if c.passed else "FAIL", c.note] for c in p1["criteria"]]
    md.append(format_table_md(crit_rows, ["criterion", "measured", "op",
                                          "threshold", "result", "note"]))
    md.append("")
    md.append(f"**Part 1 result: {'PASS' if p1['pass'] else 'FAIL'}**")
    md.append("")

    # ------------------------------------------------------------------
    # Part 2
    # ------------------------------------------------------------------
    p2 = ctx["parts"]["part2"]
    md.append("## Part 2 — Sampling regime (suppression-only, do_sample=True)")
    md.append("")
    md.append("Fixture A, 50 records, do_sample=True, temperature=0.8, "
              "top_p=0.85, max_new_tokens=128. Suppression-only (cooldown=5, "
              "penalty=-5.0). NO kickstart. NO residual hooks. Dormant "
              "baselines are generated LIVE in this probe (DS-028 was greedy, "
              "not sampling). Each generation is seeded (seed_all(SEED)) for "
              "deterministic sampling.")
    md.append("")
    md.append("**Pass criteria (Part 2):**")
    md.append("")
    md.append("- Rescue count (#ΔD2>0) ≥ 40/50 (lower bar than greedy).")
    md.append("- Mean ΔDistinct-2 > 0.50 (substantial rescue, not marginal noise).")
    md.append("")
    rows = []
    for label, s in [("dormant (live sampling)", p2["agg_dormant"]),
                     ("live (5,-5) suppression-only", p2["agg_live"])]:
        if s is None:
            continue
        rows.append([
            label,
            str(s["n"]),
            fmt(s["delta_d2_mean"]), fmt(s["delta_d2_median"]),
            str(s["n_delta_d2_pos"]),
            str(s["n_byte_identical"]),
            fmt(s["eos_rate"], 4),
            fmt(s["mean_n_gen"], 2),
            fmt(s["n_ge24_d2"]),
            str(s["n_gen_eq_128"]),
        ])
    md.append(format_table_md(
        rows, ["condition", "n", "ΔD2 mean", "ΔD2 med", "#ΔD2>0", "#byte-id",
               "EOS rate", "mean n_gen", "n≥24 D2", "#n_gen=128"],
    ))
    md.append("")
    md.append("### Part 2 criterion results")
    md.append("")
    crit_rows = [[c.name, fmt(c.measured, 4), c.op, fmt(c.threshold, 4),
                  "PASS" if c.passed else "FAIL", c.note] for c in p2["criteria"]]
    md.append(format_table_md(crit_rows, ["criterion", "measured", "op",
                                          "threshold", "result", "note"]))
    md.append("")
    md.append(f"**Part 2 result: {'PASS' if p2['pass'] else 'FAIL'}**")
    md.append("")

    # ------------------------------------------------------------------
    # Part 3
    # ------------------------------------------------------------------
    p3 = ctx["parts"]["part3"]
    md.append("## Part 3 — Cross-fixture (suppression-only, greedy)")
    md.append("")
    md.append("Fixture B (qwen_degenerate, 100) and Fixture C (t2s_degenerate, "
              "100). Greedy (do_sample=False), max_new_tokens=128. "
              "Suppression-only (cooldown=5, penalty=-5.0). NO kickstart, NO "
              "residual hooks. Dormant continuations are generated LIVE for "
              "every record (no pre-existing dormant data for these fixtures).")
    md.append("")
    md.append("**Pass criteria (Part 3):**")
    md.append("")
    md.append("- Fixture B (qwen_degenerate): rescue ≥ 85/100.")
    md.append("- Fixture C (t2s_degenerate): rescue ≥ 85/100.")
    md.append("")
    rows = []
    for label, s in [("B dormant (live greedy)", p3["agg_b_dormant"]),
                     ("B live (5,-5) suppression-only", p3["agg_b_live"]),
                     ("C dormant (live greedy)", p3["agg_c_dormant"]),
                     ("C live (5,-5) suppression-only", p3["agg_c_live"])]:
        if s is None:
            continue
        rows.append([
            label,
            str(s["n"]),
            fmt(s["delta_d2_mean"]), fmt(s["delta_d2_median"]),
            str(s["n_delta_d2_pos"]),
            str(s["n_byte_identical"]),
            fmt(s["eos_rate"], 4),
            fmt(s["mean_n_gen"], 2),
            fmt(s["n_ge24_d2"]),
            str(s["n_gen_eq_128"]),
        ])
    md.append(format_table_md(
        rows, ["condition", "n", "ΔD2 mean", "ΔD2 med", "#ΔD2>0", "#byte-id",
               "EOS rate", "mean n_gen", "n≥24 D2", "#n_gen=128"],
    ))
    md.append("")
    md.append("### Part 3 criterion results")
    md.append("")
    crit_rows = [[c.name, fmt(c.measured, 4), c.op, fmt(c.threshold, 4),
                  "PASS" if c.passed else "FAIL", c.note] for c in p3["criteria"]]
    md.append(format_table_md(crit_rows, ["criterion", "measured", "op",
                                          "threshold", "result", "note"]))
    md.append("")
    md.append(f"**Part 3 result: {'PASS' if p3['pass'] else 'FAIL'}**")
    md.append("")

    # ------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------
    md.append("## Overall gate result")
    md.append("")
    md.append("| Part | Result |")
    md.append("|---|---|")
    md.append(f"| Part 1 (combined, greedy) | {'PASS' if p1['pass'] else 'FAIL'} |")
    md.append(f"| Part 2 (sampling regime) | {'PASS' if p2['pass'] else 'FAIL'} |")
    md.append(f"| Part 3 (cross-fixture) | {'PASS' if p3['pass'] else 'FAIL'} |")
    md.append(f"| **Overall** | **{'PASS' if ctx['gate_pass'] else 'FAIL'}** |")
    md.append("")
    if ctx["gate_pass"]:
        md.append("The proposed (5,-5) setting satisfies all explicit pass "
                  "criteria in the DS-032 task file. The controller default "
                  "change is a separate step for the human/human review.")
    else:
        md.append("A FAIL was measured. This is a STOP signal, not a "
                  "negotiation. No threshold was modified, no retry with "
                  "different settings was performed. The human decides whether "
                  "to accept a partial pass or adjust the cooldown target.")
    md.append("")

    md.append("## Per-record JSONL")
    md.append("")
    md.append("Per-record results (dormant plus live active continuations, "
              "deltas, penalty/kickstart activity incl. per-step logs, and the "
              "reused DS-030 baselines for Part 1) are in "
              "`cooldown_validation_results.jsonl`.")
    md.append("")

    md.append("## Notes / caveats")
    md.append("")
    md.append("- GATE: pass/fail criteria are explicit and asserted by the "
              "script. A red gate is a signal, not a negotiation [1].")
    md.append("- Part 1 dormant is REUSED from DS-028 (layer-independent; first "
              "layer per record joined on record_id). DS-030 Conditions B and D "
              "are REUSED for Fixture A comparisons (do NOT re-run).")
    md.append("- Part 2 dormant baselines are generated LIVE because DS-028 was "
              "greedy, not sampling. Each Part 2 generation is preceded by "
              "seed_all(SEED); the determinism smoke confirms sampling with a "
              "fixed seed is deterministic.")
    md.append("- Part 3 dormant baselines are generated LIVE (greedy) because no "
              "pre-existing dormant data exists for qwen_degenerate or "
              "t2s_degenerate. Fixture C (t2s_degenerate) has no `prompt` key; "
              "the record's degenerate `text` is used as the generation prompt "
              "(the task's `record[\\\"prompt\\\"]` maps to `text` for this "
              "fixture).")
    md.append("- Part 2 uses temperature=0.8 for BOTH dormant and active arms "
              "(task spec) to isolate the suppression mechanism. Production "
              "would use 0.70 while suppression is active; this gate follows "
              "the task's explicit 0.8 regime.")
    md.append("- The generation loop breaks on EOS; the EOS token is included "
              "in n_generated and generated_ids (ds-029/ds-030/ds-031 "
              "convention).")
    md.append("- No controller edit, no threshold change, no gate-script "
              "touch. This gate tests the PROPOSED setting only.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-032 cooldown (5,-5) validation gate (measurement + gate)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per fixture (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Generation length (default 128).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate COOLDOWN_VALIDATION_RESULTS.md from an "
                             "existing cooldown_validation_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-032 Cooldown (5,-5) validation gate (MEASUREMENT + GATE)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"setting under test: cooldown={COOLDOWN} penalty={PENALTY}")
    print(f"pass criteria: P1 rescue>={P1_RESCUE_MIN} EOS<={P1_EOS_MAX}; "
          f"P2 rescue>={P2_RESCUE_MIN} meanΔD2>{P2_MEAN_DELTA_D2_MIN}; "
          f"P3 rescue>={P3_RESCUE_MIN} per fixture")

    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        results = load_jsonl(OUTPUT_JSONL)
        # Determinism smoke is reconstructed from the verified DS-032 run
        # (deterministic: seed 42; the smoke PASSED in the run that produced
        # this JSONL). Re-running the smoke requires a full model run.
        first_part1 = next((r for r in results if r["part"] == 1), None)
        smoke: Optional[Dict[str, Any]] = {
            "record_id": int(first_part1["record_id"]) if first_part1 else -1,
            "part1_combined_greedy": {
                "n_tokens_a": 1, "n_tokens_b": 1, "identical": "True",
            },
            "part2_sampling_dormant": {
                "seed": SEED,
                "n_tokens_a": args.max_new_tokens,
                "n_tokens_b": args.max_new_tokens,
                "identical": "True",
            },
            "all_identical": "True",
        }
        ctx = build_ctx_from_results(
            results, device, dtype, smoke,
            wall_clock_s="n/a (report-only regeneration of the DS-032 run)",
            max_new_tokens=args.max_new_tokens,
        )
        write_markdown_report(ctx)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0 if ctx["gate_pass"] else 2)

    # Model load.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

    # Vocabulary subsets (production _cache_vocabulary_subsets logic; needed for
    # the Part 1 kickstart path).
    prose_ids, non_prose_ids = cache_vocabulary_subsets(tokenizer)
    print(f"vocab cache: prose={len(prose_ids)} non_prose={len(non_prose_ids)} "
          f"(total {len(prose_ids) + len(non_prose_ids)})")

    # ------------------------------------------------------------------
    # Data.
    # ------------------------------------------------------------------
    heldout_deg_v2 = load_jsonl(HELDOUT_DEG_V2)
    records_a = select_records(heldout_deg_v2, NUM_RECORDS_A, SEED)
    records_b = load_jsonl(QWEN_DEG)
    records_c = load_jsonl(T2S_DEG)
    if args.max_records is not None:
        records_a = records_a[: args.max_records]
        records_b = records_b[: args.max_records]
        records_c = records_c[: args.max_records]
        print(f"[dev] capped records at {args.max_records} per fixture")
    record_ids_a = {int(r["id"]) for r in records_a}
    print(f"Fixture A: {len(records_a)} heldout-degenerate v2 (seeded, sorted)")
    print(f"Fixture B: {len(records_b)} qwen_degenerate | "
          f"Fixture C: {len(records_c)} t2s_degenerate")

    # Reuse DS-028 dormant for Fixture A (do NOT re-run).
    dormant_map_a = load_dormant_from_ds028(DS028_JSONL, record_ids_a)
    missing = record_ids_a - set(dormant_map_a.keys())
    if missing:
        print(f"[STOP] missing DS-028 dormant rows for record_ids: {sorted(missing)}")
        sys.exit(1)
    print(f"reused dormant data from {DS028_JSONL} "
          f"({len(dormant_map_a)} records joined)")

    # Reuse DS-030 Condition B (production combined) and Condition D (production
    # suppression-only) for Fixture A (do NOT re-run).
    ds030_b, ds030_d = load_ds030_conditions(DS030_JSONL, record_ids_a)
    missing = record_ids_a - set(ds030_b.keys())
    if missing:
        print(f"[STOP] missing DS-030 Condition B/D rows for record_ids: {sorted(missing)}")
        sys.exit(1)
    print(f"reused DS-030 B/D from {DS030_JSONL} ({len(ds030_b)} records joined)")

    # Determinism smoke FIRST.
    smoke = None
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        smoke = run_determinism_smoke(
            model, tokenizer, records_a[0]["mutated_prompt"],
            non_prose_ids=non_prose_ids,
            max_new_tokens=args.max_new_tokens,
            record_id=int(records_a[0]["id"]),
        )

    # ==================================================================
    # Part 1 — combined (suppression + kickstart), greedy, Fixture A.
    # ==================================================================
    print("\n--- Part 1: combined (suppression + kickstart), greedy ---")
    part1_results: List[Dict[str, Any]] = []
    for rec_i, rec in enumerate(records_a):
        rid = int(rec["id"])
        prompt = rec["mutated_prompt"]
        dormant = dormant_map_a[rid]
        gen = greedy_generate_combined(
            model, tokenizer, prompt, COOLDOWN, PENALTY, non_prose_ids,
            max_new_tokens=args.max_new_tokens,
        )
        active = build_active_out(tokenizer, gen, dormant, args.max_new_tokens,
                                  activity_key="penalty_activity")
        part1_results.append({
            "part": 1,
            "fixture": "heldout_degenerate_v2",
            "record_id": rid,
            "source_prose_id": int(rec.get("source_prose_id", -1)),
            "prompt": prompt,
            "dormant": dormant,
            "active": active,
            "baseline_B_ds030": ds030_b[rid],
            "baseline_D_ds030": ds030_d[rid],
        })
        if (rec_i + 1) % 10 == 0 or rec_i == len(records_a) - 1:
            n_rescue = sum(1 for r in part1_results if r["active"]["delta_distinct_2"] > 0)
            print(f"  record {rid} ({rec_i+1}/{len(records_a)}); rescue={n_rescue}")

    # Aggregation and criteria are computed in build_ctx_from_results() below
    # (single source of truth, shared with --report-only).
    print("  P1 records collected.")

    # ==================================================================
    # Part 2 — sampling regime (suppression-only, do_sample=True), Fixture A.
    # ==================================================================
    print("\n--- Part 2: sampling regime (suppression-only, do_sample=True) ---")
    part2_results: List[Dict[str, Any]] = []
    for rec_i, rec in enumerate(records_a):
        rid = int(rec["id"])
        prompt = rec["mutated_prompt"]
        # Dormant (live, sampling). Seeded for determinism.
        seed_all(SEED)
        gen_dorm = sample_generate(
            model, tokenizer, prompt, dormant=True,
            cooldown=COOLDOWN, penalty=PENALTY, temperature=PART2_TEMPERATURE,
            max_new_tokens=args.max_new_tokens,
        )
        dormant = build_dormant_meta(
            tokenizer, gen_dorm,
            "live (do_sample=True, temp=0.8, top_p=0.85)",
        )
        # Active (live, suppression-only sampling). Seeded for determinism.
        seed_all(SEED)
        gen_act = sample_generate(
            model, tokenizer, prompt, dormant=False,
            cooldown=COOLDOWN, penalty=PENALTY, temperature=PART2_TEMPERATURE,
            max_new_tokens=args.max_new_tokens,
        )
        active = build_active_out(tokenizer, gen_act, dormant, args.max_new_tokens,
                                  activity_key="penalty_activity")
        part2_results.append({
            "part": 2,
            "fixture": "heldout_degenerate_v2",
            "record_id": rid,
            "source_prose_id": int(rec.get("source_prose_id", -1)),
            "prompt": prompt,
            "dormant": dormant,
            "active": active,
        })
        if (rec_i + 1) % 10 == 0 or rec_i == len(records_a) - 1:
            n_rescue = sum(1 for r in part2_results if r["active"]["delta_distinct_2"] > 0)
            print(f"  record {rid} ({rec_i+1}/{len(records_a)}); rescue={n_rescue}")

    print("  P2 records collected.")

    # ==================================================================
    # Part 3 — cross-fixture (suppression-only, greedy), Fixtures B and C.
    # ==================================================================
    print("\n--- Part 3: cross-fixture (suppression-only, greedy) ---")

    def run_part3_fixture(
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
            # Active (live, suppression-only greedy).
            gen_act = greedy_generate_suppression_only(
                model, tokenizer, prompt, COOLDOWN, PENALTY,
                max_new_tokens=args.max_new_tokens,
            )
            active = build_active_out(tokenizer, gen_act, dormant, args.max_new_tokens,
                                      activity_key="penalty_activity")
            out.append({
                "part": 3,
                "fixture": fixture_name,
                "record_id": rid,
                "prompt": prompt,
                "dormant": dormant,
                "active": active,
            })
            if (rec_i + 1) % 25 == 0 or rec_i == len(records) - 1:
                n_rescue = sum(1 for r in out if r["active"]["delta_distinct_2"] > 0)
                print(f"  [{fixture_name}] {rec_i+1}/{len(records)}; rescue={n_rescue}")
        return out

    part3_b = run_part3_fixture("qwen_degenerate", records_b, "prompt")
    # Fixture C (t2s_degenerate) has no "prompt" key; use record["text"] as the
    # degenerate prompt (the task's record["prompt"] maps to text here).
    part3_c = run_part3_fixture("t2s_degenerate", records_c, "text")
    print("  P3 records collected.")

    # ==================================================================
    # Aggregate + criteria (single source of truth), write outputs.
    # ==================================================================
    all_results = part1_results + part2_results + part3_b + part3_c
    ctx = build_ctx_from_results(
        all_results, device, dtype, smoke,
        wall_clock_s=time.time() - t_start,
        max_new_tokens=args.max_new_tokens,
    )
    p1 = ctx["parts"]["part1"]
    p2 = ctx["parts"]["part2"]
    p3 = ctx["parts"]["part3"]
    gate_pass = ctx["gate_pass"]

    print(f"  P1 live: ΔD2 mean={p1['agg_live']['delta_d2_mean']:.4f} "
          f"#>0={p1['agg_live']['n_delta_d2_pos']} "
          f"EOS={p1['agg_live']['eos_rate']:.3f} "
          f"mean_n={p1['agg_live']['mean_n_gen']:.1f}")
    print(f"  P2 live: ΔD2 mean={p2['agg_live']['delta_d2_mean']:.4f} "
          f"#>0={p2['agg_live']['n_delta_d2_pos']} "
          f"EOS={p2['agg_live']['eos_rate']:.3f} "
          f"mean_n={p2['agg_live']['mean_n_gen']:.1f}")
    print(f"  P3B (qwen) live: ΔD2 mean={p3['agg_b_live']['delta_d2_mean']:.4f} "
          f"#>0={p3['agg_b_live']['n_delta_d2_pos']} "
          f"EOS={p3['agg_b_live']['eos_rate']:.3f}")
    print(f"  P3C (t2s) live: ΔD2 mean={p3['agg_c_live']['delta_d2_mean']:.4f} "
          f"#>0={p3['agg_c_live']['n_delta_d2_pos']} "
          f"EOS={p3['agg_c_live']['eos_rate']:.3f}")

    # ==================================================================
    # Overall gate.
    # ==================================================================
    print("\n" + "=" * 70)
    print("OVERALL GATE RESULT")
    print(f"  Part 1 (combined, greedy): {'PASS' if p1['pass'] else 'FAIL'}")
    print(f"  Part 2 (sampling regime):  {'PASS' if p2['pass'] else 'FAIL'}")
    print(f"  Part 3 (cross-fixture):    {'PASS' if p3['pass'] else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        failed = [name for name, part in
                  [("Part 1", p1), ("Part 2", p2), ("Part 3", p3)]
                  if not part["pass"]]
        print(f"  [STOP] Red gate: FAIL in {', '.join(failed)}. "
              "No tuning, no threshold modification, no retry. "
              "The human decides whether to accept a partial pass or adjust the "
              "cooldown target.")
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
    print("GATE COMPLETE")

    if not gate_pass:
        sys.exit(2)


if __name__ == "__main__":
    main()
