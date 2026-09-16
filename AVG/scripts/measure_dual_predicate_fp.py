#!/usr/bin/env python3
"""DS-034c: Dual-predicate FP re-verification on prose-100 (GATE, v2).

RFC-004 Amendment A3 (ratified): dual-predicate OR rule — bigram OR spectral
PR at layer 2. Gate 2.3b (DS-027) confirmed 0 false positives for PR-alone on
prose-100, hazard corpora, and schema_corpus_v2 under teacher-forced replay.
This probe re-verifies the dual-predicate under LIVE generation with the
production cadence and hysteresis per RFC A3:

- Cadence: every-step (confirmed by DS-034a at Δ = 0.7027 ms/token).
- Persistence hysteresis: >=2 consecutive is_collapsed steps (applied to the
  OR output, per RFC A3 tightening).
- Code-context immunity: spectral_collapse forced False when
  is_code_syntax_context is True (RFC A3, Path-2 integration).
- FP hardening fallback (RFC A3, Options A+B): if simple OR produces FPs,
  tighten spectral fire to PR < band_low (8.22) for >=2 consecutive steps
  and re-test. Do NOT change frozen thresholds [1].

v2 changes from the failed DS-034c v1 run (turn-limit kill at 15/550):
  1. KV-cache generation: pre-fill the prompt once, then decode one token at a
     time with past_key_values. Drops compute from O(n^2) to O(n).
  2. Batch split: prose-100 (N=100) only. Hazard/schema corpora deferred to
     DS-034c-hazard.

Ground truth (per the DS-034c task file):
  - Corpus: prose-100 (heldout partition of valid_subset_200, N=100; ds-027
    complement split). Use record["text"] as generation prompt.
  - Dormant baselines: generated LIVE with plain greedy argmax (no hooks, no
    suppression, no kickstart, no top_p). KV-cache incremental generation,
    same as active run. Do NOT reuse DS-028 dormant baselines.
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. bmm override deregistered.
  - Frozen PR threshold: T_PR(2) = 10.954796, band_low = 8.216097
    (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze).
  - Previous partial data (15 records, 0 FPs) is non-mergeable and NOT reused.

Gate criteria:
  - Primary: 0 FPs on prose-100 under simple OR (PR < T, >=2 consecutive
    steps) -> PASS.
  - Fallback (if FPs > 0): re-test with spectral fire at band_low
    (PR < 8.22, >=2 consecutive steps). 0 FPs -> PASS with hardening applied.
    FPs persist -> FAIL (STOP signal; no threshold tuning).

Environment notes (identical to ds-025/ds-033/ds-034/ds-034a/ds-034b):
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
import statistics
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
SEED = 42  # DS-034c measurement seed (matches ds-025..ds-034b)
CONTROL_SEED = 42  # valid_subset_200 control-selection seed (ds-011/ds-020)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (no EOS break in prod; EOS break here)
ANALYSIS_WINDOW = 24  # rolling 24-token PR window (Gate 2.3 / ds-025 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
T_PR_2_TASK = 10.954796  # PR < T -> spectral fire
BAND_LOW_TASK = 8.216097  # fallback hardening: PR < band_low -> spectral fire

# Production controller settings (controller.py defaults, DS-033 task).
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

# Paths
VALID_SUBSET_200 = Path("data/t2s_bench/valid_subset_200.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/dual_predicate_fp_results.jsonl")
OUTPUT_MD = Path("docs/gate23/DUAL_PREDICATE_FP_RESULTS.md")

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
    rng = random.Random(CONTROL_SEED)
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
# Layer-2 PR measurement hook (identical to DS-034b / DS-034a instrument)
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
    numerical discipline). The PR value is logged; the hook returns ``output``
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
# KV-cache generation: dormant (plain greedy argmax, no hook/penalties)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_dormant_kv(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) dormant generation with KV-cache.

    Plain argmax. No hooks, no suppression, no kickstart, no top_p, no residual
    interventions. Pre-fill the prompt once (use_cache=True), then decode one
    token at a time with past_key_values. EOS break (EOS token is included in
    generated ids, matching the ds-029..ds-033 convention).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    # Pre-fill: one forward pass over the full prompt.
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :].clone().float()

    generated: List[torch.Tensor] = []
    for _step in range(max_new_tokens):
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if int(next_token.item()) == eos_id:
            break
        # Decoding loop: forward a single token with the KV cache.
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
# KV-cache generation: active (dual-predicate OR + production logit-penalty path)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_active_dual(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    non_prose_ids: Set[int],
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) active generation under the RFC A3 dual-predicate.

    KV-cache incremental generation (pre-fill once, then single-token forwards
    with past_key_values). The layer-2 forward hook captures hidden states at
    each decoding step (seq_len==1) into a rolling 24-token ring buffer and
    computes participation_ratio() from step 23 onward.

    Dual-predicate OR rule (RFC A3, exact):
        bigram_collapse   = token_diversity < 0.30 AND trailing_ctr < 0.30
        spectral_collapse = layer_2_pr < spectral_threshold
        if is_code_syntax_context(trailing_text): spectral_collapse = False
        if bigram_collapse:   bigram_ctr += 1  else: bigram_ctr = 0
        if spectral_collapse: spectral_ctr += 1 else: spectral_ctr = 0
        is_collapsed = (bigram_ctr >= 2) OR (spectral_ctr >= 2)

    Actuation on is_collapsed: production logit-penalty path (token suppression
    cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85). The actuation
    can fire multiple times per record. Every fire is logged with (step, which
    predicate(s) triggered, PR value).
    """
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
    # the DS-034b hook pattern.
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

            # ---- 4. Dual-predicate OR rule (RFC A3, exact) ----
            bigram_collapse = bool(
                token_diversity < COLLAPSE_DIVERSITY
                and trailing_ctr < COLLAPSE_CTR
            )
            spectral_collapse = bool(
                pr is not None and pr < spectral_threshold
            )
            if is_code_context:
                spectral_collapse = False
            if bigram_collapse:
                bigram_ctr += 1
            else:
                bigram_ctr = 0
            if spectral_collapse:
                spectral_ctr += 1
            else:
                spectral_ctr = 0
            is_collapsed = bool(bigram_ctr >= 2 or spectral_ctr >= 2)

            n_bigram_collapse_steps += int(bigram_collapse)
            n_spectral_collapse_steps += int(spectral_collapse)

            # ---- 5. Actuation on is_collapsed ----
            if is_collapsed:
                if active_loop_ids:
                    for tid in active_loop_ids:
                        active_suppress[tid] = COOLDOWN
                kickstart_counter = KICKSTART_COUNTER_INIT
                n_kickstart_events += 1
                triggers: List[str] = []
                if bigram_ctr >= 2:
                    triggers.append("bigram")
                if spectral_ctr >= 2:
                    triggers.append("spectral")
                fire_events.append({
                    "step": int(step),
                    "trigger": "+".join(triggers) if triggers else "none",
                    "bigram_collapse": bool(bigram_collapse),
                    "spectral_collapse": bool(spectral_collapse),
                    "bigram_ctr": int(bigram_ctr),
                    "spectral_ctr": int(spectral_ctr),
                    "pr": pr,
                    "token_diversity": float(token_diversity),
                    "trailing_ctr": float(trailing_ctr),
                    "is_code_context": bool(is_code_context),
                })

            # ---- 6. Production logit-penalty path ----
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

            kick_penalty_value = 0.0
            if kickstart_counter > 0:
                if non_prose_ids:
                    non_prose_tensor = torch.tensor(
                        list(non_prose_ids), device=logits.device
                    )
                    kick_penalty_value = KICKSTART_PENALTIES.get(
                        kickstart_counter, -2.0
                    )
                    logits[:, non_prose_tensor] += kick_penalty_value
                kickstart_counter -= 1

            if active_suppress or kickstart_counter > 0:
                logits = sample_top_p(logits, top_p=TOP_P)

            # ---- 7. Greedy argmax (do_sample=False) ----
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            # ---- 8. Decoding loop: forward a single token with the KV cache ----
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].clone().float()

            # ---- 9. Append to input_ids for the next step's bigram ----
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
        "n_suppression_steps": n_suppression_steps,
        "n_kickstart_events": n_kickstart_events,
        "n_bigram_collapse_steps": n_bigram_collapse_steps,
        "n_spectral_collapse_steps": n_spectral_collapse_steps,
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


def build_dormant_meta(tokenizer: Any, gen: Dict[str, Any]) -> Dict[str, Any]:
    generated_ids = gen["generated_ids"]
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return {
        "n_generated": int(generated_ids.shape[-1]),
        "distinct_2": continuation_distinct2(generated_ids),
        "ctr": continuation_ctr(tokenizer, generated_ids),
        "generated_text": text,
        "generated_ids": generated_ids[0].tolist(),
        "eos_terminated": bool(
            int(generated_ids.shape[-1]) < MAX_NEW_TOKENS
            and int(generated_ids.shape[-1]) > 0
            and int(generated_ids[0, -1].item()) == tokenizer.eos_token_id
        ),
        "source": "live (greedy, KV-cache)",
    }


def build_active_meta(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
    spectral_threshold: float,
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
    out: Dict[str, Any] = {
        "spectral_threshold": float(spectral_threshold),
        "n_generated": n,
        "distinct_2": d2,
        "ctr": ctr,
        "generated_text": text,
        "generated_ids": generated_ids[0].tolist(),
        "eos_terminated": eos_terminated,
        "delta_distinct_2": float(d2 - dormant["distinct_2"]),
        "delta_ctr": float(ctr - dormant["ctr"]),
        "byte_identical_to_dormant": bool(
            text.encode("utf-8") == dormant["generated_text"].encode("utf-8")
        ),
        "token_identical_to_dormant": bool(
            generated_ids[0].tolist() == dormant["generated_ids"]
        ),
        "fire_count": gen["fire_count"],
        "fire_events": gen["fire_events"],
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
    }
    # FP criterion (task file): ΔDistinct-2 > 0 vs dormant AND the predicate
    # fired at least once (the dual-predicate triggered actuation that changed
    # the continuation on legitimate prose).
    out["is_false_positive"] = bool(
        out["delta_distinct_2"] > 0 and gen["fire_count"] > 0
    )
    return out


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    non_prose_ids: Set[int],
    spectral_threshold: float,
) -> Dict[str, Any]:
    """Determinism smoke (DS-034c): 2 prose-100 records.
      1. Dormant, generated twice — token ids must match.
      2. Active (dual-predicate, primary threshold), generated twice — token ids
         AND per-step PR log must match.
    STOP (sys.exit 1) if either fails.
    """
    print("\n--- Determinism smoke (2 prose-100 records) ---")
    pairs: List[Dict[str, Any]] = []
    all_identical = True
    for rec in records[:2]:
        rid = int(rec["id"])
        prompt = rec["text"]

        # Dormant twice.
        da = greedy_generate_dormant_kv(model, tokenizer, prompt)
        db = greedy_generate_dormant_kv(model, tokenizer, prompt)
        dorm_identical = bool(
            da["generated_ids"].shape == db["generated_ids"].shape
            and torch.equal(da["generated_ids"], db["generated_ids"])
        )
        print(f"  dormant  corpus_id={rid}: n_tokens={da['n_generated']} / "
              f"{db['n_generated']} identical={dorm_identical}")

        # Active twice.
        aa = greedy_generate_active_dual(
            model, tokenizer, prompt, non_prose_ids, spectral_threshold
        )
        ab = greedy_generate_active_dual(
            model, tokenizer, prompt, non_prose_ids, spectral_threshold
        )
        act_identical = bool(
            aa["generated_ids"].shape == ab["generated_ids"].shape
            and torch.equal(aa["generated_ids"], ab["generated_ids"])
        )
        pr_a = [p["pr"] for p in aa["pr_log"]]
        pr_b = [p["pr"] for p in ab["pr_log"]]
        pr_identical = bool(pr_a == pr_b)
        print(f"  active   corpus_id={rid}: n_tokens={aa['n_generated']} / "
              f"{ab['n_generated']} identical={act_identical} "
              f"n_pr={len(pr_a)} pr_identical={pr_identical}")

        identical = bool(dorm_identical and act_identical and pr_identical)
        all_identical = all_identical and identical
        pairs.append({
            "corpus_id": rid,
            "prompt_len": int(aa["prompt_len"]),
            "dormant_identical": str(dorm_identical),
            "active_identical": str(act_identical),
            "pr_identical": str(pr_identical),
            "identical": str(identical),
            "n_pr_values": int(len(pr_a)),
            "n_generated_active": int(aa["n_generated"]),
            "pr_a_first": pr_a[0] if pr_a else None,
            "pr_b_first": pr_b[0] if pr_b else None,
            "pr_a_last": pr_a[-1] if pr_a else None,
            "pr_b_last": pr_b[-1] if pr_b else None,
        })

    if not all_identical:
        print("[STOP] Determinism smoke FAILED: token ids or PR log differ "
              "across runs.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (dormant tokens, active tokens, "
          "PR log)")
    return {"pairs": pairs, "all_identical": "True"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
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


def aggregate_mode(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-record results for a given mode (primary/fallback)."""
    fps = [r for r in results if r["is_false_positive"]]
    fire_any = [r for r in results if r["active"]["fire_count"] > 0]
    d2_deltas = [r["active"]["delta_distinct_2"] for r in results]
    d2_deltas_firing = [
        r["active"]["delta_distinct_2"] for r in results
        if r["active"]["fire_count"] > 0
    ]
    fire_counts = [r["active"]["fire_count"] for r in results]
    pr_means = [
        r["active"]["pr_mean"] for r in results
        if r["active"]["pr_mean"] is not None
    ]
    return {
        "n_records": len(results),
        "n_false_positives": len(fps),
        "fp_rate": float(len(fps)) / len(results) if results else float("nan"),
        "fp_record_ids": [int(r["record_id"]) for r in fps],
        "n_fired": len(fire_any),
        "n_fire_steps_total": sum(fire_counts),
        "fire_count": summarize([float(x) for x in fire_counts]),
        "delta_distinct_2": summarize(d2_deltas),
        "delta_distinct_2_firing": summarize(d2_deltas_firing)
        if d2_deltas_firing else None,
        "pr_mean": summarize(pr_means) if pr_means else None,
        "n_byte_identical": int(sum(
            1 for r in results
            if r["active"]["byte_identical_to_dormant"]
        )),
        "n_eos_terminated_active": int(sum(
            1 for r in results if r["active"]["eos_terminated"]
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


def write_markdown_report(
    ctx: Dict[str, Any],
    out_path: Path,
) -> None:
    md: List[str] = []
    md.append("# DS-034c — Dual-predicate FP re-verification on prose-100 (GATE)")
    md.append("")
    md.append("> GATE REPORT (RFC-004 Amendment A3). Dual-predicate OR rule — "
              "bigram OR spectral PR at layer 2 — re-verified under LIVE "
              "generation with the production cadence and hysteresis per RFC A3. "
              "A FAIL is a STOP signal — no negotiation, no threshold tuning.")
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
    md.append("| Corpus | prose-100 (heldout partition of valid_subset_200, "
              "N=100; ds-027 complement split) |")
    md.append("| Prompt | record[\\\"text\\\"] |")
    md.append("| Decoding | greedy (do_sample=False), KV-cache incremental, "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} hidden "
              f"states (layer-2 hook, rolling ring buffer) |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] |")
    md.append(f"| Frozen T_PR(2) | {meta['t_pr']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |")
    md.append(f"| Fallback band_low(2) | {meta['band_low']:.6f} "
              "(RFC A3 Options A+B hardening) |")
    md.append("| PR instrument | participation_ratio() from core/metrics.py "
              "(existing), float32 promotion |")
    md.append("| Cadence | every-step (confirmed by DS-034a at "
              "Δ = 0.7027 ms/token) |")
    md.append("| Persistence hysteresis | ≥2 consecutive is_collapsed steps "
              "(RFC A3 tightening, applied to the OR output) |")
    md.append("| Actuation | production logit-penalty path: token suppression "
              "cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85 |")
    md.append("| Dormant | LIVE (greedy, KV-cache, no hooks/penalties) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Gate criteria
    md.append("## Gate criteria (quoted from the DS-034c task file)")
    md.append("")
    md.append("> - Primary: 0 FPs on prose-100 under simple OR (PR < T, ≥2 "
              "consecutive steps) → PASS.")
    md.append("> - Fallback (if FPs > 0): re-test with spectral fire at band_low "
              "(PR < 8.22, ≥2 consecutive steps) on the same corpus. 0 FPs → "
              "PASS with hardening applied. FPs persist → FAIL.")
    md.append("> - A FAIL is a STOP signal — do NOT negotiate, do NOT tune "
              "thresholds [1].")
    md.append("")
    md.append("FP criterion: a record is a false positive if ΔDistinct-2 > 0 "
              "vs dormant AND the predicate fired at least once (the "
              "dual-predicate triggered actuation that changed the continuation "
              "on legitimate prose).")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("Two prose-100 records: dormant generated twice AND active "
              "(dual-predicate, primary threshold) generated twice. Token ids "
              "AND the per-step PR log must match exactly. STOP if not.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = []
    for pair in sm["pairs"]:
        smoke_rows.append([
            str(pair["corpus_id"]), str(pair["prompt_len"]),
            str(pair["n_generated_active"]), str(pair["n_pr_values"]),
            pair["dormant_identical"], pair["active_identical"],
            pair["pr_identical"], pair["identical"],
        ])
    smoke_rows.append(["", "ALL", "", "", "", "", "", str(sm["all_identical"])])
    md.append(format_table_md(
        smoke_rows, ["corpus_id", "prompt_len", "n_gen", "n PR",
                     "dormant id", "active id", "PR log id", "identical"],
    ))
    md.append("")
    for pair in sm["pairs"]:
        md.append(f"PR first run A/B: {fmt(pair['pr_a_first'])} / "
                  f"{fmt(pair['pr_b_first'])}; last run A/B: "
                  f"{fmt(pair['pr_a_last'])} / {fmt(pair['pr_b_last'])}.")
        md.append("")

    tabs = ctx["tables"]

    # Verdict
    md.append("## Verdict")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| Records scored | {tabs['primary']['n_records']} |")
    md.append(f"| Primary FPs (simple OR, PR < T) | {tabs['primary']['n_false_positives']} |")
    md.append(f"| Primary fired records | {tabs['primary']['n_fired']} |")
    if tabs.get("fallback") is not None:
        md.append(f"| Fallback FPs (band_low hardening) | "
                  f"{tabs['fallback']['n_false_positives']} |")
        md.append(f"| Fallback fired records | {tabs['fallback']['n_fired']} |")
    md.append(f"| **Gate verdict** | **{tabs['verdict']}** |")
    md.append("")
    if tabs["verdict"] == "PASS":
        md.append("**PASS** — 0 false positives on prose-100 under the simple OR "
                  "(PR < T_PR(2), ≥2 consecutive steps).")
    elif tabs["verdict"] == "PASS (hardening applied)":
        md.append("**PASS with hardening applied** — primary OR produced FPs; "
                  "the fallback spectral fire at band_low (PR < 8.22, ≥2 "
                  "consecutive steps) produced 0 FPs. Per RFC A3 Options A+B, "
                  "the spectral fire threshold is tightened to band_low.")
    else:
        md.append("**FAIL** — FPs persist even after the band_low hardening. "
                  "STOP signal; no thresholds are negotiated or tuned.")
    md.append("")

    # Primary summary
    md.append("## Primary mode summary (simple OR, PR < T)")
    md.append("")
    p = tabs["primary"]
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| Records | {p['n_records']} |")
    md.append(f"| False positives | {p['n_false_positives']} |")
    md.append(f"| Records with ≥1 fire | {p['n_fired']} |")
    md.append(f"| Total fire steps | {p['n_fire_steps_total']} |")
    md.append(f"| Byte-identical to dormant | {p['n_byte_identical']} |")
    md.append(f"| EOS-terminated (active) | {p['n_eos_terminated_active']} |")
    d2 = p["delta_distinct_2"]
    md.append(f"| ΔDistinct-2 mean / median / p90 | {fmt(d2['mean'])} / "
              f"{fmt(d2['median'])} / {fmt(d2['p90'])} |")
    if p["delta_distinct_2_firing"] is not None:
        d2f = p["delta_distinct_2_firing"]
        md.append(f"| ΔDistinct-2 (firing only) mean / median / p90 | "
                  f"{fmt(d2f['mean'])} / {fmt(d2f['median'])} / "
                  f"{fmt(d2f['p90'])} |")
    fc = p["fire_count"]
    md.append(f"| Fire count per record mean / max | {fmt(fc['mean'])} / "
              f"{fmt(fc['max'])} |")
    if p["pr_mean"] is not None:
        prm = p["pr_mean"]
        md.append(f"| PR (per-record mean) mean / min / max | {fmt(prm['mean'])} "
                  f"/ {fmt(prm['min'])} / {fmt(prm['max'])} |")
    md.append("")
    if p["fp_record_ids"]:
        md.append(f"Primary FP record_ids: `{p['fp_record_ids']}`")
        md.append("")

    # Fallback summary (if run)
    if tabs.get("fallback") is not None:
        md.append("## Fallback mode summary (band_low hardening, PR < 8.22)")
        md.append("")
        f = tabs["fallback"]
        md.append("| metric | value |")
        md.append("|---|---|")
        md.append(f"| Records | {f['n_records']} |")
        md.append(f"| False positives | {f['n_false_positives']} |")
        md.append(f"| Records with ≥1 fire | {f['n_fired']} |")
        md.append(f"| Total fire steps | {f['n_fire_steps_total']} |")
        md.append(f"| Byte-identical to dormant | {f['n_byte_identical']} |")
        d2f = f["delta_distinct_2"]
        md.append(f"| ΔDistinct-2 mean / median / p90 | {fmt(d2f['mean'])} / "
                  f"{fmt(d2f['median'])} / {fmt(d2f['p90'])} |")
        if f["delta_distinct_2_firing"] is not None:
            d2ff = f["delta_distinct_2_firing"]
            md.append(f"| ΔDistinct-2 (firing only) mean / median / p90 | "
                      f"{fmt(d2ff['mean'])} / {fmt(d2ff['median'])} / "
                      f"{fmt(d2ff['p90'])} |")
        fc = f["fire_count"]
        md.append(f"| Fire count per record mean / max | {fmt(fc['mean'])} / "
                  f"{fmt(fc['max'])} |")
        md.append("")
        if f["fp_record_ids"]:
            md.append(f"Fallback FP record_ids: `{f['fp_record_ids']}`")
            md.append("")

    # Per-record table (final mode)
    final_mode = "fallback" if tabs.get("fallback") is not None else "primary"
    final_results = ctx["results_final"]
    md.append(f"## Per-record table (final mode: {final_mode})")
    md.append("")
    md.append("| record_id | corpus_id | prompt_len | n_gen | ΔD2 | byte-id | "
              "fires | bigram_steps | spectral_steps | PR mean | PR min | FP |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in final_results:
        a = r["active"]
        md.append(f"| {r['record_id']} | {r['corpus_id']} | {r['prompt_len']} | "
                  f"{a['n_generated']} | {fmt(a['delta_distinct_2'])} | "
                  f"{'Y' if a['byte_identical_to_dormant'] else 'N'} | "
                  f"{a['fire_count']} | {a['n_bigram_collapse_steps']} | "
                  f"{a['n_spectral_collapse_steps']} | "
                  f"{fmt(a['pr_mean'])} | {fmt(a['pr_min'])} | "
                  f"{'FP' if a['is_false_positive'] else '—'} |")
    md.append("")

    # Fire event detail (final mode, firing records only)
    firing = [r for r in final_results if r["active"]["fire_count"] > 0]
    if firing:
        md.append(f"## Fire events ({len(firing)} firing records, final mode: "
                  f"{final_mode})")
        md.append("")
        for r in firing:
            md.append(f"### record {r['record_id']} (corpus_id {r['corpus_id']})")
            md.append("")
            md.append("| step | trigger | bigram_ctr | spectral_ctr | PR | "
                      "token_diversity | trailing_ctr | is_code |")
            md.append("|---|---|---|---|---|---|---|---|")
            for ev in r["active"]["fire_events"]:
                md.append(f"| {ev['step']} | {ev['trigger']} | {ev['bigram_ctr']} "
                          f"| {ev['spectral_ctr']} | {fmt(ev['pr'])} | "
                          f"{fmt(ev['token_diversity'])} | "
                          f"{fmt(ev['trailing_ctr'])} | "
                          f"{'Y' if ev['is_code_context'] else 'N'} |")
            md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- GATE: the pass/fail criterion is explicit and quoted from the "
              "task file. A FAIL is a STOP signal — the script exits non-zero; "
              "no thresholds are negotiated or tuned.")
    md.append("- Live generation uses KV-cache incremental decoding (pre-fill "
              "once, then single-token forwards with past_key_values), dropping "
              "compute from O(n^2) to O(n) vs the v1 full-sequence forward "
              "passes. The layer-2 spectral hook captures hidden states at each "
              "decoding step (seq_len==1) into a rolling 24-token ring buffer, "
              "identical to the DS-034b hook pattern.")
    md.append("- The dual-predicate OR rule is implemented exactly per RFC A3: "
              "bigram OR spectral, each with its own ≥2-consecutive persistence "
              "counter, code-context immunity forcing spectral_collapse=False "
              "when is_code_syntax_context is True.")
    md.append("- Actuation on is_collapsed uses the production logit-penalty "
              "path (token suppression cooldown=8 penalty=-5.0, kickstart "
              "counter=3 with -1e4/-5.0/-2.0, top_p=0.85). The actuation can "
              "fire multiple times per record; every fire is logged.")
    md.append("- Dormant baselines are generated LIVE (greedy, KV-cache, no "
              "hooks/penalties) for all 100 records. No JSONL reuse from the "
              "failed v1 run.")
    md.append("- FP = ΔDistinct-2 > 0 AND fire_count > 0. A record whose "
              "predicate fires but whose continuation is byte-identical to "
              "dormant is an actuation-gap, not an FP.")
    md.append("- Frozen thresholds (T_PR(2) = 10.954796, band_low = 8.216097) "
              "are read from docs/gate23/FROZEN_THRESHOLDS.md and verified "
              "against the task-specified values; the script exits if they "
              "disagree.")
    md.append("- The gate verdict is scoped to prose-100 only. Hazard/schema "
              "corpora are deferred to DS-034c-hazard.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-034c dual-predicate FP re-verification on prose-100 "
                    "(GATE, RFC A3)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-034c Dual-predicate FP re-verification on prose-100 (GATE)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: {ANALYSIS_WINDOW} "
          f"| hook_layer: {HOOK_LAYER}")
    print(f"gate: 0 FPs on prose-100 -> PASS; fallback band_low -> PASS with "
          f"hardening; else FAIL")

    # ------------------------------------------------------------------
    # Frozen thresholds FIRST (Part A freeze); verify task-specified values.
    # ------------------------------------------------------------------
    t_pr, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    print(f"frozen T_PR(2)={t_pr:.6f} band_low(2)={band_low:.6f} loaded from "
          f"docs/gate23/FROZEN_THRESHOLDS.md")

    # ------------------------------------------------------------------
    # Corpus: prose-100 (heldout partition of valid_subset_200).
    # ------------------------------------------------------------------
    valid_records = load_jsonl(VALID_SUBSET_200)
    heldout = select_heldout_prose(valid_records)
    if args.max_records is not None:
        heldout = heldout[: args.max_records]
        print(f"[dev] capped heldout records at {args.max_records}")
    print(f"valid_subset_200: {len(valid_records)} | prose-100 heldout: "
          f"{len(heldout)}")
    print(f"heldout corpus_ids: {[int(r['id']) for r in heldout]}")

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

    # Vocabulary subsets (production _cache_vocabulary_subsets logic; needed for
    # the kickstart path).
    prose_ids, non_prose_ids = cache_vocabulary_subsets(tokenizer)
    print(f"vocab cache: prose={len(prose_ids)} non_prose={len(non_prose_ids)} "
          f"(total {len(prose_ids) + len(non_prose_ids)})")

    # Determinism smoke FIRST (also serves as GPU warmup).
    smoke: Dict[str, Any] = {"pairs": [], "all_identical": "True"}
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        try:
            smoke = run_determinism_smoke(
                model, tokenizer, heldout, non_prose_ids, t_pr,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during determinism smoke: {e}")
                sys.exit(1)
            raise

    # ==================================================================
    # Phase 1: dormant + active (primary, PR < T) for all records.
    # ==================================================================
    print(f"\n--- Primary run (dormant + active, spectral_threshold=T={t_pr:.6f}, "
          f"{len(heldout)} records) ---")
    results_primary: List[Dict[str, Any]] = []
    try:
        for i, rec in enumerate(heldout):
            rid = int(rec["id"])
            prompt = rec["text"]

            # Dormant (live, greedy, KV-cache).
            gen_dorm = greedy_generate_dormant_kv(model, tokenizer, prompt)
            dormant = build_dormant_meta(tokenizer, gen_dorm)

            # Active (live, dual-predicate OR + production logit-penalty path).
            gen_act = greedy_generate_active_dual(
                model, tokenizer, prompt, non_prose_ids, t_pr,
            )
            active = build_active_meta(tokenizer, gen_act, dormant, t_pr)

            results_primary.append({
                "fixture": "prose-100",
                "record_id": i,
                "corpus_id": rid,
                "seed": SEED,
                "prompt_len": int(gen_dorm["prompt_len"]),
                "mode": "primary",
                "spectral_threshold": float(t_pr),
                "dormant": dormant,
                "active": active,
                "is_false_positive": bool(active["is_false_positive"]),
            })

            if (i + 1) % 10 == 0 or i == len(heldout) - 1:
                n_fp = sum(1 for r in results_primary if r["is_false_positive"])
                n_fire = sum(
                    1 for r in results_primary
                    if r["active"]["fire_count"] > 0
                )
                print(f"  [{i+1}/{len(heldout)}] rid={rid} "
                      f"prompt_len={gen_dorm['prompt_len']} "
                      f"n_gen={gen_act['n_generated']} "
                      f"fire={gen_act['fire_count']} "
                      f"ΔD2={active['delta_distinct_2']:.4f} "
                      f"byte_id={active['byte_identical_to_dormant']} "
                      f"fp_so_far={n_fp} fired_so_far={n_fire}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during primary run: {e}")
            sys.exit(1)
        raise

    primary_agg = aggregate_mode(results_primary)
    print("\n" + "=" * 70)
    print("PRIMARY GATE (simple OR, PR < T)")
    print(f"  records: {primary_agg['n_records']}")
    print(f"  false positives: {primary_agg['n_false_positives']} "
          f"({primary_agg['fp_rate']:.4f})")
    print(f"  fired records: {primary_agg['n_fired']} | total fire steps: "
          f"{primary_agg['n_fire_steps_total']}")
    print(f"  ΔD2 mean: {primary_agg['delta_distinct_2']['mean']:.4f} | "
          f"byte-identical: {primary_agg['n_byte_identical']}")
    print("=" * 70)

    # ==================================================================
    # Fallback (if primary has FPs): active with spectral_threshold=band_low.
    # ==================================================================
    results_final = results_primary
    fallback_agg: Optional[Dict[str, Any]] = None
    verdict = "PASS"

    if primary_agg["n_false_positives"] > 0:
        print(f"\n--- Primary produced FPs. Running fallback with "
              f"spectral_threshold=band_low={band_low:.6f} ---")
        results_fallback: List[Dict[str, Any]] = []
        try:
            for i, rec in enumerate(heldout):
                rid = int(rec["id"])
                prompt = rec["text"]

                # Dormant is the same; reuse the primary dormant meta.
                dormant = results_primary[i]["dormant"]

                gen_act = greedy_generate_active_dual(
                    model, tokenizer, prompt, non_prose_ids, band_low,
                )
                active = build_active_meta(tokenizer, gen_act, dormant, band_low)

                results_fallback.append({
                    "fixture": "prose-100",
                    "record_id": i,
                    "corpus_id": rid,
                    "seed": SEED,
                    "prompt_len": int(gen_act["prompt_len"]),
                    "mode": "fallback_hardened",
                    "spectral_threshold": float(band_low),
                    "dormant": dormant,
                    "active": active,
                    "is_false_positive": bool(active["is_false_positive"]),
                })

                if (i + 1) % 10 == 0 or i == len(heldout) - 1:
                    n_fp = sum(
                        1 for r in results_fallback if r["is_false_positive"]
                    )
                    n_fire = sum(
                        1 for r in results_fallback
                        if r["active"]["fire_count"] > 0
                    )
                    print(f"  [fallback {i+1}/{len(heldout)}] rid={rid} "
                          f"n_gen={gen_act['n_generated']} "
                          f"fire={gen_act['fire_count']} "
                          f"ΔD2={active['delta_distinct_2']:.4f} "
                          f"fp_so_far={n_fp} fired_so_far={n_fire}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during fallback run: {e}")
                sys.exit(1)
            raise

        fallback_agg = aggregate_mode(results_fallback)
        print("\n" + "=" * 70)
        print("FALLBACK GATE (band_low hardening, PR < 8.22)")
        print(f"  records: {fallback_agg['n_records']}")
        print(f"  false positives: {fallback_agg['n_false_positives']} "
              f"({fallback_agg['fp_rate']:.4f})")
        print(f"  fired records: {fallback_agg['n_fired']} | total fire steps: "
              f"{fallback_agg['n_fire_steps_total']}")
        print("=" * 70)

        results_final = results_fallback
        if fallback_agg["n_false_positives"] == 0:
            verdict = "PASS (hardening applied)"
        else:
            verdict = "FAIL"

    # ==================================================================
    # Aggregate (single source of truth), write outputs.
    # ==================================================================
    tables = {
        "primary": primary_agg,
        "fallback": fallback_agg,
        "verdict": verdict,
    }

    print("\n" + "=" * 70)
    print("DUAL-PREDICATE FP GATE VERDICT")
    print(f"  primary FPs: {primary_agg['n_false_positives']}")
    if fallback_agg is not None:
        print(f"  fallback FPs: {fallback_agg['n_false_positives']}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    # Write JSONL: all primary lines, then fallback lines if run.
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results_primary:
            f.write(json.dumps(r) + "\n")
        if fallback_agg is not None:
            for r in results_fallback:
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
            "t_pr": t_pr,
            "band_low": band_low,
            "verdict": verdict,
            "primary_fps": primary_agg["n_false_positives"],
            "fallback_fps": fallback_agg["n_false_positives"]
            if fallback_agg is not None else None,
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} "
          f"({len(results_primary) + (len(results_fallback) if fallback_agg is not None else 0)} "
          f"record lines + sidecar lines)")

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
        "t_pr": t_pr,
        "band_low": band_low,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "results_final": results_final,
    }
    write_markdown_report(ctx, OUTPUT_MD)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    if verdict == "FAIL":
        print("[STOP] DS-034c GATE: FAIL — FPs persist after band_low "
              "hardening. Do NOT wire the dual-predicate into the controller.")
        sys.exit(1)
    print(f"DS-034c GATE: {verdict} — measurement complete")


if __name__ == "__main__":
    main()
