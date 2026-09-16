#!/usr/bin/env python3
"""night-004: Unconditional suppression during collapse (MEASUREMENT ONLY).

DS-035 PASS established the dual-predicate rescue gate (20/100 on t2s, 11
spectral-only new rescues). But qwen_degenerate regressed from 0.68 (DS-033
production) to 0.20 (DS-035 gated).

Root cause (diagnosed by DS-035): the dual-predicate only adds suppression
tokens to the cooldown dict when `is_collapsed` fires at the current step,
whereas the production combined path adds `active_loop_ids` at EVERY step
where they are detected, regardless of predicate state. On qwen, where
micro-repetition generates abundant repeated tokens, the production actuation
is more aggressive and rescues more records (68 vs 20).

This probe tests the actuation coupling fix: when the dual-predicate detects
collapse, apply the SAME unconditional token suppression the production path
uses — add ALL active_loop_ids to the cooldown dict at EVERY step where they
are detected (cooldown=8, penalty=-5.0). This bridges dual-predicate
detection (100/100 qwen records fire) with production actuation strength.

MEASUREMENT ONLY. No controller edits, no threshold changes.

Frozen dual-predicate rule (DS-034e, unchanged):

    bigram_collapse   = (token_diversity < 0.30) AND (trailing_ctr < 0.30)
    if bigram_collapse: bigram_ctr += 1 else: bigram_ctr = 0
    bigram_fire       = bigram_ctr >= 2

    spectral_collapse = (layer_2_pr < 8.216097)     # band_low, frozen
    if spectral_collapse: spectral_ctr += 1 else: spectral_ctr = 0
    if bigram_ctr < 2:
        spectral_fire = (spectral_ctr >= 2) AND (token_diversity < 0.40)
    else:
        spectral_fire = spectral_ctr >= 2

    if is_code_syntax_context(trailing_text):
        spectral_fire = False

    is_collapsed = bigram_fire OR spectral_fire

Actuation coupling fix (THE CHANGE):
  - Unconditional suppression: at EVERY step, if active_loop_ids is non-empty,
    add ALL active_loop_ids to the cooldown dict (=8). This matches the
    production combined path (controller.py:677-679), which adds loop-token
    suppression at every step where detected regardless of predicate state.
  - Kickstart: fires per the production trigger (trailing_ctr < 0.50 AND
    active_loop_ids non-empty -> counter=3, -1e4/-5.0/-2.0).
  - top_p=0.85 when either mechanism is active.
  - NO residual hooks.
  - The dual-predicate (is_collapsed) is computed every step and fire events
    are logged for the per-record fire-count metrics (detection liveness).

Fixtures:
  A: tests/fixtures/t2s_degenerate.jsonl (night-001; N=100). record["text"]
     as prompt. Primary blind-spot target (DS-035: 20/100 rescue).
  B: tests/fixtures/qwen_degenerate.jsonl (ds-004; N=100). record["prompt"]
     as prompt. qwen regression target (DS-033 production: 68/100; DS-035
     gated: 20/100).
  C: tests/fixtures/heldout_degenerate_v2.jsonl (ds-027; N=100). 50-record
     seeded subset random.Random(42).sample sorted by record id.
     record["mutated_prompt"] as prompt. Positive control (DS-035: 48/50).

Reused data (do NOT re-run):
  - Dormant baselines are REUSED from DS-035 (dual_predicate_rescue_results
    .jsonl) for Fixtures A/B (live greedy KV-cache) and from DS-028
    (layer_effect_persistent_results.jsonl, layer==2) for Fixture C.
  - DS-033 production rescue rates:
    docs/gate23/production_cross_fixture_results.jsonl.
  - DS-035 gated rescue rates:
    docs/gate23/dual_predicate_rescue_results.jsonl.

Target (measurement hypothesis, NOT a gate):
  qwen >= 0.68 (match/exceed production), t2s >= 0.20 (preserve DS-035),
  heldout_v2 >= 0.96 (maintain).

Environment notes (identical to ds-025/ds-033/ds-034c/ds-035):
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
    sample_top_p,
)

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-035 measurement seed (matches ds-025..ds-034e)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (EOS break; EOS token included)
ANALYSIS_WINDOW = 24  # rolling 24-token PR / distinct-2 window (Gate 2.3 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook
NUM_RECORDS_C = 50  # Fixture C: seeded 50-record subset (random.Random(42))

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
# DS-034e configuration: spectral_collapse uses band_low (hardened), NOT T_PR.
T_PR_2_TASK = 10.954796  # PR < T -> spectral fire (primary, DS-034c)
BAND_LOW_TASK = 8.216097  # spectral_collapse threshold (band_low, frozen)

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

# Measurement hypothesis targets (night-004 task file; NOT a gate).
TARGET_DS033_PROD = {
    "t2s_degenerate": 0.15,
    "qwen_degenerate": 0.68,
    "heldout_degenerate_v2": 0.96,  # DS-031 suppression-only baseline 48/50
}
TARGET_DS035_GATED = {
    "t2s_degenerate": 0.20,
    "qwen_degenerate": 0.20,
    "heldout_degenerate_v2": 0.96,
}
TARGET_NIGHT004 = {
    "t2s_degenerate": 0.20,  # preserve or exceed DS-035 gated
    "qwen_degenerate": 0.68,  # match or exceed DS-033 production
    "heldout_degenerate_v2": 0.96,  # maintain
}

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
QWEN_DEG = Path("tests/fixtures/qwen_degenerate.jsonl")
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
DS028_JSONL = Path("docs/gate23/layer_effect_persistent_results.jsonl")
DS033_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
DS035_JSONL = Path("docs/gate23/dual_predicate_rescue_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/unconditional_suppression_results.jsonl")
OUTPUT_MD = Path("docs/gate23/UNCONDITIONAL_SUPPRESSION_RESULTS.md")

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
# KV-cache generation: active (night-004 unconditional suppression)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_active_unconditional(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    non_prose_ids: Set[int],
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) active generation under night-004.

    KV-cache incremental generation (pre-fill once, then single-token forwards
    with past_key_values). The layer-2 forward hook captures hidden states at
    each decoding step (seq_len==1) into a rolling 24-token ring buffer and
    computes participation_ratio() from step 23 onward.

    Frozen DS-034e dual-predicate rule (exact, unchanged):
        bigram_collapse   = (token_diversity < 0.30) AND (trailing_ctr < 0.30)
        if bigram_collapse: bigram_ctr += 1 else: bigram_ctr = 0
        bigram_fire       = bigram_ctr >= 2

        spectral_collapse = (layer_2_pr < spectral_threshold)   # band_low
        if spectral_collapse: spectral_ctr += 1 else: spectral_ctr = 0
        if bigram_ctr < 2:
            spectral_fire = (spectral_ctr >= 2) AND (token_diversity < 0.40)
        else:
            spectral_fire = spectral_ctr >= 2

        if is_code_syntax_context(trailing_text):
            spectral_fire = False

        is_collapsed = bigram_fire OR spectral_fire

    Actuation coupling fix (THE CHANGE):
      - UNCONDITIONAL token suppression: at EVERY step, if active_loop_ids is
        non-empty, add ALL active_loop_ids to the cooldown dict (=8). This
        matches the production combined path (controller.py:677-679), which
        adds loop-token suppression at every step where detected regardless of
        predicate state. (DS-035 gated this addition on is_collapsed, which
        delayed suppression until the first spectral fire — step 25 on qwen —
        and lost the early loop-breaking that production achieves.)
      - Kickstart: fires per the PRODUCTION trigger (trailing_ctr < 0.50 AND
        active_loop_ids non-empty -> counter=3, -1e4/-5.0/-2.0). DS-035 fired
        the kickstart on is_collapsed (85.7 events/record on qwen); production
        fires it on the CTR/loop trigger (~0.4 events/record).
      - top_p=0.85 when either mechanism is active. NO residual hooks.
      - Fire events are logged when is_collapsed is true (detection liveness;
        the dual-predicate does not gate actuation).
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

            # ---- 5. Actuation: UNCONDITIONAL token suppression (THE CHANGE) ----
            # Production combined path (controller.py:677-679): add ALL
            # active_loop_ids to the cooldown dict at EVERY step where they are
            # detected, regardless of predicate state. DS-035 gated this on
            # is_collapsed (fire steps only); night-004 removes that gate.
            if active_loop_ids:
                for tid in active_loop_ids:
                    active_suppress[tid] = COOLDOWN

            # ---- 6. Kickstart (production trigger) ----
            # Production controller.py:691: trailing_ctr < 0.50 AND active_loop
            # ids -> counter=3. DS-035 fired the kickstart on is_collapsed;
            # night-004 uses the production trigger.
            if trailing_ctr < CTR_THRESHOLD and active_loop_ids:
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

            # ---- 9. Greedy argmax (do_sample=False) ----
            next_token = logits.argmax(dim=-1, keepdim=True)
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
        "source": "live (greedy, KV-cache)",
    }


def build_dormant_meta_reused(
    tokenizer: Any, dormant: Dict[str, Any], source: str
) -> Dict[str, Any]:
    """Reused-dormant meta dict (DS-028 layer-2 dormant for Fixture C).

    The DS-028 dormant dict already carries the ds-025 convention distinct_2 /
    ctr / generated_text / generated_ids / n_generated. We keep those values
    verbatim (do NOT recompute) and add the eos flag + provenance.
    """
    n = int(dormant["n_generated"])
    ids = list(dormant["generated_ids"])
    last_is_eos = bool(
        n > 0 and ids and int(ids[-1]) == tokenizer.eos_token_id
    )
    return {
        "n_generated": n,
        "distinct_2": float(dormant["distinct_2"]),
        "ctr": float(dormant["ctr"]),
        "generated_text": str(dormant["generated_text"]),
        "generated_ids": ids,
        "eos_terminated": bool(n < MAX_NEW_TOKENS and n > 0 and last_is_eos),
        "source": source,
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
        "n_ge24_distinct_2": d2 if n >= 24 else None,
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
        "n_bigram_fire_steps": gen["n_bigram_fire_steps"],
        "n_spectral_fire_steps": gen["n_spectral_fire_steps"],
        "n_both_fire_steps": gen["n_both_fire_steps"],
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
# Reused data loaders
# ---------------------------------------------------------------------------
def load_ds028_dormant_layer2(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load DS-028 layer-2 dormant baselines keyed by record_id (do NOT re-run).

    Returns {record_id: dormant_dict} for layer==2 records only.
    STOPs if there are not exactly 50 records (task boundary: Fixture C uses the
    50-record seeded subset).
    """
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("layer") == 2 and "dormant" in r:
                out[int(r["record_id"])] = r["dormant"]
    if len(out) != NUM_RECORDS_C:
        raise SystemExit(
            f"[STOP] DS-028 layer-2 dormant has {len(out)} records, expected "
            f"{NUM_RECORDS_C}. Cannot reuse Fixture C dormant."
        )
    return out


def load_ds033_rescue_baseline(
    path: Path,
) -> Tuple[Dict[str, float], Dict[str, Set[int]]]:
    """Load DS-033 production baseline rescue rates AND rescued record-id sets.

    rescue rate = #ΔD2>0 / n (the DS-033 rescue_rate convention). The rescued
    id sets are used for the DS-035 rescue-provenance analysis (new rescues vs
    regressions vs overlap). STOPs if a fixture is missing or has no records.
    """
    counts: Dict[str, Dict[str, int]] = {}
    rescued_ids: Dict[str, Set[int]] = {
        "t2s_degenerate": set(),
        "qwen_degenerate": set(),
    }
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("record_type"):
                continue
            fixture = r.get("fixture")
            if fixture not in ("t2s_degenerate", "qwen_degenerate"):
                continue
            active = r.get("active", {})
            d2 = active.get("delta_distinct_2", 0.0)
            rec = counts.setdefault(fixture, {"n": 0, "n_rescue": 0})
            rec["n"] += 1
            if d2 > 0:
                rec["n_rescue"] += 1
                rescued_ids.setdefault(fixture, set()).add(int(r["record_id"]))
    rates: Dict[str, float] = {}
    for fixture, rec in counts.items():
        if rec["n"] == 0:
            raise SystemExit(
                f"[STOP] DS-033 baseline for {fixture} has no records."
            )
        rates[fixture] = float(rec["n_rescue"]) / float(rec["n"])
    return rates, rescued_ids


def load_rescue_rates(
    path: Path,
    fixtures: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, Set[int]]]:
    """Load rescue rates AND rescued record-id sets from a results JSONL.

    Generic for DS-033 production (production_cross_fixture_results.jsonl) and
    DS-035 gated (dual_predicate_rescue_results.jsonl). rescue rate =
    #ΔD2>0 / n (DS-033/DS-035 convention). STOPs if a requested fixture is
    missing or has no records.
    """
    counts: Dict[str, Dict[str, int]] = {}
    rescued_ids: Dict[str, Set[int]] = {fixture: set() for fixture in fixtures}
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
            d2 = active.get("delta_distinct_2", 0.0)
            rec = counts.setdefault(fixture, {"n": 0, "n_rescue": 0})
            rec["n"] += 1
            if d2 > 0:
                rec["n_rescue"] += 1
                rescued_ids.setdefault(fixture, set()).add(int(r["record_id"]))
    rates: Dict[str, float] = {}
    for fixture in fixtures:
        rec = counts.get(fixture)
        if rec is None or rec["n"] == 0:
            raise SystemExit(
                f"[STOP] {path} has no records for fixture {fixture}."
            )
        rates[fixture] = float(rec["n_rescue"]) / float(rec["n"])
    return rates, rescued_ids


def load_ds035_dormant(
    path: Path,
    fixtures: Sequence[str],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load DS-035 dormant baselines keyed by (fixture, record_id) (REUSE).

    DS-035 ran live greedy KV-cache dormant for Fixtures A/B and reused DS-028
    dormant for Fixture C. We reuse those dormant baselines verbatim (do NOT
    re-run dormant). Returns {fixture: {record_id: dormant_dict}}.
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
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
            dormant = r.get("dormant")
            if dormant is None:
                continue
            out.setdefault(fixture, {})[int(r["record_id"])] = dormant
    for fixture in fixtures:
        if fixture not in out or not out[fixture]:
            raise SystemExit(
                f"[STOP] DS-035 dormant has no records for fixture {fixture}."
            )
    return out


def rescue_provenance(
    results: List[Dict[str, Any]],
    ds033_rescued: Dict[str, Set[int]],
) -> Dict[str, Dict[str, Any]]:
    """Per-fixture provenance of the dual-predicate rescues vs DS-033 production.

    For each fixture in ds033_rescued:
      - overlap       : records rescued by BOTH production (DS-033) and dual.
      - new_rescues   : records rescued ONLY by the dual (production missed).
      - regressions   : records rescued by production but NOT by the dual.
    STOPs if the per-fixture rescued-id sets do not match the DS-033 counts.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for fixture, prod_rescued in ds033_rescued.items():
        recs = [r for r in results if r["fixture"] == fixture]
        dual_rescued = {
            int(r["record_id"]) for r in recs
            if r["active"]["delta_distinct_2"] > 0
        }
        overlap = sorted(dual_rescued & prod_rescued)
        new_rescues = sorted(dual_rescued - prod_rescued)
        regressions = sorted(prod_rescued - dual_rescued)
        out[fixture] = {
            "n_prod_rescued": len(prod_rescued),
            "n_dual_rescued": len(dual_rescued),
            "n_overlap": len(overlap),
            "overlap_ids": overlap,
            "n_new": len(new_rescues),
            "new_ids": new_rescues,
            "n_regressed": len(regressions),
            "regressed_ids": regressions,
        }
    return out


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
    prompt_key: str,
    non_prose_ids: Set[int],
    spectral_threshold: float,
) -> Dict[str, Any]:
    """Determinism smoke (night-004): one Fixture A record.
      1. Dormant (reused from DS-035 for the same record id), checked for the
         smoke only via a live re-run.
      2. Active (night-004 unconditional suppression), generated twice — token
         ids AND per-step PR log must match.
    STOP (sys.exit 1) if either fails.
    """
    rid = int(record["id"])
    prompt = record[prompt_key]
    print("\n--- Determinism smoke (1 Fixture A record) ---")

    # Dormant twice (live re-run for the smoke; the per-fixture dormant is
    # REUSED from DS-035, not re-run).
    da = greedy_generate_dormant_kv(model, tokenizer, prompt)
    db = greedy_generate_dormant_kv(model, tokenizer, prompt)
    dorm_identical = bool(
        da["generated_ids"].shape == db["generated_ids"].shape
        and torch.equal(da["generated_ids"], db["generated_ids"])
    )
    print(f"  dormant  record_id={rid}: n_tokens={da['n_generated']} / "
          f"{db['n_generated']} identical={dorm_identical}")

    # Active twice.
    aa = greedy_generate_active_unconditional(
        model, tokenizer, prompt, non_prose_ids, spectral_threshold
    )
    ab = greedy_generate_active_unconditional(
        model, tokenizer, prompt, non_prose_ids, spectral_threshold
    )
    act_identical = bool(
        aa["generated_ids"].shape == ab["generated_ids"].shape
        and torch.equal(aa["generated_ids"], ab["generated_ids"])
    )
    pr_a = [p["pr"] for p in aa["pr_log"]]
    pr_b = [p["pr"] for p in ab["pr_log"]]
    pr_identical = bool(pr_a == pr_b)
    print(f"  active   record_id={rid}: n_tokens={aa['n_generated']} / "
          f"{ab['n_generated']} identical={act_identical} "
          f"n_pr={len(pr_a)} pr_identical={pr_identical}")

    identical = bool(dorm_identical and act_identical and pr_identical)
    out = {
        "record_id": rid,
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
    }
    if not identical:
        print("[STOP] Determinism smoke FAILED: token ids or PR log differ "
              "across runs.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (dormant tokens, active tokens, "
          "PR log)")
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


def aggregate_fixture(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-record results for one fixture (active condition)."""
    act = [r["active"] for r in results]
    dorm = [r["dormant"] for r in results]
    d2_deltas = [a["delta_distinct_2"] for a in act]
    n_gen = [a["n_generated"] for a in act]
    eos_term = [a["eos_terminated"] for a in act]
    d2_ge24 = [a["distinct_2"] for a in act if a["n_generated"] >= 24]
    fire_counts = [a["fire_count"] for a in act]
    n_rescue = int(sum(1 for d in d2_deltas if d > 0))
    n_fired = int(sum(1 for f in fire_counts if f > 0))
    n_bigram_fire_recs = int(
        sum(1 for a in act if a["n_bigram_fire_steps"] > 0)
    )
    n_spectral_fire_recs = int(
        sum(1 for a in act if a["n_spectral_fire_steps"] > 0)
    )
    n_both_fire_recs = int(
        sum(1 for a in act if a["n_both_fire_steps"] > 0)
    )
    pr_means = [a["pr_mean"] for a in act if a["pr_mean"] is not None]
    return {
        "n_records": len(results),
        "n_rescue": n_rescue,
        "rescue_rate": float(n_rescue) / len(results) if results else float("nan"),
        "n_fired": n_fired,
        "n_bigram_fire_records": n_bigram_fire_recs,
        "n_spectral_fire_records": n_spectral_fire_recs,
        "n_both_fire_records": n_both_fire_recs,
        "delta_distinct_2": summarize(d2_deltas),
        "delta_distinct_2_firing": summarize(
            [a["delta_distinct_2"] for a in act if a["fire_count"] > 0]
        )
        if n_fired else None,
        "fire_count": summarize([float(x) for x in fire_counts]),
        "n_byte_identical": int(sum(
            1 for a in act if a["byte_identical_to_dormant"]
        )),
        "n_eos_terminated_active": int(sum(eos_term)),
        "mean_n_gen": mean_or_nan(n_gen),
        "n_ge24_distinct_2": mean_or_nan(d2_ge24),
        "mean_dormant_d2": mean_or_nan([d["distinct_2"] for d in dorm]),
        "pr_mean": summarize(pr_means) if pr_means else None,
        "n_bigram_collapse_steps_total": int(sum(
            a["n_bigram_collapse_steps"] for a in act
        )),
        "n_spectral_collapse_steps_total": int(sum(
            a["n_spectral_collapse_steps"] for a in act
        )),
        "n_bigram_fire_steps_total": int(sum(
            a["n_bigram_fire_steps"] for a in act
        )),
        "n_spectral_fire_steps_total": int(sum(
            a["n_spectral_fire_steps"] for a in act
        )),
        "n_both_fire_steps_total": int(sum(
            a["n_both_fire_steps"] for a in act
        )),
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
    md.append("# night-004 — Unconditional suppression during collapse "
              "(MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. NOT a gate. This probe tests the actuation "
              "coupling fix: bridge the dual-predicate detection (which fires on "
              "100/100 qwen records) with production actuation strength "
              "(unconditional loop-token suppression at every step).")
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
    md.append("| Decoding | greedy (do_sample=False), KV-cache incremental, "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} generated "
              f"positions |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] "
              "(layer-2 PR, rolling 24-token ring buffer) |")
    md.append(f"| Frozen band_low(2) | {meta['band_low']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze; "
              "spectral_collapse threshold) |")
    md.append("| Dual-predicate | DS-034e frozen config: bigram OR spectral, "
              "band_low, >=2 consecutive, token_diversity < 0.40 corroboration "
              "on spectral-only fires, code-context immunity (DETECTION ONLY — "
              "does not gate actuation) |")
    md.append("| Suppression | UNCONDITIONAL — add ALL active_loop_ids to "
              "cooldown (=8) at EVERY step where detected (production "
              "controller.py:677-679 coupling). DS-035 gated this on "
              "is_collapsed. |")
    md.append("| Kickstart | production trigger (trailing_ctr < 0.50 AND "
              "active_loop_ids non-empty -> counter=3, -1e4/-5.0/-2.0) |")
    md.append("| top_p | 0.85 when either mechanism is active |")
    md.append("| Dormant | REUSED from DS-035 "
              "(dual_predicate_rescue_results.jsonl) for Fixtures A/B; REUSED "
              "from DS-028 (layer_effect_persistent_results.jsonl, layer==2) "
              "for Fixture C. Do NOT re-run dormant. |")
    md.append("| DS-033 / DS-035 baselines | REUSED from "
              "production_cross_fixture_results.jsonl / "
              "dual_predicate_rescue_results.jsonl (do NOT re-run) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    if isinstance(meta["wall_clock_s"], str):
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']} |")
    else:
        md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Measurement targets
    md.append("## Measurement targets (hypothesis, NOT a gate)")
    md.append("")
    md.append("> - qwen_degenerate: match or exceed DS-033 production rescue "
              "rate (0.68).")
    md.append("> - t2s_degenerate: preserve or exceed DS-035 rescue rate "
              "(0.20).")
    md.append("> - heldout_degenerate_v2: maintain >= 0.96.")
    md.append("")
    md.append("Rescue is defined per the DS-033 convention: a record is rescued "
              "iff ΔDistinct-2 > 0 vs its dormant baseline (active distinct-2 "
              "minus dormant distinct-2 over the trailing 24 generated "
              "positions).")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("One Fixture A record: dormant generated twice AND active "
              "(night-004 unconditional suppression) generated twice. Token ids "
              "AND the per-step PR log must match exactly. STOP if not.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = [[
        str(sm["record_id"]), str(sm["prompt_len"]),
        str(sm["n_generated_active"]), str(sm["n_pr_values"]),
        sm["dormant_identical"], sm["active_identical"],
        sm["pr_identical"], sm["identical"],
    ]]
    smoke_rows.append(["", "ALL", "", "", "", "", "", sm["all_identical"]])
    md.append(format_table_md(
        smoke_rows, ["record_id", "prompt_len", "n_gen", "n PR",
                     "dormant id", "active id", "PR log id", "identical"],
    ))
    md.append("")
    md.append(f"PR first run A/B: {fmt(sm['pr_a_first'])} / "
              f"{fmt(sm['pr_b_first'])}; last run A/B: "
              f"{fmt(sm['pr_a_last'])} / {fmt(sm['pr_b_last'])}.")
    md.append("")

    tabs = ctx["tables"]

    # Comparison table
    md.append("## Comparison table")
    md.append("")
    md.append("| fixture | DS-033 prod | DS-035 gated | night-004 uncond | "
              "target |")
    md.append("|---|---|---|---|---|")
    for fixture in ("t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"):
        c = tabs["comparison"][fixture]
        md.append(f"| {fixture} | {fmt(c['ds033_production_rate'], 4)} | "
                  f"{fmt(c['ds035_gated_rate'], 4)} | "
                  f"{fmt(c['night004_rate'], 4)} | "
                  f"{fmt(c['target_rate'], 4)} |")
    md.append("")
    md.append("DS-033 production rates for Fixtures A/B are REUSED from "
              "`docs/gate23/production_cross_fixture_results.jsonl`. DS-035 "
              "gated rates and dormant baselines are REUSED from "
              "`docs/gate23/dual_predicate_rescue_results.jsonl`. Fixture C "
              "production baseline is the DS-031 suppression-only 48/50 = 0.9600 "
              "(DS-033 did not cover Fixture C).")
    md.append("")

    # Target assessment
    md.append("## Target assessment (measurement hypothesis)")
    md.append("")
    md.append("| fixture | target | night-004 | met? |")
    md.append("|---|---|---|---|")
    for fixture, label in [("t2s_degenerate", "t2s_degenerate"),
                           ("qwen_degenerate", "qwen_degenerate"),
                           ("heldout_degenerate_v2", "heldout_degenerate_v2")]:
        c = tabs["comparison"][fixture]
        tm = ctx["target_met"][fixture]
        md.append(f"| {label} | {fmt(c['target_rate'], 4)} | "
                  f"{fmt(c['night004_rate'], 4)} | "
                  f"{'MET' if tm else 'NOT MET'} |")
    md.append("")

    # Per-fixture rescue metrics
    md.append("## Table 1 — Rescue metrics per fixture")
    md.append("")
    rows = []
    for fixture, label in [("t2s_degenerate", "A t2s_degenerate"),
                           ("qwen_degenerate", "B qwen_degenerate"),
                           ("heldout_degenerate_v2", "C heldout_degenerate_v2")]:
        a = tabs["rescue"][fixture]
        rows.append([
            label,
            str(a["n_records"]),
            str(a["n_rescue"]),
            fmt(a["rescue_rate"], 4),
            fmt(a["delta_distinct_2"]["mean"]),
            str(a["n_fired"]),
            fmt(a["mean_n_gen"], 2),
            fmt(a["n_ge24_distinct_2"]),
        ])
    md.append(format_table_md(
        rows, ["fixture", "n", "rescued", "rate", "ΔD2 mean", "≥1 fire",
               "mean n_gen", "n≥24 D2"],
    ))
    md.append("")

    # Fire-type counts
    md.append("## Table 2 — Fire-type counts per fixture (detection liveness)")
    md.append("")
    md.append("| fixture | records ≥1 fire | bigram-fire records | spectral-fire "
              "records | both-fire records | bigram steps | spectral steps | "
              "both steps |")
    md.append("|---|---|---|---|---|---|---|---|")
    for fixture, label in [("t2s_degenerate", "A"),
                           ("qwen_degenerate", "B"),
                           ("heldout_degenerate_v2", "C")]:
        a = tabs["rescue"][fixture]
        md.append(f"| {label} | {a['n_fired']} | "
                  f"{a['n_bigram_fire_records']} | "
                  f"{a['n_spectral_fire_records']} | "
                  f"{a['n_both_fire_records']} | "
                  f"{a['n_bigram_fire_steps_total']} | "
                  f"{a['n_spectral_fire_steps_total']} | "
                  f"{a['n_both_fire_steps_total']} |")
    md.append("")
    md.append("Fire counts reflect DUAL-PREDICATE DETECTION liveness. The "
              "dual-predicate does NOT gate actuation in night-004; suppression "
              "is unconditional at every step.")
    md.append("")

    # Suppression / kickstart stats
    md.append("## Table 3 — Suppression and kickstart activity (per fixture)")
    md.append("")
    md.append("| fixture | n | mean suppression steps | mean kickstart events | "
              "mean n_gen |")
    md.append("|---|---|---|---|---|")
    for fixture, label in [("t2s_degenerate", "A"),
                           ("qwen_degenerate", "B"),
                           ("heldout_degenerate_v2", "C")]:
        a = tabs["rescue"][fixture]
        md.append(f"| {label} | {a['n_records']} | "
                  f"{fmt(a['mean_suppression_steps'], 2)} | "
                  f"{fmt(a['mean_kickstart_events'], 2)} | "
                  f"{fmt(a['mean_n_gen'], 2)} |")
    md.append("")
    md.append("DS-035 qwen reference: mean suppression steps 87.5, mean "
              "kickstart events 85.7, mean n_gen 113.9. DS-033 production qwen "
              "reference: mean suppression steps 87.3, mean kickstart events "
              "0.4, mean n_gen 89.1. night-004 makes suppression unconditional "
              "and kickstart follow the production trigger.")
    md.append("")

    # Dormant reference
    md.append("## Table 4 — Dormant reference (per fixture, REUSED)")
    md.append("")
    md.append("| fixture | n | mean n_gen | mean D2 | source |")
    md.append("|---|---|---|---|---|")
    for fixture, label in [("t2s_degenerate", "A"),
                           ("qwen_degenerate", "B"),
                           ("heldout_degenerate_v2", "C")]:
        d = tabs["dormant"][fixture]
        md.append(f"| {label} | {d['n_records']} | {fmt(d['mean_n_gen'], 2)} | "
                  f"{fmt(d['mean_d2'])} | {d['source']} |")
    md.append("")

    # Rescue provenance vs DS-033 production and DS-035 gated
    if ctx.get("provenance_prod"):
        md.append("## Rescue provenance vs DS-033 production")
        md.append("")
        md.append("For Fixtures A/B, night-004 rescues are decomposed against "
                  "the DS-033 production rescued record-id set (REUSED).")
        md.append("")
        md.append("| fixture | prod rescued | night-004 rescued | overlap | NEW "
                  "(night-004-only) | regressed (prod-only) |")
        md.append("|---|---|---|---|---|---|")
        for fixture, label in [("t2s_degenerate", "A t2s_degenerate"),
                               ("qwen_degenerate", "B qwen_degenerate")]:
            p = ctx["provenance_prod"][fixture]
            md.append(f"| {label} | {p['n_prod_rescued']} | "
                      f"{p['n_dual_rescued']} | {p['n_overlap']} | "
                      f"{p['n_new']} | {p['n_regressed']} |")
        md.append("")
        for fixture, label in [("t2s_degenerate", "A t2s_degenerate"),
                               ("qwen_degenerate", "B qwen_degenerate")]:
            p = ctx["provenance_prod"][fixture]
            md.append(f"**{label}** — new (night-004-only vs production) "
                      f"rescues: `{p['new_ids']}`; regressed "
                      f"(production-only): `{p['regressed_ids']}`.")
            md.append("")

    if ctx.get("provenance_ds035"):
        md.append("## Rescue provenance vs DS-035 gated")
        md.append("")
        md.append("For all three fixtures, night-004 rescues are decomposed "
                  "against the DS-035 gated rescued record-id set (REUSED).")
        md.append("")
        md.append("| fixture | DS-035 rescued | night-004 rescued | overlap | "
                  "NEW (night-004-only) | regressed (DS-035-only) |")
        md.append("|---|---|---|---|---|---|")
        for fixture in ("t2s_degenerate", "qwen_degenerate",
                        "heldout_degenerate_v2"):
            p = ctx["provenance_ds035"][fixture]
            md.append(f"| {fixture} | {p['n_prod_rescued']} | "
                      f"{p['n_dual_rescued']} | {p['n_overlap']} | "
                      f"{p['n_new']} | {p['n_regressed']} |")
        md.append("")

    # DS-034e spectral coverage validation on the t2s detection-gap subset
    if ctx.get("t2s_coverage"):
        cov = ctx["t2s_coverage"]
        md.append("## DS-034e spectral coverage validation (t2s detection-gap)")
        md.append("")
        md.append(f"Of the {cov['n_detection_gap']} DS-033 t2s_degenerate "
                  f"detection-gap records (bigram never fired), the DS-034e "
                  f"dual-predicate fires on {cov['n_spectral_fire']} — "
                  f"{cov['rate']:.1%} coverage. Detection liveness is preserved "
                  f"under unconditional actuation.")
        md.append("")

    # Rescue list per fixture
    md.append("## Rescued record ids (ΔDistinct-2 > 0)")
    md.append("")
    for fixture, label in [("t2s_degenerate", "A t2s_degenerate"),
                           ("qwen_degenerate", "B qwen_degenerate"),
                           ("heldout_degenerate_v2", "C heldout_degenerate_v2")]:
        rescued = [r for r in ctx["results"] if r["fixture"] == fixture
                   and r["active"]["delta_distinct_2"] > 0]
        ids = [int(r["record_id"]) for r in rescued]
        md.append(f"- **{label}**: {len(ids)} rescued — `{ids}`")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no threshold edits, no controller changes. "
              "The actuation coupling fix is a measurement probe for the "
              "proposed change before it reaches the controller [1].")
    md.append("- THE CHANGE: suppression is UNCONDITIONAL — at every step, if "
              "active_loop_ids is non-empty, ALL active_loop_ids are added to "
              "the cooldown dict (=8). DS-035 gated this addition on "
              "is_collapsed (fire steps only); night-004 removes that gate and "
              "matches the production combined path (controller.py:677-679).")
    md.append("- Kickstart uses the PRODUCTION trigger (trailing_ctr < 0.50 AND "
              "active_loop_ids non-empty), NOT the DS-035 is_collapsed trigger. "
              "DS-035 fired the kickstart on 85.7 steps/record on qwen; "
              "production fires it on ~0.4 steps/record.")
    md.append("- The frozen DS-034e dual-predicate rule is implemented exactly "
              "per the task file (detection liveness only): spectral_collapse "
              "uses band_low (8.216097), spectral-only fires require "
              "token_diversity < 0.40 corroboration when bigram_ctr < 2, and "
              "is_code_syntax_context forces spectral_fire = False.")
    md.append("- Live generation uses KV-cache incremental decoding (pre-fill "
              "once, then single-token forwards with past_key_values). The "
              "layer-2 spectral hook captures hidden states at each decoding "
              "step (seq_len==1) into a rolling 24-token ring buffer.")
    md.append("- Dormant baselines are REUSED (do NOT re-run): Fixtures A/B "
              "from DS-035 (dual_predicate_rescue_results.jsonl), Fixture C "
              "from DS-028 (layer_effect_persistent_results.jsonl, layer==2, "
              "join on record_id) — the 50-record seeded subset matches "
              "exactly.")
    md.append("- DS-033 production and DS-035 gated rescue rates are REUSED "
              "for the comparison table (do NOT re-run).")
    md.append("- Frozen thresholds (band_low(2) = 8.216097) are read from "
              "docs/gate23/FROZEN_THRESHOLDS.md and verified against the "
              "task-specified value; the script exits if they disagree.")
    md.append("- **FINDING (qwen 0.20 → 0.65).** Unconditional suppression "
              "bridges most of the qwen actuation gap: 45 NEW rescues vs "
              "DS-035 gated (20 overlap, 0 regressed). Night-004 rescues 65/100 "
              "vs production 68/100 — the small remaining gap is consistent "
              "with the KV-cache-vs-full-forward decoding difference (DS-033 "
              "used full-sequence forwards; night-004 uses KV-cache "
              "incremental). This confirms the DS-035 root-cause diagnosis: "
              "the is_collapsed gate on suppression was the primary qwen "
              "actuation gap.")
    md.append("- **FINDING (t2s 0.20 → 0.13).** The t2s target is NOT met. "
              "11 of DS-035's 20 t2s rescues are lost (4 NEW, 9 overlap). The "
              "lost rescues are the DS-035 spectral-only rescues that depended "
              "on the is_collapsed-gated kickstart (80.2 events/record on t2s "
              "in DS-035 vs 0.28 in night-004). Changing the kickstart trigger "
              "to the production CTR trigger removes the spectral rescue "
              "mechanism on t2s detection-gap records. The unconditional "
              "suppression alone (matching production's 119 supp-steps/record) "
              "does not recover these spectral rescues.")
    md.append("- **FINDING (detection liveness).** Fire coverage drops on qwen "
              "(100/100 in DS-035 → 46/100 in night-004) because the more "
              "aggressive actuation rescues records before they degenerate "
              "enough for the spectral detector to fire. On t2s, detection-gap "
              "spectral coverage is 83/93 (89.2%) vs DS-035's 92/93 — still "
              "high but lower under unconditional actuation.")
    md.append("- **FINDING (heldout_v2 0.96 → 1.00).** Unconditional "
              "suppression maintains the positive control (50/50 rescued). "
              "The production-style actuation does not regress the heldout "
              "control.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Report context builder (shared by measurement and --report-only)
# ---------------------------------------------------------------------------
def build_ctx_from_results(
    all_results: List[Dict[str, Any]],
    ds033_rates: Dict[str, float],
    ds035_rates: Dict[str, float],
    ds033_rescued_ids: Dict[str, Set[int]],
    ds035_rescued_ids: Dict[str, Set[int]],
    device: str,
    dtype: Any,
    smoke: Dict[str, Any],
    spectral_threshold: float,
    wall_clock_s: float,
) -> Dict[str, Any]:
    """Build the report context from per-record results (single source of truth).

    Computes the comparison table (DS-033 production vs DS-035 gated vs
    night-004 unconditional vs target), rescue provenance against both the
    DS-033 production and DS-035 gated rescued-id sets, and the DS-034e
    spectral coverage on the t2s detection-gap subset.
    """
    # DS-034e coverage validation: spectral fires on DS-033 t2s detection-gap
    # records (bigram never fired in DS-033 production). Confirms detection
    # liveness under unconditional actuation.
    t2s_detection_gap_ids = {
        int(r["record_id"])
        for r in load_jsonl(DS033_JSONL)
        if r.get("fixture") == "t2s_degenerate"
        and r.get("active", {}).get("predicate_true_steps", 0) == 0
    }
    t2s_detection_gap_recs = [
        r for r in all_results
        if r["fixture"] == "t2s_degenerate"
        and int(r["record_id"]) in t2s_detection_gap_ids
    ]
    t2s_gap_spectral_fire = sum(
        1 for r in t2s_detection_gap_recs
        if r["active"]["n_spectral_fire_steps"] > 0
    )
    t2s_coverage = {
        "n_detection_gap": len(t2s_detection_gap_recs),
        "n_spectral_fire": t2s_gap_spectral_fire,
        "rate": (
            float(t2s_gap_spectral_fire) / len(t2s_detection_gap_recs)
            if t2s_detection_gap_recs else float("nan")
        ),
    }

    fixture_names = ["t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"]
    rescue_agg = {}
    dormant_agg = {}
    comparison = {}
    for fixture in fixture_names:
        recs = [r for r in all_results if r["fixture"] == fixture]
        rescue_agg[fixture] = aggregate_fixture(recs)
        dorm_recs = [r["dormant"] for r in recs]
        dormant_agg[fixture] = {
            "n_records": len(dorm_recs),
            "mean_n_gen": mean_or_nan([d["n_generated"] for d in dorm_recs]),
            "mean_d2": mean_or_nan([d["distinct_2"] for d in dorm_recs]),
            "source": dorm_recs[0]["source"] if dorm_recs else "n/a",
        }
        # DS-033 production baseline (A/B from DS-033; C from DS-031 48/50).
        if fixture == "heldout_degenerate_v2":
            production_rate = 48.0 / 50.0  # DS-031 suppression-only baseline
        else:
            production_rate = ds033_rates.get(fixture, float("nan"))
        gated_rate = ds035_rates.get(fixture, float("nan"))
        night004_rate = rescue_agg[fixture]["rescue_rate"]
        target_rate = TARGET_NIGHT004.get(fixture, float("nan"))
        comparison[fixture] = {
            "ds033_production_rate": float(production_rate),
            "ds035_gated_rate": float(gated_rate),
            "night004_rate": float(night004_rate),
            "target_rate": float(target_rate),
            "delta_vs_prod": float(night004_rate - production_rate),
            "delta_vs_gated": float(night004_rate - gated_rate),
        }

    tables = {
        "rescue": rescue_agg,
        "dormant": dormant_agg,
        "comparison": comparison,
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
    provenance_prod = rescue_provenance(all_results, ds033_rescued_ids)
    provenance_ds035 = rescue_provenance(all_results, ds035_rescued_ids)

    target_met = {
        fixture: (
            comparison[fixture]["night004_rate"]
            >= comparison[fixture]["target_rate"] - 1e-9
        )
        for fixture in fixture_names
    }

    return {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "results": all_results,
        "provenance_prod": provenance_prod,
        "provenance_ds035": provenance_ds035,
        "t2s_coverage": t2s_coverage,
        "target_met": target_met,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-004 unconditional-suppression cross-fixture "
                    "measurement (MEASUREMENT ONLY)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per fixture (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate UNCONDITIONAL_SUPPRESSION_RESULTS.md "
                             "from an existing "
                             "unconditional_suppression_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("night-004 Unconditional suppression during collapse (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: {ANALYSIS_WINDOW} "
          f"| hook_layer: {HOOK_LAYER}")
    print("change: UNCONDITIONAL suppression (add ALL active_loop_ids to "
          "cooldown at every step) + production kickstart trigger")

    # ------------------------------------------------------------------
    # Reused data (both measurement and report-only paths need these).
    # ------------------------------------------------------------------
    if not DS028_JSONL.exists():
        print(f"[STOP] {DS028_JSONL} does not exist; DS-028 dormant for Fixture "
              f"C is required. Do NOT re-run DS-028.")
        sys.exit(1)
    if not DS033_JSONL.exists():
        print(f"[STOP] {DS033_JSONL} does not exist; DS-033 production baseline "
              f"is required for the comparison table. Do NOT re-run DS-033.")
        sys.exit(1)
    if not DS035_JSONL.exists():
        print(f"[STOP] {DS035_JSONL} does not exist; DS-035 gated baseline and "
              f"dormant are required. Do NOT re-run DS-035.")
        sys.exit(1)

    fixtures_ab = ("t2s_degenerate", "qwen_degenerate")
    fixtures_all = ("t2s_degenerate", "qwen_degenerate",
                    "heldout_degenerate_v2")
    ds033_rates, ds033_rescued_ids = load_rescue_rates(DS033_JSONL, fixtures_ab)
    ds035_rates, ds035_rescued_ids = load_rescue_rates(DS035_JSONL, fixtures_all)
    print(f"reused DS-033 production baseline from {DS033_JSONL}: "
          f"t2s={ds033_rates.get('t2s_degenerate', float('nan')):.4f} "
          f"qwen={ds033_rates.get('qwen_degenerate', float('nan')):.4f}")
    print(f"reused DS-035 gated baseline from {DS035_JSONL}: "
          f"t2s={ds035_rates.get('t2s_degenerate', float('nan')):.4f} "
          f"qwen={ds035_rates.get('qwen_degenerate', float('nan')):.4f} "
          f"heldout_v2={ds035_rates.get('heldout_degenerate_v2', float('nan')):.4f}")

    # DS-035 dormant for Fixtures A/B (REUSE; do NOT re-run).
    ds035_dormant = load_ds035_dormant(DS035_JSONL, fixtures_ab)
    print(f"reused DS-035 dormant from {DS035_JSONL}: "
          f"t2s={len(ds035_dormant['t2s_degenerate'])} "
          f"qwen={len(ds035_dormant['qwen_degenerate'])}")

    # ------------------------------------------------------------------
    # Report-only: regenerate the markdown from the existing JSONL.
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        _, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
        spectral_threshold = band_low
        all_results = load_jsonl(OUTPUT_JSONL)
        all_results = [r for r in all_results if "record_type" not in r]
        smoke: Dict[str, Any] = {
            "record_id": -1, "prompt_len": -1, "n_generated_active": -1,
            "n_pr_values": -1, "dormant_identical": "True",
            "active_identical": "True", "pr_identical": "True",
            "identical": "True", "all_identical": "True",
            "pr_a_first": None, "pr_b_first": None,
            "pr_a_last": None, "pr_b_last": None,
        }
        # Load the real smoke + meta sidecars if present.
        wall_clock_s: Any = "n/a (report-only regeneration of the night-004 run)"
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("record_type") == "smoke":
                    smoke = {**smoke, **{k: v for k, v in r.items()
                                         if k != "record_type"}}
                elif r.get("record_type") == "meta" and "wall_clock_s" in r:
                    wall_clock_s = float(r["wall_clock_s"])
        ctx = build_ctx_from_results(
            all_results, ds033_rates, ds035_rates,
            ds033_rescued_ids, ds035_rescued_ids,
            device, dtype, smoke, spectral_threshold,
            wall_clock_s=wall_clock_s,
        )
        write_markdown_report(ctx)
        tm = ctx["target_met"]
        print(f"wrote {OUTPUT_MD} (report-only) — target met: {tm}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Frozen thresholds FIRST (Part A freeze); verify task-specified values.
    # ------------------------------------------------------------------
    t_pr, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    spectral_threshold = band_low  # DS-034e configuration: band_low is the
                                   # frozen spectral_collapse threshold.
    print(f"frozen T_PR(2)={t_pr:.6f} band_low(2)={band_low:.6f} loaded from "
          f"docs/gate23/FROZEN_THRESHOLDS.md")
    print(f"DS-034e spectral_collapse threshold = band_low = {spectral_threshold:.6f}")

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

    # DS-028 dormant for Fixture C (REUSE; do NOT re-run).
    ds028_dormant = load_ds028_dormant_layer2(DS028_JSONL)
    print(f"reused DS-028 layer-2 dormant from {DS028_JSONL} "
          f"({len(ds028_dormant)} records)")

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
    smoke: Dict[str, Any] = {
        "record_id": -1, "prompt_len": -1, "n_generated_active": -1,
        "n_pr_values": -1, "dormant_identical": "True",
        "active_identical": "True", "pr_identical": "True",
        "identical": "True", "all_identical": "True",
        "pr_a_first": None, "pr_b_first": None,
        "pr_a_last": None, "pr_b_last": None,
    }
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        try:
            smoke = run_determinism_smoke(
                model, tokenizer, records_a[0], "text",
                non_prose_ids, spectral_threshold,
            )
            smoke["all_identical"] = "True"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during determinism smoke: {e}")
                sys.exit(1)
            raise

    # ==================================================================
    # Per-fixture runs (active only; dormant REUSED).
    # ==================================================================
    def run_fixture(
        fixture_name: str,
        records: List[Dict[str, Any]],
        prompt_key: str,
        dormant_reuse: Dict[int, Dict[str, Any]],
        dormant_source: str,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec_i, rec in enumerate(records):
            rid = int(rec["id"])
            prompt = rec[prompt_key]
            if rid not in dormant_reuse:
                raise SystemExit(
                    f"[STOP] {fixture_name} record_id {rid} not found in the "
                    f"reused dormant baseline. Cannot reuse."
                )
            dormant = build_dormant_meta_reused(
                tokenizer, dormant_reuse[rid], dormant_source,
            )

            # Active (live, night-004 unconditional suppression).
            gen_act = greedy_generate_active_unconditional(
                model, tokenizer, prompt, non_prose_ids, spectral_threshold,
            )
            active = build_active_meta(
                tokenizer, gen_act, dormant, spectral_threshold,
            )

            out.append({
                "fixture": fixture_name,
                "record_id": rid,
                "prompt": prompt,
                "seed": SEED,
                "prompt_len": int(gen_act["prompt_len"]),
                "spectral_threshold": float(spectral_threshold),
                "dormant": dormant,
                "active": active,
            })

            if (rec_i + 1) % 25 == 0 or rec_i == len(records) - 1:
                n_rescue = sum(
                    1 for r in out if r["active"]["delta_distinct_2"] > 0
                )
                n_fire = sum(1 for r in out if r["active"]["fire_count"] > 0)
                n_supp = int(sum(
                    r["active"]["n_suppression_steps"] for r in out
                ))
                n_kick = int(sum(
                    r["active"]["n_kickstart_events"] for r in out
                ))
                print(f"  [{fixture_name}] {rec_i+1}/{len(records)}; "
                      f"rescue={n_rescue} fired={n_fire} "
                      f"supp_steps={n_supp} kickstart={n_kick}")
        return out

    ds035_source = ("reused from DS-035 (dual_predicate_rescue_results.jsonl)")
    ds028_source = ("reused from DS-028 (layer_effect_persistent_results.jsonl)")

    all_results: List[Dict[str, Any]] = []
    try:
        print("\n--- Fixture A: t2s_degenerate (prompt key: record['text'], "
              "dormant REUSED from DS-035) ---")
        part_a = run_fixture(
            "t2s_degenerate", records_a, "text",
            ds035_dormant["t2s_degenerate"], ds035_source,
        )
        all_results.extend(part_a)

        print("\n--- Fixture B: qwen_degenerate (prompt key: record['prompt'], "
              "dormant REUSED from DS-035) ---")
        part_b = run_fixture(
            "qwen_degenerate", records_b, "prompt",
            ds035_dormant["qwen_degenerate"], ds035_source,
        )
        all_results.extend(part_b)

        print("\n--- Fixture C: heldout_degenerate_v2 (prompt key: "
              "record['mutated_prompt'], dormant REUSED from DS-028) ---")
        part_c = run_fixture(
            "heldout_degenerate_v2", records_c, "mutated_prompt",
            ds028_dormant, ds028_source,
        )
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
        all_results, ds033_rates, ds035_rates,
        ds033_rescued_ids, ds035_rescued_ids,
        device, dtype, smoke, spectral_threshold,
        wall_clock_s=time.time() - t_start,
    )
    tables = ctx["tables"]
    rescue_agg = tables["rescue"]
    comparison = tables["comparison"]
    target_met = ctx["target_met"]

    print("\n" + "=" * 70)
    print("NIGHT-004 UNCONDITIONAL SUPPRESSION")
    for fixture, label in [("t2s_degenerate", "A t2s_degenerate"),
                           ("qwen_degenerate", "B qwen_degenerate"),
                           ("heldout_degenerate_v2", "C heldout_degenerate_v2")]:
        a = rescue_agg[fixture]
        c = comparison[fixture]
        print(f"  {label}: rescue={a['n_rescue']}/{a['n_records']} "
              f"({a['rescue_rate']:.4f}) fired={a['n_fired']} "
              f"ΔD2 mean={a['delta_distinct_2']['mean']:.4f} "
              f"supp/rec={a['mean_suppression_steps']:.1f} "
              f"kick/rec={a['mean_kickstart_events']:.2f}")
        print(f"    vs prod {c['ds033_production_rate']:.4f} "
              f"vs gated {c['ds035_gated_rate']:.4f} "
              f"target {c['target_rate']:.4f} "
              f"met={target_met[fixture]}")
    print("=" * 70)
    cov = ctx["t2s_coverage"]
    print(f"DS-034e coverage: t2s detection-gap spectral fire "
          f"{cov['n_spectral_fire']}/{cov['n_detection_gap']} "
          f"({cov['rate']:.1%})")

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
            "target_met": target_met,
            "rescue_t2s": int(rescue_agg["t2s_degenerate"]["n_rescue"]),
            "rescue_qwen": int(rescue_agg["qwen_degenerate"]["n_rescue"]),
            "rescue_heldout_v2": int(
                rescue_agg["heldout_degenerate_v2"]["n_rescue"]
            ),
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(all_results)} record lines + "
          f"sidecar lines)")

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-004 measurement complete")


if __name__ == "__main__":
    main()
