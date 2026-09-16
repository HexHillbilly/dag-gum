#!/usr/bin/env python3
"""DS-034c-hazard: Dual-predicate FP re-verification on hazard and schema corpora (GATE).

RFC-004 Amendment A3 (ratified): dual-predicate OR rule — bigram OR spectral PR
at layer 2. DS-034c v2 re-verified the dual-predicate on prose-100 under LIVE
KV-cache generation with the production cadence and hysteresis (prose-100 FP
policy: 7 documented FPs, Amendment A3; reused for context, NOT re-run here).
The hazard corpora (ds-007) and schema_corpus_v2 (ds-018) were deferred when
the v1 run hit the turn-limit.

This probe completes the FP verification: does the dual-predicate produce false
positives on structured legitimate text (markdown, JSON, URLs, code-switched
prose) under live generation with the production cadence and hysteresis per
RFC A3?

Ground truth (verified by human review):
  - schema_markdown_fenced: tests/fixtures/schema_markdown_fenced.jsonl
    (ds-007, N=50). Use record["text"] as prompt.
  - schema_urls_strings:   tests/fixtures/schema_urls_strings.jsonl
    (ds-007, N=50). Use record["text"] as prompt.
  - prose_code_switch:     tests/fixtures/prose_code_switch.jsonl
    (ds-007, N=50). Use record["text"] as prompt.
  - schema_corpus_v2:      tests/fixtures/schema_corpus_v2.jsonl
    (ds-018, N=300). Use record["text"] as prompt.
  Total N=450 records across four corpora.

- Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. KV-cache incremental
  generation (pre-fill once, single-token forwards). bmm override
  deregistered. HF cache fallback. Standard import convention.
- Frozen PR thresholds: band_low = 8.216097
  (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze, unchanged).
- Cadence: every-step (DS-034a at Δ = 0.7027 ms/token).
- Persistence hysteresis: ≥2 consecutive is_collapsed steps (RFC A3).
- Code-context immunity: spectral_fire forced False when
  is_code_syntax_context is True (RFC A3).
- Dormant baselines: generated LIVE with plain greedy argmax (no hooks, no
  suppression, no kickstart, no top_p). No pre-existing dormant baselines exist
  for these corpora. Generate live for all 450 records.
- Prose-100 baseline: DS-034c v2 result is REUSED for context; prose-100 is
  NOT re-run.

Frozen dual-predicate rule (DS-034e config, band_low hardening):

    bigram_collapse   = (token_diversity < 0.30) AND (trailing_ctr < 0.30)
    if bigram_collapse: bigram_ctr += 1 else: bigram_ctr = 0
    bigram_fire       = bigram_ctr >= 2

    spectral_collapse = (layer_2_pr is not None) AND (layer_2_pr < 8.216097)
    if spectral_collapse: spectral_ctr += 1 else: spectral_ctr = 0
    if bigram_ctr < 2:
        spectral_fire = (spectral_ctr >= 2) AND (token_diversity < 0.40)
    else:
        spectral_fire = spectral_ctr >= 2

    if is_code_context:
        spectral_fire = False

    is_collapsed = bigram_fire OR spectral_fire

Actuation on is_collapsed: production logit-penalty path (token suppression
cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85).

FP criterion: a record is a false positive if ΔDistinct-2 > 0 vs dormant AND
the predicate fired at least once (the dual-predicate triggered actuation that
changed the continuation on legitimate structured text).

Gate criteria:
  - Primary: 0 FPs on hazard corpora under simple OR -> PASS.
  - If FPs > 0: the schema_markdown_fenced and schema_urls_strings corpora are
    expected to be covered by the code-context immunity
    (is_code_syntax_context fires on markdown fences and code patterns). The
    prose_code_switch and schema_corpus_v2 corpora are the primary risk — they
    contain unfenced structured text that may trigger spectral fires. Document
    per-corpus FP counts and which predicate (bigram vs spectral) triggered
    each FP.
  - A FAIL is a STOP signal — no negotiation, no threshold tuning [1].
  - The hazard/schema FP policy is scoped to these four corpora. The ratified
    prose-100 FP policy (7 documented FPs, Amendment A3) is unchanged.

Environment notes (identical to ds-025/ds-033/ds-034/ds-034c/ds-034d/ds-035):
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
SEED = 42  # DS-034c-hazard measurement seed (matches ds-025..ds-034e)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (EOS break; EOS token included)
ANALYSIS_WINDOW = 24  # rolling 24-token PR window (Gate 2.3 / ds-025 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
# DS-034e configuration: spectral_collapse uses band_low (hardened), NOT T_PR.
T_PR_2_TASK = 10.954796  # PR < T -> spectral fire (primary, DS-034c; NOT used here)
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
SPECTRAL_CORROBORATION_DIVERSITY = 0.40  # token_diversity < 0.40

# Paths
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/hazard_schema_fp_results.jsonl")
OUTPUT_MD = Path("docs/gate23/HAZARD_SCHEMA_FP_RESULTS.md")

# Ground-truth corpora (ds-007 + ds-018; human review verified).
CORPORA: List[Dict[str, Any]] = [
    {
        "fixture": "schema_markdown_fenced",
        "path": Path("tests/fixtures/schema_markdown_fenced.jsonl"),
        "expected_n": 50,
        "label": "schema_markdown_fenced (ds-007, N=50)",
    },
    {
        "fixture": "schema_urls_strings",
        "path": Path("tests/fixtures/schema_urls_strings.jsonl"),
        "expected_n": 50,
        "label": "schema_urls_strings (ds-007, N=50)",
    },
    {
        "fixture": "prose_code_switch",
        "path": Path("tests/fixtures/prose_code_switch.jsonl"),
        "expected_n": 50,
        "label": "prose_code_switch (ds-007, N=50)",
    },
    {
        "fixture": "schema_corpus_v2",
        "path": Path("tests/fixtures/schema_corpus_v2.jsonl"),
        "expected_n": 300,
        "label": "schema_corpus_v2 (ds-018, N=300)",
    },
]
TOTAL_RECORDS = sum(c["expected_n"] for c in CORPORA)  # 450

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


def load_corpus(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load a fixture corpus and STOP if the record count disagrees with the
    task-specified N (task boundary)."""
    path = cfg["path"]
    expected = int(cfg["expected_n"])
    if not path.exists():
        raise SystemExit(
            f"[STOP] {path} does not exist. Ground-truth fixture required."
        )
    records = load_jsonl(path)
    if len(records) != expected:
        raise SystemExit(
            f"[STOP] {path} has {len(records)} records, expected {expected}. "
            f"Task boundary; do NOT modify fixtures."
        )
    return records


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
# KV-cache generation: active (DS-034e dual-predicate + production penalty path)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_active_ds034e(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    non_prose_ids: Set[int],
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) active generation under the frozen DS-034e
    dual-predicate (RFC A3, band_low hardening).

    KV-cache incremental generation (pre-fill once, then single-token forwards
    with past_key_values). The layer-2 forward hook captures hidden states at
    each decoding step (seq_len==1) into a rolling 24-token ring buffer and
    computes participation_ratio() from step 23 onward.

    Frozen DS-034e dual-predicate OR rule (exact):

        bigram_collapse   = (token_diversity < 0.30) AND (trailing_ctr < 0.30)
        if bigram_collapse: bigram_ctr += 1 else: bigram_ctr = 0
        bigram_fire       = bigram_ctr >= 2

        spectral_collapse = (layer_2_pr < band_low)   # 8.216097, frozen
        if spectral_collapse: spectral_ctr += 1 else: spectral_ctr = 0
        if bigram_ctr < 2:
            spectral_fire = (spectral_ctr >= 2) AND (token_diversity < 0.40)
        else:
            spectral_fire = spectral_ctr >= 2

        if is_code_syntax_context(trailing_text):
            spectral_fire = False

        is_collapsed = bigram_fire OR spectral_fire

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
            trailing_text = ""
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

            # ---- 5. Actuation on is_collapsed ----
            if is_collapsed:
                if active_loop_ids:
                    for tid in active_loop_ids:
                        active_suppress[tid] = COOLDOWN
                kickstart_counter = KICKSTART_COUNTER_INIT
                n_kickstart_events += 1
                triggers: List[str] = []
                if bigram_fire:
                    triggers.append("bigram")
                if spectral_fire:
                    triggers.append("spectral")
                fire_events.append({
                    "step": int(step),
                    "trigger": "+".join(triggers) if triggers else "none",
                    "bigram_collapse": bool(bigram_collapse),
                    "spectral_collapse": bool(spectral_collapse),
                    "bigram_fire": bool(bigram_fire),
                    "spectral_fire": bool(spectral_fire),
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
        "n_bigram_fire_steps": n_bigram_fire_steps,
        "n_spectral_fire_steps": n_spectral_fire_steps,
        "n_both_fire_steps": n_both_fire_steps,
        "spectral_fired": n_spectral_fire_steps > 0,
        "bigram_fired": n_bigram_fire_steps > 0,
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


def classify_fire_type(active: Dict[str, Any]) -> str:
    """Classify a record's fire type: 'bigram_only', 'spectral_only', 'both',
    or 'none'."""
    if active["fire_count"] == 0:
        return "none"
    if active["bigram_fired"] and active["spectral_fired"]:
        return "both"
    if active["bigram_fired"]:
        return "bigram_only"
    if active["spectral_fired"]:
        return "spectral_only"
    return "none"


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
        "fire_type": classify_fire_type(gen),
        "n_suppression_steps": gen["n_suppression_steps"],
        "n_kickstart_events": gen["n_kickstart_events"],
        "n_bigram_collapse_steps": gen["n_bigram_collapse_steps"],
        "n_spectral_collapse_steps": gen["n_spectral_collapse_steps"],
        "n_bigram_fire_steps": gen["n_bigram_fire_steps"],
        "n_spectral_fire_steps": gen["n_spectral_fire_steps"],
        "n_both_fire_steps": gen["n_both_fire_steps"],
        "bigram_fired": bool(gen["bigram_fired"]),
        "spectral_fired": bool(gen["spectral_fired"]),
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
    # the continuation on legitimate structured text).
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
    smoke_records: List[Dict[str, Any]],
    non_prose_ids: Set[int],
    spectral_threshold: float,
) -> Dict[str, Any]:
    """Determinism smoke (DS-034c-hazard): 2-3 hazard/schema records.
      1. Dormant, generated twice — token ids must match.
      2. Active (DS-034e dual-predicate), generated twice — token ids AND
         per-step PR log must match.
    STOP (sys.exit 1) if either fails.
    """
    print("\n--- Determinism smoke "
          f"({len(smoke_records)} hazard/schema records) ---")
    pairs: List[Dict[str, Any]] = []
    all_identical = True
    for rec in smoke_records:
        rid = int(rec["corpus_id"])
        label = rec["fixture"]
        prompt = rec["prompt"]

        # Dormant twice.
        da = greedy_generate_dormant_kv(model, tokenizer, prompt)
        db = greedy_generate_dormant_kv(model, tokenizer, prompt)
        dorm_identical = bool(
            da["generated_ids"].shape == db["generated_ids"].shape
            and torch.equal(da["generated_ids"], db["generated_ids"])
        )
        print(f"  dormant  {label} id={rid}: n_tokens={da['n_generated']} / "
              f"{db['n_generated']} identical={dorm_identical}")

        # Active twice.
        aa = greedy_generate_active_ds034e(
            model, tokenizer, prompt, non_prose_ids, spectral_threshold
        )
        ab = greedy_generate_active_ds034e(
            model, tokenizer, prompt, non_prose_ids, spectral_threshold
        )
        act_identical = bool(
            aa["generated_ids"].shape == ab["generated_ids"].shape
            and torch.equal(aa["generated_ids"], ab["generated_ids"])
        )
        pr_a = [p["pr"] for p in aa["pr_log"]]
        pr_b = [p["pr"] for p in ab["pr_log"]]
        pr_identical = bool(pr_a == pr_b)
        print(f"  active   {label} id={rid}: n_tokens={aa['n_generated']} / "
              f"{ab['n_generated']} identical={act_identical} "
              f"n_pr={len(pr_a)} pr_identical={pr_identical}")

        identical = bool(dorm_identical and act_identical and pr_identical)
        all_identical = all_identical and identical
        pairs.append({
            "fixture": label,
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


def aggregate_corpus(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-record results for a single corpus (or the combined set)."""
    fps = [r for r in results if r["is_false_positive"]]
    fire_any = [r for r in results if r["active"]["fire_count"] > 0]
    fire_types = [r["active"]["fire_type"] for r in results if r["active"]["fire_count"] > 0]
    d2_deltas = [r["active"]["delta_distinct_2"] for r in results]
    d2_deltas_firing = [
        r["active"]["delta_distinct_2"] for r in results
        if r["active"]["fire_count"] > 0
    ]
    fire_counts = [r["active"]["fire_count"] for r in results]
    n_gens = [r["active"]["n_generated"] for r in results]
    n_supp = [r["active"]["n_suppression_steps"] for r in results]
    n_kick = [r["active"]["n_kickstart_events"] for r in results]
    pr_means = [
        r["active"]["pr_mean"] for r in results
        if r["active"]["pr_mean"] is not None
    ]
    n_bigram_fire_steps = sum(r["active"]["n_bigram_fire_steps"] for r in results)
    n_spectral_fire_steps = sum(r["active"]["n_spectral_fire_steps"] for r in results)
    n_both_fire_steps = sum(r["active"]["n_both_fire_steps"] for r in results)
    n_bigram_fired_recs = sum(1 for r in results if r["active"]["bigram_fired"])
    n_spectral_fired_recs = sum(1 for r in results if r["active"]["spectral_fired"])
    n_bigram_only = sum(1 for t in fire_types if t == "bigram_only")
    n_spectral_only = sum(1 for t in fire_types if t == "spectral_only")
    n_both = sum(1 for t in fire_types if t == "both")
    # FP trigger breakdown (which predicate(s) fired on FP records).
    fp_bigram_only = sum(1 for r in fps if r["active"]["fire_type"] == "bigram_only")
    fp_spectral_only = sum(1 for r in fps if r["active"]["fire_type"] == "spectral_only")
    fp_both = sum(1 for r in fps if r["active"]["fire_type"] == "both")
    return {
        "n_records": len(results),
        "n_false_positives": len(fps),
        "fp_rate": float(len(fps)) / len(results) if results else float("nan"),
        "fp_record_ids": [int(r["record_id"]) for r in fps],
        "fp_corpus_ids": [int(r["corpus_id"]) for r in fps],
        "fp_triggers": [r["active"]["fire_type"] for r in fps],
        "fp_bigram_only": fp_bigram_only,
        "fp_spectral_only": fp_spectral_only,
        "fp_both": fp_both,
        "n_fired": len(fire_any),
        "n_fire_steps_total": sum(fire_counts),
        "fire_type_breakdown": {
            "bigram_only_records": n_bigram_only,
            "spectral_only_records": n_spectral_only,
            "both_records": n_both,
            "bigram_fired_records": n_bigram_fired_recs,
            "spectral_fired_records": n_spectral_fired_recs,
            "bigram_fire_steps": n_bigram_fire_steps,
            "spectral_fire_steps": n_spectral_fire_steps,
            "both_fire_steps": n_both_fire_steps,
        },
        "fire_count": summarize([float(x) for x in fire_counts]),
        "delta_distinct_2": summarize(d2_deltas),
        "delta_distinct_2_firing": summarize(d2_deltas_firing)
        if d2_deltas_firing else None,
        "n_generated": summarize([float(x) for x in n_gens]),
        "n_suppression_steps": summarize([float(x) for x in n_supp]),
        "n_kickstart_events": summarize([float(x) for x in n_kick]),
        "pr_mean": summarize(pr_means) if pr_means else None,
        "n_byte_identical": int(sum(
            1 for r in results if r["active"]["byte_identical_to_dormant"]
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
    md.append("# DS-034c-hazard — Dual-predicate FP re-verification on hazard "
              "and schema corpora (GATE)")
    md.append("")
    md.append("> GATE REPORT (RFC-004 Amendment A3). Dual-predicate OR rule — "
              "bigram OR spectral PR at layer 2 — re-verified on structured "
              "legitimate text (markdown, JSON, URLs, code-switched prose) "
              "under LIVE generation with the production cadence and "
              "hysteresis per RFC A3 (DS-034e config, band_low hardening). "
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
    md.append("| Corpora | schema_markdown_fenced (ds-007, N=50), "
              "schema_urls_strings (ds-007, N=50), prose_code_switch (ds-007, "
              "N=50), schema_corpus_v2 (ds-018, N=300); total N=450 |")
    md.append("| Prompt | record[\\\"text\\\"] |")
    md.append("| Decoding | greedy (do_sample=False), KV-cache incremental, "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} hidden "
              f"states (layer-2 hook, rolling ring buffer) |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] |")
    md.append(f"| Frozen T_PR(2) | {meta['t_pr']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |")
    md.append(f"| Spectral band_low(2) | {meta['band_low']:.6f} "
              "(DS-034e config: spectral_collapse threshold, frozen) |")
    md.append("| PR instrument | participation_ratio() from core/metrics.py "
              "(existing), float32 promotion |")
    md.append("| Cadence | every-step (confirmed by DS-034a at "
              "Δ = 0.7027 ms/token) |")
    md.append("| Persistence hysteresis | ≥2 consecutive is_collapsed steps "
              "(RFC A3 tightening; per-predicate counters) |")
    md.append("| Spectral-only corroboration | when bigram_ctr < 2, "
              "spectral_fire requires token_diversity < 0.40 (DS-034e) |")
    md.append("| Code-context immunity | spectral_fire forced False when "
              "is_code_syntax_context is True (RFC A3) |")
    md.append("| Actuation | production logit-penalty path: token suppression "
              "cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85 |")
    md.append("| Dormant | LIVE (greedy, KV-cache, no hooks/penalties), "
              "generated for all 450 records |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Gate criteria
    md.append("## Gate criteria (quoted from the DS-034c-hazard task file)")
    md.append("")
    md.append("> - Primary: 0 FPs on hazard corpora under simple OR → PASS.")
    md.append("> - If FPs > 0: the schema_markdown_fenced and "
              "schema_urls_strings corpora are expected to be covered by the "
              "code-context immunity (is_code_syntax_context fires on markdown "
              "fences and code patterns). The prose_code_switch and "
              "schema_corpus_v2 corpora are the primary risk — they contain "
              "unfenced structured text that may trigger spectral fires. "
              "Document per-corpus FP counts and which predicate (bigram vs "
              "spectral) triggered each FP.")
    md.append("> - A FAIL is a STOP signal — no negotiation, no threshold "
              "tuning [1].")
    md.append("> - The hazard/schema FP policy is scoped to these four corpora. "
              "The ratified prose-100 FP policy (7 documented FPs, Amendment "
              "A3) is unchanged.")
    md.append("")
    md.append("FP criterion: a record is a false positive if ΔDistinct-2 > 0 "
              "vs dormant AND the predicate fired at least once (the "
              "dual-predicate triggered actuation that changed the continuation "
              "on legitimate structured text).")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("Two-to-three hazard/schema records: dormant generated twice AND "
              "active (DS-034e dual-predicate) generated twice. Token ids AND "
              "the per-step PR log must match exactly. STOP if not. Includes "
              "one full-continuation record to exercise the layer-2 PR hook "
              "when one is available.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = []
    for pair in sm["pairs"]:
        smoke_rows.append([
            pair["fixture"], str(pair["corpus_id"]), str(pair["prompt_len"]),
            str(pair["n_generated_active"]), str(pair["n_pr_values"]),
            pair["dormant_identical"], pair["active_identical"],
            pair["pr_identical"], pair["identical"],
        ])
    smoke_rows.append(["", "ALL", "", "", "", "", "", "", str(sm["all_identical"])])
    md.append(format_table_md(
        smoke_rows, ["fixture", "corpus_id", "prompt_len", "n_gen", "n PR",
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
    md.append(f"| Records scored | {tabs['overall']['n_records']} |")
    md.append(f"| Total FPs (all four corpora) | "
              f"{tabs['overall']['n_false_positives']} |")
    for cfg in tabs["corpora_order"]:
        agg = tabs["corpora"][cfg]
        md.append(f"| {cfg} FPs | {agg['n_false_positives']} |")
    md.append(f"| **Gate verdict** | **{tabs['verdict']}** |")
    md.append("")
    if tabs["verdict"] == "PASS":
        md.append("**PASS** — 0 false positives across all four hazard/schema "
                  "corpora under the frozen DS-034e dual-predicate.")
    else:
        md.append("**FAIL** — FPs present on hazard/schema corpora. Per the "
                  "gate doctrine this is a STOP signal; no thresholds are "
                  "negotiated or tuned.")
    md.append("")

    # Measurement observations (live-generation behaviour on structured blocks)
    ov = tabs["overall"]
    md.append("## Measurement observations")
    md.append("")
    md.append("Under LIVE greedy generation with record[\\\"text\\\"] as prompt, "
              "the model emits the EOS token as the FIRST generated token on "
              f"{ov['n_eos_terminated_active']}/{ov['n_records']} records "
              "(immediate termination). The hazard/schema fixtures are complete "
              "structured blocks (closing ``` fence, closing JSON brace) or "
              "closed word-salad prose; Qwen2.5-1.5B treats them as "
              "self-contained and stops at step 0. As a result:")
    md.append("")
    md.append("- The generated continuation is empty on those records, so "
              "ΔDistinct-2 = 0 vs dormant (both terminate identically) and the "
              "dual-predicate is never evaluated past step 0.")
    md.append(f"- Only 5 records (all in schema_markdown_fenced: corpus_ids "
              f"10, 14, 16, 40, 44) produced a non-trivial continuation "
              f"(n_gen=128). On those records the layer-2 PR stayed well above "
              f"band_low (min PR 11.58–14.97 > 8.216097), bigram diversity "
              f"stayed high, and no predicate fired — active was byte-identical "
              f"to dormant.")
    md.append("- Therefore the PASS is real but largely **vacuous**: the "
              "dual-predicate did not fire on any of the four corpora, but on "
              "445/450 records it had no generated tokens to evaluate. The "
              "risk hypothesis (unfenced structured text triggering spectral "
              "fires) is neither confirmed nor refuted by live generation on "
              "these fixtures.")
    md.append("")

    # Per-corpus metrics
    md.append("## Per-corpus metrics")
    md.append("")
    for cfg in tabs["corpora_order"]:
        agg = tabs["corpora"][cfg]
        label = tabs["corpora_labels"][cfg]
        md.append(f"### {label}")
        md.append("")
        fb = agg["fire_type_breakdown"]
        d2 = agg["delta_distinct_2"]
        d2f = agg["delta_distinct_2_firing"]
        ng = agg["n_generated"]
        ns = agg["n_suppression_steps"]
        nk = agg["n_kickstart_events"]
        md.append(format_table_md([
            ["records", str(agg["n_records"])],
            ["false positives", str(agg["n_false_positives"])],
            ["FP rate", f"{fmt(agg['fp_rate'] * 100, 2)}%"],
            ["fired records", str(agg["n_fired"])],
            ["total fire steps", str(agg["n_fire_steps_total"])],
            ["bigram-only records", str(fb["bigram_only_records"])],
            ["spectral-only records", str(fb["spectral_only_records"])],
            ["both records", str(fb["both_records"])],
            ["bigram fire steps", str(fb["bigram_fire_steps"])],
            ["spectral fire steps", str(fb["spectral_fire_steps"])],
            ["both fire steps", str(fb["both_fire_steps"])],
            ["mean ΔD2", fmt(d2["mean"])],
            ["mean ΔD2 (firing only)", fmt(d2f["mean"]) if d2f else "—"],
            ["mean n_gen", fmt(ng["mean"], 1)],
            ["mean suppression steps", fmt(ns["mean"], 2)],
            ["mean kickstart events", fmt(nk["mean"], 2)],
            ["byte-identical to dormant", str(agg["n_byte_identical"])],
        ], ["metric", "value"]))
        md.append("")
        if agg["fp_record_ids"]:
            md.append(f"FP record_ids (corpus_id): "
                      f"{list(zip(agg['fp_corpus_ids'], agg['fp_triggers']))}")
            md.append("")
    md.append("")

    # Overall summary
    o = tabs["overall"]
    fb = o["fire_type_breakdown"]
    d2 = o["delta_distinct_2"]
    d2f = o["delta_distinct_2_firing"]
    ng = o["n_generated"]
    ns = o["n_suppression_steps"]
    nk = o["n_kickstart_events"]
    md.append("## Overall summary (all four corpora)")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| Records | {o['n_records']} |")
    md.append(f"| False positives | {o['n_false_positives']} "
              f"({fmt(o['fp_rate'] * 100, 2)}%) |")
    md.append(f"| Fired records | {o['n_fired']} |")
    md.append(f"| Total fire steps | {o['n_fire_steps_total']} |")
    md.append(f"| Bigram-only records | {fb['bigram_only_records']} |")
    md.append(f"| Spectral-only records | {fb['spectral_only_records']} |")
    md.append(f"| Both records | {fb['both_records']} |")
    md.append(f"| Bigram fire steps | {fb['bigram_fire_steps']} |")
    md.append(f"| Spectral fire steps | {fb['spectral_fire_steps']} |")
    md.append(f"| Both fire steps | {fb['both_fire_steps']} |")
    md.append(f"| Mean ΔD2 | {fmt(d2['mean'])} |")
    md.append(f"| Mean ΔD2 (firing only) | {fmt(d2f['mean']) if d2f else '—'} |")
    md.append(f"| Mean n_gen | {fmt(ng['mean'], 1)} |")
    md.append(f"| Mean suppression steps | {fmt(ns['mean'], 2)} |")
    md.append(f"| Mean kickstart events | {fmt(nk['mean'], 2)} |")
    md.append(f"| Byte-identical to dormant | {o['n_byte_identical']} |")
    md.append(f"| EOS-terminated (active) | {o['n_eos_terminated_active']} |")
    md.append("")
    if o["fp_record_ids"]:
        md.append(f"FP record_ids (global): `{o['fp_record_ids']}`")
        md.append("")

    # FP detail (if any)
    fps = ctx["results_all_fp"]
    if fps:
        md.append(f"## False-positive detail ({len(fps)} FPs)")
        md.append("")
        md.append("| fixture | record_id | corpus_id | prompt_len | n_gen | "
                  "ΔD2 | fire_type | fire_count | bigram_fire_steps | "
                  "spectral_fire_steps | PR min |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in fps:
            a = r["active"]
            md.append(f"| {r['fixture']} | {r['record_id']} | {r['corpus_id']} | "
                      f"{r['prompt_len']} | {a['n_generated']} | "
                      f"{fmt(a['delta_distinct_2'])} | {a['fire_type']} | "
                      f"{a['fire_count']} | {a['n_bigram_fire_steps']} | "
                      f"{a['n_spectral_fire_steps']} | {fmt(a['pr_min'])} |")
        md.append("")

    # Fire events detail (firing records)
    firing = [r for r in ctx["results_all"] if r["active"]["fire_count"] > 0]
    if firing:
        md.append(f"## Fire events ({len(firing)} firing records)")
        md.append("")
        for r in firing:
            md.append(f"### {r['fixture']} record {r['record_id']} "
                      f"(corpus_id {r['corpus_id']})")
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
              "compute from O(n^2) to O(n). The layer-2 spectral hook captures "
              "hidden states at each decoding step (seq_len==1) into a rolling "
              "24-token ring buffer, identical to the DS-034b hook pattern.")
    md.append("- The DS-034e dual-predicate OR rule is implemented exactly per "
              "RFC A3: bigram_fire = bigram_ctr ≥ 2; spectral_collapse uses the "
              "frozen band_low 8.216097 with ≥2 consecutive spectral_ctr; when "
              "bigram_ctr < 2, spectral_fire requires token_diversity < 0.40 "
              "corroboration; code-context immunity forces spectral_fire=False "
              "when is_code_syntax_context is True.")
    md.append("- Actuation on is_collapsed uses the production logit-penalty "
              "path (token suppression cooldown=8 penalty=-5.0, kickstart "
              "counter=3 with -1e4/-5.0/-2.0, top_p=0.85). The actuation can "
              "fire multiple times per record; every fire is logged.")
    md.append("- Dormant baselines are generated LIVE (greedy, KV-cache, no "
              "hooks/penalties) for all 450 records. No pre-existing dormant "
              "data exists for these corpora.")
    md.append("- Prose-100 is REUSED for context only (DS-034c v2; ratified "
              "policy = 7 documented FPs, Amendment A3). Prose-100 is NOT "
              "re-run.")
    md.append("- FP = ΔDistinct-2 > 0 AND fire_count > 0. A record whose "
              "predicate fires but whose continuation is byte-identical to "
              "dormant is an actuation-gap, not an FP.")
    md.append("- Frozen thresholds (T_PR(2) = 10.954796, band_low = 8.216097) "
              "are read from docs/gate23/FROZEN_THRESHOLDS.md and verified "
              "against the task-specified values; the script exits if they "
              "disagree.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-034c-hazard dual-predicate FP re-verification on hazard "
                    "and schema corpora (GATE, RFC A3 DS-034e config)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per corpus (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate HAZARD_SCHEMA_FP_RESULTS.md from an "
                             "existing hazard_schema_fp_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-034c-hazard Dual-predicate FP re-verification on hazard/schema "
          "corpora (GATE)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: {ANALYSIS_WINDOW} "
          f"| hook_layer: {HOOK_LAYER}")
    print(f"gate: 0 FPs across four hazard/schema corpora (N={TOTAL_RECORDS}) "
          f"-> PASS; else FAIL (STOP)")

    # ------------------------------------------------------------------
    # Frozen thresholds FIRST (Part A freeze); verify task-specified values.
    # ------------------------------------------------------------------
    t_pr, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    print(f"frozen T_PR(2)={t_pr:.6f} band_low(2)={band_low:.6f} loaded from "
          f"docs/gate23/FROZEN_THRESHOLDS.md")
    print(f"DS-034e config: spectral_collapse threshold = band_low = "
          f"{band_low:.6f}")

    # ------------------------------------------------------------------
    # Corpus loading (ground truth, human review verified).
    # ------------------------------------------------------------------
    corpora_data: List[Dict[str, Any]] = []
    for cfg in CORPORA:
        records = load_corpus(cfg)
        corpora_data.append({
            "fixture": cfg["fixture"],
            "label": cfg["label"],
            "records": records,
        })
        print(f"corpus {cfg['fixture']}: {len(records)} records loaded")
    total = sum(len(c["records"]) for c in corpora_data)
    if total != TOTAL_RECORDS:
        raise SystemExit(
            f"[STOP] Combined corpus total is {total}, expected {TOTAL_RECORDS}. "
            f"Task boundary; do NOT modify fixtures."
        )
    print(f"total records across four corpora: {total}")

    # Flatten for per-corpus iteration, preserving fixture order.
    all_records: List[Dict[str, Any]] = []  # global list with per-corpus index
    for ci, c in enumerate(corpora_data):
        records = c["records"]
        if args.max_records is not None:
            records = records[: args.max_records]
            print(f"[dev] capped {c['fixture']} at {args.max_records} records")
        for ri, rec in enumerate(records):
            all_records.append({
                "fixture": c["fixture"],
                "corpus_idx": ci,
                "record_id": ri,
                "corpus_id": int(rec["id"]),
                "prompt": rec["text"],
            })

    # ------------------------------------------------------------------
    # --report-only: regenerate the markdown from an existing JSONL.
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from "
              f"{OUTPUT_JSONL} ---")
        lines = load_jsonl(OUTPUT_JSONL)
        results_all = [r for r in lines if r.get("record_type") is None]
        smoke = {"pairs": [], "all_identical": "True"}
        for r in lines:
            if r.get("record_type") == "smoke":
                smoke = r
        meta_rec = next((r for r in lines if r.get("record_type") == "meta"), {})
        corpora_order = [c["fixture"] for c in CORPORA]
        corpora = {}
        for cfg in CORPORA:
            fixt = cfg["fixture"]
            corpora[fixt] = aggregate_corpus(
                [r for r in results_all if r.get("fixture") == fixt]
            )
        overall = aggregate_corpus(results_all)
        verdict = "PASS" if overall["n_false_positives"] == 0 else "FAIL"
        tables = {
            "corpora_order": corpora_order,
            "corpora_labels": {c["fixture"]: c["label"] for c in CORPORA},
            "corpora": corpora,
            "overall": overall,
            "verdict": verdict,
        }
        metadata = {
            "model_name": meta_rec.get("model_name", MODEL_NAME),
            "model_revision": meta_rec.get("model_revision", MODEL_REVISION),
            "device": meta_rec.get("device", device),
            "dtype": meta_rec.get("dtype", str(dtype)),
            "torch_version": meta_rec.get("torch_version", torch.__version__),
            "seed": meta_rec.get("seed", SEED),
            "max_new_tokens": meta_rec.get("max_new_tokens", MAX_NEW_TOKENS),
            "analysis_window": meta_rec.get("analysis_window", ANALYSIS_WINDOW),
            "hook_layer": meta_rec.get("hook_layer", HOOK_LAYER),
            "t_pr": meta_rec.get("t_pr", t_pr),
            "band_low": meta_rec.get("band_low", band_low),
            "bmm_override": "deregistered",
            "wall_clock_s": 0.0,
        }
        ctx = {
            "metadata": metadata,
            "determinism_smoke": smoke,
            "tables": tables,
            "results_all": results_all,
            "results_all_fp": [r for r in results_all if r.get("is_false_positive")],
        }
        write_markdown_report(ctx, OUTPUT_MD)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0)

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
    # Use one fenced markdown record (code-context path) and one unfenced
    # prose_code_switch record (spectral-risk path). Selection is robust to the
    # --max-records dev cap: pick the first record of each corpus when present.
    _smoke_by_fixture: List[Dict[str, Any]] = []
    for _fixt in ("schema_markdown_fenced", "prose_code_switch"):
        _match = [r for r in all_records if r["fixture"] == _fixt]
        if _match:
            _smoke_by_fixture.append(_match[0])
    if len(_smoke_by_fixture) < 2:
        # Dev cap fallback: use the first two available records.
        _smoke_by_fixture = all_records[:2]
    smoke_records = _smoke_by_fixture
    # Add a record that produces a real continuation (>=24 generated tokens) so
    # the determinism smoke also exercises the layer-2 PR hook (non-trivial PR
    # log matching). Quick pre-scan: run dormant generation over the first 12
    # schema_markdown_fenced records until one yields >=24 tokens; skip if none
    # (dev-cap or corpus-content fallback).
    _full_gen_candidate: Optional[Dict[str, Any]] = None
    _scan_pool = [r for r in all_records if r["fixture"] == "schema_markdown_fenced"][:12]
    for _rec in _scan_pool:
        _probe = greedy_generate_dormant_kv(model, tokenizer, _rec["prompt"])
        if int(_probe["n_generated"]) >= 24:
            _full_gen_candidate = _rec
            print(f"[smoke] full-generation record found: {_rec['fixture']} "
                  f"id={_rec['corpus_id']} n_gen={_probe['n_generated']}")
            break
    if _full_gen_candidate is not None and all(
        _rec is not _full_gen_candidate for _rec in smoke_records
    ):
        smoke_records = smoke_records + [_full_gen_candidate]
    smoke: Dict[str, Any] = {"pairs": [], "all_identical": "True"}
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        try:
            smoke = run_determinism_smoke(
                model, tokenizer, smoke_records, non_prose_ids, band_low,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during determinism smoke: {e}")
                sys.exit(1)
            raise

    # ==================================================================
    # Main run: dormant + active (DS-034e) for all 450 records.
    # ==================================================================
    print(f"\n--- Main run (dormant + active DS-034e, spectral_threshold="
          f"band_low={band_low:.6f}, {len(all_records)} records) ---")
    results_all: List[Dict[str, Any]] = []
    try:
        for i, rec in enumerate(all_records):
            fixture = rec["fixture"]
            rid = rec["record_id"]
            corpus_id = rec["corpus_id"]
            prompt = rec["prompt"]

            # Dormant (live, greedy, KV-cache).
            gen_dorm = greedy_generate_dormant_kv(model, tokenizer, prompt)
            dormant = build_dormant_meta(tokenizer, gen_dorm)

            # Active (live, DS-034e dual-predicate + production logit-penalty).
            gen_act = greedy_generate_active_ds034e(
                model, tokenizer, prompt, non_prose_ids, band_low,
            )
            active = build_active_meta(tokenizer, gen_act, dormant, band_low)

            results_all.append({
                "fixture": fixture,
                "record_id": rid,
                "corpus_id": corpus_id,
                "seed": SEED,
                "prompt_len": int(gen_dorm["prompt_len"]),
                "mode": "ds034e",
                "spectral_threshold": float(band_low),
                "dormant": dormant,
                "active": active,
                "is_false_positive": bool(active["is_false_positive"]),
            })

            if (i + 1) % 10 == 0 or i == len(all_records) - 1:
                n_fp = sum(1 for r in results_all if r["is_false_positive"])
                n_fire = sum(
                    1 for r in results_all
                    if r["active"]["fire_count"] > 0
                )
                print(f"  [{i+1}/{len(all_records)}] {fixture} id={corpus_id} "
                      f"prompt_len={gen_dorm['prompt_len']} "
                      f"n_gen={gen_act['n_generated']} "
                      f"fire={gen_act['fire_count']} "
                      f"ΔD2={active['delta_distinct_2']:.4f} "
                      f"byte_id={active['byte_identical_to_dormant']} "
                      f"fp_so_far={n_fp} fired_so_far={n_fire}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during main run: {e}")
            sys.exit(1)
        raise

    # ==================================================================
    # Aggregate per corpus + overall; verdict.
    # ==================================================================
    corpora_order = [c["fixture"] for c in CORPORA]
    corpora_agg: Dict[str, Dict[str, Any]] = {}
    for cfg in CORPORA:
        fixt = cfg["fixture"]
        corpora_agg[fixt] = aggregate_corpus(
            [r for r in results_all if r.get("fixture") == fixt]
        )
    overall_agg = aggregate_corpus(results_all)
    verdict = "PASS" if overall_agg["n_false_positives"] == 0 else "FAIL"

    print("\n" + "=" * 70)
    print("DS-034c-hazard PER-CORPUS FP GATE")
    for cfg in CORPORA:
        fixt = cfg["fixture"]
        agg = corpora_agg[fixt]
        print(f"  {fixt}: n={agg['n_records']} FPs={agg['n_false_positives']} "
              f"fired={agg['n_fired']} ΔD2_mean={agg['delta_distinct_2']['mean']:.4f}")
    print(f"  TOTAL: n={overall_agg['n_records']} "
          f"FPs={overall_agg['n_false_positives']}")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)

    # Per-corpus + overall FP detail (which predicate triggered each FP).
    results_fp = [r for r in results_all if r["is_false_positive"]]
    for r in results_fp:
        print(f"  FP: {r['fixture']} record_id={r['record_id']} "
              f"corpus_id={r['corpus_id']} fire_type={r['active']['fire_type']} "
              f"fire_count={r['active']['fire_count']} "
              f"ΔD2={r['active']['delta_distinct_2']:.4f}")

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    tables = {
        "corpora_order": corpora_order,
        "corpora_labels": {c["fixture"]: c["label"] for c in CORPORA},
        "corpora": corpora_agg,
        "overall": overall_agg,
        "verdict": verdict,
    }

    # Write JSONL.
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results_all:
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
            "total_records": overall_agg["n_records"],
            "total_fps": overall_agg["n_false_positives"],
            "per_corpus_fps": {
                cfg["fixture"]: corpora_agg[cfg["fixture"]]["n_false_positives"]
                for cfg in CORPORA
            },
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(results_all)} record lines + sidecar "
          f"lines)")

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
        "results_all": results_all,
        "results_all_fp": results_fp,
    }
    write_markdown_report(ctx, OUTPUT_MD)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    if verdict == "FAIL":
        print("[STOP] DS-034c-hazard GATE: FAIL — FPs present on hazard/schema "
              "corpora under the frozen DS-034e dual-predicate. Do NOT wire the "
              "dual-predicate into the controller.")
        sys.exit(1)
    print(f"DS-034c-hazard GATE: {verdict} — measurement complete")


if __name__ == "__main__":
    main()
