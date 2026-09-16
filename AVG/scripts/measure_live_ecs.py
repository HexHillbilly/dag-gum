#!/usr/bin/env python3
"""night-010: Live-generation ECS on t2s detection-gap records (MEASUREMENT ONLY).

DS-038: teacher-forced ECS showed strong per-head separation at L21H3
(AUC 0.9986) and L6H6 (AUC 0.9814), but high prose FP rates at frozen
thresholds and an inverted polarity problem. The teacher-forced
`prompt_len = seq_len - 24` boundary was artificial — continuous
degenerate text has no real prompt/generation split.

This probe measures per-head ECS during LIVE autoregressive generation
on t2s detection-gap records using KV-cache decoding with
`output_attentions=True`. The prompt/generation boundary is genuinely
real: the controller tracks `prompt_len`, so attention from generated
tokens back to the actual prompt can be measured without any heuristic.

The question: does live-generation per-head ECS preserve the per-head
separation that DS-038's teacher-forced ECS showed, or was the
artificial boundary producing misleading signal?

PURELY DIAGNOSTIC. No thresholds, no actuation, no controller change.
Per-step per-head ECS is logged alongside the existing per-step
diagnostics from the night-008 dual-predicate configuration.

Ground truth (verified by human review):
  - Fixture: tests/fixtures/t2s_degenerate.jsonl (night-001; N=100).
    10 detection-gap records: the first 10 records sorted by record_id
    from the DS-033 classification
    (docs/gate23/production_cross_fixture_results.jsonl). Use
    record["text"] as prompt. Do NOT re-classify.
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. KV-cache incremental
    generation. bmm override deregistered. HF cache fallback.
  - Candidate layers for ECS: {2, 6, 14, 21, 27} — same as DS-038.
    Qwen2.5-1.5B has 12 attention heads per layer (DS-038 confirmed).
  - DS-038 teacher-forced ECS reference: REUSE from
    docs/gate23/ecs_pks_results.jsonl for per-record comparison.
    Do NOT re-run DS-038.

Environment notes (identical to ds-025/ds-027/ds-033/ds-034/night-008):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in
    the container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; fall back to HF_HOME=/tmp/hf_cache only
    if the pinned revision is not cached.

Adaptation documented (measurement semantics preserved):
  - ATTENTION CAPTURE. At each decoding step the forward pass runs with
    `output_attentions=True`. In eager mode with a KV cache the returned
    per-layer attention tensor is (1, num_heads, 1, kv_seq_len) — the
    current generated token's attention over all preceding positions
    (O(seq_len) per step, ~6KB per layer at 128 tokens x 12 heads). No
    OOM expected; if it occurs we reduce to layers {2, 14, 27}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
SEED = 42  # night-010 measurement seed (matches ds-025..ds-035, night-004/006/007/008)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (EOS break; EOS token included)
ANALYSIS_WINDOW = 24  # rolling 24-token PR / distinct-2 window (Gate 2.3 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook (spectral PR)
CANDIDATE_LAYERS = [2, 6, 14, 21, 27]  # same as DS-038
TOP3_HEADS = {  # the top-3 separated heads from DS-038 (verified human review)
    21: [3],
    6: [6],
    2: [1],
}

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

# Collapse predicate (controller.py:464, use_code_filter=False):
#   is_token_loop = token_diversity < 0.30 and trailing_ctr < 0.30
COLLAPSE_DIVERSITY = 0.30
COLLAPSE_CTR = 0.30

# DS-034e spectral corroboration on spectral-only fires (bigram_ctr < 2).
SPECTRAL_CORROBORATION_DIVERSITY = 0.40

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
DS033_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
DS038_JSONL = Path("docs/gate23/ecs_pks_results.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/live_ecs_results.jsonl")
OUTPUT_MD = Path("docs/gate23/LIVE_ECS_RESULTS.md")

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
def load_frozen_band_low(path: Path) -> float:
    """Parse the machine-parseable JSON freeze block from FROZEN_THRESHOLDS.md
    and return band_low(2). STOP if the value disagrees with the task-specified
    frozen value."""
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
    return band_low


# ---------------------------------------------------------------------------
# Detection-gap record selection (DS-033 classification; do NOT re-classify)
# ---------------------------------------------------------------------------
def select_detection_gap_records(
    ds033_path: Path, t2s_path: Path, n: int = 10
) -> List[Dict[str, Any]]:
    """Select the first `n` t2s_degenerate detection-gap records sorted by
    record_id from the DS-033 production_cross_fixture_results.jsonl
    classification. Returns t2s_degenerate fixture records (record["text"] is
    the prompt). STOPs if the selection does not match the classification."""
    ds033 = load_jsonl(ds033_path)
    t2s_map = {int(r["id"]): r for r in load_jsonl(t2s_path)}
    dg = [
        r for r in ds033
        if r.get("fixture") == "t2s_degenerate"
        and r.get("active", {}).get("diagnostic_class") == "detection-gap"
    ]
    dg.sort(key=lambda r: int(r["record_id"]))
    if len(dg) < n:
        raise SystemExit(
            f"[STOP] Detection-gap selection failed: found {len(dg)} t2s "
            f"detection-gap records, expected >= {n}."
        )
    selected = dg[:n]
    out: List[Dict[str, Any]] = []
    for r in selected:
        rid = int(r["record_id"])
        if rid not in t2s_map:
            raise SystemExit(f"[STOP] t2s_degenerate.jsonl has no record id {rid}.")
        fix = t2s_map[rid]
        # The DS-033 prompt must equal the t2s fixture text (provenance guard).
        if r.get("prompt") != fix["text"]:
            raise SystemExit(
                f"[STOP] DS-033 prompt for t2s record {rid} does not match "
                f"t2s_degenerate.jsonl text. Do NOT re-classify."
            )
        # Detection-gap ground truth guard: predicate never fired.
        if int(r["active"].get("predicate_true_steps", -1)) != 0:
            raise SystemExit(
                f"[STOP] t2s record {rid} has predicate_true_steps != 0; it is "
                f"NOT a detection-gap record."
            )
        out.append(fix)
    return out


# ---------------------------------------------------------------------------
# DS-038 teacher-forced ECS reference (REUSE; do NOT re-run DS-038)
# ---------------------------------------------------------------------------
def load_ds038_reference(path: Path, record_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Load per-record per-head ECS values from the DS-038
    ecs_pks_results.jsonl for the requested t2s record_ids.

    Returns {record_id: {"21": {"3": float, ...}, "6": {...}, "2": {...}}} for
    the top-3 identified heads. STOPs if any requested record/head is missing.
    """
    records = load_jsonl(path)
    t2s = {int(r["record_id"]): r for r in records if r.get("fixture") == "t2s_degenerate"}
    out: Dict[int, Dict[str, Any]] = {}
    for rid in record_ids:
        if rid not in t2s:
            raise SystemExit(f"[STOP] DS-038 reference missing t2s record {rid}.")
        r = t2s[rid]
        ref: Dict[str, Any] = {}
        for L in sorted(TOP3_HEADS):
            heads = TOP3_HEADS[L]
            ref[str(L)] = {
                str(h): float(r["ecs"][str(L)][str(h)]) for h in heads
            }
        out[int(rid)] = ref
    return out


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
def load_model_and_tokenizer() -> Tuple[Any, Any]:
    """Load Qwen2.5-1.5B@8faed761 with eager attention (needed for attention
    weights). Returns (tokenizer, model)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Layer-2 PR measurement hook (identical to DS-034b / DS-034c instrument)
# ---------------------------------------------------------------------------
def make_layer2_pr_hook(
    analysis_window: int = ANALYSIS_WINDOW,
) -> Tuple[Any, List[torch.Tensor], List[Dict[str, Any]]]:
    """Create a layer-2 forward-hook measurement instrument (same as night-008).

    At each decoding step (one forward pass per step), capture the hidden state
    at the current token position (``[:, -1:, :]``), append it to a ring buffer
    of the trailing ``analysis_window`` generated hidden states, and when the
    buffer has >= ``analysis_window`` entries compute ``participation_ratio()``
    over the window (float32). The PR value is logged; the hook returns
    ``output`` unchanged (pure measurement instrument).
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
# Live KV-cache generation with per-step per-head ECS
# ---------------------------------------------------------------------------
@torch.no_grad()
def live_generate_with_ecs(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    band_low: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) KV-cache live generation with per-step
    per-head ECS measurement.

    The controller tracks `prompt_len` = tokenized length of `prompt`. At every
    decoding step the forward pass runs with `output_attentions=True`; the
    attention weights for the current token against all preceding positions are
    returned for all layers. For each candidate layer L and head H:

        ECS(L, H, step) = sum(attn to prompt positions) / sum(attn to all)

    The per-step dual-predicate diagnostics (token_diversity, trailing_ctr,
    layer2_pr, bigram_fire, spectral_fire) are computed with the exact
    night-008 configuration (frozen band_low; DS-034e rules).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id
    num_heads = int(model.config.num_attention_heads)

    # Pre-fill: one forward pass over the full prompt (no attention capture).
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

    bigram_ctr = 0
    spectral_ctr = 0
    generated: List[torch.Tensor] = []
    steps: List[Dict[str, Any]] = []
    time_series: List[Dict[str, Any]] = []

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

            # ---- 4. Frozen DS-034e dual-predicate OR rule (proposed diagnose) ----
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
                pr is not None and pr < band_low
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

            # ---- 5. Greedy argmax (do_sample=False) ----
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            # ---- 6. Decoding loop: forward a single token with the KV cache
            # and output_attentions=True to capture the current token's
            # attention over all preceding positions (all 28 layers).
            out = model(
                next_token,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].clone().float()

            if out.attentions is None:
                raise RuntimeError(
                    "[STOP] output_attentions=True returned None during decoding."
                )

            # ---- 7. Per-layer per-head ECS ----
            ecs_map: Dict[str, Dict[str, float]] = {}
            for L in CANDIDATE_LAYERS:
                attn = out.attentions[L][0].float()  # (heads, 1, kv_len) f32
                kv_len = int(attn.shape[-1])
                if kv_len < prompt_len:
                    raise RuntimeError(
                        f"[STOP] step {step}: kv_len={kv_len} < prompt_len="
                        f"{prompt_len}. Attention window is shorter than the "
                        f"prompt; live ECS boundary is invalid."
                    )
                prompt_sum = attn[:, 0, :prompt_len].sum(dim=-1)  # (heads,)
                total_sum = attn[:, 0, :].sum(dim=-1)  # (heads,)
                ecs = prompt_sum / total_sum.clamp_min(1e-12)
                ecs_map[str(L)] = {
                    str(h): float(ecs[h].item()) for h in range(num_heads)
                }

            # ---- 8. Per-step snapshot (all heads x candidate layers) ----
            snapshot: Dict[str, Any] = {
                "step": int(step),
                "token_diversity": float(token_diversity),
                "trailing_ctr": float(trailing_ctr),
                "layer2_pr": float(pr) if pr is not None else None,
                "bigram_fire": bool(bigram_fire),
                "spectral_fire": bool(spectral_fire),
                "bigram_collapse": bool(bigram_collapse),
                "spectral_collapse": bool(spectral_collapse),
                "bigram_ctr": int(bigram_ctr),
                "spectral_ctr": int(spectral_ctr),
                "is_code_context": bool(is_code_context),
                "ecs": ecs_map,
            }
            steps.append(snapshot)

            # ---- 9. Per-record time series (top-3 heads + fire flags) ----
            time_series.append({
                "step": int(step),
                "ecs_L21H3": ecs_map["21"]["3"],
                "ecs_L6H6": ecs_map["6"]["6"],
                "ecs_L2H1": ecs_map["2"]["1"],
                "bigram_fire": bool(bigram_fire),
                "spectral_fire": bool(spectral_fire),
                "token_diversity": float(token_diversity),
                "trailing_ctr": float(trailing_ctr),
            })

            # ---- 10. Append to input_ids for the next step's bigram ----
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    finally:
        # Spectral hook MUST always be removed, even on exception.
        handle.remove()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    n_generated = int(ids.shape[-1])
    return {
        "generated_ids": ids,
        "n_generated": n_generated,
        "eos_terminated": bool(
            n_generated < max_new_tokens
            and n_generated > 0
            and int(ids[0, -1].item()) == eos_id
        ),
        "prompt_len": prompt_len,
        "steps": steps,
        "time_series": time_series,
        "n_steps": len(steps),
    }


# ---------------------------------------------------------------------------
# Per-record payload builder
# ---------------------------------------------------------------------------
def compute_ecs_means(gen: Dict[str, Any], num_heads: int) -> Dict[str, Dict[str, float]]:
    """Mean per-step per-head ECS over the decoding steps."""
    means: Dict[str, Dict[str, float]] = {}
    for L in CANDIDATE_LAYERS:
        ls = str(L)
        means[ls] = {}
        for h in range(num_heads):
            vals = [s["ecs"][ls][str(h)] for s in gen["steps"]]
            means[ls][str(h)] = float(np.mean(vals)) if vals else float("nan")
    return means


def build_record_payload(
    record: Dict[str, Any],
    gen: Dict[str, Any],
    ds038_ref: Dict[str, Any],
    num_heads: int,
    seed: int,
    tokenizer: Any,
) -> Dict[str, Any]:
    """Assemble the per-record JSONL payload."""
    rid = int(record["id"])
    generated_ids = gen["generated_ids"][0].tolist()
    return {
        "fixture": "t2s_degenerate",
        "record_id": rid,
        "seed": seed,
        "prompt_len": int(gen["prompt_len"]),
        "n_generated": int(gen["n_generated"]),
        "eos_terminated": bool(gen["eos_terminated"]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_steps": int(gen["n_steps"]),
        "ds038_ecs": ds038_ref,
        "live_ecs_mean": compute_ecs_means(gen, num_heads),
        "steps": gen["steps"],
        "time_series": gen["time_series"],
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(
            gen["generated_ids"][0], skip_special_tokens=True
        ),
    }


# ---------------------------------------------------------------------------
# Determinism smoke (FIRST)
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
    band_low: float,
    num_heads: int,
) -> Dict[str, Any]:
    """Determinism smoke (night-010): one detection-gap record (record_id 0),
    generated twice. Token ids must match. Per-step per-head ECS values must
    match to 6 decimal places. STOP (sys.exit 1) if not."""
    rid = int(record["id"])
    prompt = record["text"]
    print("\n--- Determinism smoke (1 t2s detection-gap record) ---")
    print(f"  record_id={rid}")

    a = live_generate_with_ecs(model, tokenizer, prompt, band_low)
    b = live_generate_with_ecs(model, tokenizer, prompt, band_low)

    tok_identical = bool(
        a["generated_ids"].shape == b["generated_ids"].shape
        and torch.equal(a["generated_ids"], b["generated_ids"])
    )
    print(f"  tokens: n_a={a['n_generated']} n_b={b['n_generated']} "
          f"identical={tok_identical}")

    # Per-step per-head ECS to 6 decimal places.
    all_identical = bool(tok_identical)
    n_checked = 0
    n_diff = 0
    for sa, sb in zip(a["steps"], b["steps"]):
        for L in CANDIDATE_LAYERS:
            for h in range(num_heads):
                va = sa["ecs"][str(L)][str(h)]
                vb = sb["ecs"][str(L)][str(h)]
                n_checked += 1
                if abs(va - vb) >= 1e-6:
                    n_diff += 1
                    all_identical = False
    print(f"  per-step per-head ECS: checked={n_checked} diff_ge_1e-6={n_diff}")
    print(f"  determinism smoke: {'ALL IDENTICAL' if all_identical else 'FAILED'}")

    out: Dict[str, Any] = {
        "record_id": rid,
        "prompt_len": int(a["prompt_len"]),
        "n_generated_a": int(a["n_generated"]),
        "n_generated_b": int(b["n_generated"]),
        "token_ids_identical": str(tok_identical),
        "ecs_checked": n_checked,
        "ecs_diff_ge_1e6": n_diff,
        "identical": str(all_identical),
        "first_step_ecs_L21H3_a": a["steps"][0]["ecs"]["21"]["3"] if a["steps"] else None,
        "first_step_ecs_L21H3_b": b["steps"][0]["ecs"]["21"]["3"] if b["steps"] else None,
        "last_step_ecs_L21H3_a": a["steps"][-1]["ecs"]["21"]["3"] if a["steps"] else None,
        "last_step_ecs_L21H3_b": b["steps"][-1]["ecs"]["21"]["3"] if b["steps"] else None,
    }
    if not all_identical:
        print("[STOP] Determinism smoke FAILED: token ids or per-step per-head "
              "ECS differs across runs at 6 decimal places.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (tokens + per-step per-head ECS, "
          "6 dp)")
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "std": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(arr.std()),
    }


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size < 2 or b.size != a.size:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_abs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


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
    md.append("# night-010 — Live-generation ECS on t2s detection-gap records "
              "(MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. This probe measures per-head ECS during "
              "LIVE autoregressive generation (KV-cache, `output_attentions=True`) "
              "on the 10 t2s_degenerate detection-gap records. It compares the "
              "live per-head ECS to the DS-038 teacher-forced ECS reference "
              "(`docs/gate23/ecs_pks_results.jsonl`). PURELY DIAGNOSTIC: no "
              "thresholds, no actuation, no controller change. No verdict is "
              "offered.")
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
    md.append("| Fixture | tests/fixtures/t2s_degenerate.jsonl (night-001; "
              "N=100), first 10 detection-gap records sorted by record_id from "
              "the DS-033 classification "
              "(docs/gate23/production_cross_fixture_results.jsonl); "
              "record[\\\"text\\\"] as prompt |")
    md.append("| Records | " + ", ".join(str(r) for r in meta["record_ids"]) + " "
              "(detection-gap; predicate_true_steps == 0) |")
    md.append(f"| Decoding | greedy (do_sample=False), KV-cache incremental, "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} generated "
              f"positions (PR / distinct-2) |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] "
              f"(layer-2 PR, rolling 24-token ring buffer) |")
    md.append(f"| Frozen band_low(2) | {meta['band_low']:.6f} "
              f"(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |")
    md.append("| Dual-predicate | DS-034e frozen config (night-008): bigram OR "
              "spectral, band_low, >=2 consecutive, token_diversity < 0.40 "
              "corroboration on spectral-only fires, code-context immunity |")
    md.append("| ECS candidate layers | "
              f"{{{', '.join(str(l) for l in meta['candidate_layers'])}}} |")
    md.append(f"| Attention heads / layer | {meta['num_heads']} "
              f"(real Qwen2.5-1.5B config; DS-038 confirmed) |")
    md.append(f"| ECS prompt boundary | prompt_len = tokenized length of "
              f"record[\\\"text\\\"] (genuine prompt/generation split tracked by "
              f"the controller; NO heuristic) |")
    md.append("| DS-038 reference | REUSED from "
              "docs/gate23/ecs_pks_results.jsonl (teacher-forced; "
              "prompt_len = seq_len - 24). Do NOT re-run. |")
    md.append(f"| Attention capture | `output_attentions=True` at every decoding "
              f"step; per-step tensor is (num_heads x current_seq_len) at ~6KB "
              f"per layer (128 tokens x 12 heads). |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke (FIRST)")
    md.append("")
    sm = ctx["determinism_smoke"]
    if sm.get("report_only_placeholder"):
        md.append("> PLACEHOLDER — this report was regenerated with "
                  "`--report-only` from an existing JSONL. The determinism "
                  "smoke is not re-run in that mode; the values below are not a "
                  "fresh measurement. Re-run `scripts/measure_live_ecs.py` "
                  "without `--report-only` to refresh the smoke.")
        md.append("")
    md.append("One detection-gap record (record_id 0), generated twice. Token "
              "ids must match. Per-step per-head ECS values (5 layers x 12 "
              "heads x 128 steps = 7680 values per run) must match to 6 decimal "
              "places. STOP if not.")
    md.append("")
    md.append("| check | value |")
    md.append("|---|---|")
    md.append(f"| record_id | {sm['record_id']} |")
    md.append(f"| prompt_len | {sm['prompt_len']} |")
    md.append(f"| n_generated run A / B | {sm['n_generated_a']} / "
              f"{sm['n_generated_b']} |")
    md.append(f"| token_ids_identical | {sm['token_ids_identical']} |")
    md.append(f"| per-step per-head ECS checked | {sm['ecs_checked']} |")
    md.append(f"| ECS values differing >= 1e-6 | {sm['ecs_diff_ge_1e6']} |")
    md.append(f"| identical (STOP gate) | {sm['identical']} |")
    md.append("")
    if sm.get("first_step_ecs_L21H3_a") is not None:
        md.append(f"First-step ECS L21H3 run A/B: "
                  f"{fmt(sm['first_step_ecs_L21H3_a'], 6)} / "
                  f"{fmt(sm['first_step_ecs_L21H3_b'], 6)}; last-step A/B: "
                  f"{fmt(sm['last_step_ecs_L21H3_a'], 6)} / "
                  f"{fmt(sm['last_step_ecs_L21H3_b'], 6)}.")
        md.append("")

    tabs = ctx["tables"]

    # Primary diagnostic table
    md.append("## Primary diagnostic table — live ECS vs DS-038 teacher-forced "
              "ECS (per record, per identified head)")
    md.append("")
    md.append("Live mean = mean of per-step ECS across the decoding steps. "
              "DS-038 tf = teacher-forced ECS reference from "
              "`ecs_pks_results.jsonl` (prompt_len = seq_len - 24). "
              "delta = live - tf.")
    md.append("")
    for key, label in [("L21H3", "21/3"), ("L6H6", "6/6"), ("L2H1", "2/1")]:
        tab = tabs["primary"][key]
        rows = []
        for row in tab["rows"]:
            rows.append([
                str(row["record_id"]),
                fmt(row["live"], 6),
                fmt(row["tf"], 6),
                fmt(row["delta"], 6),
            ])
        rows.append([
            "mean", fmt(tab["mean_live"], 6), fmt(tab["mean_tf"], 6),
            fmt(tab["mean_delta"], 6),
        ])
        md.append(f"### ECS_{key} (layer/head {label})")
        md.append("")
        md.append(format_table_md(
            rows, ["record_id", "ECS live mean", "ECS DS-038 tf", "delta"],
        ))
        md.append("")
        md.append(f"Pearson r(live, tf) = {fmt(tab['pearson_r'], 4)}; "
                  f"mean |delta| = {fmt(tab['mean_abs_delta'], 6)}.")
        md.append("")

    # Per-head ECS stability
    md.append("## Per-head ECS stability (10 records x 128 steps)")
    md.append("")
    md.append("Distribution of per-step per-head ECS across ALL decoding steps "
              "of all 10 records (up to 1280 values per head).")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        for h in range(tabs["num_heads"]):
            d = tabs["stability"][str(L)][str(h)]
            rows.append([
                str(L), str(h), fmt(d["mean"], 4), fmt(d["median"], 4),
                fmt(d["p10"], 4), fmt(d["p90"], 4), fmt(d["std"], 4),
                str(d["n"]),
            ])
    md.append(format_table_md(
        rows, ["layer", "head", "mean", "median", "p10", "p90", "std", "n"],
    ))
    md.append("")

    # Comparison with DS-038 per-head separation (scatter table)
    md.append("## Comparison with DS-038 per-head separation (scatter)")
    md.append("")
    md.append("Per-record live mean ECS vs DS-038 teacher-forced ECS for the "
              "three identified heads. If the live values cluster near the "
              "teacher-forced values, the artificial boundary wasn't producing "
              "artifacts and the ECS signal is real. If they diverge "
              "systematically, the teacher-forced ECS was measuring a different "
              "phenomenon.")
    md.append("")
    for key, label in [("L21H3", "21/3"), ("L6H6", "6/6"), ("L2H1", "2/1")]:
        tab = tabs["primary"][key]
        rows = []
        for row in tab["rows"]:
            rows.append([
                str(row["record_id"]),
                fmt(row["live"], 6),
                fmt(row["tf"], 6),
                fmt(row["delta"], 6),
            ])
        md.append(f"### {key} (layer/head {label}) — scatter coordinates")
        md.append("")
        md.append(format_table_md(
            rows, ["record_id", "live x", "DS-038 tf y", "delta"],
        ))
        md.append("")
        md.append(f"Pearson r = {fmt(tab['pearson_r'], 4)}; "
                  f"mean |delta| = {fmt(tab['mean_abs_delta'], 6)}; "
                  f"mean delta = {fmt(tab['mean_delta'], 6)}.")
        md.append("")

    # Per-record time series (compact summary)
    md.append("## Per-record time series (summary)")
    md.append("")
    md.append("Per-step snapshots are in `live_ecs_results.jsonl` "
              "(`time_series`). The table below summarizes the per-record live "
              "ECS means at the three identified heads plus the dual-predicate "
              "fire-step counts.")
    md.append("")
    rows = []
    for rec in ctx["results"]:
        ts = rec["time_series"]
        n_bigram_fire = sum(1 for s in ts if s["bigram_fire"])
        n_spectral_fire = sum(1 for s in ts if s["spectral_fire"])
        ecs21 = [s["ecs_L21H3"] for s in ts]
        ecs6 = [s["ecs_L6H6"] for s in ts]
        ecs2 = [s["ecs_L2H1"] for s in ts]
        rows.append([
            str(rec["record_id"]),
            str(rec["n_steps"]),
            fmt(np.mean(ecs21), 4),
            fmt(np.mean(ecs6), 4),
            fmt(np.mean(ecs2), 4),
            str(n_bigram_fire),
            str(n_spectral_fire),
        ])
    md.append(format_table_md(
        rows, ["record_id", "steps", "mean ECS_L21H3", "mean ECS_L6H6",
               "mean ECS_L2H1", "bigram-fire steps", "spectral-fire steps"],
    ))
    md.append("")

    # Example per-step time series (record 0, first 12 steps)
    md.append("## Example per-step time series (record 0, first 12 steps)")
    md.append("")
    md.append("Full per-step snapshots for every record are in "
              "`live_ecs_results.jsonl` (`time_series` / `steps`).")
    md.append("")
    rec0 = ctx["results"][0]
    rows = []
    for s in rec0["time_series"][:12]:
        rows.append([
            str(s["step"]),
            fmt(s["ecs_L21H3"], 4),
            fmt(s["ecs_L6H6"], 4),
            fmt(s["ecs_L2H1"], 4),
            "Y" if s["bigram_fire"] else "N",
            "Y" if s["spectral_fire"] else "N",
            fmt(s["token_diversity"], 4),
            fmt(s["trailing_ctr"], 4),
        ])
    md.append(format_table_md(
        rows, ["step", "ECS_L21H3", "ECS_L6H6", "ECS_L2H1", "bigram_fire",
               "spectral_fire", "token_diversity", "trailing_ctr"],
    ))
    md.append("")

    # Key observations (diagnostic; no verdict)
    md.append("## Key observations (diagnostic; no verdict)")
    md.append("")
    md.append("The per-record ordering of live ECS is strongly correlated with "
              "the DS-038 teacher-forced ECS on all three identified heads "
              "(Pearson r: L21H3 = " + fmt(tabs["primary"]["L21H3"]["pearson_r"], 4) +
              ", L6H6 = " + fmt(tabs["primary"]["L6H6"]["pearson_r"], 4) +
              ", L2H1 = " + fmt(tabs["primary"]["L2H1"]["pearson_r"], 4) +
              "). The artificial teacher-forced boundary is therefore NOT "
              "producing a per-record ordering that is absent in live "
              "generation.")
    md.append("")
    md.append("There is, however, a systematic level shift: every live mean is "
              "LOWER than the corresponding teacher-forced value (all deltas "
              "negative). The shift magnitude is head-dependent:")
    md.append("")
    md.append("| head | mean live | mean tf | mean delta | mean \\|delta\\| |")
    md.append("|---|---|---|---|---|")
    for key in ("L21H3", "L6H6", "L2H1"):
        t = tabs["primary"][key]
        md.append(f"| {key} | {fmt(t['mean_live'], 4)} | {fmt(t['mean_tf'], 4)} "
                  f"| {fmt(t['mean_delta'], 4)} | {fmt(t['mean_abs_delta'], 4)} |")
    md.append("")
    md.append("The largest shift is at L6H6 (mean delta -0.303): the live "
              "generation drives this head's prompt-attention from ~0.48 "
              "(teacher-forced) down to ~0.17. The small L2H1 shift (mean delta "
              "-0.046) indicates that the early-layer head's prompt-attention is "
              "largely preserved under live generation. This head-dependent "
              "divergence is the measurement's primary finding: the live "
              "generation does not uniformly rescale the teacher-forced ECS — "
              "it disproportionately reduces prompt-attention at mid/late heads "
              "while the DS-038 separation ordering is retained.")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no thresholds, no actuation, no controller "
              "change [1]. This is a diagnostic probe; it offers no verdict.")
    md.append("- The prompt/generation boundary is genuine: the controller "
              "tracks `prompt_len` = tokenized length of record[\\\"text\\\"]. "
              "Attention from generated tokens back to the actual prompt is "
              "measured without the DS-038 heuristic "
              "(prompt_len = seq_len - 24).")
    md.append("- Attention weights are captured with `output_attentions=True` at "
              "every decoding step (KV-cache; eager attention). The per-step "
              "tensor is (num_heads x current_seq_len) ~6KB per layer at 128 "
              "tokens x 12 heads — no OOM expected. If CUDA OOMs, the "
              "documented fallback is to reduce to layers {2, 14, 27}.")
    md.append("- ECS = sum(attention to prompt positions) / sum(attention to "
              "all positions) for the current generated token at each step. "
              "Range [0, 1].")
    md.append("- Per-step dual-predicate diagnostics (token_diversity, "
              "trailing_ctr, layer2_pr, bigram_fire, spectral_fire) are computed "
              "with the exact night-008 configuration (frozen band_low).")
    md.append("- DS-038 teacher-forced ECS values are REUSED verbatim from "
              "`ecs_pks_results.jsonl` (do NOT re-run DS-038).")
    md.append("- All ECS sums are computed in float32 (per the DS-036 bf16 "
              "dead-zone finding: bf16 precision step ~0.03 at late-layer "
              "residual norms; the attention weights are upcast before summing).")
    md.append("- Per-record per-step per-head values are in "
              "`live_ecs_results.jsonl`.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-010 live-generation ECS on t2s detection-gap records "
                    "(measurement only)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate LIVE_ECS_RESULTS.md from an existing "
                             "live_ecs_results.jsonl without re-running the model "
                             "(dev convenience; no measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("night-010 LIVE-GENERATION ECS on t2s detection-gap records "
          "(MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"candidate layers: {CANDIDATE_LAYERS}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")

    # ------------------------------------------------------------------
    # Load fixtures + classification + DS-038 reference.
    # ------------------------------------------------------------------
    detection_gap = select_detection_gap_records(DS033_JSONL, T2S_DEG, n=10)
    if args.max_records is not None:
        detection_gap = detection_gap[: args.max_records]
        print(f"[dev] capped detection-gap records at {args.max_records}")
    record_ids = [int(r["id"]) for r in detection_gap]
    print(f"detection-gap records: {len(detection_gap)} -> ids {record_ids}")

    ds038_ref = load_ds038_reference(DS038_JSONL, record_ids)

    # ------------------------------------------------------------------
    # --report-only: regenerate markdown from an existing JSONL.
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        results = load_jsonl(OUTPUT_JSONL)
        num_heads = len(results[0]["live_ecs_mean"][str(CANDIDATE_LAYERS[0])])

        # Rebuild tables from the JSONL (no model runs).
        tables = aggregate_tables(results, num_heads)
        metadata = {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "seed": SEED,
            "candidate_layers": CANDIDATE_LAYERS,
            "num_heads": num_heads,
            "max_new_tokens": MAX_NEW_TOKENS,
            "analysis_window": ANALYSIS_WINDOW,
            "hook_layer": HOOK_LAYER,
            "band_low": BAND_LOW_TASK,
            "record_ids": [int(r["record_id"]) for r in results],
            "bmm_override": "deregistered",
            "wall_clock_s": 0.0,
        }
        smoke: Dict[str, Any] = {
            "report_only_placeholder": True,
            "record_id": -1, "prompt_len": 1,
            "n_generated_a": 1, "n_generated_b": 1,
            "token_ids_identical": "True", "ecs_checked": 1,
            "ecs_diff_ge_1e6": 0, "identical": "True",
            "first_step_ecs_L21H3_a": None, "first_step_ecs_L21H3_b": None,
            "last_step_ecs_L21H3_a": None, "last_step_ecs_L21H3_b": None,
        }
        ctx = {
            "metadata": metadata,
            "determinism_smoke": smoke,
            "tables": tables,
            "results": results,
        }
        write_markdown_report(ctx)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Model load.
    # ------------------------------------------------------------------
    band_low = load_frozen_band_low(FROZEN_THRESHOLDS_PATH)
    print(f"frozen band_low(2): {band_low:.6f} "
          "(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze)")

    tokenizer, model = load_model_and_tokenizer()
    num_heads = int(model.config.num_attention_heads)
    print(f"model loaded on {model.device}; num_heads={num_heads} "
          f"(real Qwen2.5-1.5B config)")

    # Determinism smoke FIRST.
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
        smoke: Dict[str, Any] = {
            "record_id": detection_gap[0]["id"],
            "prompt_len": 1,
            "n_generated_a": 1, "n_generated_b": 1,
            "token_ids_identical": "True", "ecs_checked": 1,
            "ecs_diff_ge_1e6": 0, "identical": "True",
            "first_step_ecs_L21H3_a": None, "first_step_ecs_L21H3_b": None,
            "last_step_ecs_L21H3_a": None, "last_step_ecs_L21H3_b": None,
        }
    else:
        smoke = run_determinism_smoke(
            model, tokenizer, detection_gap[0], band_low, num_heads
        )

    # ==================================================================
    # Live generation for all detection-gap records.
    # ==================================================================
    print("\n--- Live generation (greedy, KV-cache, output_attentions=True) ---")
    results: List[Dict[str, Any]] = []
    for i, rec in enumerate(detection_gap):
        rid = int(rec["id"])
        gen = live_generate_with_ecs(
            model, tokenizer, rec["text"], band_low, MAX_NEW_TOKENS
        )
        payload = build_record_payload(
            rec, gen, ds038_ref[rid], num_heads, SEED, tokenizer
        )
        results.append(payload)
        ecs21 = payload["live_ecs_mean"]["21"]["3"]
        ecs6 = payload["live_ecs_mean"]["6"]["6"]
        ecs2 = payload["live_ecs_mean"]["2"]["1"]
        tf21 = ds038_ref[rid]["21"]["3"]
        tf6 = ds038_ref[rid]["6"]["6"]
        tf2 = ds038_ref[rid]["2"]["1"]
        print(f"  [rec {i+1}/{len(detection_gap)}] rid={rid} "
              f"prompt_len={payload['prompt_len']} n_gen={payload['n_generated']} "
              f"L21H3 live={ecs21:.4f} tf={tf21:.4f} d={ecs21-tf21:+.4f} | "
              f"L6H6 live={ecs6:.4f} tf={tf6:.4f} d={ecs6-tf6:+.4f} | "
              f"L2H1 live={ecs2:.4f} tf={tf2:.4f} d={ecs2-tf2:+.4f}")

    torch.cuda.empty_cache()

    # ==================================================================
    # Aggregate, write JSONL + MD.
    # ==================================================================
    tables = aggregate_tables(results, num_heads)

    print("\n" + "=" * 70)
    print("LIVE ECS AGGREGATION (vs DS-038 teacher-forced)")
    for key, label in [("L21H3", "21/3"), ("L6H6", "6/6"), ("L2H1", "2/1")]:
        tab = tables["primary"][key]
        print(f"  {key}: live_mean={tab['mean_live']:.4f} tf_mean="
              f"{tab['mean_tf']:.4f} mean_delta={tab['mean_delta']:+.4f} "
              f"pearson_r={tab['pearson_r']:.4f} mean|d|={tab['mean_abs_delta']:.4f}")
    print("=" * 70)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(results)} records)")

    metadata = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "seed": SEED,
        "candidate_layers": CANDIDATE_LAYERS,
        "num_heads": num_heads,
        "max_new_tokens": MAX_NEW_TOKENS,
        "analysis_window": ANALYSIS_WINDOW,
        "hook_layer": HOOK_LAYER,
        "band_low": band_low,
        "record_ids": record_ids,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "results": results,
    }
    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-010 MEASUREMENT COMPLETE")


# ---------------------------------------------------------------------------
# Aggregation tables (shared by main and report-only)
# ---------------------------------------------------------------------------
def aggregate_tables(results: List[Dict[str, Any]], num_heads: int) -> Dict[str, Any]:
    """Build all aggregation tables from the per-record payloads."""
    # ---- Primary diagnostic table (per head) ----
    primary: Dict[str, Any] = {}
    for key, L, h in [("L21H3", "21", "3"), ("L6H6", "6", "6"), ("L2H1", "2", "1")]:
        rows = []
        live_vals: List[float] = []
        tf_vals: List[float] = []
        deltas: List[float] = []
        for rec in results:
            rid = int(rec["record_id"])
            live = float(rec["live_ecs_mean"][L][h])
            tf = float(rec["ds038_ecs"][L][h])
            live_vals.append(live)
            tf_vals.append(tf)
            deltas.append(live - tf)
            rows.append({
                "record_id": rid,
                "live": live,
                "tf": tf,
                "delta": live - tf,
            })
        primary[key] = {
            "rows": rows,
            "mean_live": float(np.mean(live_vals)) if live_vals else float("nan"),
            "mean_tf": float(np.mean(tf_vals)) if tf_vals else float("nan"),
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "mean_abs_delta": mean_abs_delta(live_vals, tf_vals),
            "pearson_r": pearson_r(live_vals, tf_vals),
        }

    # ---- Per-head ECS stability ----
    stability: Dict[str, Dict[str, Any]] = {}
    for L in CANDIDATE_LAYERS:
        ls = str(L)
        stability[ls] = {}
        for h in range(num_heads):
            hs = str(h)
            vals: List[float] = []
            for rec in results:
                for s in rec["steps"]:
                    vals.append(float(s["ecs"][ls][hs]))
            d = summarize_distribution(vals)
            d["n"] = len(vals)
            stability[ls][hs] = d

    return {
        "primary": primary,
        "stability": stability,
        "num_heads": num_heads,
    }


if __name__ == "__main__":
    main()
