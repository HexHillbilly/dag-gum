#!/usr/bin/env python3
"""DS-030: Logit-penalty decomposition — suppression vs kickstart (MEASUREMENT ONLY).

DS-029 measured the combined logit-penalty path (token suppression + kickstart)
and found it rescues all 50 heldout-degenerate v2 records under greedy decoding,
with the residual path contributing zero marginal effect. But DS-029's condition B
bundled BOTH logit mechanisms. This probe decomposes them:

  Condition B  — Combined (RUN LIVE): token suppression + kickstart, no residual
                 hooks. Production logit-penalty path (controller.py generate(),
                 steps 1/2/4; NO residual hooks, NO diagnose()).
  Condition D  — Token-suppression-only (NEW): NO kickstart, NO residual hooks.
  Condition E  — Kickstart-only (NEW): NO token suppression, NO residual hooks.

Dormant is REUSED from DS-028 (docs/gate23/layer_effect_persistent_results.jsonl;
join on record_id; do NOT re-run dormant).

All three live conditions (B, D, E) run in THIS process on the same model load
(eliminates cross-session drift). Greedy decoding (do_sample=False),
max_new_tokens=128, prompt = record["mutated_prompt"]. SEED=42.

Determinism smoke FIRST: 1 record, each live condition, twice — token ids must
match exactly; else STOP. If CUDA OOMs: STOP.

MEASUREMENT ONLY. No threshold is created or modified, no gate script is touched,
no governor/controller.py edit, no assert on outcomes, and no green/red verdict
and no mechanism story are offered.

Production logit-penalty budget (ground truth, controller.py):
  cooldown_steps=8             (controller.py:137)
  suppression penalty -5.0     (controller.py:745)
  kickstart penalties -1e4/-5.0/-2.0  (controller.py:753-758)
  top_p=0.85                   (controller.py:138, 663)
  kickstart trigger trailing_ctr < 0.50 AND active_loop_ids (controller.py:691)
  CTR sub-sampling every_k=2, or token_diversity < 0.40 (controller.py:136, 683)
  trailing generated window 16 (controller.py:686)

Condition B follows the production sub-sampling (every_k=2). Condition E computes
trailing_ctr at EVERY step (per the DS-030 task spec), which is a slightly more
sensitive kickstart trigger than production. Condition D needs no CTR at all.

top_p (controller.py:762-763) is applied when the suppression dict is non-empty
OR the post-decrement kickstart counter is > 0, exactly matching production:
on the third kickstart step (counter 1 -> 0), the -2.0 kickstart penalty is
applied but top_p is not.

Environment notes (identical to ds-025/ds-026/ds-028):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; the pinned model is ALREADY present in the
    read-only cache, so no HF_HOME fallback is forced (a fallback to /tmp would
    trigger a re-download of the 1.5B weights). If the model were missing, the
    loader would fall back to /tmp/hf_cache.
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
SEED = 42  # DS-030 measurement seed (matches ds-025 / ds-026 / ds-027 / ds-028)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128
ANALYSIS_WINDOW = 24  # trailing generated positions for distinct-2
NUM_RECORDS = 50

# Production logit-penalty budget (controller.py).
COOLDOWN_STEPS = 8            # controller.py:137 (cooldown_steps=8)
SUPPRESS_PENALTY = -5.0       # controller.py:745 (next_logits[:, tid] -= 5.0)
KICKSTART_PENALTIES: Dict[int, float] = {3: -1e4, 2: -5.0, 1: -2.0}  # controller.py:753-758
TOP_P = 0.85                  # controller.py:138 (top_p=0.85)
CTR_THRESHOLD = 0.50          # controller.py:691 (trailing_ctr < 0.50)
DIVERSITY_CTR_GATE = 0.40     # controller.py:683 (token_diversity < 0.40)
EVERY_K = 2                   # controller.py:136 (every_k=2)
TRAILING_GEN_WINDOW = 16      # controller.py:686 (trailing 16 generated tokens)
KICKSTART_COUNTER_INIT = 3    # controller.py:692

# Paths
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
DS028_JSONL = Path("docs/gate23/layer_effect_persistent_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/penalty_decomposition_results.jsonl")
OUTPUT_MD = Path("docs/gate23/PENALTY_DECOMPOSITION_RESULTS.md")

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
                    "source": "ds028_layer_effect_persistent_results.jsonl",
                }
    return out


# ---------------------------------------------------------------------------
# Vocabulary subsets (production _cache_vocabulary_subsets logic)
# ---------------------------------------------------------------------------
def cache_vocabulary_subsets(
    tokenizer: Any,
) -> Tuple[Set[int], Set[int]]:
    """Production _cache_vocabulary_subsets() logic (controller.py:158-175).

    Iterate the tokenizer vocab (151,643 ids for Qwen2.5-1.5B); classify each
    id as prose (single a/I/A or len>=2 alpha-only with a vowel) or non-prose.
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
# Greedy generation under a logit-penalty condition
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_condition(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    condition: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    non_prose_ids: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) generation under a logit-penalty condition.

    condition:
      "B"  combined: token suppression + kickstart, production sub-sampling
           for trailing-CTR (every_k=2 or diversity<0.40), no residual hooks.
      "D"  token-suppression only (NO kickstart, NO vocab steering).
      "E"  kickstart only (NO token suppression; trailing-CTR every step).

    The loop mirrors the controller's logit-penalty path (controller.py
    generate(), steps 1/2/4) with EOS-termination (the controller itself does
    not break on EOS; DS-029's measurement loop did, and the early-EOS pattern
    is a first-class outcome here). Penalties are applied before argmax at
    every step, including the prefill step (step 0).
    """
    assert condition in ("B", "D", "E"), condition
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

    for controller_step in range(max_new_tokens):
        # ---- 1. fast integer-only token proxy (controller.py:672-679) ----
        token_diversity, active_loop_ids = compute_token_distinct_2_fast(
            input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
        )

        if condition in ("B", "D") and active_loop_ids:
            for tid in active_loop_ids:
                active_suppress[tid] = COOLDOWN_STEPS
        max_cooldown_size = max(max_cooldown_size, len(active_suppress))

        # ---- 2. trailing-CTR / kickstart detection ----
        # Condition E: every step (DS-030 spec). Condition B: production
        # sub-sampling (diversity < 0.40 OR step % every_k == 0).
        trailing_ctr = 1.0
        if condition in ("B", "E"):
            if condition == "E":
                compute_ctr = True
            else:
                compute_ctr = (
                    token_diversity < DIVERSITY_CTR_GATE
                    or controller_step % EVERY_K == 0
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

        # ---- 4a. token-suppression penalties (B, D) ----
        n_suppressed = 0
        if condition in ("B", "D") and active_suppress:
            for tid, steps_left in list(active_suppress.items()):
                if steps_left > 0:
                    logits[:, tid] += SUPPRESS_PENALTY  # -5.0 (controller.py:745)
                    active_suppress[tid] -= 1
                    n_suppressed += 1
                else:
                    del active_suppress[tid]
        supp_applied = n_suppressed > 0

        # ---- 4b. kickstart penalties (B, E) ----
        kick_penalty_value = 0.0
        if condition in ("B", "E") and kickstart_counter > 0:
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

        # ---- 4c. top_p (controller.py:762-763) ----
        top_p_applied = False
        if active_suppress or kickstart_counter > 0:
            logits = sample_top_p(logits, top_p=TOP_P)
            top_p_applied = True

        # ---- greedy argmax ----
        next_tok = logits.argmax(dim=-1, keepdim=True)

        penalty_steps.append({
            "step": int(controller_step),
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


def aggregate_penalty_activity(gen: Dict[str, Any]) -> Dict[str, Any]:
    steps = gen["penalty_steps"]
    return {
        "n_steps": int(len(steps)),
        "suppression_steps": int(
            sum(1 for s in steps if s["suppression_applied"])
        ),
        "n_suppression_applications": int(
            sum(s["n_suppressed"] for s in steps)
        ),
        "kickstart_events": int(gen["n_kickstart_events"]),
        "kickstart_steps": int(
            sum(1 for s in steps if s["kickstart_applied"])
        ),
        "top_p_steps": int(sum(1 for s in steps if s["top_p_applied"])),
        "max_cooldown_size": int(gen["max_cooldown_size"]),
        "per_step": steps,
    }


def build_condition_out(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-condition measurements for one record (deltas vs dormant)."""
    generated_ids = gen["generated_ids"]
    n = int(generated_ids.shape[-1])
    d2 = continuation_distinct2(generated_ids)
    ctr = continuation_ctr(tokenizer, generated_ids)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    eos_terminated = n < MAX_NEW_TOKENS
    return {
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
        "penalty_activity": aggregate_penalty_activity(gen),
    }


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
    """1 record, each live condition (B, D, E), twice — ids must match."""
    print("\n--- Determinism smoke (1 record, each live condition, twice) ---")
    out: Dict[str, Any] = {"record_id": record_id, "per_condition": {}}
    for cond in ("B", "D", "E"):
        a = greedy_generate_condition(
            model, tokenizer, prompt, cond,
            max_new_tokens=max_new_tokens, non_prose_ids=non_prose_ids,
        )
        b = greedy_generate_condition(
            model, tokenizer, prompt, cond,
            max_new_tokens=max_new_tokens, non_prose_ids=non_prose_ids,
        )
        ids_a = a["generated_ids"]
        ids_b = b["generated_ids"]
        identical = bool(torch.equal(ids_a, ids_b))
        print(f"  cond {cond}: run A tokens={ids_a.shape[-1]}, "
              f"run B tokens={ids_b.shape[-1]}, identical={identical}")
        if not identical:
            print(f"[STOP] Determinism smoke FAILED for condition {cond}: "
                  "token ids differ across replays.")
            sys.exit(1)
        out["per_condition"][cond] = {
            "n_tokens_a": int(ids_a.shape[-1]),
            "n_tokens_b": int(ids_b.shape[-1]),
            "identical": str(identical),
        }
    out["all_identical"] = "True"
    print("  determinism smoke: ALL IDENTICAL (B, D, E)")
    return out


# ---------------------------------------------------------------------------
# Summaries
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
    md.append("# DS-030 — Logit-penalty decomposition: suppression vs kickstart")
    md.append("")
    md.append("> MEASUREMENT REPORT. This document reports measured values only.")
    md.append("> No threshold is created or modified, no gate script is touched,")
    md.append("> no governor/controller.py edit, and no green/red verdict and no")
    md.append("> mechanism story are offered. Decomposition conclusions are the")
    md.append("> human's act.")
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
    md.append("| Records | "
              f"{meta['n_records']} seeded heldout-degenerate v2 "
              "(random.Random(42).sample, sorted) |")
    md.append("| Decoding | greedy (do_sample=False), "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append("| Analysis window | "
              f"trailing {meta['analysis_window']} generated positions |")
    md.append("| Dormant | REUSED from DS-028 "
              "(docs/gate23/layer_effect_persistent_results.jsonl, join on record_id) |")
    md.append(f"| Conditions | B (combined, live), D (suppression-only, live), "
              f"E (kickstart-only, live) |")
    md.append("| Suppression budget | "
              f"cooldown={meta['suppression']['cooldown_steps']}, "
              f"penalty={meta['suppression']['penalty']} "
              "(controller.py:137,745) |")
    md.append("| Kickstart budget | "
              f"penalties={meta['kickstart']['penalties']}, "
              f"trigger CTR<{meta['kickstart']['ctr_threshold']} "
              "AND active_loop_ids (controller.py:691,753-758) |")
    md.append("| top_p | " + f"{meta['top_p']} (controller.py:138,663) |")
    md.append(f"| B CTR sub-sampling | every_k={meta['every_k']} or "
              f"diversity<{meta['diversity_ctr_gate']} (production, "
              f"controller.py:136,683); E computes CTR every step |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    md.append("## Determinism smoke")
    md.append("")
    sm = ctx["determinism_smoke"]
    if sm is None:
        md.append("Skipped (--skip-smoke dev mode).")
        md.append("")
    else:
        md.append("One heldout-degenerate record, each live condition (B, D, E),")
        md.append("generated twice on the same model load; generated token ids must")
        md.append("match exactly.")
        md.append("")
        rows = []
        for cond in ("B", "D", "E"):
            p = sm["per_condition"][cond]
            rows.append([cond, str(p["n_tokens_a"]), str(p["n_tokens_b"]), p["identical"]])
        rows.append(["", "", "ALL", sm["all_identical"]])
        md.append(format_table_md(
            rows, ["condition", "run A tokens", "run B tokens", "identical"],
        ))
        md.append("")

    md.append("## Method summary")
    md.append("")
    md.append("- Per record, per live condition {B, D, E}: one greedy generation")
    md.append("  from the record's mutated_prompt; dormant is reused from DS-028.")
    md.append("- Condition B (combined): token suppression + kickstart, no")
    md.append("  residual hooks; production CTR sub-sampling (every_k=2 or")
    md.append("  diversity<0.40); EOS-terminating loop.")
    md.append("- Condition D (suppression-only): token suppression ONLY, applied")
    md.append("  at every decoding step; cooldown initial/reset 8, -5.0 penalty;")
    md.append("  top_p=0.85 when suppression active; NO kickstart, NO vocab cache.")
    md.append("- Condition E (kickstart-only): kickstart vocabulary steering ONLY;")
    md.append("  prose/non-prose sets via production _cache_vocabulary_subsets;")
    md.append("  trailing-CTR every step; kickstart_counter=3 when CTR<0.50 AND")
    md.append("  active_loop_ids; penalties -1e4/-5.0/-2.0; top_p=0.85 when active;")
    md.append("  NO token suppression.")
    md.append("- Distinct-2 on trailing 24 generated tokens (ds-025 convention);")
    md.append("  CTR is the coherent-token ratio of the decoded continuation.")
    md.append("- Δ vs dormant = condition − dormant (per record).")
    md.append("")

    md.append("## Primary aggregation table")
    md.append("")
    md.append("ΔDistinct-2 / ΔCTR are per-record deltas vs the DS-028 dormant")
    md.append("continuation; #byte-id counts records whose decoded continuation is")
    md.append("byte-identical to dormant; EOS rate is the fraction of records that")
    md.append("terminated early (n_generated < max_new_tokens).")
    md.append("")
    agg = ctx["aggregation"]
    rows = []
    for cond in ("dormant", "B", "D", "E"):
        s = agg[cond]
        rows.append([
            cond,
            str(s["n"]),
            fmt(s["delta_d2_mean"]), fmt(s["delta_d2_median"]),
            str(s["n_delta_d2_pos"]),
            str(s["n_byte_identical"]),
            fmt(s["delta_ctr_mean"]),
            fmt(s["eos_rate"], 4),
            fmt(s["mean_n_gen"], 2),
            str(s["n_gen_eq_1"]),
            str(s["n_gen_eq_128"]),
        ])
    md.append(format_table_md(
        rows, ["cond", "n", "ΔD2 mean", "ΔD2 med", "#ΔD2>0", "#byte-id",
               "ΔCTR mean", "EOS rate", "mean n_gen", "n_gen=1", "n_gen=128"],
    ))
    md.append("")

    md.append("## Continuation length distribution per condition")
    md.append("")
    rows = []
    for cond in ("dormant", "B", "D", "E"):
        s = agg[cond]["length_hist"]
        rows.append([
            cond,
            str(s["n_gen_1"]), str(s["n_gen_2_23"]),
            str(s["n_gen_24_127"]), str(s["n_gen_128"]),
            fmt(s["mean"], 2), fmt(s["median"], 2),
        ])
    md.append(format_table_md(
        rows, ["cond", "n_gen=1", "n_gen=2..23", "n_gen=24..127", "n_gen=128",
               "mean n_gen", "median n_gen"],
    ))
    md.append("")

    md.append("## Distinct-2 for records reaching n>=24 vs n<24")
    md.append("")
    md.append("Distinct-2 is measured over the trailing 24 generated tokens;")
    md.append("records with fewer than 24 generated tokens use the cold-start")
    md.append("fallback (Distinct-2 = 1.0 when n<4). Splitting by n>=24 / n<24")
    md.append("separates prose-recovery continuations from early-EOS continuations.")
    md.append("")
    rows = []
    for cond in ("dormant", "B", "D", "E"):
        s = agg[cond]["d2_split"]
        rows.append([cond, "n>=24", str(s["ge24"]["n"]), fmt(s["ge24"]["mean"]), fmt(s["ge24"]["median"])])
        rows.append([cond, "n<24", str(s["lt24"]["n"]), fmt(s["lt24"]["mean"]), fmt(s["lt24"]["median"])])
    md.append(format_table_md(
        rows, ["cond", "subset", "n", "d2 mean", "d2 med"],
    ))
    md.append("")

    md.append("## Penalty activity stats (per record, per live condition)")
    md.append("")
    md.append("Aggregated over the 50 records of each live condition. Suppression")
    md.append("counts are steps with at least one active suppression penalty;")
    md.append("n_suppression_applications is the total (token, step) penalty")
    md.append("applications; kickstart_events counts times kickstart_counter was")
    md.append("set to 3; kickstart_steps counts steps with a kickstart penalty.")
    md.append("")
    rows = []
    for cond in ("B", "D", "E"):
        s = agg[cond]["penalty_activity"]
        rows.append([
            cond,
            str(s["suppression_steps_mean"]), str(s["n_suppression_applications_mean"]),
            str(s["kickstart_events_mean"]), str(s["kickstart_steps_mean"]),
            str(s["top_p_steps_mean"]), str(s["max_cooldown_size_mean"]),
        ])
    md.append(format_table_md(
        rows, ["cond", "supp steps", "n supp appl", "kick events", "kick steps",
               "top_p steps", "max cooldown"],
    ))
    md.append("")

    md.append("## Per-record interaction table")
    md.append("")
    md.append("D/E/B rescues = per-record ΔDistinct-2 > 0 vs dormant. D ≡ E = D and")
    md.append("E generated token ids are identical. D ≠ dormant / E ≠ dormant = the")
    md.append("condition's decoded continuation differs from dormant. Genuine")
    md.append("interaction would be B rescues with D≠E and both ≠ dormant;")
    md.append("degenerate nulls are B also failing, or D≡E≡dormant.")
    md.append("")
    rows = []
    for rec in ctx["interaction_rows"]:
        rows.append([
            str(rec["record_id"]),
            "Y" if rec["d_rescues"] else "N",
            "Y" if rec["e_rescues"] else "N",
            "Y" if rec["b_rescues"] else "N",
            "Y" if rec["d_eq_e"] else "N",
            "Y" if rec["d_neq_dormant"] else "N",
            "Y" if rec["e_neq_dormant"] else "N",
        ])
    md.append(format_table_md(
        rows, ["record_id", "D rescues", "E rescues", "B rescues",
               "D ≡ E?", "D ≠ dormant?", "E ≠ dormant?"],
    ))
    md.append("")
    it = ctx["interaction_summary"]
    md.append("Interaction-table summary counts:")
    md.append("")
    md.append("| metric | count |")
    md.append("|---|---|")
    md.append(f"| D rescues (ΔD2>0) | {it['n_d_rescues']} |")
    md.append(f"| E rescues (ΔD2>0) | {it['n_e_rescues']} |")
    md.append(f"| B rescues (ΔD2>0) | {it['n_b_rescues']} |")
    md.append(f"| D ≡ E (token-identical) | {it['n_d_eq_e']} |")
    md.append(f"| D ≠ dormant | {it['n_d_neq_dormant']} |")
    md.append(f"| E ≠ dormant | {it['n_e_neq_dormant']} |")
    md.append(f"| both D and E null while B rescues | {it['n_both_null_b_rescues']} |")
    md.append(f"| genuine-interaction shape (B rescues, D≠E, both ≠ dormant) | "
              f"{it['n_genuine_interaction_shape']} |")
    md.append(f"| degenerate-null shape (B fails, or D≡E≡dormant) | "
              f"{it['n_degenerate_null_shape']} |")
    md.append("")
    md.append("These are measurement categories, not a verdict.")
    md.append("")

    md.append("## Per-record JSONL")
    md.append("")
    md.append("Per-record results (dormant plus live B/D/E continuations, deltas,")
    md.append("penalty activity incl. per-step logs) are in")
    md.append("`penalty_decomposition_results.jsonl`.")
    md.append("")

    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY. No verdict, no mechanism story.")
    md.append("- Condition B uses the production CTR sub-sampling (every_k=2 or")
    md.append("  diversity<0.40) and the production EOS-terminating loop;")
    md.append("  Condition E computes trailing-CTR at every step (DS-030 spec), so")
    md.append("  E's kickstart trigger is slightly more sensitive than B's.")
    md.append("- top_p follows the production post-decrement check (controller.py:762):")
    md.append("  on the third kickstart step (counter 1->0), the -2.0 penalty is")
    md.append("  applied but top_p is not, for both B and E.")
    md.append("- The controller's generate() does not break on EOS; the DS-029")
    md.append("  measurement loop did (early-EOS is a first-class outcome here).")
    md.append("- Dormant data is reused from DS-028 and is layer-independent;")
    md.append("  the first layer entry per record was joined on record_id.")
    md.append("- No layer sweep: logit mechanisms are layer-independent.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-030 logit-penalty decomposition (measurement only)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Greedy generation length (default 128).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-030 Logit-penalty decomposition (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"conditions: B (combined), D (suppression-only), E (kickstart-only) — live")
    print(f"suppression: cooldown={COOLDOWN_STEPS} penalty={SUPPRESS_PENALTY}")
    print(f"kickstart: penalties={KICKSTART_PENALTIES} "
          f"ctr<{CTR_THRESHOLD} every_k={EVERY_K}")

    # Model load.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

    # Vocabulary subsets (production _cache_vocabulary_subsets logic).
    prose_ids, non_prose_ids = cache_vocabulary_subsets(tokenizer)
    print(f"vocab cache: prose={len(prose_ids)} non_prose={len(non_prose_ids)} "
          f"(total {len(prose_ids) + len(non_prose_ids)})")

    # Data.
    heldout_deg = load_jsonl(HELDOUT_DEG_V2)
    records = select_records(heldout_deg, NUM_RECORDS, SEED)
    if args.max_records is not None:
        records = records[: args.max_records]
        print(f"[dev] capped records at {args.max_records}")
    record_ids = {int(r["id"]) for r in records}
    print(f"selected {len(records)} heldout-degenerate v2 records (seeded, sorted)")

    # Dormant reuse from DS-028 (join on record_id; do NOT re-run dormant).
    dormant_map = load_dormant_from_ds028(DS028_JSONL, record_ids)
    missing = record_ids - set(dormant_map.keys())
    if missing:
        print(f"[STOP] missing dormant rows in DS-028 for record_ids: {sorted(missing)}")
        sys.exit(1)
    print(f"reused dormant data from {DS028_JSONL} "
          f"({len(dormant_map)} records joined)")

    # Determinism smoke FIRST.
    smoke = None
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        smoke = run_determinism_smoke(
            model, tokenizer, records[0]["mutated_prompt"],
            non_prose_ids=non_prose_ids,
            max_new_tokens=args.max_new_tokens,
            record_id=int(records[0]["id"]),
        )

    # ------------------------------------------------------------------
    # Measurement loop.
    # ------------------------------------------------------------------
    print("\n--- Measurement loop ---")
    results: List[Dict[str, Any]] = []

    for rec_i, rec in enumerate(records):
        rid = int(rec["id"])
        prompt = rec["mutated_prompt"]
        dormant = dormant_map[rid]
        print(f"  record {rid} ({rec_i+1}/{len(records)})")

        rec_out: Dict[str, Any] = {
            "record_id": rid,
            "source_prose_id": int(rec.get("source_prose_id", -1)),
            "prompt": prompt,
            "dormant": dormant,
        }
        for cond in ("B", "D", "E"):
            gen = greedy_generate_condition(
                model, tokenizer, prompt, cond,
                max_new_tokens=args.max_new_tokens,
                non_prose_ids=non_prose_ids,
            )
            rec_out[cond] = build_condition_out(tokenizer, gen, dormant)
        results.append(rec_out)

        if (rec_i + 1) % 10 == 0 or rec_i == len(records) - 1:
            n_b_rescue = sum(1 for r in results if r["B"]["delta_distinct_2"] > 0)
            n_d_rescue = sum(1 for r in results if r["D"]["delta_distinct_2"] > 0)
            n_e_rescue = sum(1 for r in results if r["E"]["delta_distinct_2"] > 0)
            print(f"    ... {len(results)} records; "
                  f"rescue(ΔD2>0) B={n_b_rescue} D={n_d_rescue} E={n_e_rescue}")

    # ------------------------------------------------------------------
    # Aggregation.
    # ------------------------------------------------------------------
    print("\n--- Aggregating ---")
    agg: Dict[str, Dict[str, Any]] = {}
    for cond in ("dormant", "B", "D", "E"):
        if cond == "dormant":
            cond_recs = [dict(r["dormant"], **{"record_id": r["record_id"]})
                         for r in results]
            delta_d2 = [0.0 for _ in cond_recs]
            delta_ctr = [0.0 for _ in cond_recs]
            n_byte_id = len(cond_recs)
            eos_term = [c["n_generated"] < args.max_new_tokens for c in cond_recs]
            n_gen = [c["n_generated"] for c in cond_recs]
        else:
            cond_recs = [r[cond] for r in results]
            delta_d2 = [c["delta_distinct_2"] for c in cond_recs]
            delta_ctr = [c["delta_ctr"] for c in cond_recs]
            n_byte_id = sum(1 for c in cond_recs if c["byte_identical_to_dormant"])
            eos_term = [c["eos_terminated"] for c in cond_recs]
            n_gen = [c["n_generated"] for c in cond_recs]

        d2_all = [c["distinct_2"] for c in cond_recs]
        d2_ge24 = [c["distinct_2"] for c in cond_recs if c["n_generated"] >= 24]
        d2_lt24 = [c["distinct_2"] for c in cond_recs if c["n_generated"] < 24]

        agg[cond] = {
            "n": len(cond_recs),
            "delta_d2_mean": float(np.mean(delta_d2)) if delta_d2 else float("nan"),
            "delta_d2_median": float(np.median(delta_d2)) if delta_d2 else float("nan"),
            "n_delta_d2_pos": int(sum(1 for d in delta_d2 if d > 0)),
            "n_byte_identical": int(n_byte_id),
            "delta_ctr_mean": float(np.mean(delta_ctr)) if delta_ctr else float("nan"),
            "eos_rate": float(np.mean(eos_term)) if eos_term else float("nan"),
            "mean_n_gen": float(np.mean(n_gen)) if n_gen else float("nan"),
            "n_gen_eq_1": int(sum(1 for n in n_gen if n == 1)),
            "n_gen_eq_128": int(sum(1 for n in n_gen if n == 128)),
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
        if cond != "dormant":
            pa = [c["penalty_activity"] for c in cond_recs]
            agg[cond]["penalty_activity"] = {
                "suppression_steps_mean": float(
                    np.mean([p["suppression_steps"] for p in pa])
                ),
                "n_suppression_applications_mean": float(
                    np.mean([p["n_suppression_applications"] for p in pa])
                ),
                "kickstart_events_mean": float(
                    np.mean([p["kickstart_events"] for p in pa])
                ),
                "kickstart_steps_mean": float(
                    np.mean([p["kickstart_steps"] for p in pa])
                ),
                "top_p_steps_mean": float(
                    np.mean([p["top_p_steps"] for p in pa])
                ),
                "max_cooldown_size_mean": float(
                    np.mean([p["max_cooldown_size"] for p in pa])
                ),
            }
        else:
            agg[cond]["penalty_activity"] = {
                "suppression_steps_mean": 0.0,
                "n_suppression_applications_mean": 0.0,
                "kickstart_events_mean": 0.0,
                "kickstart_steps_mean": 0.0,
                "top_p_steps_mean": 0.0,
                "max_cooldown_size_mean": 0.0,
            }

    # Per-record interaction table.
    interaction_rows = []
    n_d_rescues = n_e_rescues = n_b_rescues = 0
    n_d_eq_e = 0
    n_d_neq_dormant = n_e_neq_dormant = 0
    n_both_null_b_rescues = 0
    n_genuine = 0
    n_degenerate = 0
    for r in results:
        d_rescues = r["D"]["delta_distinct_2"] > 0
        e_rescues = r["E"]["delta_distinct_2"] > 0
        b_rescues = r["B"]["delta_distinct_2"] > 0
        d_eq_e = r["D"]["generated_ids"] == r["E"]["generated_ids"]
        d_neq_dormant = not r["D"]["byte_identical_to_dormant"]
        e_neq_dormant = not r["E"]["byte_identical_to_dormant"]
        interaction_rows.append({
            "record_id": r["record_id"],
            "d_rescues": d_rescues,
            "e_rescues": e_rescues,
            "b_rescues": b_rescues,
            "d_eq_e": d_eq_e,
            "d_neq_dormant": d_neq_dormant,
            "e_neq_dormant": e_neq_dormant,
        })
        n_d_rescues += int(d_rescues)
        n_e_rescues += int(e_rescues)
        n_b_rescues += int(b_rescues)
        n_d_eq_e += int(d_eq_e)
        n_d_neq_dormant += int(d_neq_dormant)
        n_e_neq_dormant += int(e_neq_dormant)
        if (not d_rescues) and (not e_rescues) and b_rescues:
            n_both_null_b_rescues += 1
        if b_rescues and (not d_eq_e) and d_neq_dormant and e_neq_dormant:
            n_genuine += 1
        if (not b_rescues) or (d_eq_e and (not d_neq_dormant) and (not e_neq_dormant)):
            n_degenerate += 1

    interaction_summary = {
        "n_d_rescues": n_d_rescues,
        "n_e_rescues": n_e_rescues,
        "n_b_rescues": n_b_rescues,
        "n_d_eq_e": n_d_eq_e,
        "n_d_neq_dormant": n_d_neq_dormant,
        "n_e_neq_dormant": n_e_neq_dormant,
        "n_both_null_b_rescues": n_both_null_b_rescues,
        "n_genuine_interaction_shape": n_genuine,
        "n_degenerate_null_shape": n_degenerate,
    }

    for cond in ("B", "D", "E"):
        s = agg[cond]
        print(f"  {cond}: ΔD2 mean={s['delta_d2_mean']:.4f} med="
              f"{s['delta_d2_median']:.4f} #>0={s['n_delta_d2_pos']} "
              f"byte-id={s['n_byte_identical']} ΔCTR mean={s['delta_ctr_mean']:.4f} "
              f"EOS={s['eos_rate']:.3f} mean_n={s['mean_n_gen']:.1f}")
    print(f"  interaction: D rescue={n_d_rescues} E rescue={n_e_rescues} "
          f"B rescue={n_b_rescues} D≡E={n_d_eq_e} "
          f"both-null-B-rescue={n_both_null_b_rescues}")

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
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
        "n_records": len(records),
        "analysis_window": ANALYSIS_WINDOW,
        "max_new_tokens": args.max_new_tokens,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
        "suppression": {
            "cooldown_steps": COOLDOWN_STEPS,
            "penalty": SUPPRESS_PENALTY,
        },
        "kickstart": {
            "penalties": KICKSTART_PENALTIES,
            "ctr_threshold": CTR_THRESHOLD,
            "counter_init": KICKSTART_COUNTER_INIT,
        },
        "top_p": TOP_P,
        "every_k": EVERY_K,
        "diversity_ctr_gate": DIVERSITY_CTR_GATE,
        "trailing_gen_window": TRAILING_GEN_WINDOW,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "aggregation": agg,
        "interaction_rows": interaction_rows,
        "interaction_summary": interaction_summary,
    }

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("MEASUREMENT COMPLETE (no verdict, no mechanism story offered)")


if __name__ == "__main__":
    main()
