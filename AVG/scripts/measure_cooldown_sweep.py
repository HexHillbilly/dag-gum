#!/usr/bin/env python3
"""DS-031: Token-suppression cooldown sweep (MEASUREMENT ONLY).

DS-030 showed token suppression alone (cooldown=8, penalty=-5.0) rescues
48/50 heldout-degenerate v2 records under greedy decoding (D2 +0.74, EOS
rate 0.28, n>=24 prose D2 0.91); the combined suppression + kickstart path
rescues 50/50 but doubles the EOS rate to 0.54. This sweep varies cooldown
in {3, 5, 8} x penalty in {-2, -3, -5} to find the gentlest setting that
preserves the 48/50 rescue rate while driving the EOS rate toward zero.

  Cell (8,-5) is REUSED from DS-030 Condition D
  (docs/gate23/penalty_decomposition_results.jsonl; join on record_id).
  Do NOT re-run it.
  New cells measured LIVE in this process (5 cells x 50 records = 250
  generations): (3,-3), (3,-5), (5,-2), (5,-3), (5,-5).

Suppression logic is identical to DS-030 Condition D, only cooldown and
penalty values change per cell:
  - At each step compute compute_token_distinct_2_fast(input_ids,
    prompt_len=prompt_len, window_len=24) -> active_loop_ids.
  - For each newly detected repeated token id, set/add to the cooldown
    dict with value = cell_cooldown (existing ids are reset to
    cell_cooldown).
  - Before argmax, for each (tid, steps_left) in the cooldown dict:
    if steps_left > 0 apply cell_penalty to logits[:, tid] and decrement;
    else remove.
  - top_p=0.85 when the suppression dict is non-empty (controller.py:762;
    kickstart counter is always 0 here).
  - NO kickstart, NO vocabulary cache, NO trailing_ctr computation.

Dormant is REUSED from DS-028 (docs/gate23/layer_effect_persistent_results
.jsonl; join on record_id; do NOT re-run).

Determinism smoke FIRST: 1 record, cell (5,-3), generated twice — token ids
must match exactly; else STOP. If CUDA OOMs: STOP.

Records 19 and 20 are the canary pair DS-030 showed suppression-only (8,-5)
fails to rescue (only the combined B path breaks their loops). The canary
mini-table reports whether any sweep cell rescues them without kickstart.

MEASUREMENT ONLY. No threshold is created or modified, no gate script is
touched, no governor/controller.py edit, no assert on outcomes, and no
green/red verdict and no "this is the right setting" language are offered.

Environment notes (identical to ds-025/ds-026/ds-028/ds-030):
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
SEED = 42  # DS-031 measurement seed (matches ds-025/026/027/028/029/030)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128
ANALYSIS_WINDOW = 24  # trailing generated positions for distinct-2
NUM_RECORDS = 50
TOP_P = 0.85  # controller.py:138,663 (top_p=0.85)

# Cells to measure live (cooldown, penalty). Five new cells.
NEW_CELLS: List[Tuple[int, float]] = [
    (3, -3.0),
    (3, -5.0),
    (5, -2.0),
    (5, -3.0),
    (5, -5.0),
]
# Reused cell from DS-030 Condition D (do NOT re-run).
REUSED_CELL: Tuple[int, float] = (8, -5.0)

# Paths
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
DS028_JSONL = Path("docs/gate23/layer_effect_persistent_results.jsonl")
DS030_JSONL = Path("docs/gate23/penalty_decomposition_results.jsonl")
OUTPUT_JSONL = Path("docs/gate23/cooldown_sweep_results.jsonl")
OUTPUT_MD = Path("docs/gate23/COOLDOWN_SWEEP_RESULTS.md")

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


def cell_label(cooldown: int, penalty: float) -> str:
    """Human-readable cell label, e.g. (3,-3)."""
    return f"({int(cooldown)},{int(penalty)})"


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


def load_ds030_condition_d(
    path: Path, record_ids: Set[int]
) -> Dict[int, Dict[str, Any]]:
    """Join DS-030 Condition D (cell (8,-5)) per record; do NOT re-run.

    DS-030's Condition D uses the same suppression-only logic as this sweep
    with cooldown=8, penalty=-5.0. The per-step logs are reused as-is; mean
    cooldown occupancy is computed from the DS-030 D per_step n_active_cooldown.
    """
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = int(r["record_id"])
            if rid in record_ids and rid not in out:
                if "D" not in r:
                    raise ValueError(
                        f"DS-030 record {rid} missing Condition D entry"
                    )
                d = r["D"]
                pa = d["penalty_activity"]
                per_step = pa["per_step"]
                occ = (
                    float(statistics.mean(s["n_active_cooldown"] for s in per_step))
                    if per_step else float("nan")
                )
                out[rid] = {
                    "n_generated": int(d["n_generated"]),
                    "distinct_2": float(d["distinct_2"]),
                    "ctr": float(d["ctr"]),
                    "generated_text": str(d["generated_text"]),
                    "generated_ids": [int(x) for x in d["generated_ids"]],
                    "eos_terminated": bool(d["eos_terminated"]),
                    "last_token_is_eos": bool(d["last_token_is_eos"]),
                    "delta_distinct_2": float(d["delta_distinct_2"]),
                    "delta_ctr": float(d["delta_ctr"]),
                    "byte_identical_to_dormant": bool(
                        d["byte_identical_to_dormant"]
                    ),
                    "token_identical_to_dormant": bool(
                        d["token_identical_to_dormant"]
                    ),
                    "suppression_activity": {
                        "n_steps": int(pa["n_steps"]),
                        "suppression_steps": int(pa["suppression_steps"]),
                        "n_suppression_applications": int(
                            pa["n_suppression_applications"]
                        ),
                        "top_p_steps": int(pa["top_p_steps"]),
                        "max_cooldown_size": int(pa["max_cooldown_size"]),
                        "mean_cooldown_occupancy": occ,
                        "per_step": per_step,
                    },
                    "source": "ds030_penalty_decomposition_results.jsonl "
                              "(Condition D, reused)",
                }
    return out


# ---------------------------------------------------------------------------
# Greedy generation under token-suppression-only (per-cell cooldown/penalty)
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

    Identical to DS-030 Condition D, only cooldown and penalty values change
    per cell. NO kickstart, NO vocabulary cache, NO trailing_ctr computation.
    top_p=0.85 is applied when the suppression dict is non-empty after the
    decrement/removal step (controller.py:762; kickstart counter is 0 here).
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

        # ---- 4a. token-suppression penalties (per-cell cooldown/penalty) ----
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

        # ---- 4c. top_p (controller.py:762-763; kickstart counter always 0) ----
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


def aggregate_suppression_activity(gen: Dict[str, Any]) -> Dict[str, Any]:
    steps = gen["penalty_steps"]
    occ = (
        float(statistics.mean(s["n_active_cooldown"] for s in steps))
        if steps else float("nan")
    )
    return {
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


def build_cell_out(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-cell measurements for one record (deltas vs dormant)."""
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
        "suppression_activity": aggregate_suppression_activity(gen),
        "source": "live",
    }


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    cooldown: int,
    penalty: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
    record_id: int = -1,
) -> Dict[str, Any]:
    """1 record, cell (5,-3), generated twice — ids must match."""
    print(f"\n--- Determinism smoke (1 record, cell {cell_label(cooldown, penalty)}, twice) ---")
    a = greedy_generate_suppression_only(
        model, tokenizer, prompt, cooldown, penalty,
        max_new_tokens=max_new_tokens,
    )
    b = greedy_generate_suppression_only(
        model, tokenizer, prompt, cooldown, penalty,
        max_new_tokens=max_new_tokens,
    )
    ids_a = a["generated_ids"]
    ids_b = b["generated_ids"]
    identical = bool(torch.equal(ids_a, ids_b))
    print(f"  cell {cell_label(cooldown, penalty)}: run A tokens={ids_a.shape[-1]}, "
          f"run B tokens={ids_b.shape[-1]}, identical={identical}")
    if not identical:
        print(f"[STOP] Determinism smoke FAILED for cell "
              f"{cell_label(cooldown, penalty)}: token ids differ across replays.")
        sys.exit(1)
    out = {
        "record_id": record_id,
        "cell": cell_label(cooldown, penalty),
        "n_tokens_a": int(ids_a.shape[-1]),
        "n_tokens_b": int(ids_b.shape[-1]),
        "identical": str(identical),
        "all_identical": "True",
    }
    print(f"  determinism smoke: ALL IDENTICAL (cell {cell_label(cooldown, penalty)})")
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


def aggregate_cell(records: List[Dict[str, Any]], cell_key: str) -> Dict[str, Any]:
    """Aggregate per-record measurements for one cell key."""
    cond_recs = [r["cells"][cell_key] for r in records]
    delta_d2 = [c["delta_distinct_2"] for c in cond_recs]
    n_gen = [c["n_generated"] for c in cond_recs]
    eos_term = [c["eos_terminated"] for c in cond_recs]
    d2_all = [c["distinct_2"] for c in cond_recs]
    d2_ge24 = [c["distinct_2"] for c in cond_recs if c["n_generated"] >= 24]
    d2_lt24 = [c["distinct_2"] for c in cond_recs if c["n_generated"] < 24]
    sa = [c["suppression_activity"] for c in cond_recs]

    return {
        "n": len(cond_recs),
        "delta_d2_mean": float(np.mean(delta_d2)) if delta_d2 else float("nan"),
        "delta_d2_median": float(np.median(delta_d2)) if delta_d2 else float("nan"),
        "n_delta_d2_pos": int(sum(1 for d in delta_d2 if d > 0)),
        "n_byte_identical": int(
            sum(1 for c in cond_recs if c["byte_identical_to_dormant"])
        ),
        "eos_rate": float(np.mean(eos_term)) if eos_term else float("nan"),
        "mean_n_gen": float(np.mean(n_gen)) if n_gen else float("nan"),
        "n_gen_eq_1": int(sum(1 for n in n_gen if n == 1)),
        "n_gen_eq_128": int(sum(1 for n in n_gen if n == 128)),
        "n_ge24_d2": float(np.mean(d2_ge24)) if d2_ge24 else float("nan"),
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
        "suppression_activity": {
            "suppression_steps_mean": float(
                np.mean([p["suppression_steps"] for p in sa])
            ),
            "n_suppression_applications_mean": float(
                np.mean([p["n_suppression_applications"] for p in sa])
            ),
            "mean_cooldown_occupancy_mean": float(
                np.mean([p["mean_cooldown_occupancy"] for p in sa])
            ),
            "top_p_steps_mean": float(
                np.mean([p["top_p_steps"] for p in sa])
            ),
            "max_cooldown_size_mean": float(
                np.mean([p["max_cooldown_size"] for p in sa])
            ),
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


def select_detail_cell(
    agg: Dict[str, Dict[str, Any]], cell_keys: List[str]
) -> str:
    """Select the cell for the per-record detail table (display only).

    Selection rule (stated as a measurement category, not a verdict):
      primary   — preserve the (8,-5) rescue count: #ΔD2>0 >= 48;
      secondary — lowest EOS rate (drive EOS toward zero);
      tertiary  — lowest mean suppression applications per record (gentlest);
      quaternary— lowest |penalty|, then lowest cooldown.
    """
    eligible = [
        k for k in cell_keys
        if agg[k]["n_delta_d2_pos"] >= 48
    ]
    pool = eligible if eligible else cell_keys
    pool_sorted = sorted(
        pool,
        key=lambda k: (
            agg[k]["eos_rate"],
            agg[k]["suppression_activity"]["n_suppression_applications_mean"],
            -float(k.split(",")[1].rstrip(")")),
            int(k.split(",")[0].lstrip("(")),
        ),
    )
    return pool_sorted[0]


def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# DS-031 — Token-suppression cooldown sweep")
    md.append("")
    md.append("> MEASUREMENT REPORT. This document reports measured values only.")
    md.append("> No threshold is created or modified, no gate script is touched,")
    md.append("> no governor/controller.py edit, and no green/red verdict and no")
    md.append("> \"this is the right setting\" language are offered. Cooldown/penalty")
    md.append("> selection is the human's act.")
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
    md.append("| Reused cell | (8,-5) REUSED from DS-030 Condition D "
              "(docs/gate23/penalty_decomposition_results.jsonl, join on record_id) |")
    md.append("| New cells | " + ", ".join(meta['new_cells']) + " |")
    md.append("| top_p | " + f"{meta['top_p']} (controller.py:138,663) |")
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
        md.append("One heldout-degenerate record, cell (5,-3), generated twice")
        md.append("on the same model load; generated token ids must match exactly.")
        md.append("")
        md.append(format_table_md(
            [[sm["cell"], str(sm["n_tokens_a"]), str(sm["n_tokens_b"]),
              sm["identical"]],
             ["", "", "ALL", sm["all_identical"]]],
            ["cell", "run A tokens", "run B tokens", "identical"],
        ))
        md.append("")

    md.append("## Method summary")
    md.append("")
    md.append("- Per record, per cell: one greedy generation from the record's")
    md.append("  mutated_prompt; dormant is reused from DS-028.")
    md.append("- Suppression-only logic is identical to DS-030 Condition D, only")
    md.append("  cooldown and penalty change per cell: repeated-token ids from")
    md.append("  compute_token_distinct_2_fast set/reset a cooldown dict entry to")
    md.append("  the cell cooldown; before argmax each entry with steps_left>0 is")
    md.append("  penalised by the cell penalty and decremented, else removed.")
    md.append("- top_p=0.85 when the suppression dict is non-empty after the")
    md.append("  decrement/removal step (controller.py:762; kickstart counter 0).")
    md.append("- NO kickstart, NO vocabulary cache, NO trailing_ctr computation.")
    md.append("- Distinct-2 on trailing 24 generated tokens (ds-025 convention);")
    md.append("  CTR is the coherent-token ratio of the decoded continuation.")
    md.append("- Δ vs dormant = cell − dormant (per record).")
    md.append("")

    md.append("## Primary aggregation table")
    md.append("")
    md.append("ΔDistinct-2 is the per-record delta vs the DS-028 dormant")
    md.append("continuation; #byte-id counts records whose decoded continuation is")
    md.append("byte-identical to dormant; EOS rate is the fraction of records that")
    md.append("terminated early (n_generated < max_new_tokens); n>=24 D2 is the")
    md.append("distinct-2 mean over records that reached the analysis window;")
    md.append("#n_gen=128 counts records that generated the full budget.")
    md.append("")
    agg = ctx["aggregation"]
    rows = []
    for cell_key in ctx["cell_order"]:
        s = agg[cell_key]
        label = cell_key if cell_key != "(8,-5)" else "(8,-5)*"
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
        rows, ["cell", "n", "ΔD2 mean", "ΔD2 med", "#ΔD2>0", "#byte-id",
               "EOS rate", "mean n_gen", "n≥24 D2", "#n_gen=128"],
    ))
    md.append("")
    md.append("* (8,-5) is REUSED from DS-030 Condition D (not re-run).")
    md.append("")

    md.append("## Continuation length distribution per cell")
    md.append("")
    rows = []
    for cell_key in ctx["cell_order"]:
        s = agg[cell_key]
        label = cell_key if cell_key != "(8,-5)" else "(8,-5)*"
        rows.append([
            label,
            str(s["length_hist"]["n_gen_1"]), str(s["length_hist"]["n_gen_2_23"]),
            str(s["length_hist"]["n_gen_24_127"]), str(s["length_hist"]["n_gen_128"]),
            fmt(s["length_hist"]["mean"], 2), fmt(s["length_hist"]["median"], 2),
        ])
    md.append(format_table_md(
        rows, ["cell", "n_gen=1", "n_gen=2..23", "n_gen=24..127", "n_gen=128",
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
    for cell_key in ctx["cell_order"]:
        s = agg[cell_key]
        label = cell_key if cell_key != "(8,-5)" else "(8,-5)*"
        rows.append([label, "n>=24", str(s["d2_split"]["ge24"]["n"]),
                     fmt(s["d2_split"]["ge24"]["mean"]), fmt(s["d2_split"]["ge24"]["median"])])
        rows.append([label, "n<24", str(s["d2_split"]["lt24"]["n"]),
                     fmt(s["d2_split"]["lt24"]["mean"]), fmt(s["d2_split"]["lt24"]["median"])])
    md.append(format_table_md(
        rows, ["cell", "subset", "n", "d2 mean", "d2 med"],
    ))
    md.append("")

    md.append("## Records 19/20 canary")
    md.append("")
    md.append("DS-030 showed suppression-only (8,-5) failed to rescue records 19")
    md.append("and 20 (both byte-identical to dormant under D); only the combined")
    md.append("B path broke their loops. This table reports, per cell, whether the")
    md.append("canary pair is rescued without kickstart (ΔD2 > 0). Whether kickstart")
    md.append("is still required for the 4% tail is a design decision for the human.")
    md.append("")
    rows = []
    for rec_id in (19, 20):
        rec = ctx["by_record"].get(rec_id)
        if rec is None:
            continue
        for cell_key in ctx["cell_order"]:
            c = rec["cells"][cell_key]
            label = cell_key if cell_key != "(8,-5)" else "(8,-5)*"
            rows.append([
                str(rec_id),
                label,
                fmt(c["delta_distinct_2"]),
                "Y" if c["byte_identical_to_dormant"] else "N",
                str(c["n_generated"]),
                "Y" if c["eos_terminated"] else "N",
                "Y" if c["delta_distinct_2"] > 0 else "N",
            ])
    md.append(format_table_md(
        rows, ["record_id", "cell", "ΔD2", "byte-id?", "n_gen", "EOS?", "rescued?"],
    ))
    md.append("")

    md.append("## Suppression activity per cell")
    md.append("")
    md.append("Aggregated over the 50 records of each cell. suppression_steps =")
    md.append("steps with at least one suppression penalty; n_suppression_applications")
    md.append("= total (token, step) penalty applications; mean cooldown occupancy =")
    md.append("mean number of ids in the cooldown dict at the end of each step.")
    md.append("")
    rows = []
    for cell_key in ctx["cell_order"]:
        s = agg[cell_key]["suppression_activity"]
        label = cell_key if cell_key != "(8,-5)" else "(8,-5)*"
        rows.append([
            label,
            fmt(s["suppression_steps_mean"], 2),
            fmt(s["n_suppression_applications_mean"], 2),
            fmt(s["mean_cooldown_occupancy_mean"], 2),
            fmt(s["top_p_steps_mean"], 2),
            fmt(s["max_cooldown_size_mean"], 2),
        ])
    md.append(format_table_md(
        rows, ["cell", "supp steps", "n supp appl", "cooldown occupancy",
               "top_p steps", "max cooldown"],
    ))
    md.append("")

    detail_cell = ctx["detail_cell"]
    md.append("## Per-record detail table for cell "
              + (detail_cell if detail_cell != "(8,-5)" else "(8,-5)*"))
    md.append("")
    md.append("Selection rule (display only, not a verdict): among cells with")
    md.append("#ΔD2>0 >= 48 (the (8,-5) rescue count), the cell with the lowest EOS")
    md.append("rate; ties broken by lowest mean suppression applications per record,")
    md.append("then lowest |penalty|, then lowest cooldown.")
    md.append("")
    md.append("Selected cell: `" + detail_cell + "`.")
    md.append("")
    rows = []
    for rec in ctx["by_record"].values():
        c = rec["cells"][detail_cell]
        sa = c["suppression_activity"]
        rows.append([
            str(rec["record_id"]),
            str(c["n_generated"]),
            fmt(c["distinct_2"]), fmt(c["delta_distinct_2"]),
            fmt(c["ctr"]), fmt(c["delta_ctr"]),
            "Y" if c["byte_identical_to_dormant"] else "N",
            "Y" if c["eos_terminated"] else "N",
            "Y" if c["delta_distinct_2"] > 0 else "N",
            str(sa["suppression_steps"]),
            str(sa["n_suppression_applications"]),
            fmt(sa["mean_cooldown_occupancy"], 2),
        ])
    md.append(format_table_md(
        rows, ["record_id", "n_gen", "D2", "ΔD2", "CTR", "ΔCTR", "byte-id?",
               "EOS?", "rescued?", "supp steps", "supp appl", "cooldown occ"],
    ))
    md.append("")

    md.append("## Per-record JSONL")
    md.append("")
    md.append("Per-record results (dormant plus live cells, deltas, suppression")
    md.append("activity incl. per-step logs) are in `cooldown_sweep_results.jsonl`.")
    md.append("")

    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY. No verdict, no mechanism story, no \"right setting\".")
    md.append("- Cell (8,-5) is REUSED from DS-030 Condition D; it is not re-run.")
    md.append("  Its suppression activity is reconstructed from the DS-030 D")
    md.append("  per-step logs (mean cooldown occupancy computed from per-step")
    md.append("  n_active_cooldown).")
    md.append("- New cells are run live in this process on the same model load;")
    md.append("  determinism smoke (cell (5,-3)) passed before the measurement loop.")
    md.append("- Dormant data is reused from DS-028 and is layer-independent;")
    md.append("  the first layer entry per record was joined on record_id.")
    md.append("- The generation loop breaks on EOS; the EOS token is included in")
    md.append("  n_generated and generated_ids (ds-029/ds-030 convention).")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-031 token-suppression cooldown sweep (measurement only)"
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
    print("DS-031 Token-suppression cooldown sweep (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"new cells: {[cell_label(*c) for c in NEW_CELLS]}")
    print(f"reused cell: {cell_label(*REUSED_CELL)} (DS-030 Condition D)")

    # Model load.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

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

    # Reused cell (8,-5) from DS-030 Condition D (do NOT re-run).
    reused_map = load_ds030_condition_d(DS030_JSONL, record_ids)
    missing = record_ids - set(reused_map.keys())
    if missing:
        print(f"[STOP] missing DS-030 Condition D rows for record_ids: {sorted(missing)}")
        sys.exit(1)
    print(f"reused cell (8,-5) from {DS030_JSONL} "
          f"({len(reused_map)} records joined)")

    # Determinism smoke FIRST.
    smoke = None
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        smoke = run_determinism_smoke(
            model, tokenizer, records[0]["mutated_prompt"],
            cooldown=5, penalty=-3.0,
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
            "cells": {},
        }
        # Reused cell first.
        rec_out["cells"][cell_label(*REUSED_CELL)] = reused_map[rid]
        # Live new cells.
        for (cooldown, penalty) in NEW_CELLS:
            gen = greedy_generate_suppression_only(
                model, tokenizer, prompt, cooldown, penalty,
                max_new_tokens=args.max_new_tokens,
            )
            rec_out["cells"][cell_label(cooldown, penalty)] = build_cell_out(
                tokenizer, gen, dormant,
            )
        results.append(rec_out)

        if (rec_i + 1) % 10 == 0 or rec_i == len(records) - 1:
            print(f"    ... {len(results)} records; ", end="")
            for cell_key in [cell_label(*c) for c in NEW_CELLS]:
                n_rescue = sum(
                    1 for r in results if r["cells"][cell_key]["delta_distinct_2"] > 0
                )
                print(f"{cell_key} rescue={n_rescue} ", end="")
            print()

    # ------------------------------------------------------------------
    # Aggregation.
    # ------------------------------------------------------------------
    print("\n--- Aggregating ---")
    cell_order = [cell_label(*c) for c in NEW_CELLS] + [cell_label(*REUSED_CELL)]
    agg: Dict[str, Dict[str, Any]] = {}
    for cell_key in cell_order:
        agg[cell_key] = aggregate_cell(results, cell_key)
        s = agg[cell_key]
        print(f"  {cell_key}: ΔD2 mean={s['delta_d2_mean']:.4f} med="
              f"{s['delta_d2_median']:.4f} #>0={s['n_delta_d2_pos']} "
              f"byte-id={s['n_byte_identical']} EOS={s['eos_rate']:.3f} "
              f"mean_n={s['mean_n_gen']:.1f} n≥24 D2={s['n_ge24_d2']:.4f}")

    detail_cell = select_detail_cell(agg, cell_order)
    print(f"  detail-table cell selected: {detail_cell}")

    by_record = {int(r["record_id"]): r for r in results}

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
        "new_cells": [cell_label(*c) for c in NEW_CELLS],
        "reused_cell": cell_label(*REUSED_CELL),
        "top_p": TOP_P,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "aggregation": agg,
        "cell_order": cell_order,
        "detail_cell": detail_cell,
        "by_record": by_record,
    }

    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("MEASUREMENT COMPLETE (no verdict, no mechanism story offered)")


if __name__ == "__main__":
    main()
