#!/usr/bin/env python3
"""
DS-024 — Gate 2.2 AUROC measurement.

MEASUREMENT ONLY. No thresholds are created or modified, no gate scripts are
touched, no governor/controller.py edits, no asserts on outcomes, and no
green/red verdict is produced. The green/red ratification belongs to humans.

Method (per the task file):
  1. REPLAY (teacher-forced): one forward pass over prompt+text (qwen_degenerate)
     or text (t2s_degenerate, prose control) with output_hidden_states=True.
     Analysis window = LAST 24 token positions.
  2. Per record, per layer (1..28; layer 0 embeddings recorded as datum):
     participation_ratio, singular_value_spectrum_entropy, effective_rank.
     Per record (text-level): compute_coherent_token_ratio,
     compute_token_distinct_2_fast.
  3. Rank-based Mann-Whitney AUROC (numpy only) per signal per layer, per
     contrast (primary: qwen_degenerate vs prose; secondary: t2s_degenerate vs
     prose). Direction reported explicitly; separation magnitude = max(AUC, 1-AUC).
  4. Determinism smoke FIRST: 2 records replayed twice; metric tuples must be
     byte-identical, otherwise STOP and report.
  5. Engagement-evidence arm: ActiveVarietyGovernor (intervene=True,
     use_code_filter=True, calibrated on the fox text) on 20 seeded
     qwen_degenerate prompts, decoding per each corpus record's own params,
     max_new_tokens=128; plus the canonical word-loop probe (temp 0.7, top_p
     0.9, 64 tokens) as a positive-control reference.

Environment notes:
  - torch 2.13 CUDA `bmm_outer_product` Triton override is deregistered
    (C compiler absent; see docs/PROBE_B_RECONSTRUCTION.md §6). This is an
    env-only adaptation; no protected file is modified.
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount. Verified once.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Env-only adaptation: deregister torch 2.13's CUDA bmm Triton override
# (requires a Triton JIT C compiler absent from this container).
# ---------------------------------------------------------------------------
try:
    from torch._native import triton_utils as _triton_utils

    _triton_utils.deregister_op_overrides()
    print("[env] torch bmm Triton override deregistered")
except Exception as _env_e:  # pragma: no cover - env dependent
    print(f"[env] bmm override deregistration skipped: {_env_e}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from AVG.core.metrics import (  # noqa: E402
    compute_coherent_token_ratio,
    effective_rank,
    participation_ratio,
    singular_value_spectrum_entropy,
)
from AVG.governor.controller import (  # noqa: E402
    ActiveVarietyGovernor,
    compute_token_distinct_2_fast,
)

# ---------------------------------------------------------------------------
# Constants (SEED=42 everywhere)
# ---------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
QWEN_DEG_PATH = Path("tests/fixtures/qwen_degenerate.jsonl")
T2S_DEG_PATH = Path("tests/fixtures/t2s_degenerate.jsonl")
VALID_200_PATH = Path("data/t2s_bench/valid_subset_200.jsonl")
ANALYSIS_WINDOW = 24
N_LAYERS = 28  # Qwen2.5-1.5B transformer layers (layer 0 = embeddings datum)
CONTROL_N = 100
ENGAGEMENT_N = 20
ENGAGEMENT_MAX_NEW_TOKENS = 128
WORD_LOOP_PROMPT = "word word word word word word word word word word"
WORD_LOOP_TEMP = 0.7
WORD_LOOP_TOP_P = 0.9
WORD_LOOP_MAX_NEW_TOKENS = 64
CALIBRATION_TEXT = "The quick brown fox jumps over the lazy dog."

# The 100 control ids MUST match the list printed in
# docs/SUBSTRATE_LEAKAGE_PROBE.md; if they do not, STOP and report.
EXPECTED_CONTROL_IDS = [
    0, 24, 28, 32, 36, 46, 49, 52, 53, 54, 56, 57, 58, 65, 71, 72, 78, 80,
    82, 83, 98, 101, 107, 114, 117, 119, 121, 122, 128, 135, 140, 142, 148,
    150, 152, 157, 161, 172, 176, 181, 182, 186, 189, 196, 204, 205, 209,
    214, 219, 222, 232, 235, 236, 255, 256, 258, 259, 260, 271, 276, 282,
    283, 290, 308, 309, 316, 319, 321, 328, 329, 331, 332, 339, 343, 349,
    350, 355, 359, 360, 376, 388, 390, 396, 404, 407, 412, 413, 415, 417,
    433, 444, 454, 458, 466, 469, 470, 473, 474, 490, 496,
]

# Per-layer signal names used for per-layer AUROC.
LAYER_SIGNALS = ["pr", "svse", "er"]
# Text-level signal names used for text-level AUROC.
TEXT_SIGNALS = ["ctr", "distinct_2"]


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


def select_prose_control(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """random.Random(42).sample(records, 100) sorted by corpus id."""
    rng = random.Random(SEED)
    selected = rng.sample(records, CONTROL_N)
    selected_sorted = sorted(selected, key=lambda r: r["id"])
    ids = [int(r["id"]) for r in selected_sorted]
    if ids != EXPECTED_CONTROL_IDS:
        raise SystemExit(
            "[STOP] Prose control ids do NOT match docs/SUBSTRATE_LEAKAGE_PROBE.md. "
            f"got {ids}"
        )
    return selected_sorted


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_windowed_layer_metrics(
    hidden_state: torch.Tensor,
) -> Dict[str, float]:
    """Compute PR / SVSE / effective_rank on a (1, seq, hidden) hidden state,
    restricted to the LAST 24 token positions."""
    win = hidden_state[0, -ANALYSIS_WINDOW:, :].unsqueeze(0)  # (1, 24, hidden)
    return {
        "pr": float(participation_ratio(win).item()),
        "svse": float(singular_value_spectrum_entropy(win, normalize=True).item()),
        "er": float(effective_rank(win).item()),
    }


def replay_record(
    model: torch.nn.Module,
    tokenizer: Any,
    input_text: str,
) -> Tuple[List[torch.Tensor], int]:
    """One teacher-forced forward pass over ``input_text``.

    Returns (hidden_states, n_tokens); hidden_states[0] = embeddings and
    hidden_states[i] = output of layer i (i = 1..28).
    """
    enc = tokenizer(input_text, return_tensors="pt").to(model.device)
    n_tokens = int(enc["input_ids"].shape[-1])
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states, n_tokens


def build_record_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    corpus: str,
    record: Dict[str, Any],
    input_text: str,
) -> Dict[str, Any]:
    """Build the per-record, per-layer metric payload."""
    hidden_states, n_tokens = replay_record(model, tokenizer, input_text)
    layers: Dict[str, Dict[str, float]] = {}
    for layer in range(N_LAYERS + 1):  # 0..28 (0 = embeddings datum)
        layers[str(layer)] = compute_windowed_layer_metrics(hidden_states[layer])

    # Text-level signals over the same LAST-24 window convention.
    enc = tokenizer(input_text, return_tensors="pt").to(model.device)
    d2, _repeated = compute_token_distinct_2_fast(
        enc["input_ids"], prompt_len=0, window_len=ANALYSIS_WINDOW
    )
    ctr = compute_coherent_token_ratio(record["text"])

    return {
        "corpus": corpus,
        "record_id": int(record["id"]),
        "n_tokens": n_tokens,
        "text": input_text,
        "text_signals": {"ctr": float(ctr), "distinct_2": float(d2)},
        "layers": layers,
    }


def record_metric_tuple(payload: Dict[str, Any]) -> Tuple[Any, ...]:
    """Determinism-smoke tuple: all per-layer + text-level values."""
    parts: List[Any] = []
    for layer in range(N_LAYERS + 1):
        m = payload["layers"][str(layer)]
        parts.append((m["pr"], m["svse"], m["er"]))
    parts.append((payload["text_signals"]["ctr"], payload["text_signals"]["distinct_2"]))
    return tuple(parts)


# ---------------------------------------------------------------------------
# AUROC (rank-based Mann-Whitney, numpy only)
# ---------------------------------------------------------------------------
def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean rank)."""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def mann_whitney_auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Rank-based Mann-Whitney U / (n_pos * n_neg).

    positives = degenerate-class values; negatives = prose-control values.
    AUC > 0.5  -> degenerate-higher
    AUC < 0.5  -> degenerate-lower
    """
    pos = np.asarray(positives, dtype=np.float64)
    neg = np.asarray(negatives, dtype=np.float64)
    n_pos = len(pos)
    n_neg = len(neg)
    all_vals = np.concatenate([pos, neg])
    ranks = average_ranks(all_vals)
    rank_sum_pos = float(ranks[:n_pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auc_direction(auc: float) -> str:
    return "degenerate-higher" if auc >= 0.5 else "degenerate-lower"


def separation_magnitude(auc: float) -> float:
    return float(max(auc, 1.0 - auc))


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n_non_finite = int(np.count_nonzero(~np.isfinite(arr)))
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_non_finite": n_non_finite,
    }


# ---------------------------------------------------------------------------
# AUROC tables
# ---------------------------------------------------------------------------
def build_auroc_tables(
    degenerate: List[Dict[str, Any]],
    prose: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-contrast AUROC tables for per-layer and text-level signals."""
    result: Dict[str, Any] = {}

    # Per-layer signals.
    for sig in LAYER_SIGNALS:
        per_layer = []
        for layer in range(N_LAYERS + 1):
            pos = [r["layers"][str(layer)][sig] for r in degenerate]
            neg = [r["layers"][str(layer)][sig] for r in prose]
            auc = mann_whitney_auc(pos, neg)
            per_layer.append(
                {
                    "layer": layer,
                    "auc": auc,
                    "direction": auc_direction(auc),
                    "separation": separation_magnitude(auc),
                }
            )
        # Best layer = max separation magnitude.
        best = max(per_layer, key=lambda e: e["separation"])
        result[sig] = {
            "signal": sig,
            "per_layer": per_layer,
            "best_layer": best["layer"],
            "best_auc": best["auc"],
            "best_direction": best["direction"],
            "best_separation": best["separation"],
            # Distributions at the best layer.
            "best_layer_dist": {
                "degenerate": summarize_distribution(
                    [r["layers"][str(best["layer"])][sig] for r in degenerate]
                ),
                "prose": summarize_distribution(
                    [r["layers"][str(best["layer"])][sig] for r in prose]
                ),
            },
        }

    # Text-level signals.
    for sig in TEXT_SIGNALS:
        pos = [r["text_signals"][sig] for r in degenerate]
        neg = [r["text_signals"][sig] for r in prose]
        auc = mann_whitney_auc(pos, neg)
        result[sig] = {
            "signal": sig,
            "auc": auc,
            "direction": auc_direction(auc),
            "separation": separation_magnitude(auc),
            "dist": {
                "degenerate": summarize_distribution(pos),
                "prose": summarize_distribution(neg),
            },
        }

    return result


# ---------------------------------------------------------------------------
# Engagement arm
# ---------------------------------------------------------------------------
def run_engagement_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    qwen_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Governor engagement evidence on 20 seeded qwen_degenerate prompts plus
    the canonical word-loop positive-control probe."""
    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        use_code_filter=True,
    )
    trusted = [tokenizer(CALIBRATION_TEXT, return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)
    print(f"[engagement] governor calibrated on {CALIBRATION_TEXT!r}")

    # 20 seeded prompts: random.Random(42).sample(100, 20), sorted by id.
    selected = sorted(
        random.Random(SEED).sample(qwen_records, ENGAGEMENT_N),
        key=lambda r: r["id"],
    )
    selected_ids = [int(r["id"]) for r in selected]

    prompt_results: List[Dict[str, Any]] = []
    for rec in selected:
        d = rec["decoding"]
        seed_all(SEED)  # SEED=42 everywhere; applied immediately before generate
        with torch.no_grad():
            out = governor.generate(
                rec["prompt"],
                max_new_tokens=ENGAGEMENT_MAX_NEW_TOKENS,
                temperature=d["temperature"],
                top_p=d["top_p"],
                do_sample=d["do_sample"],
                repetition_penalty=d["repetition_penalty"],
                intervene=True,
            )
        log = out["intervention_log"]
        deltas = [r["residual_delta_norm"] for r in log]
        prompt_results.append(
            {
                "record_id": int(rec["id"]),
                "prompt": rec["prompt"],
                "decoding_params": d,
                "n_decisions": len(out["decisions"]),
                "n_injections": len(log),
                "intervened": bool(out["intervened"]),
                "injection_mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                "injection_max_delta": float(np.max(deltas)) if deltas else 0.0,
                "injection_layers": list(
                    dict.fromkeys(r["layer"] for r in log)
                ),
                "gen_text_head": (out.get("gen_only_text") or "")[:200],
            }
        )
        print(
            f"[engagement] id={rec['id']} decisions={len(out['decisions'])} "
            f"injections={len(log)}"
        )

    # Word-loop positive-control reference.
    seed_all(SEED)
    with torch.no_grad():
        out = governor.generate(
            WORD_LOOP_PROMPT,
            max_new_tokens=WORD_LOOP_MAX_NEW_TOKENS,
            temperature=WORD_LOOP_TEMP,
            top_p=WORD_LOOP_TOP_P,
            intervene=True,
        )
    log = out["intervention_log"]
    deltas = [r["residual_delta_norm"] for r in log]
    word_loop_result = {
        "probe": "word_loop",
        "prompt": WORD_LOOP_PROMPT,
        "temperature": WORD_LOOP_TEMP,
        "top_p": WORD_LOOP_TOP_P,
        "max_new_tokens": WORD_LOOP_MAX_NEW_TOKENS,
        "n_decisions": len(out["decisions"]),
        "n_injections": len(log),
        "intervened": bool(out["intervened"]),
        "injection_mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "injection_max_delta": float(np.max(deltas)) if deltas else 0.0,
        "injection_layers": list(dict.fromkeys(r["layer"] for r in log)),
        "gen_text_head": (out.get("gen_only_text") or "")[:200],
    }
    print(
        f"[engagement] word_loop decisions={len(out['decisions'])} "
        f"injections={len(log)}"
    )

    return {
        "selected_ids": selected_ids,
        "prompt_results": prompt_results,
        "word_loop": word_loop_result,
        "calibration_text": CALIBRATION_TEXT,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _fmt_dist_value(v: float) -> str:
    """Format a distribution statistic, preserving non-finite values verbatim."""
    if v != v:  # NaN
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.4f}"


def write_markdown_report(
    ctx: Dict[str, Any],
    auroc: Dict[str, Any],
    engagement: Dict[str, Any],
    out_path: Path,
) -> None:
    """Write docs/gate22/GATE22_AUROC_RESULTS.md (strictly factual)."""
    md: List[str] = []

    md.append("# DS-024 — Gate 2.2 AUROC measurement results")
    md.append("")
    md.append("> MEASUREMENT REPORT. This document reports measured values only.")
    md.append("> No threshold is created or modified, no gate script is touched,")
    md.append("> and no green/red verdict is offered. Ratification belongs to")
    md.append("> human review and humans.")
    md.append("")
    md.append("## Gate context (quoted from the task greenlight line)")
    md.append("")
    md.append("> \"AUROC >= 0.92 vs a degenerate corpus, SEED=42, engagement evidence,")
    md.append("> STOP-DON'T-NEGOTIATE.\"")
    md.append("")
    md.append("This report MEASURES and REPORTS. It does not render the verdict.")
    md.append("")

    # Run metadata
    md.append("## Run metadata")
    md.append("")
    meta = ctx["metadata"]
    md.append(format_table_md(
        [
            ["Model", meta["model_name"]],
            ["Revision", meta["model_revision"]],
            ["Device", meta["device"]],
            ["dtype", meta["dtype"]],
            ["torch", meta["torch_version"]],
            ["transformers", meta["transformers_version"]],
            ["numpy", meta["numpy_version"]],
            ["SEED", str(meta["seed"])],
            ["Control selection seed", str(meta["control_seed"])],
            ["Analysis window", f"last {ANALYSIS_WINDOW} token positions"],
            ["Layers", f"0 (embeddings datum) .. {N_LAYERS}"],
            ["bmm Triton override", meta["bmm_override"]],
            ["Wall clock (s)", f"{meta['wall_clock_s']:.1f}"],
        ],
        ["Field", "Value"],
    ))
    md.append("")
    md.append("Corpus sha256 (qwen_degenerate):")
    md.append("")
    md.append("```text")
    md.append(meta["qwen_degenerate_sha256"])
    md.append("```")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("Two qwen_degenerate records were replayed twice (teacher-forced, ")
    md.append("single forward pass each). The full metric tuple (29 layers x 3 ")
    md.append("per-layer signals + 2 text-level signals) must be identical across ")
    md.append("the two runs.")
    md.append("")
    smoke_rows = []
    for row in ctx["determinism_smoke"]["pairs"]:
        smoke_rows.append([str(row["record_id"]), str(row["n_tokens"]), row["identical"]])
    smoke_rows.append(["", "ALL", str(ctx["determinism_smoke"]["all_identical"])])
    md.append(format_table_md(smoke_rows, ["record_id", "n_tokens", "identical"]))
    md.append("")

    # Numerical notes
    md.append("## Numerical notes")
    md.append("")
    md.append("- All three per-layer SVD signals force float32 internally "
             "(`core/metrics.py`). SVD is computed on-device for the 24x1536 "
             "analysis window (24*1536 < 1,000,000).")
    md.append("- `singular_value_spectrum_entropy(normalize=True)` divides by "
             "`log(rank + eps)` with `eps=1e-8`; for a rank-1 analysis window "
             "(e.g. the layer-0 embeddings of a fully repeated 24-token tail) "
             "`rank + eps` rounds to `1.0` in float32, so the normalized "
             "entropy is `inf`. Non-finite counts are reported in every "
             "distribution table (`n_non_finite` column); rank-based AUROC "
             "handles `inf` correctly (top rank for `+inf`).")
    md.append("")

    # Per-contrast AUROC tables
    for contrast_name, contrast_key in [
        ("Primary contrast (qwen_degenerate vs prose)", "primary"),
        ("Secondary contrast (t2s_degenerate vs prose)", "secondary"),
    ]:
        md.append(f"## {contrast_name}")
        md.append("")
        c = auroc[contrast_key]
        md.append("### Per-layer signal AUROC (degenerate = positive class)")
        md.append("")
        for sig in LAYER_SIGNALS:
            s = c[sig]
            md.append(f"#### {sig.upper()} — best layer: {s['best_layer']} "
                      f"(AUC {s['best_auc']:.4f}, {s['best_direction']}, "
                      f"sep {s['best_separation']:.4f})")
            md.append("")
            rows = []
            for entry in s["per_layer"]:
                rows.append([
                    str(entry["layer"]),
                    f"{entry['auc']:.4f}",
                    entry["direction"],
                    f"{entry['separation']:.4f}",
                ])
            md.append(format_table_md(rows, ["layer", "AUC", "direction", "separation"]))
            md.append("")
            md.append(f"Best-layer distributions ({sig.upper()}, layer {s['best_layer']}):")
            md.append("")
            d_rows = []
            for cls in ["degenerate", "prose"]:
                d = s["best_layer_dist"][cls]
                d_rows.append([
                    cls,
                    _fmt_dist_value(d["mean"]),
                    _fmt_dist_value(d["median"]),
                    _fmt_dist_value(d["p10"]),
                    _fmt_dist_value(d["p90"]),
                    str(d["n_non_finite"]),
                ])
            md.append(format_table_md(
                d_rows, ["class", "mean", "median", "p10", "p90", "n_non_finite"]
            ))
            md.append("")

        md.append("### Text-level signal AUROC")
        md.append("")
        t_rows = []
        for sig in TEXT_SIGNALS:
            s = c[sig]
            t_rows.append([
                sig,
                f"{s['auc']:.4f}",
                s["direction"],
                f"{s['separation']:.4f}",
            ])
        md.append(format_table_md(
            t_rows, ["signal", "AUC", "direction", "separation"]
        ))
        md.append("")
        md.append("Text-level distributions:")
        md.append("")
        for sig in TEXT_SIGNALS:
            s = c[sig]
            md.append(f"#### {sig}")
            md.append("")
            dist_rows = []
            for cls in ["degenerate", "prose"]:
                d = s["dist"][cls]
                dist_rows.append([
                    cls,
                    _fmt_dist_value(d["mean"]),
                    _fmt_dist_value(d["median"]),
                    _fmt_dist_value(d["p10"]),
                    _fmt_dist_value(d["p90"]),
                    str(d["n_non_finite"]),
                ])
            md.append(format_table_md(
                dist_rows, ["class", "mean", "median", "p10", "p90", "n_non_finite"]
            ))
            md.append("")

    # Engagement arm
    md.append("## Engagement-evidence arm")
    md.append("")
    md.append(f"Governor: `ActiveVarietyGovernor(model, tokenizer=tokenizer, "
              f"use_code_filter=True)`, calibrated on "
              f"`{engagement['calibration_text']!r}`.")
    md.append("")
    md.append(f"20 seeded qwen_degenerate prompts (seed {SEED}), decoding per "
              f"each corpus record's own params, `max_new_tokens="
              f"{ENGAGEMENT_MAX_NEW_TOKENS}`.")
    md.append("")
    md.append(f"Selected record ids: `{engagement['selected_ids']}`")
    md.append("")
    md.append("### Per-prompt engagement results")
    md.append("")
    e_rows = []
    for pr in engagement["prompt_results"]:
        e_rows.append([
            str(pr["record_id"]),
            str(pr["n_decisions"]),
            str(pr["n_injections"]),
            str(pr["intervened"]),
            f"{pr['injection_mean_delta']:.4f}",
            f"{pr['injection_max_delta']:.4f}",
            str(pr["injection_layers"]),
        ])
    md.append(format_table_md(
        e_rows,
        ["id", "decisions", "injections", "intervened", "mean_delta", "max_delta", "layers"],
    ))
    md.append("")
    md.append("### Word-loop positive-control reference")
    md.append("")
    wl = engagement["word_loop"]
    md.append(format_table_md(
        [
            ["Prompt", f"`{wl['prompt']}`"],
            ["temperature / top_p / max_new_tokens",
             f"{wl['temperature']} / {wl['top_p']} / {wl['max_new_tokens']}"],
            ["decisions", str(wl["n_decisions"])],
            ["injections", str(wl["n_injections"])],
            ["intervened", str(wl["intervened"])],
            ["mean_delta", f"{wl['injection_mean_delta']:.4f}"],
            ["max_delta", f"{wl['injection_max_delta']:.4f}"],
            ["layers", str(wl["injection_layers"])],
        ],
        ["Field", "Value"],
    ))
    md.append("")
    md.append("### Engagement-arm observations (strictly factual)")
    md.append("")
    obs = engagement.get("observations", [])
    for line in obs:
        md.append(f"- {line}")
    md.append("")

    # Data
    md.append("## Data")
    md.append("")
    md.append("- Per-record per-layer signal values: `docs/gate22/gate22_results.jsonl`")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


def build_observations(engagement: Dict[str, Any], auroc: Dict[str, Any]) -> List[str]:
    """Strictly factual bullet observations for the engagement arm."""
    obs: List[str] = []
    prompt_results = engagement["prompt_results"]
    n_fired = sum(1 for p in prompt_results if p["n_injections"] > 0)
    total_inj = sum(p["n_injections"] for p in prompt_results)
    mean_fired = sum(p["n_injections"] for p in prompt_results) / len(prompt_results)
    max_inj = max((p["n_injections"] for p in prompt_results), default=0)
    obs.append(
        f"On {len(prompt_results)} seeded qwen_degenerate prompts, the governor "
        f"fired on {n_fired}/{len(prompt_results)} prompts, logging {total_inj} "
        f"total residual injections (mean {mean_fired:.1f} per prompt, max {max_inj})."
    )
    if prompt_results:
        obs.append(
            f"Prompt-level injection counts ranged from "
            f"{min(p['n_injections'] for p in prompt_results)} to "
            f"{max_inj}."
        )
    wl = engagement["word_loop"]
    obs.append(
        f"The word-loop positive-control probe logged "
        f"{wl['n_injections']} injection(s) across "
        f"{wl['max_new_tokens']} generated tokens "
        f"(mean delta {wl['injection_mean_delta']:.4f} L2)."
    )
    # Factual per-contrast best-layer summaries (context for the AUROC tables).
    for contrast_name, contrast_key in [
        ("qwen_degenerate vs prose", "primary"),
        ("t2s_degenerate vs prose", "secondary"),
    ]:
        c = auroc[contrast_key]
        bits = []
        for sig in LAYER_SIGNALS:
            s = c[sig]
            bits.append(
                f"{sig} best AUC {s['best_auc']:.4f} at layer "
                f"{s['best_layer']} ({s['best_direction']})"
            )
        for sig in TEXT_SIGNALS:
            s = c[sig]
            bits.append(
                f"{sig} AUC {s['auc']:.4f} ({s['direction']})"
            )
        obs.append(f"{contrast_name}: " + "; ".join(bits) + ".")
    return obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 2.2 AUROC measurement")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap on per-corpus records (for smoke/dev runs).")
    parser.add_argument("--skip-engagement", action="store_true",
                        help="Skip the engagement arm (dev only).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-024 Gate 2.2 AUROC measurement")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {args.model}@{args.revision}")
    print(f"torch: {torch.__version__} | transformers: {_tf_version()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device} | layers: {N_LAYERS}")

    # ------------------------------------------------------------------
    # Load corpora
    # ------------------------------------------------------------------
    qwen_records = load_jsonl(QWEN_DEG_PATH)
    t2s_records = load_jsonl(T2S_DEG_PATH)
    valid_200 = load_jsonl(VALID_200_PATH)
    prose_records = select_prose_control(valid_200)
    print(f"qwen_degenerate: {len(qwen_records)} | t2s_degenerate: {len(t2s_records)} | prose control: {len(prose_records)}")
    print("prose control ids match docs/SUBSTRATE_LEAKAGE_PROBE.md: TRUE")

    if args.max_records is not None:
        qwen_records = qwen_records[: args.max_records]
        t2s_records = t2s_records[: args.max_records]
        prose_records = prose_records[: args.max_records]
        print(f"[dev] capped each corpus at {args.max_records} records")

    # ------------------------------------------------------------------
    # Determinism smoke FIRST
    # ------------------------------------------------------------------
    print("\n--- Determinism smoke ---")
    smoke_pairs = []
    all_identical = True
    for rec in qwen_records[:2]:
        input_text = rec["prompt"] + rec["text"]
        a = build_record_metrics(model, tokenizer, "qwen_degenerate", rec, input_text)
        b = build_record_metrics(model, tokenizer, "qwen_degenerate", rec, input_text)
        ta = record_metric_tuple(a)
        tb = record_metric_tuple(b)
        identical = ta == tb
        all_identical = all_identical and identical
        smoke_pairs.append(
            {"record_id": int(rec["id"]), "n_tokens": a["n_tokens"], "identical": str(identical)}
        )
        print(f"  record {rec['id']}: identical={identical} n_tokens={a['n_tokens']}")
    if not all_identical:
        print("[STOP] Determinism smoke FAILED: metric tuples differ across replays.")
        print("Aborting; see report.")
        # Write what we have so far? The task says STOP and report. We exit.
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL")

    # ------------------------------------------------------------------
    # Replay all records and build per-record metrics
    # ------------------------------------------------------------------
    print("\n--- Replaying corpora (teacher-forced) ---")
    results: List[Dict[str, Any]] = []

    def process_corpus(corpus: str, records: List[Dict[str, Any]], join_prompt: bool) -> None:
        for i, rec in enumerate(records):
            if join_prompt:
                input_text = rec["prompt"] + rec["text"]
            else:
                input_text = rec["text"]
            payload = build_record_metrics(model, tokenizer, corpus, rec, input_text)
            results.append(payload)
            if (i + 1) % 25 == 0 or i == len(records) - 1:
                print(f"  [{corpus}] {i+1}/{len(records)}")

    process_corpus("qwen_degenerate", qwen_records, join_prompt=True)
    process_corpus("t2s_degenerate", t2s_records, join_prompt=False)
    process_corpus("prose", prose_records, join_prompt=False)

    # ------------------------------------------------------------------
    # AUROC per signal per layer, per contrast
    # ------------------------------------------------------------------
    print("\n--- Computing AUROC ---")
    qwen_res = [r for r in results if r["corpus"] == "qwen_degenerate"]
    t2s_res = [r for r in results if r["corpus"] == "t2s_degenerate"]
    prose_res = [r for r in results if r["corpus"] == "prose"]

    auroc = {
        "primary": build_auroc_tables(qwen_res, prose_res),
        "secondary": build_auroc_tables(t2s_res, prose_res),
    }
    for contrast_name, contrast_key in [("primary", "primary"), ("secondary", "secondary")]:
        c = auroc[contrast_key]
        print(f"\n[{contrast_name}]")
        for sig in LAYER_SIGNALS:
            s = c[sig]
            print(f"  {sig}: best AUC {s['best_auc']:.4f} at layer {s['best_layer']} "
                  f"({s['best_direction']}, sep {s['best_separation']:.4f})")
        for sig in TEXT_SIGNALS:
            s = c[sig]
            print(f"  {sig}: AUC {s['auc']:.4f} ({s['direction']}, sep {s['separation']:.4f})")

    # ------------------------------------------------------------------
    # Engagement arm
    # ------------------------------------------------------------------
    print("\n--- Engagement-evidence arm ---")
    engagement: Dict[str, Any] = {}
    if args.skip_engagement:
        print("[dev] engagement arm skipped")
    else:
        engagement = run_engagement_arm(model, tokenizer, qwen_records)
        engagement["observations"] = build_observations(engagement, auroc)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    qwen_sha = _sha256_of_file(QWEN_DEG_PATH)
    metadata = {
        "model_name": args.model,
        "model_revision": args.revision,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": _tf_version(),
        "numpy_version": np.__version__,
        "seed": SEED,
        "control_seed": SEED,
        "bmm_override": "deregistered",
        "qwen_degenerate_sha256": qwen_sha,
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": {
            "pairs": smoke_pairs,
            "all_identical": str(all_identical),
        },
    }

    out_jsonl = Path("docs/gate22/gate22_results.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_jsonl} ({len(results)} records)")

    out_md = Path("docs/gate22/GATE22_AUROC_RESULTS.md")
    if engagement:
        write_markdown_report(ctx, auroc, engagement, out_md)
    else:
        # Engagement arm skipped: write report with placeholder observations.
        engagement_placeholder = {
            "calibration_text": CALIBRATION_TEXT,
            "selected_ids": [],
            "prompt_results": [],
            "word_loop": {
                "probe": "word_loop",
                "prompt": WORD_LOOP_PROMPT,
                "temperature": WORD_LOOP_TEMP,
                "top_p": WORD_LOOP_TOP_P,
                "max_new_tokens": WORD_LOOP_MAX_NEW_TOKENS,
                "n_decisions": 0,
                "n_injections": 0,
                "intervened": False,
                "injection_mean_delta": 0.0,
                "injection_max_delta": 0.0,
                "injection_layers": [],
                "gen_text_head": "",
            },
            "observations": [],
        }
        write_markdown_report(ctx, auroc, engagement_placeholder, out_md)
    print(f"wrote {out_md}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("MEASUREMENT COMPLETE (no verdict offered)")


def _tf_version() -> str:
    import transformers as _tf

    return _tf.__version__


def _sha256_of_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
