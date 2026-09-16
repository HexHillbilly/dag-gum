#!/usr/bin/env python3
"""DS-034: Spectral PR liveness on t2s_degenerate (MEASUREMENT ONLY).

DS-033 (8,-5) cross-fixture diagnostic showed detection-gap dominates on
t2s_degenerate: 93/100 records have predicate_true_steps == 0 (the bigram
detector never fires on macro-syntax motifs). The shadow PR diagnostic hook
(55-85% layer span) sees a mean PR of 5.63 on detection-gap records, well
below the frozen T_PR(2) = 10.954796.

This probe tests the EXISTING spectral PR instrument on t2s_degenerate:
does PR cross the frozen threshold on records where the bigram is silent?
Zero new code. Zero new architecture. Uses:
  - participation_ratio() from core/metrics.py (existing instrument)
  - FROZEN_THRESHOLDS.md ds-025 freeze (existing thresholds, T_PR per layer)
  - DS-033 detection-gap classification from
    docs/gate23/production_cross_fixture_results.jsonl (REUSED, not re-computed)

Decision rule (for the reader; this script offers no verdict):
  - If PR fires where the bigram predicate does not -> dual-predicate fix
    (bigram OR spectral -> logit-penalty path).
  - If PR is also silent -> open macro-syntax window design.

Method (per the DS-034 task file):
  1. REPLAY (teacher-forced): one forward pass over each t2s_degenerate record
     with output_hidden_states=True. Input text = record["text"].
     Analysis window = LAST 24 token positions (Gate 2.3 / ds-025 convention).
  2. Per record, per RFC-004 candidate layer {2,12,26}: extract hidden states
     at that layer, compute participation_ratio() over the trailing 24
     positions, record pr and whether pr < T_PR(layer) (the frozen threshold).
  3. Join each record against its DS-033 classification (detection-gap vs
     actuation-gap vs rescued). Primary question is about the 93 detection-gap
     (bigram-blind) records; non-detection-gap records are logged for
     comparison.
  4. Determinism smoke FIRST: 2 records replayed twice. Per-layer PR must be
     identical. STOP if not.

Environment notes (identical to ds-025/ds-027/ds-033):
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from AVG.core.metrics import participation_ratio  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-034 measurement seed (matches ds-025..ds-033)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ANALYSIS_WINDOW = 24  # trailing token positions (Gate 2.3 / ds-025 convention)
CANDIDATE_LAYERS = ["2", "12", "26"]  # RFC-004 candidate layers

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
DS033_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/spectral_pr_t2s_results.jsonl")
OUTPUT_MD = Path("docs/gate23/SPECTRAL_PR_T2S_RESULTS.md")

# DS-033 diagnostic classes (reused; do NOT re-compute).
CLASS_DETECTION_GAP = "detection-gap"
CLASS_ACTUATION_GAP = "actuation-gap"
CLASS_RESCUED = "rescued"

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
# Frozen thresholds (read from the Part A freeze file; do NOT re-derive)
# ---------------------------------------------------------------------------
def load_frozen_thresholds(path: Path) -> Dict[str, Any]:
    """Parse the machine-parseable JSON freeze block from FROZEN_THRESHOLDS.md.

    This is the SINGLE source of truth; DS-034 does NOT re-derive any
    threshold. Returns {"layers": {layer: {"pr": {...}, "svse": {...}}}}.
    """
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md has no ```json freeze block.")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md freeze block is unterminated.")
    block = text[start:end]
    freeze = json.loads(block)
    for layer in CANDIDATE_LAYERS:
        if layer not in freeze["layers"]:
            raise SystemExit(
                f"[STOP] FROZEN_THRESHOLDS.md missing candidate layer {layer}."
            )
        for sig in ("pr", "svse"):
            for key in ("T", "band_low", "band_high"):
                if key not in freeze["layers"][layer][sig]:
                    raise SystemExit(
                        f"[STOP] FROZEN_THRESHOLDS.md missing {layer}.{sig}.{key}."
                    )
    return freeze


# ---------------------------------------------------------------------------
# Replay / metric helpers
# ---------------------------------------------------------------------------
def replay_record(
    model: torch.nn.Module,
    tokenizer: Any,
    input_text: str,
) -> Tuple[Tuple[torch.Tensor, ...], int]:
    """Teacher-forced single forward pass with output_hidden_states=True."""
    enc = tokenizer(input_text, return_tensors="pt").to(model.device)
    n_tokens = int(enc["input_ids"].shape[-1])
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states, n_tokens


def windowed_pr(hidden_state: torch.Tensor) -> float:
    """participation_ratio() over the last-24 token window (Gate 2.3 conv)."""
    win = hidden_state[0, -ANALYSIS_WINDOW:, :].unsqueeze(0)
    return float(participation_ratio(win).item())


def build_record_payload(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
    ds033: Optional[Dict[str, Any]],
    t_pr: Dict[str, float],
) -> Dict[str, Any]:
    """Per-record spectral PR payload at the RFC-004 candidate layers."""
    input_text = record["text"]
    hidden_states, n_tokens = replay_record(model, tokenizer, input_text)
    layers: Dict[str, Dict[str, Any]] = {}
    for layer in CANDIDATE_LAYERS:
        hs = hidden_states[int(layer)]
        pr = windowed_pr(hs)
        layers[layer] = {
            "pr": pr,
            "t_pr": t_pr[layer],
            "pr_below_t": bool(pr < t_pr[layer]),
        }

    # Earliest layer where PR crosses the frozen threshold.
    first_fire: Optional[str] = None
    for layer in CANDIDATE_LAYERS:
        if layers[layer]["pr_below_t"]:
            first_fire = layer
            break
    fires_at_any = first_fire is not None

    # Reused DS-033 classification (do NOT re-compute).
    diag_class = None
    predicate_true_steps = None
    if ds033 is not None:
        diag_class = ds033["active"]["diagnostic_class"]
        predicate_true_steps = int(ds033["active"]["predicate_true_steps"])

    payload: Dict[str, Any] = {
        "fixture": "t2s_degenerate",
        "record_id": int(record["id"]),
        "seed": SEED,
        "diagnostic_class": diag_class,
        "predicate_true_steps": predicate_true_steps,
        "detection_gap": bool(diag_class == CLASS_DETECTION_GAP),
        "n_tokens": n_tokens,
        "text": input_text,
        "layers": layers,
        "fires_at_any_layer": bool(fires_at_any),
        "first_fire_layer": first_fire,
    }
    return payload


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    t_pr: Dict[str, float],
) -> Dict[str, Any]:
    """Determinism smoke (DS-034): 2 records replayed twice; per-layer PR must
    be identical. STOP (sys.exit 1) if not."""
    print("\n--- Determinism smoke (2 t2s_degenerate records replayed twice) ---")
    pairs: List[Dict[str, Any]] = []
    all_identical = True
    for rec in records[:2]:
        rid = int(rec["id"])
        a = build_record_payload(model, tokenizer, rec, None, t_pr)
        b = build_record_payload(model, tokenizer, rec, None, t_pr)
        ta = tuple((a["layers"][l]["pr"], a["layers"][l]["pr_below_t"])
                   for l in CANDIDATE_LAYERS)
        tb = tuple((b["layers"][l]["pr"], b["layers"][l]["pr_below_t"])
                   for l in CANDIDATE_LAYERS)
        identical = ta == tb
        all_identical = all_identical and identical
        pairs.append({
            "record_id": rid,
            "n_tokens": a["n_tokens"],
            "pr_a": {l: a["layers"][l]["pr"] for l in CANDIDATE_LAYERS},
            "pr_b": {l: b["layers"][l]["pr"] for l in CANDIDATE_LAYERS},
            "identical": str(identical),
        })
        print(f"  record {rid}: identical={identical} n_tokens={a['n_tokens']} "
              f"pr2a={a['layers']['2']['pr']:.6f} pr2b={b['layers']['2']['pr']:.6f}")
    if not all_identical:
        print("[STOP] Determinism smoke FAILED: per-layer PR differs across "
              "replays.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (per-layer PR, 2 records x 2 replays)")
    return {"pairs": pairs, "all_identical": str(all_identical)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "n_non_finite": 0,
        }
    n_non_finite = int(np.count_nonzero(~np.isfinite(arr)))
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_non_finite": n_non_finite,
    }


def aggregate_tables(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    det = [r for r in results if r["detection_gap"]]
    res = [r for r in results if r["diagnostic_class"] == CLASS_RESCUED]
    act = [r for r in results if r["diagnostic_class"] == CLASS_ACTUATION_GAP]

    # Table 1 — per-layer PR distributions on detection-gap records.
    table1: Dict[str, Dict[str, Any]] = {}
    for layer in CANDIDATE_LAYERS:
        prs = [r["layers"][layer]["pr"] for r in det]
        n_below = sum(1 for r in det if r["layers"][layer]["pr_below_t"])
        dist = summarize_distribution(prs)
        table1[layer] = {
            "n": len(det),
            "mean_pr": dist["mean"],
            "median_pr": dist["median"],
            "p10_pr": dist["p10"],
            "p90_pr": dist["p90"],
            "n_pr_below_t": n_below,
            "pr_below_t_rate": float(n_below) / len(det) if det else float("nan"),
        }

    # Table 2 — detection-gap records: PR liveness summary.
    n_any = sum(1 for r in det if r["fires_at_any_layer"])
    n_layer2 = sum(1 for r in det if r["layers"]["2"]["pr_below_t"])
    n_silent = sum(1 for r in det if not r["fires_at_any_layer"])
    table2 = {
        "n_detection_gap": len(det),
        "n_fires_at_any_layer": n_any,
        "n_fires_at_layer_2": n_layer2,
        "n_silent_all_layers": n_silent,
    }

    # Table 3 — per-layer PR comparison: detection-gap vs rescued.
    table3: Dict[str, List[Dict[str, Any]]] = {}
    for layer in CANDIDATE_LAYERS:
        d_prs = [r["layers"][layer]["pr"] for r in det]
        r_prs = [r["layers"][layer]["pr"] for r in res]
        d_dist = summarize_distribution(d_prs)
        r_dist = summarize_distribution(r_prs)
        table3[layer] = [
            {"class": CLASS_DETECTION_GAP, "n": len(det),
             "mean_pr": d_dist["mean"], "median_pr": d_dist["median"]},
            {"class": CLASS_RESCUED, "n": len(res),
             "mean_pr": r_dist["mean"], "median_pr": r_dist["median"]},
        ]

    return {
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "n_records": len(results),
        "n_detection_gap": len(det),
        "n_actuation_gap": len(act),
        "n_rescued": len(res),
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


def write_markdown_report(
    ctx: Dict[str, Any],
    freeze: Dict[str, Any],
    out_path: Path,
) -> None:
    md: List[str] = []
    md.append("# DS-034 — Spectral PR liveness on t2s_degenerate "
              "(MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. This probe tests the EXISTING spectral PR "
              "instrument on t2s_degenerate: does PR cross the frozen "
              "threshold on records where the bigram detector is silent? Zero "
              "new code, zero new architecture, no threshold edits, no "
              "controller changes, no new detection logic. No verdict is "
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
    md.append("| Fixture | tests/fixtures/t2s_degenerate.jsonl "
              "(night-001; N=100), record[\\\"text\\\"] input |")
    md.append("| Analysis window | "
              f"last {meta['analysis_window']} token positions |")
    md.append("| Candidate layers | "
              f"{{{', '.join(str(l) for l in meta['candidate_layers'])}}} (RFC-004) |")
    md.append(f"| PR instrument | participation_ratio() from core/metrics.py "
              f"(existing) |")
    md.append("| Thresholds | docs/gate23/FROZEN_THRESHOLDS.md (ds-025 freeze), "
              "T_PR per layer; not re-derived |")
    md.append("| DS-033 classification | REUSED from "
              "docs/gate23/production_cross_fixture_results.jsonl "
              "(do NOT re-compute) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    sm = ctx["determinism_smoke"]
    md.append("Two t2s_degenerate records replayed twice (teacher-forced, single "
              "forward pass each). The per-layer PR tuple at layers {2,12,26} "
              "must be identical across the two runs.")
    md.append("")
    smoke_rows = []
    for pair in sm["pairs"]:
        smoke_rows.append([
            str(pair["record_id"]), str(pair["n_tokens"]),
            fmt(pair["pr_a"]["2"]), fmt(pair["pr_b"]["2"]),
            pair["identical"],
        ])
    smoke_rows.append(["", "ALL", "", "", str(sm["all_identical"])])
    md.append(format_table_md(
        smoke_rows, ["record_id", "n_tokens", "PR layer2 run A",
                     "PR layer2 run B", "identical"],
    ))
    md.append("")

    # Frozen thresholds restated
    md.append("## Frozen thresholds (restated from Part A freeze)")
    md.append("")
    md.append("Thresholds are read from `docs/gate23/FROZEN_THRESHOLDS.md`; "
              "nothing here re-derives them. Criterion for \"PR fires\": "
              "PR < T_PR(layer).")
    md.append("")
    thr_rows = []
    for layer in CANDIDATE_LAYERS:
        t = freeze["layers"][layer]
        thr_rows.append([
            layer, "PR", fmt(t["pr"]["T"], 6), fmt(t["pr"]["band_low"], 6),
            fmt(t["pr"]["band_high"], 6),
        ])
    md.append(format_table_md(
        thr_rows, ["layer", "signal", "T_PR", "band_low", "band_high"],
    ))
    md.append("")

    tabs = ctx["tables"]

    # Table 1
    md.append("## Table 1 — Per-layer PR distributions on detection-gap records")
    md.append("")
    md.append(f"Detection-gap records: {tabs['n_detection_gap']} (predicate_true_"
              f"steps == 0 in DS-033; bigram detector never fired).")
    md.append("")
    t1_rows = []
    for layer in CANDIDATE_LAYERS:
        t1 = tabs["table1"][layer]
        t1_rows.append([
            layer, str(t1["n"]), fmt(t1["mean_pr"]), fmt(t1["median_pr"]),
            fmt(t1["p10_pr"]), fmt(t1["p90_pr"]), str(t1["n_pr_below_t"]),
            fmt(t1["pr_below_t_rate"], 4),
        ])
    md.append(format_table_md(
        t1_rows, ["layer", "n", "mean PR", "median PR", "p10 PR", "p90 PR",
                  "#PR<T", "PR<T rate"],
    ))
    md.append("")

    # Table 2
    md.append("## Table 2 — Detection-gap records: PR liveness summary")
    md.append("")
    t2 = tabs["table2"]
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| Total detection-gap records | {t2['n_detection_gap']} |")
    md.append(f"| PR fires at any layer | {t2['n_fires_at_any_layer']} |")
    md.append(f"| PR fires at layer 2 | {t2['n_fires_at_layer_2']} |")
    md.append(f"| PR silent at all layers | {t2['n_silent_all_layers']} |")
    md.append("")

    # Table 3
    md.append("## Table 3 — Per-layer PR comparison: detection-gap vs rescued")
    md.append("")
    md.append("Rescued records (n=%d) are the non-detection-gap reference where "
              "the bigram predicate fired and ΔDistinct-2 moved." %
              tabs["n_rescued"])
    md.append("")
    t3_rows = []
    for layer in CANDIDATE_LAYERS:
        for row in tabs["table3"][layer]:
            t3_rows.append([
                layer, row["class"], str(row["n"]),
                fmt(row["mean_pr"]), fmt(row["median_pr"]),
            ])
    md.append(format_table_md(
        t3_rows, ["layer", "class", "n", "mean PR", "median PR"],
    ))
    md.append("")

    # Non-detection-gap records (comparison log)
    md.append("## Non-detection-gap records (comparison log)")
    md.append("")
    md.append("The 7 non-detection-gap records (predicate_true_steps > 0 in "
              "DS-033) are logged per-record for comparison; the primary "
              "question concerns the 93 detection-gap blind spots.")
    md.append("")
    nd_rows = []
    for r in ctx["results"]:
        if r["detection_gap"]:
            continue
        nd_rows.append([
            str(r["record_id"]), r["diagnostic_class"],
            str(r["predicate_true_steps"]),
            fmt(r["layers"]["2"]["pr"]), fmt(r["layers"]["12"]["pr"]),
            fmt(r["layers"]["26"]["pr"]),
            str(r["fires_at_any_layer"]),
            r["first_fire_layer"] if r["first_fire_layer"] else "—",
        ])
    md.append(format_table_md(
        nd_rows, ["record_id", "class", "pred_true_steps", "PR L2", "PR L12",
                  "PR L26", "fires any", "first fire"],
    ))
    md.append("")

    # Interpretation
    md.append("## Interpretation (measurement, not a verdict)")
    md.append("")
    md.append("Per the DS-033 locked triad decision rule, this probe feeds the "
              "human decision:")
    md.append("")
    md.append("- If **PR fires where the bigram predicate does not** → "
              "dual-predicate fix (bigram OR spectral → logit-penalty path).")
    md.append("- If **PR is also silent** → open macro-syntax window design.")
    md.append("")
    md.append("This report MEASURES per-layer PR and the fire rate on "
              "detection-gap records. It does not render the verdict.")
    md.append("")

    # Notes
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no threshold edits, no controller changes, "
              "no new detection logic. Uses the EXISTING participation_ratio "
              "instrument and EXISTING frozen thresholds.")
    md.append("- DS-033 detection-gap classification is REUSED from "
              "production_cross_fixture_results.jsonl (join on fixture + "
              "record_id). Do NOT re-compute or re-classify.")
    md.append("- Input text is `record[\"text\"]` from t2s_degenerate.jsonl "
              "(the whole degenerate text). Analysis window = last 24 token "
              "positions, matching the Gate 2.3 / ds-025 convention.")
    md.append("- Hidden states are indexed as `out.hidden_states[int(layer)]`, "
              "identical to score_gate23.py (ds-025/027) so that the measured "
              "PR is directly comparable to the frozen thresholds.")
    md.append("- The criterion for \"PR fires\" is `PR < T_PR(layer)` (the "
              "frozen midpoint threshold), per the DS-034 task parentheticals.")
    md.append("- Per-record results are in `spectral_pr_t2s_results.jsonl`.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-034 spectral PR liveness on t2s_degenerate "
                    "(measurement only)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate SPECTRAL_PR_T2S_RESULTS.md from an "
                             "existing spectral_pr_t2s_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-034 Spectral PR liveness on t2s_degenerate (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"candidate layers: {CANDIDATE_LAYERS}")
    print(f"analysis window: last {ANALYSIS_WINDOW} token positions")

    # ------------------------------------------------------------------
    # Frozen thresholds FIRST (Part A freeze).
    # ------------------------------------------------------------------
    freeze = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    t_pr = {
        layer: float(freeze["layers"][layer]["pr"]["T"])
        for layer in CANDIDATE_LAYERS
    }
    print("frozen thresholds loaded from docs/gate23/FROZEN_THRESHOLDS.md")
    for layer in CANDIDATE_LAYERS:
        print(f"  layer {layer}: T_PR={t_pr[layer]:.6f}")

    # ------------------------------------------------------------------
    # Load fixture + DS-033 classification (reused).
    # ------------------------------------------------------------------
    records = load_jsonl(T2S_DEG)
    if args.max_records is not None:
        records = records[: args.max_records]
        print(f"[dev] capped records at {args.max_records}")
    print(f"t2s_degenerate records: {len(records)}")

    if not DS033_JSONL.exists():
        print(f"[STOP] {DS033_JSONL} does not exist; DS-033 classification is "
              "required for the diagnostic join. Do NOT re-compute DS-033.")
        sys.exit(1)
    ds033 = load_jsonl(DS033_JSONL)
    ds033_by_id: Dict[int, Dict[str, Any]] = {}
    for rec in ds033:
        if rec.get("fixture") == "t2s_degenerate":
            ds033_by_id[int(rec["record_id"])] = rec
    missing = [int(r["id"]) for r in records
               if int(r["id"]) not in ds033_by_id]
    if missing:
        print(f"[STOP] DS-033 JSONL join failed: {len(missing)} t2s_degenerate "
              f"record_ids missing from production_cross_fixture_results.jsonl: "
              f"{sorted(missing)[:20]}")
        sys.exit(1)
    n_det = sum(1 for r in records
                if ds033_by_id[int(r["id"])]["active"]["diagnostic_class"]
                == CLASS_DETECTION_GAP)
    print(f"DS-033 join OK: {len(ds033_by_id)} t2s_degenerate records; "
          f"detection-gap={n_det}")

    # ------------------------------------------------------------------
    # --report-only: regenerate the markdown from an existing JSONL without
    # loading the model (dev convenience; no measurement is performed).
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        results = load_jsonl(OUTPUT_JSONL)
        smoke: Dict[str, Any] = {
            "pairs": [{"record_id": -1, "n_tokens": 1,
                       "pr_a": {"2": 0.0}, "pr_b": {"2": 0.0},
                       "identical": "True"}],
            "all_identical": "True",
        }
        tables = aggregate_tables(results)
        metadata = {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "seed": SEED,
            "analysis_window": ANALYSIS_WINDOW,
            "candidate_layers": CANDIDATE_LAYERS,
            "bmm_override": "deregistered",
            "wall_clock_s": 0.0,
        }
        ctx = {
            "metadata": metadata,
            "determinism_smoke": smoke,
            "tables": tables,
            "results": results,
        }
        write_markdown_report(ctx, freeze, OUTPUT_MD)
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

    # Determinism smoke FIRST.
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
        smoke = {"pairs": [], "all_identical": "True"}
    else:
        smoke = run_determinism_smoke(model, tokenizer, records, t_pr)

    # ==================================================================
    # Replay all records (teacher-forced, single forward pass each).
    # ==================================================================
    print("\n--- Replaying t2s_degenerate (teacher-forced) ---")
    results: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        ds033_rec = ds033_by_id.get(int(rec["id"]))
        payload = build_record_payload(model, tokenizer, rec, ds033_rec, t_pr)
        results.append(payload)
        if (i + 1) % 20 == 0 or i == len(records) - 1:
            print(f"  [{i+1}/{len(records)}] rid={payload['record_id']} "
                  f"class={payload['diagnostic_class']} "
                  f"pr2={payload['layers']['2']['pr']:.4f} "
                  f"pr12={payload['layers']['12']['pr']:.4f} "
                  f"pr26={payload['layers']['26']['pr']:.4f} "
                  f"fire_any={payload['fires_at_any_layer']}")

    # ==================================================================
    # Aggregate (single source of truth), write outputs.
    # ==================================================================
    tables = aggregate_tables(results)

    print("\n" + "=" * 70)
    print("SPECTRAL PR LIVENESS AGGREGATION (detection-gap records)")
    for layer in CANDIDATE_LAYERS:
        t1 = tables["table1"][layer]
        print(f"  layer {layer}: n={t1['n']} mean={t1['mean_pr']:.4f} "
              f"median={t1['median_pr']:.4f} p10={t1['p10_pr']:.4f} "
              f"p90={t1['p90_pr']:.4f} #PR<T={t1['n_pr_below_t']} "
              f"rate={t1['pr_below_t_rate']:.4f}")
    t2 = tables["table2"]
    print(f"  liveness: fires_any={t2['n_fires_at_any_layer']} "
          f"fires_L2={t2['n_fires_at_layer_2']} "
          f"silent_all={t2['n_silent_all_layers']}")
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
        "analysis_window": ANALYSIS_WINDOW,
        "candidate_layers": CANDIDATE_LAYERS,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "results": results,
    }
    write_markdown_report(ctx, freeze, OUTPUT_MD)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("DS-034 MEASUREMENT COMPLETE")


if __name__ == "__main__":
    main()
