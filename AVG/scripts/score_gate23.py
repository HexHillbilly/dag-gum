#!/usr/bin/env python3
"""DS-025: Gate 2.3 acceptance scoring (Part C).

MEASUREMENT ONLY. No thresholds are invented beyond the frozen rule in
``docs/gate23/FROZEN_THRESHOLDS.md`` (Part A), no gate scripts are touched, no
governor/controller.py edits, no asserts on outcomes, and no green/red verdict
and no layer recommendation are produced. Per-layer results are reported;
earliest-passing layer selection is the human's act per the greenlight.

Method (per the DS-025 task file / greenlight):
  1. REPLAY (teacher-forced): one forward pass over each record with
     ``output_hidden_states=True``. Input text is ``prompt + text`` for
     heldout-degenerate and ``text`` for the prose/hazard/schema sets.
     Analysis window = LAST 24 token positions.
  2. Per record, per candidate layer {2,12,26}: participation_ratio and
     singular_value_spectrum_entropy (``core/metrics.py``), plus the RFC-004
     §7.4 v_traj SHADOW metric (log only; decides nothing, not scored).
  3. Per record: ``is_code_syntax_context`` (``core/metrics.py``) for the
     Stage-1 routing measurement.
  4. Apply the FROZEN thresholds with the RFC-004 fusion rule:
     PR primary; if PR is below its band -> fire; if PR is inside its
     ambiguity band -> fire only if SVSE agrees (SVSE below T_SVSE) and NOT
     both-inside-bands (SVSE below its own lower band edge); if PR is above
     its band -> no fire.
  5. Per layer: rank-based Mann-Whitney AUROC (numpy only, direction explicit)
     for PR and SVSE on the held-out contrast (heldout-degenerate vs
     heldout-100 prose); fusion-rule recall on heldout-degenerate; false-
     positive counts on heldout prose + 3 hazard corpora + schema_corpus_v2;
     is_code_syntax_context routing rate per set.
  6. Determinism smoke FIRST: 2 records replayed twice; the per-layer metric
     tuple must be byte-identical, otherwise STOP and report.

Environment notes:
  - torch 2.13 CUDA ``bmm_outer_product`` Triton override is deregistered
    (C compiler absent; see docs/PROBE_B_RECONSTRUCTION.md §6). Env-only.
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount. Verified once.
  - HF cache is read-only here; set HF_HOME to a writable fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np
import torch

# Standard root-as-package header. From scripts/, parents[2] resolves to '/';
# `import AVG.*` resolves via the /AVG mount. (Protocol import convention.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Env-only adaptation: deregister torch 2.13's CUDA bmm Triton override.
try:
    from torch._native import triton_utils as _triton_utils

    _triton_utils.deregister_op_overrides()
    print("[env] torch bmm Triton override deregistered")
except Exception as _env_e:  # pragma: no cover - env dependent
    print(f"[env] bmm override deregistration skipped: {_env_e}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from AVG.core.metrics import (  # noqa: E402
    is_code_syntax_context,
    participation_ratio,
    singular_value_spectrum_entropy,
)
from AVG.governor.controller import compute_token_distinct_2_fast  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42  # replay seed (all runs seeded; matches Gate 2.2 measurement seed)
CONTROL_SEED = 42  # valid_subset_200 prose-control selection seed
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ANALYSIS_WINDOW = 24
CANDIDATE_LAYERS = ["2", "12", "26"]
CONTROL_N = 100
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")

# Paths to replayed corpora.
HELDOUT_DEG_PATH = Path("tests/fixtures/heldout_degenerate.jsonl")
VALID_200_PATH = Path("data/t2s_bench/valid_subset_200.jsonl")
HAZARD_PATHS = {
    "schema_markdown_fenced": Path("tests/fixtures/schema_markdown_fenced.jsonl"),
    "schema_urls_strings": Path("tests/fixtures/schema_urls_strings.jsonl"),
    "prose_code_switch": Path("tests/fixtures/prose_code_switch.jsonl"),
}
SCHEMA_V2_PATH = Path("tests/fixtures/schema_corpus_v2.jsonl")

# The Gate 2.2 prose-control ids (docs/SUBSTRATE_LEAKAGE_PROBE.md). The heldout
# prose slice is the complement within valid_subset_200.
EXPECTED_CONTROL_IDS = [
    0, 24, 28, 32, 36, 46, 49, 52, 53, 54, 56, 57, 58, 65, 71, 72, 78, 80,
    82, 83, 98, 101, 107, 114, 117, 119, 121, 122, 128, 135, 140, 142, 148,
    150, 152, 157, 161, 172, 176, 181, 182, 186, 189, 196, 204, 205, 209,
    214, 219, 222, 232, 235, 236, 255, 256, 258, 259, 260, 271, 276, 282,
    283, 290, 308, 309, 316, 319, 321, 328, 329, 331, 332, 339, 343, 349,
    350, 355, 359, 360, 376, 388, 390, 396, 404, 407, 412, 413, 415, 417,
    433, 444, 454, 458, 466, 469, 470, 473, 474, 490, 496,
]


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
# Frozen thresholds (read from the Part A freeze file)
# ---------------------------------------------------------------------------
def load_frozen_thresholds(path: Path) -> Dict[str, Any]:
    """Parse the machine-parseable JSON freeze block from FROZEN_THRESHOLDS.md.

    This is the SINGLE source of truth; the scoring path does NOT re-derive
    any threshold. Returns {"layers": {layer: {"pr": {...}, "svse": {...}}}}.
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
    # Validate candidate layers are present.
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
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_heldout_prose(valid_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heldout-100 prose = complement of the Gate 2.2 control 100 within
    valid_subset_200. STOPs if the control selection does not match the
    documented ids or the complement is not exactly 100."""
    rng = random.Random(CONTROL_SEED)
    selected = sorted(
        rng.sample(valid_records, CONTROL_N), key=lambda r: int(r["id"])
    )
    control_ids = [int(r["id"]) for r in selected]
    if control_ids != EXPECTED_CONTROL_IDS:
        raise SystemExit(
            "[STOP] Prose control ids do NOT match docs/SUBSTRATE_LEAKAGE_PROBE.md. "
            f"got {control_ids}"
        )
    control_set = set(control_ids)
    heldout = sorted(
        (r for r in valid_records if int(r["id"]) not in control_set),
        key=lambda r: int(r["id"]),
    )
    if len(heldout) != CONTROL_N:
        raise SystemExit(
            f"[STOP] Complement split failed: {len(heldout)} heldout prose records, "
            f"expected {CONTROL_N}."
        )
    return heldout


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_windowed_layer_metrics(
    hidden_state: torch.Tensor,
) -> Dict[str, float]:
    """PR / SVSE on a (1, seq, hidden) hidden state, last-24 window."""
    win = hidden_state[0, -ANALYSIS_WINDOW:, :].unsqueeze(0)
    return {
        "pr": float(participation_ratio(win).item()),
        "svse": float(singular_value_spectrum_entropy(win, normalize=True).item()),
    }


def compute_v_traj(hidden_state: torch.Tensor) -> float:
    """RFC-004 §7.4 SHADOW candidate: normalized mean cosine step-distance
    between consecutive hidden-state deltas over the last-24 window.

    Consecutive deltas d_t = h_t - h_{t-1}; cosine step-distance between
    d_t and d_{t+1} is (1 - cos(d_t, d_{t+1})), which lies in [0, 2];
    normalized by 2 -> [0, 1]. LOG ONLY. It decides nothing and is not scored
    against thresholds (RFC-004 §7.4).
    """
    win = hidden_state[0, -ANALYSIS_WINDOW:, :].to(dtype=torch.float32)
    if win.shape[0] < 3:
        return 0.0
    deltas = win[1:] - win[:-1]  # (W-1, hidden)
    d1 = deltas[:-1]
    d2 = deltas[1:]
    sims = torch.nn.functional.cosine_similarity(d1, d2, dim=-1)
    dists = 1.0 - sims
    return float(dists.mean().item() / 2.0)


def replay_record(
    model: torch.nn.Module,
    tokenizer: Any,
    input_text: str,
) -> Tuple[List[torch.Tensor], int]:
    enc = tokenizer(input_text, return_tensors="pt").to(model.device)
    n_tokens = int(enc["input_ids"].shape[-1])
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return out.hidden_states, n_tokens


def build_record_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    set_name: str,
    record: Dict[str, Any],
    input_text: str,
    include_distinct2: bool,
) -> Dict[str, Any]:
    """Per-record metric payload for the candidate layers."""
    hidden_states, n_tokens = replay_record(model, tokenizer, input_text)
    layers: Dict[str, Dict[str, float]] = {}
    for layer in CANDIDATE_LAYERS:
        hs = hidden_states[int(layer)]
        m = compute_windowed_layer_metrics(hs)
        m["v_traj"] = compute_v_traj(hs)
        layers[layer] = m

    payload: Dict[str, Any] = {
        "set": set_name,
        "record_id": int(record.get("id", record.get("record_id", -1))),
        "n_tokens": n_tokens,
        "text": input_text,
        "is_code": bool(is_code_syntax_context(input_text)),
        "layers": layers,
    }
    if include_distinct2:
        enc = tokenizer(input_text, return_tensors="pt").to(model.device)
        d2, _repeated = compute_token_distinct_2_fast(
            enc["input_ids"], prompt_len=0, window_len=ANALYSIS_WINDOW
        )
        payload["text_signals"] = {"distinct_2": float(d2)}
    return payload


def record_metric_tuple(payload: Dict[str, Any]) -> Tuple[Any, ...]:
    """Determinism-smoke tuple: per-layer pr/svse/v_traj + text flags."""
    parts: List[Any] = []
    for layer in CANDIDATE_LAYERS:
        m = payload["layers"][layer]
        parts.append((m["pr"], m["svse"], m["v_traj"]))
    parts.append((payload["is_code"],))
    if "text_signals" in payload:
        parts.append((payload["text_signals"]["distinct_2"],))
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
    """Rank-based Mann-Whitney U / (n_pos * n_neg). positives = degenerate-class
    values; negatives = prose-control values. AUC > 0.5 -> degenerate-higher;
    AUC < 0.5 -> degenerate-lower."""
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


# ---------------------------------------------------------------------------
# Fusion rule
# ---------------------------------------------------------------------------
def fusion_fire(pr: float, svse: float, thr: Dict[str, Any]) -> bool:
    """RFC-004 fusion rule using the frozen thresholds (task greenlight).

    Greenlight parenthetical: "inside band: fire only if SVSE below T_SVSE;
    both inside bands: no fire."
    """
    low_pr = thr["pr"]["band_low"]
    high_pr = thr["pr"]["band_high"]
    t_svse = thr["svse"]["T"]
    low_svse = thr["svse"]["band_low"]
    high_svse = thr["svse"]["band_high"]

    if pr < low_pr:
        return True
    if pr > high_pr:
        return False
    # PR is inside its ambiguity band.
    if svse >= t_svse:
        return False
    if low_svse <= svse <= high_svse:
        return False  # both inside bands: no fire (conservative)
    return True


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
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
# Report writer
# ---------------------------------------------------------------------------
def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _fmt_dist_value(v: float) -> str:
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate 2.3 acceptance scoring (DS-025 / DS-027 measurement)"
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap per set (for smoke/dev runs).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip the determinism smoke (dev only).")
    parser.add_argument("--heldout-deg-path", type=str,
                        default=str(HELDOUT_DEG_PATH),
                        help="Held-out degenerate fixture path (default: "
                             "tests/fixtures/heldout_degenerate.jsonl).")
    parser.add_argument("--heldout-prompt-key", type=str, default="prompt",
                        help="Record key holding the input prompt for the "
                             "held-out degenerate fixture (default: 'prompt'; "
                             "use 'mutated_prompt' for the v2 fixture).")
    parser.add_argument("--output-jsonl", type=str,
                        default="docs/gate23/gate23_results.jsonl",
                        help="Per-record JSONL output path.")
    parser.add_argument("--output-md", type=str,
                        default="docs/gate23/GATE23_RESULTS.md",
                        help="Markdown report output path.")
    parser.add_argument("--report-title", type=str,
                        default="DS-025 — Gate 2.3 acceptance scoring results",
                        help="Markdown report H1 title.")
    parser.add_argument("--profile-path", type=str,
                        default="docs/gate23/HELDOUT_PROFILE.md",
                        help="Held-out slice profile document referenced by "
                             "the report.")
    parser.add_argument("--criteria-header", type=str,
                        default="Greenlight acceptance criteria (quoted from RFC-004 §4)",
                        help="Section header for the quoted acceptance criteria.")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-025 Gate 2.3 acceptance scoring (Part C)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {args.model}@{args.revision}")
    print(f"torch: {torch.__version__}")

    # ------------------------------------------------------------------
    # Load frozen thresholds FIRST (Part A freeze).
    # ------------------------------------------------------------------
    freeze = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    print("frozen thresholds loaded from docs/gate23/FROZEN_THRESHOLDS.md")
    for layer in CANDIDATE_LAYERS:
        t = freeze["layers"][layer]
        print(f"  layer {layer}: PR T={t['pr']['T']:.6f} "
              f"band=[{t['pr']['band_low']:.6f},{t['pr']['band_high']:.6f}] | "
              f"SVSE T={t['svse']['T']:.6f} "
              f"band=[{t['svse']['band_low']:.6f},{t['svse']['band_high']:.6f}]")

    # ------------------------------------------------------------------
    # Model load.
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

    # ------------------------------------------------------------------
    # Load corpora.
    # ------------------------------------------------------------------
    heldout_deg_path = Path(args.heldout_deg_path)
    heldout_deg = load_jsonl(heldout_deg_path)
    valid_200 = load_jsonl(VALID_200_PATH)
    heldout_prose = select_heldout_prose(valid_200)
    hazards = {name: load_jsonl(path) for name, path in HAZARD_PATHS.items()}
    schema_v2 = load_jsonl(SCHEMA_V2_PATH)

    if args.max_records is not None:
        heldout_deg = heldout_deg[: args.max_records]
        heldout_prose = heldout_prose[: args.max_records]
        hazards = {k: v[: args.max_records] for k, v in hazards.items()}
        schema_v2 = schema_v2[: args.max_records]
        print(f"[dev] capped each set at {args.max_records} records")

    n_hazards = {k: len(v) for k, v in hazards.items()}
    print(f"heldout_degenerate: {len(heldout_deg)} | heldout prose: {len(heldout_prose)} "
          f"| hazards: {n_hazards} | schema_corpus_v2: {len(schema_v2)}")

    # ------------------------------------------------------------------
    # Determinism smoke FIRST (2 heldout-degenerate records replayed twice).
    # ------------------------------------------------------------------
    print("\n--- Determinism smoke ---")
    smoke_pairs = []
    all_identical = True
    if args.skip_smoke:
        print("  skipped (dev)")
    else:
        for rec in heldout_deg[:2]:
            input_text = rec[args.heldout_prompt_key] + rec["text"]
            a = build_record_metrics(model, tokenizer, "heldout_degenerate", rec,
                                     input_text, include_distinct2=True)
            b = build_record_metrics(model, tokenizer, "heldout_degenerate", rec,
                                     input_text, include_distinct2=True)
            ta = record_metric_tuple(a)
            tb = record_metric_tuple(b)
            identical = ta == tb
            all_identical = all_identical and identical
            smoke_pairs.append(
                {"record_id": int(rec["id"]), "n_tokens": a["n_tokens"],
                 "identical": str(identical)}
            )
            print(f"  record {rec['id']}: identical={identical} n_tokens={a['n_tokens']}")
        if not all_identical:
            print("[STOP] Determinism smoke FAILED: metric tuples differ across replays.")
            sys.exit(1)
        print("  determinism smoke: ALL IDENTICAL")

    # ------------------------------------------------------------------
    # Replay all sets.
    # ------------------------------------------------------------------
    print("\n--- Replaying corpora (teacher-forced) ---")
    results: List[Dict[str, Any]] = []

    def process_set(set_name: str, records: List[Dict[str, Any]], join_prompt: bool,
                    include_d2: bool) -> None:
        for i, rec in enumerate(records):
            if join_prompt:
                input_text = rec[args.heldout_prompt_key] + rec["text"]
            else:
                input_text = rec["text"]
            payload = build_record_metrics(model, tokenizer, set_name, rec,
                                           input_text, include_distinct2=include_d2)
            results.append(payload)
            if (i + 1) % 50 == 0 or i == len(records) - 1:
                print(f"  [{set_name}] {i+1}/{len(records)}")

    process_set("heldout_degenerate", heldout_deg, join_prompt=True, include_d2=True)
    process_set("heldout_prose", heldout_prose, join_prompt=False, include_d2=True)
    for name, recs in hazards.items():
        process_set(name, recs, join_prompt=False, include_d2=True)
    process_set("schema_corpus_v2", schema_v2, join_prompt=False, include_d2=True)

    # ------------------------------------------------------------------
    # Per-layer scoring.
    # ------------------------------------------------------------------
    print("\n--- Scoring per layer ---")
    deg = [r for r in results if r["set"] == "heldout_degenerate"]
    prose = [r for r in results if r["set"] == "heldout_prose"]
    set_names = ["heldout_prose"] + list(HAZARD_PATHS.keys()) + ["schema_corpus_v2"]

    scoring: Dict[str, Any] = {}
    for layer in CANDIDATE_LAYERS:
        thr = freeze["layers"][layer]
        # AUROC: heldout-degenerate vs heldout-100 prose.
        deg_pr = [r["layers"][layer]["pr"] for r in deg]
        prose_pr = [r["layers"][layer]["pr"] for r in prose]
        deg_svse = [r["layers"][layer]["svse"] for r in deg]
        prose_svse = [r["layers"][layer]["svse"] for r in prose]
        pr_auc = mann_whitney_auc(deg_pr, prose_pr)
        svse_auc = mann_whitney_auc(deg_svse, prose_svse)

        # Fusion-rule decisions.
        fires_deg = [fusion_fire(r["layers"][layer]["pr"], r["layers"][layer]["svse"], thr)
                     for r in deg]
        recall = float(sum(fires_deg)) / len(fires_deg)
        fp_counts: Dict[str, int] = {}
        routing: Dict[str, float] = {}
        for set_name in set_names:
            recs = [r for r in results if r["set"] == set_name]
            fires = [fusion_fire(r["layers"][layer]["pr"], r["layers"][layer]["svse"], thr)
                     for r in recs]
            fp_counts[set_name] = int(sum(fires))
            routing[set_name] = float(sum(1 for r in recs if r["is_code"])) / len(recs)
        # Stage-1 routing for the two held-out contrast sets as well.
        routing["heldout_degenerate"] = float(sum(1 for r in deg if r["is_code"])) / len(deg)
        routing["heldout_prose"] = float(sum(1 for r in prose if r["is_code"])) / len(prose)

        scoring[layer] = {
            "pr_auc": pr_auc,
            "pr_auc_direction": auc_direction(pr_auc),
            "pr_separation": separation_magnitude(pr_auc),
            "svse_auc": svse_auc,
            "svse_auc_direction": auc_direction(svse_auc),
            "svse_separation": separation_magnitude(svse_auc),
            "recall_heldout_deg": recall,
            "fp_counts": fp_counts,
            "routing": routing,
            "distributions": {
                "deg_pr": summarize_distribution(deg_pr),
                "prose_pr": summarize_distribution(prose_pr),
                "deg_svse": summarize_distribution(deg_svse),
                "prose_svse": summarize_distribution(prose_svse),
            },
        }
        print(f"  layer {layer}: PR AUC={pr_auc:.4f} ({scoring[layer]['pr_auc_direction']}, "
              f"sep {scoring[layer]['pr_separation']:.4f}) | "
              f"SVSE AUC={svse_auc:.4f} ({scoring[layer]['svse_auc_direction']}, "
              f"sep {scoring[layer]['svse_separation']:.4f}) | "
              f"recall={recall:.4f} | FP={fp_counts}")

    # ------------------------------------------------------------------
    # v_traj shadow distributions (per layer, over all replayed records).
    # ------------------------------------------------------------------
    vtraj_by_layer: Dict[str, Dict[str, float]] = {}
    for layer in CANDIDATE_LAYERS:
        vals = [r["layers"][layer]["v_traj"] for r in results]
        vtraj_by_layer[layer] = summarize_distribution(vals)
        print(f"  v_traj layer {layer}: mean={vtraj_by_layer[layer]['mean']:.4f} "
              f"p10={vtraj_by_layer[layer]['p10']:.4f} "
              f"p90={vtraj_by_layer[layer]['p90']:.4f}")

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    out_jsonl = Path(args.output_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_jsonl} ({len(results)} records)")

    metadata = {
        "model_name": args.model,
        "model_revision": args.revision,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "seed": SEED,
        "control_seed": CONTROL_SEED,
        "analysis_window": f"last {ANALYSIS_WINDOW} token positions",
        "candidate_layers": CANDIDATE_LAYERS,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": {
            "pairs": smoke_pairs,
            "all_identical": str(all_identical),
        },
        "scoring": scoring,
        "vtraj": vtraj_by_layer,
        "n_sets": {name: len([r for r in results if r["set"] == name])
                   for name in ["heldout_degenerate", "heldout_prose"] + set_names},
        "distinct2": {
            "heldout_degenerate": summarize_distribution(
                [r["text_signals"]["distinct_2"] for r in deg]
            ),
            "heldout_prose": summarize_distribution(
                [r["text_signals"]["distinct_2"] for r in prose]
            ),
        },
        "report_title": args.report_title,
        "profile_path": args.profile_path,
        "output_jsonl": str(out_jsonl),
        "criteria_header": args.criteria_header,
    }

    out_md = Path(args.output_md)
    write_markdown_report(ctx, freeze, out_md)
    print(f"wrote {out_md}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("MEASUREMENT COMPLETE (no verdict offered)")


def write_markdown_report(ctx: Dict[str, Any], freeze: Dict[str, Any], out_path: Path) -> None:
    """Write the Gate 2.3 acceptance scoring report (strictly factual, no verdict)."""
    md: List[str] = []
    md.append(f"# {ctx['report_title']}")
    md.append("")
    md.append("> MEASUREMENT REPORT. This document reports measured values only.")
    md.append("> No threshold is created or modified, no gate script is touched,")
    md.append("> and no green/red verdict and no layer recommendation are offered.")
    md.append("> Earliest-passing layer selection is the human's act per the greenlight.")
    md.append("")

    # Greenlight criteria (quoted, not verdicted).
    md.append(f"## {ctx['criteria_header']}")
    md.append("")
    md.append("> 4. Acceptance (all required): AUROC >= 0.98 on held-out contrasts;")
    md.append("> 0 false positives on prose + hazard + schema corpora (including")
    md.append("> Stage-1 routing measurement on structured-legitimate text); minimum")
    md.append("> recall >= 0.95 on the held-out hard-degenerate slice.")
    md.append("")
    md.append("This report MEASURES and REPORTS per layer. It does not render the verdict.")
    md.append("")

    # Run metadata
    md.append("## Run metadata")
    md.append("")
    meta = ctx["metadata"]
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(f"| Model | {meta['model_name']} |")
    md.append(f"| Revision | {meta['model_revision']} |")
    md.append(f"| Device | {meta['device']} |")
    md.append(f"| dtype | {meta['dtype']} |")
    md.append(f"| torch | {meta['torch_version']} |")
    md.append(f"| SEED | {meta['seed']} |")
    md.append(f"| Control selection seed | {meta['control_seed']} |")
    md.append(f"| Analysis window | {meta['analysis_window']} |")
    md.append(f"| Candidate layers | {meta['candidate_layers']} |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Held-out slice characterization (Distinct-2)
    md.append("## Held-out slice characterization (Distinct-2)")
    md.append("")
    md.append("Distinct-2 is the offline text-level diversity measure (last-24-token")
    md.append("window) used to curate the held-out degenerate slice. It is reported")
    md.append("here as strictly factual slice characterization; it is not a decision")
    md.append("signal in the RFC-004 Stage-2 detector.")
    md.append("")
    d2 = ctx["distinct2"]
    d2_rows = [
        ["heldout_degenerate", _fmt_dist_value(d2["heldout_degenerate"]["mean"]),
         _fmt_dist_value(d2["heldout_degenerate"]["median"]),
         _fmt_dist_value(d2["heldout_degenerate"]["p10"]),
         _fmt_dist_value(d2["heldout_degenerate"]["p90"])],
        ["heldout-100 prose", _fmt_dist_value(d2["heldout_prose"]["mean"]),
         _fmt_dist_value(d2["heldout_prose"]["median"]),
         _fmt_dist_value(d2["heldout_prose"]["p10"]),
         _fmt_dist_value(d2["heldout_prose"]["p90"])],
    ]
    md.append(format_table_md(
        d2_rows, ["set", "mean", "median", "p10", "p90"]
    ))
    md.append("")
    md.append(f"The full grid/curation profile is in `{ctx['profile_path']}`.")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("Two heldout-degenerate records were replayed twice (teacher-forced,")
    md.append("single forward pass each). The per-layer metric tuple (PR/SVSE/v_traj")
    md.append("at layers {2,12,26} + is_code + distinct_2) must be identical across")
    md.append("the two runs.")
    md.append("")
    smoke_rows = []
    for row in ctx["determinism_smoke"]["pairs"]:
        smoke_rows.append([str(row["record_id"]), str(row["n_tokens"]), row["identical"]])
    smoke_rows.append(["", "ALL", str(ctx["determinism_smoke"]["all_identical"])])
    md.append(format_table_md(smoke_rows, ["record_id", "n_tokens", "identical"]))
    md.append("")

    # Frozen thresholds restated
    md.append("## Frozen thresholds (restated from Part A freeze)")
    md.append("")
    md.append("Thresholds are read from `docs/gate23/FROZEN_THRESHOLDS.md`; nothing")
    md.append("here re-derives them.")
    md.append("")
    thr_rows = []
    for layer in CANDIDATE_LAYERS:
        t = freeze["layers"][layer]
        thr_rows.append([
            layer, "PR",
            _fmt_dist_value(t["pr"]["T"]), _fmt_dist_value(t["pr"]["band_low"]),
            _fmt_dist_value(t["pr"]["band_high"]),
        ])
        thr_rows.append([
            layer, "SVSE",
            _fmt_dist_value(t["svse"]["T"]), _fmt_dist_value(t["svse"]["band_low"]),
            _fmt_dist_value(t["svse"]["band_high"]),
        ])
    md.append(format_table_md(thr_rows, ["layer", "signal", "T", "band_low", "band_high"]))
    md.append("")

    # Per-layer acceptance tables
    md.append("## Per-layer acceptance scoring (fusion rule)")
    md.append("")
    md.append("Fusion rule (RFC-004 + greenlight parenthetical): PR primary. If PR is")
    md.append("below its band -> fire. If PR is inside its ambiguity band -> fire only")
    md.append("if SVSE agrees (SVSE below T_SVSE) and not both-inside-bands (SVSE below")
    md.append("its own lower band edge). If PR is above its band -> no fire.")
    md.append("")
    sc = ctx["scoring"]
    acc_rows = []
    for layer in CANDIDATE_LAYERS:
        s = sc[layer]
        acc_rows.append([
            layer,
            f"{s['pr_auc']:.4f}", s["pr_auc_direction"], f"{s['pr_separation']:.4f}",
            f"{s['svse_auc']:.4f}", s["svse_auc_direction"], f"{s['svse_separation']:.4f}",
            f"{s['recall_heldout_deg']:.4f}",
            str(s["fp_counts"]["heldout_prose"]),
            str(s["fp_counts"]["schema_markdown_fenced"]),
            str(s["fp_counts"]["schema_urls_strings"]),
            str(s["fp_counts"]["prose_code_switch"]),
            str(s["fp_counts"]["schema_corpus_v2"]),
        ])
    md.append(format_table_md(
        acc_rows,
        ["layer", "PR AUC", "PR dir", "PR sep", "SVSE AUC", "SVSE dir", "SVSE sep",
         "recall(deg)",
         "FP prose", "FP schema_md", "FP schema_urls", "FP prose_code", "FP schema_v2"],
    ))
    md.append("")
    md.append("- AUROC direction is with heldout-degenerate as the positive class;")
    md.append("  degenerate-LOWER means AUC < 0.5 (PR/SVSE lower for degenerate).")
    md.append("  Separation = max(AUC, 1-AUC); the `AUROC >= 0.98` criterion is read")
    md.append("  against separation (equivalently |1-AUC| for a degenerate-lower signal).")
    md.append("- recall(deg) = fraction of the 100 heldout-degenerate records that fire.")
    md.append("- FP <set> = number of records in that legitimate set that fire under the")
    md.append("  fusion rule (any fire on a legitimate set is a false positive).")
    md.append("")

    # Stage-1 routing measurement
    md.append("## Stage-1 routing measurement (is_code_syntax_context)")
    md.append("")
    md.append("`is_code_syntax_context(text)` is the Stage-1 context router as-is; the")
    md.append("rate is the fraction of records in each set flagged as code/schema")
    md.append("context (routed to Path-2 immunity and therefore not eligible for Stage-2).")
    md.append("")
    routing_rows = []
    n_sets = ctx["n_sets"]
    for set_name in ["heldout_degenerate", "heldout_prose", "schema_markdown_fenced",
                     "schema_urls_strings", "prose_code_switch", "schema_corpus_v2"]:
        # is_code_syntax_context is a set-level property; use the layer-2 entry
        # (identical across layers).
        rate = sc["2"]["routing"][set_name]
        routing_rows.append([set_name, str(n_sets.get(set_name, 0)), f"{rate:.4f}"])
    md.append(format_table_md(
        routing_rows, ["set", "n_records", "routing_rate"],
    ))
    md.append("")

    # Per-layer distribution summaries for the held-out contrast
    md.append("## Held-out contrast distributions (per layer)")
    md.append("")
    for layer in CANDIDATE_LAYERS:
        d = sc[layer]["distributions"]
        md.append(f"### Layer {layer}")
        md.append("")
        dist_rows = [
            ["PR", "degenerate", _fmt_dist_value(d["deg_pr"]["mean"]),
             _fmt_dist_value(d["deg_pr"]["median"]), _fmt_dist_value(d["deg_pr"]["p10"]),
             _fmt_dist_value(d["deg_pr"]["p90"]), str(d["deg_pr"]["n_non_finite"])],
            ["PR", "prose", _fmt_dist_value(d["prose_pr"]["mean"]),
             _fmt_dist_value(d["prose_pr"]["median"]), _fmt_dist_value(d["prose_pr"]["p10"]),
             _fmt_dist_value(d["prose_pr"]["p90"]), str(d["prose_pr"]["n_non_finite"])],
            ["SVSE", "degenerate", _fmt_dist_value(d["deg_svse"]["mean"]),
             _fmt_dist_value(d["deg_svse"]["median"]), _fmt_dist_value(d["deg_svse"]["p10"]),
             _fmt_dist_value(d["deg_svse"]["p90"]), str(d["deg_svse"]["n_non_finite"])],
            ["SVSE", "prose", _fmt_dist_value(d["prose_svse"]["mean"]),
             _fmt_dist_value(d["prose_svse"]["median"]), _fmt_dist_value(d["prose_svse"]["p10"]),
             _fmt_dist_value(d["prose_svse"]["p90"]), str(d["prose_svse"]["n_non_finite"])],
        ]
        md.append(format_table_md(
            dist_rows, ["signal", "class", "mean", "median", "p10", "p90", "n_non_finite"]
        ))
        md.append("")

    # v_traj shadow log
    md.append("## v_traj SHADOW log (RFC-004 §7.4)")
    md.append("")
    md.append("v_traj is the independent review candidate shadow metric: normalized mean cosine")
    md.append("step-distance between consecutive hidden-state deltas over the last-24")
    md.append("window. It is LOGGED ONLY, decides NOTHING, and is NOT scored against")
    md.append("thresholds.")
    md.append("")
    v_rows = []
    for layer in CANDIDATE_LAYERS:
        v = ctx["vtraj"][layer]
        v_rows.append([
            layer, _fmt_dist_value(v["mean"]), _fmt_dist_value(v["median"]),
            _fmt_dist_value(v["p10"]), _fmt_dist_value(v["p90"]),
            str(v["n_non_finite"]),
        ])
    md.append(format_table_md(
        v_rows, ["layer", "mean", "median", "p10", "p90", "n_non_finite"]
    ))
    md.append("")
    md.append(f"Per-record v_traj values are in `{ctx['output_jsonl']}`.")
    md.append("")

    # Data
    md.append("## Data")
    md.append("")
    md.append(f"- Per-record per-layer values: `{ctx['output_jsonl']}`")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
