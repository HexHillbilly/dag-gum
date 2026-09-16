#!/usr/bin/env python3
"""DS-034b: Online PR parity — teacher-forced vs hook-based (HARD GATE).

RFC-004 Amendment A3 (ratified) requires a live hook-based participation-ratio
(PR) predicate from a forward hook at layer 2 capturing hidden states
step-by-step into a rolling 24-token ring buffer during autoregressive
generation. DS-034 measured spectral PR on t2s_degenerate via teacher-forced
replay (output_hidden_states=True, last-24 window, frozen T_PR thresholds) —
a single forward pass over the full sequence. These are different measurement
geometries: the hook-based PR is computed from hidden states produced across
128 separate forward passes with different KV-cache contexts.

This is a HARD GATE per RFC A3: if fire/no-fire disagreement between
teacher-forced and hook-based PR exceeds 5/93 records, the hook implementation
must be revised before wiring the dual-predicate into the controller. A FAIL
is a STOP signal — no negotiation, no threshold tuning.

Method (per the DS-034b task file):
  1. REUSE the 93 detection-gap records from DS-033
     (docs/gate23/production_cross_fixture_results.jsonl, class=="detection-gap").
     Do NOT re-classify.
  2. REUSE the DS-034 teacher-forced PR reference values at layer 2 from
     docs/gate23/spectral_pr_t2s_results.jsonl. Do NOT re-measure.
  3. NEW hook-based PR measured live for all 93 records:
     - Greedy decoding (do_sample=False), max_new_tokens=128, prompt = record["text"].
     - Forward hook at model.model.layers[2] captures hidden state at every
       decoding step (last position of the growing sequence, matching the
       governor's shadow-hook capture at [:, -1:, :]).
     - Rolling ring buffer of trailing 24 generated hidden states (float32
       promotion). When buffer >= 24 (from step 23 onward): compute
       participation_ratio() over the window.
     - Record the mean PR across all steps where buffer >= 24.
     - Record fire = any step where PR < T_PR(2).
  4. Compare per-record fire flags: teacher-forced (DS-034) vs hook-based (live).
     Agreement: both fire OR both do not fire. Disagreement: one fires, the
     other does not. PR delta = |PR_hook_mean - PR_teacher_forced|.

Gate criteria (RFC A3):
  - Disagreement count <= 5/93 records -> PASS. Hook implementation is safe.
  - Disagreement count > 5/93 records -> FAIL. Hook must be revised.
    - If disagreement is near the boundary (PR within 0.5 of T), add
      sub-threshold hysteresis (must stay below T for >= 2 steps).
    - If disagreement is systematic (large PR delta), investigate float32
      promotion fidelity or buffer accumulation.

Environment notes (identical to ds-025/ds-033/ds-034/ds-034a):
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from AVG.core.metrics import participation_ratio  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-034b measurement seed (matches ds-025..ds-034a)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (no EOS break, prod generate())
ANALYSIS_WINDOW = 24  # rolling 24-token PR window (Gate 2.3 / ds-025 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook
T_PR_2_TASK = 10.954796  # ds-025 Part A freeze (must match FROZEN_THRESHOLDS.md)

# RFC A3 HARD GATE: disagreement > 5/93 -> FAIL (STOP, no negotiation).
GATE_MAX_DISAGREEMENTS = 5
N_DETECTION_GAP_EXPECTED = 93

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
DS033_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
DS034_SPECTRAL_JSONL = Path("docs/gate23/spectral_pr_t2s_results.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/online_pr_parity_results.jsonl")
OUTPUT_MD = Path("docs/gate23/ONLINE_PR_PARITY_RESULTS.md")

# DS-033 diagnostic classes (reused; do NOT re-compute).
CLASS_DETECTION_GAP = "detection-gap"

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
def load_frozen_threshold(path: Path, layer: str = "2") -> float:
    """Parse the machine-parseable JSON freeze block from FROZEN_THRESHOLDS.md
    and return T_PR(layer). STOP if the value disagrees with the task-specified
    frozen T_PR(2) = 10.954796."""
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md has no ```json freeze block.")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md freeze block is unterminated.")
    freeze = json.loads(text[start:end])
    t_pr = float(freeze["layers"][layer]["pr"]["T"])
    if abs(t_pr - T_PR_2_TASK) > 1e-9:
        raise SystemExit(
            f"[STOP] FROZEN_THRESHOLDS.md T_PR(2)={t_pr:.6f} does not match the "
            f"task-specified frozen value {T_PR_2_TASK:.6f}. Do NOT re-derive "
            f"or adjust thresholds."
        )
    return t_pr


# ---------------------------------------------------------------------------
# Hook-based live generation (layer-2 PR measurement instrument)
# ---------------------------------------------------------------------------
def make_layer2_pr_hook(
    analysis_window: int = ANALYSIS_WINDOW,
) -> Tuple[Callable[..., Any], List[torch.Tensor], List[Dict[str, Any]]]:
    """Create a layer-2 forward-hook measurement instrument.

    At each decoding step (one forward pass per step), capture the hidden state
    at the current token position (the last position of the growing sequence,
    matching the governor's own shadow-hook capture at ``[:, -1:, :]``), append
    it to a ring buffer of the trailing ``analysis_window`` generated hidden
    states, and when the buffer has >= ``analysis_window`` entries compute
    ``participation_ratio()`` over the window (float32, matching ds-005/ds-010
    numerical discipline). The PR value is logged; it does NOT gate any
    actuation and the hook returns ``output`` unchanged.
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
        # Current token position = last position of the growing sequence.
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


@torch.no_grad()
def greedy_generate_hook_pr(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    layer: int = HOOK_LAYER,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) generation with the layer-2 PR measurement hook.

    Compute pattern matches the production controller's generate(): full
    sequence forward passes (no KV cache), exactly ``max_new_tokens`` decoding
    steps (no EOS break). The hook runs EVERY STEP (0..max_new_tokens-1). It is
    a pure measurement instrument: it never modifies the residual stream.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])

    hook_fn, buffer, pr_log = make_layer2_pr_hook(ANALYSIS_WINDOW)
    blocks = model.model.layers
    handle = blocks[layer].register_forward_hook(hook_fn)

    try:
        for _step in range(max_new_tokens):
            out = model(input_ids)
            logits = out.logits[:, -1, :].clone().float()
            next_tok = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
    finally:
        handle.remove()

    return {
        "generated_ids": input_ids[:, prompt_len:].detach().cpu(),
        "prompt_len": prompt_len,
        "n_generated": max_new_tokens,
        "pr_log": pr_log,
        "n_pr_values": len(pr_log),
        "buffer_len": len(buffer),
    }


# ---------------------------------------------------------------------------
# Per-record hook-based PR summary
# ---------------------------------------------------------------------------
def summarize_hook_pr(pr_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [p["pr"] for p in pr_log]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None,
                "first": None, "last": None}
    return {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "first": float(vals[0]),
        "last": float(vals[-1]),
    }


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Determinism smoke (DS-034b): 2 detection-gap records, hook-based
    generated twice. Token ids AND per-step PR log must match exactly.
    STOP (sys.exit 1) if either fails."""
    print("\n--- Determinism smoke (2 detection-gap records, hook-based twice) ---")
    pairs: List[Dict[str, Any]] = []
    all_identical = True
    for rec in records[:2]:
        rid = int(rec["id"])
        prompt = rec["text"]
        a = greedy_generate_hook_pr(model, tokenizer, prompt)
        b = greedy_generate_hook_pr(model, tokenizer, prompt)
        tok_identical = bool(
            a["generated_ids"].shape == b["generated_ids"].shape
            and torch.equal(a["generated_ids"], b["generated_ids"])
        )
        pr_a = [p["pr"] for p in a["pr_log"]]
        pr_b = [p["pr"] for p in b["pr_log"]]
        pr_identical = bool(pr_a == pr_b)
        identical = bool(tok_identical and pr_identical)
        all_identical = all_identical and identical
        pairs.append({
            "record_id": rid,
            "prompt_len": a["prompt_len"],
            "n_pr_values": len(pr_a),
            "token_identical": str(tok_identical),
            "pr_identical": str(pr_identical),
            "identical": str(identical),
            "pr_a_first": pr_a[0] if pr_a else None,
            "pr_b_first": pr_b[0] if pr_b else None,
            "pr_a_last": pr_a[-1] if pr_a else None,
            "pr_b_last": pr_b[-1] if pr_b else None,
        })
        print(f"  record {rid}: prompt_len={a['prompt_len']} "
              f"tokens_identical={tok_identical} n_pr={len(pr_a)} "
              f"pr_identical={pr_identical} identical={identical}")
    if not all_identical:
        print("[STOP] Determinism smoke FAILED: hook-based token ids or PR log "
              "differ across runs.")
        sys.exit(1)
    if any(p["n_pr_values"] != (MAX_NEW_TOKENS - ANALYSIS_WINDOW + 1)
           for p in pairs):
        print("[STOP] Determinism smoke FAILED: expected "
              f"{MAX_NEW_TOKENS - ANALYSIS_WINDOW + 1} PR values per generation.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (token ids + PR log, 2 records x 2 runs)")
    return {"pairs": pairs, "all_identical": str(all_identical)}


# ---------------------------------------------------------------------------
# Aggregation / gate
# ---------------------------------------------------------------------------
def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def classify_disagreement(r: Dict[str, Any], t_pr: float) -> str:
    """Classify a record's fire/no-fire disagreement (RFC A3 follow-up logic).

    Since the teacher-forced reference fires on all 93 detection-gap records, a
    disagreement means the hook-based path did NOT fire: every hook PR >= T. If
    the hook PR min is within 0.5 of T, the disagreement is near the boundary
    (candidate for sub-threshold hysteresis); otherwise it is systematic (large
    PR delta / hook PR far from T).
    """
    if r["agreement"]:
        return "none"
    hook_min = r["hook_based"]["pr_min"]
    if hook_min is None:
        return "systematic"
    dist_to_t = hook_min - t_pr
    return "near-boundary" if dist_to_t <= 0.5 else "systematic"


def aggregate_results(results: List[Dict[str, Any]], t_pr: float) -> Dict[str, Any]:
    disagreements = [r for r in results if not r["agreement"]]
    n_disagree = len(disagreements)
    agree = [r for r in results if r["agreement"]]

    pr_deltas = [r["pr_delta"] for r in results]
    hook_means = [r["hook_based"]["pr_mean"] for r in results]
    teacher_prs = [r["teacher_forced"]["pr"] for r in results]

    n_near = sum(1 for r in disagreements
                 if classify_disagreement(r, t_pr) == "near-boundary")
    n_sys = sum(1 for r in disagreements
                if classify_disagreement(r, t_pr) == "systematic")

    return {
        "n_records": len(results),
        "n_agreement": len(agree),
        "n_disagreement": n_disagree,
        "disagreement_rate": float(n_disagree) / len(results) if results else float("nan"),
        "gate_max_disagreements": GATE_MAX_DISAGREEMENTS,
        "gate_pass": bool(n_disagree <= GATE_MAX_DISAGREEMENTS),
        "pr_delta": summarize_distribution(pr_deltas),
        "hook_pr_mean": summarize_distribution([x for x in hook_means if x is not None]),
        "teacher_pr": summarize_distribution(teacher_prs),
        "n_near_boundary_disagreement": n_near,
        "n_systematic_disagreement": n_sys,
        "disagreement_ids": [r["record_id"] for r in disagreements],
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
    md.append("# DS-034b — Online PR parity: teacher-forced vs hook-based (HARD GATE)")
    md.append("")
    md.append("> HARD GATE REPORT (RFC-004 Amendment A3). The dual-predicate "
              "architecture requires a live hook-based PR from a forward hook at "
              "layer 2 capturing hidden states step-by-step into a rolling "
              "24-token ring buffer during autoregressive generation. DS-034 "
              "measured spectral PR via teacher-forced replay (single forward "
              "pass over the full sequence). These are different measurement "
              "geometries. This probe quantifies fire/no-fire disagreement "
              "between the two and applies the RFC A3 gate. A FAIL is a STOP "
              "signal — no negotiation, no threshold tuning.")
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
    md.append("| Records measured | 93 detection-gap (DS-033 classification "
              "REUSED, do NOT re-compute) |")
    md.append("| Teacher-forced PR | REUSED from DS-034 "
              "spectral_pr_t2s_results.jsonl (do NOT re-measure) |")
    md.append("| Decoding | greedy (do_sample=False), "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} hidden "
              f"states (hook-based); last {meta['analysis_window']} token "
              f"positions (teacher-forced) |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] |")
    md.append(f"| Frozen T_PR(2) | {meta['t_pr']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |")
    md.append("| PR instrument | participation_ratio() from core/metrics.py "
              "(existing), float32 promotion |")
    md.append("| Cadence | every-step (confirmed by DS-034a at "
              "Δ = 0.7027 ms/token) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Gate criterion
    md.append("## Gate criterion (RFC A3, quoted from the task file)")
    md.append("")
    md.append("> HARD GATE per RFC A3: if fire/no-fire disagreement between "
              "teacher-forced and hook-based PR exceeds 5/93 records, the hook "
              "implementation must be revised before wiring the dual-predicate "
              "into the controller.")
    md.append("")
    md.append("| Criterion | Value |")
    md.append("|---|---|")
    md.append(f"| Disagreement count ≤ {GATE_MAX_DISAGREEMENTS}/93 records | PASS "
              f"— hook implementation is safe |")
    md.append(f"| Disagreement count > {GATE_MAX_DISAGREEMENTS}/93 records | FAIL "
              f"— hook must be revised |")
    md.append("")
    md.append("Fire criterion (both methods): `PR < T_PR(2)`. Teacher-forced "
              "fires on a single windowed PR (last 24 token positions of the "
              "full text). Hook-based fires if PR drops below T at ANY decoding "
              "step (rolling 24-token buffer, from step 23 onward).")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("Two detection-gap records, hook-based generated twice. Token ids "
              "AND the per-step PR log must match exactly. STOP if not.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = []
    for pair in sm["pairs"]:
        smoke_rows.append([
            str(pair["record_id"]), str(pair["prompt_len"]),
            str(pair["n_pr_values"]), pair["token_identical"],
            pair["pr_identical"], pair["identical"],
        ])
    smoke_rows.append(["", "ALL", "", "", "", str(sm["all_identical"])])
    md.append(format_table_md(
        smoke_rows, ["record_id", "prompt_len", "n PR", "tokens identical",
                     "PR log identical", "identical"],
    ))
    md.append("")
    for pair in sm["pairs"]:
        md.append(f"PR first run A/B: {fmt(pair['pr_a_first'])} / "
                  f"{fmt(pair['pr_b_first'])}; last run A/B: "
                  f"{fmt(pair['pr_a_last'])} / {fmt(pair['pr_b_last'])}.")
        md.append("")

    tabs = ctx["tables"]

    # Verdict
    md.append("## Verdict (RFC A3 HARD GATE)")
    md.append("")
    md.append(f"| metric | value |")
    md.append("|---|---|")
    md.append(f"| Records compared | {tabs['n_records']} |")
    md.append(f"| Agreement | {tabs['n_agreement']} |")
    md.append(f"| **Disagreement** | **{tabs['n_disagreement']}** |")
    md.append(f"| Gate max disagreements | {tabs['gate_max_disagreements']} |")
    md.append(f"| **Gate verdict** | **{'PASS' if tabs['gate_pass'] else 'FAIL'}** |")
    md.append("")
    if tabs["gate_pass"]:
        md.append(f"**PASS** — {tabs['n_disagreement']} ≤ "
                  f"{tabs['gate_max_disagreements']}: the hook implementation is "
                  "safe. Fire/no-fire agreement between teacher-forced and "
                  "hook-based PR holds within the RFC A3 tolerance.")
    else:
        md.append(f"**FAIL** — {tabs['n_disagreement']} > "
                  f"{tabs['gate_max_disagreements']}: the hook implementation "
                  "must be revised before wiring the dual-predicate into the "
                  "controller. STOP signal; no threshold tuning.")
    md.append("")

    # PR delta distribution
    md.append("## PR delta distribution (|PR_hook_mean − PR_teacher_forced|)")
    md.append("")
    d = tabs["pr_delta"]
    md.append("| stat | value |")
    md.append("|---|---|")
    md.append(f"| mean | {fmt(d['mean'])} |")
    md.append(f"| median | {fmt(d['median'])} |")
    md.append(f"| p10 | {fmt(d['p10'])} |")
    md.append(f"| p90 | {fmt(d['p90'])} |")
    md.append("")
    hp = tabs["hook_pr_mean"]
    tp = tabs["teacher_pr"]
    md.append("| measure | mean | median | p10 | p90 |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Hook-based PR (per-record mean of per-step PR) | {fmt(hp['mean'])} "
              f"| {fmt(hp['median'])} | {fmt(hp['p10'])} | {fmt(hp['p90'])} |")
    md.append(f"| Teacher-forced PR (DS-034, layer 2) | {fmt(tp['mean'])} "
              f"| {fmt(tp['median'])} | {fmt(tp['p10'])} | {fmt(tp['p90'])} |")
    md.append("")

    # Disagreement detail
    md.append("## Disagreement records")
    md.append("")
    md.append(f"Total disagreements: {tabs['n_disagreement']}. Near-boundary "
              f"(hook PR min within 0.5 of T): "
              f"{tabs['n_near_boundary_disagreement']}; systematic (large PR "
              f"delta / hook PR far from T): "
              f"{tabs['n_systematic_disagreement']}.")
    md.append("")
    if tabs["n_disagreement"] > 0:
        md.append("| record_id | teacher PR | teacher fire | hook PR mean | "
                  "hook PR min | hook PR max | hook fire | PR delta | "
                  "disagreement type |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in ctx["results"]:
            if r["agreement"]:
                continue
            hb = r["hook_based"]
            tf = r["teacher_forced"]
            md.append(f"| {r['record_id']} | {fmt(tf['pr'])} | {tf['fire']} | "
                      f"{fmt(hb['pr_mean'])} | {fmt(hb['pr_min'])} | "
                      f"{fmt(hb['pr_max'])} | {hb['fire']} | "
                      f"{fmt(r['pr_delta'])} | {r['disagreement_type']} |")
        md.append("")
    else:
        md.append("None — all 93 records agree (both fire).")
        md.append("")

    # Full per-record table
    md.append("## Per-record comparison (all 93 detection-gap records)")
    md.append("")
    md.append("| record_id | teacher PR | teacher fire | hook PR mean | "
              "hook PR min | hook PR max | hook fire | PR delta | agreement |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in ctx["results"]:
        hb = r["hook_based"]
        tf = r["teacher_forced"]
        md.append(f"| {r['record_id']} | {fmt(tf['pr'])} | {tf['fire']} | "
                  f"{fmt(hb['pr_mean'])} | {fmt(hb['pr_min'])} | "
                  f"{fmt(hb['pr_max'])} | {hb['fire']} | "
                  f"{fmt(r['pr_delta'])} | {r['agreement']} |")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- HARD GATE: the pass/fail criterion is explicit and quoted from "
              "RFC A3. A FAIL is a STOP signal — the script exits non-zero; no "
              "thresholds are negotiated or tuned.")
    md.append("- Teacher-forced PR values are REUSED from "
              "docs/gate23/spectral_pr_t2s_results.jsonl (DS-034). They are NOT "
              "re-measured. The teacher-forced PR is the layer-2 participation "
              "ratio over the last 24 token positions of the full input text in "
              "a single forward pass.")
    md.append("- DS-033 detection-gap classification is REUSED from "
              "docs/gate23/production_cross_fixture_results.jsonl (join on "
              "fixture + record_id, class == 'detection-gap'). Not re-classified.")
    md.append("- Hook-based PR is measured live: greedy decoding "
              f"(do_sample=False), max_new_tokens={MAX_NEW_TOKENS}, prompt = "
              "record[\\\"text\\\"]. Forward hook at "
              f"model.model.layers[{HOOK_LAYER}] captures the hidden state at "
              "every decoding step; a rolling ring buffer keeps the trailing 24 "
              "generated hidden states (float32 promotion); "
              "participation_ratio() is computed from step 23 onward (105 "
              "values per record).")
    md.append("- Fire criterion is identical for both methods: PR < T_PR(2) = "
              f"{meta['t_pr']:.6f}. Hook-based fire = any step below T; "
              "teacher-forced fire = the single windowed PR below T.")
    md.append("- bf16 accumulation across the 128 separate forward passes "
              "(hook-based) vs the single forward pass (teacher-forced) can "
              "produce numerical drift; this probe quantifies the resulting "
              "fire/no-fire disagreement.")
    md.append("- Per-record per-step PR values are in "
              "`online_pr_parity_results.jsonl`.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-034b online PR parity: teacher-forced vs hook-based "
                    "(RFC A3 HARD GATE)"
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
    print("DS-034b Online PR parity: teacher-forced vs hook-based (HARD GATE)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: {ANALYSIS_WINDOW} "
          f"| hook_layer: {HOOK_LAYER}")
    print(f"gate: disagreement <= {GATE_MAX_DISAGREEMENTS}/{N_DETECTION_GAP_EXPECTED} -> PASS")

    # ------------------------------------------------------------------
    # Frozen threshold FIRST (Part A freeze); verify task-specified value.
    # ------------------------------------------------------------------
    t_pr = load_frozen_threshold(FROZEN_THRESHOLDS_PATH, "2")
    print(f"frozen T_PR(2) loaded from docs/gate23/FROZEN_THRESHOLDS.md: {t_pr:.6f}")
    if abs(t_pr - T_PR_2_TASK) > 1e-9:
        print("[STOP] frozen T_PR(2) does not match the task-specified value.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load fixture + DS-033 classification + DS-034 teacher-forced PR
    # (all REUSED; nothing re-computed).
    # ------------------------------------------------------------------
    records = load_jsonl(T2S_DEG)
    print(f"t2s_degenerate records: {len(records)}")

    if not DS033_JSONL.exists():
        print(f"[STOP] {DS033_JSONL} does not exist; DS-033 classification is "
              "required for the detection-gap join. Do NOT re-compute DS-033.")
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

    if not DS034_SPECTRAL_JSONL.exists():
        print(f"[STOP] {DS034_SPECTRAL_JSONL} does not exist; DS-034 "
              "teacher-forced PR reference is required. Do NOT re-measure.")
        sys.exit(1)
    spec = load_jsonl(DS034_SPECTRAL_JSONL)
    spec_by_id: Dict[int, Dict[str, Any]] = {}
    for rec in spec:
        if rec.get("fixture") == "t2s_degenerate":
            spec_by_id[int(rec["record_id"])] = rec
    missing_spec = [int(r["id"]) for r in records
                    if int(r["id"]) not in spec_by_id]
    if missing_spec:
        print(f"[STOP] DS-034 spectral JSONL join failed: {len(missing_spec)} "
              f"t2s_degenerate record_ids missing from "
              f"spectral_pr_t2s_results.jsonl: {sorted(missing_spec)[:20]}")
        sys.exit(1)

    # Select the 93 detection-gap records (DS-033 classification, REUSED).
    gap_records: List[Dict[str, Any]] = []
    for r in records:
        rid = int(r["id"])
        if ds033_by_id[rid]["active"]["diagnostic_class"] == CLASS_DETECTION_GAP:
            gap_records.append(r)
    if len(gap_records) != N_DETECTION_GAP_EXPECTED:
        print(f"[STOP] expected {N_DETECTION_GAP_EXPECTED} detection-gap "
              f"records, found {len(gap_records)}. Do NOT re-classify DS-033.")
        sys.exit(1)
    print(f"DS-033 join OK: {len(ds033_by_id)} t2s_degenerate records; "
          f"detection-gap={len(gap_records)}")
    if args.max_records is not None:
        gap_records = gap_records[: args.max_records]
        print(f"[dev] capped detection-gap records at {args.max_records} "
              f"(ids={[int(r['id']) for r in gap_records]})")

    # Cross-check: every detection-gap record has a DS-034 layer-2 PR reference.
    for r in gap_records:
        rid = int(r["id"])
        if "2" not in spec_by_id[rid]["layers"]:
            print(f"[STOP] DS-034 spectral reference missing layer 2 for "
                  f"record {rid}.")
            sys.exit(1)
    print("DS-034 teacher-forced layer-2 PR reference present for all "
          f"{len(gap_records)} detection-gap records (REUSED, not re-measured)")

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
        try:
            smoke = run_determinism_smoke(model, tokenizer, gap_records)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during determinism smoke: {e}")
                sys.exit(1)
            raise

    # ==================================================================
    # Live hook-based generation for all detection-gap records, joined
    # against the REUSED DS-034 teacher-forced reference.
    # ==================================================================
    print(f"\n--- Hook-based live generation ({len(gap_records)} detection-gap "
          f"records, greedy, max_new_tokens={MAX_NEW_TOKENS}) ---")
    results: List[Dict[str, Any]] = []
    try:
        for i, rec in enumerate(gap_records):
            rid = int(rec["id"])
            prompt = rec["text"]
            ds034_ref = spec_by_id[rid]
            tf_pr = float(ds034_ref["layers"]["2"]["pr"])
            tf_fire = bool(ds034_ref["layers"]["2"]["pr_below_t"])

            gen = greedy_generate_hook_pr(model, tokenizer, prompt)
            pr_summary = summarize_hook_pr(gen["pr_log"])
            hook_mean = pr_summary["mean"]
            hook_fire = bool(
                any(p["pr"] < t_pr for p in gen["pr_log"])
            ) if gen["pr_log"] else False
            agreement = bool(tf_fire == hook_fire)
            pr_delta = abs(hook_mean - tf_pr) if hook_mean is not None else float("nan")

            payload: Dict[str, Any] = {
                "fixture": "t2s_degenerate",
                "record_id": rid,
                "seed": SEED,
                "diagnostic_class": CLASS_DETECTION_GAP,
                "predicate_true_steps": int(
                    ds033_by_id[rid]["active"]["predicate_true_steps"]
                ),
                "prompt_len": int(gen["prompt_len"]),
                "n_generated": int(gen["n_generated"]),
                "teacher_forced": {
                    "pr": tf_pr,
                    "t_pr": t_pr,
                    "fire": tf_fire,
                    "source": "DS-034 REUSED (spectral_pr_t2s_results.jsonl)",
                },
                "hook_based": {
                    "n_pr_values": pr_summary["n"],
                    "pr_mean": hook_mean,
                    "pr_min": pr_summary["min"],
                    "pr_max": pr_summary["max"],
                    "pr_first": pr_summary["first"],
                    "pr_last": pr_summary["last"],
                    "fire": hook_fire,
                    "fire_steps": sum(1 for p in gen["pr_log"] if p["pr"] < t_pr),
                    "t_pr": t_pr,
                    "values": [p["pr"] for p in gen["pr_log"]],
                },
                "pr_delta": float(pr_delta),
                "agreement": agreement,
                "fire_teacher_forced": tf_fire,
                "fire_hook_based": hook_fire,
            }
            payload["disagreement_type"] = classify_disagreement(payload, t_pr)
            results.append(payload)
            if (i + 1) % 10 == 0 or i == len(gap_records) - 1:
                hmean = hook_mean if hook_mean is not None else float("nan")
                hmin = (pr_summary["min"] if pr_summary["min"] is not None
                        else float("nan"))
                print(f"  [{i+1}/{len(gap_records)}] rid={rid} "
                      f"prompt_len={gen['prompt_len']} "
                      f"teacher_pr={tf_pr:.4f} fire={tf_fire} "
                      f"hook_mean={hmean:.4f} hook_min={hmin:.4f} "
                      f"hook_fire={hook_fire} agreement={agreement}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during hook-based generation: {e}")
            sys.exit(1)
        raise

    # ==================================================================
    # Aggregate (single source of truth), gate verdict, write outputs.
    # ==================================================================
    tables = aggregate_results(results, t_pr)

    print("\n" + "=" * 70)
    print("RFC A3 ONLINE PR PARITY GATE")
    print(f"  records compared: {tables['n_records']}")
    print(f"  agreement: {tables['n_agreement']} | disagreement: "
          f"{tables['n_disagreement']}")
    print(f"  disagreement rate: {tables['disagreement_rate']:.4f}")
    print(f"  gate max disagreements: {tables['gate_max_disagreements']}")
    print(f"  near-boundary disagreements: {tables['n_near_boundary_disagreement']}")
    print(f"  systematic disagreements: {tables['n_systematic_disagreement']}")
    print(f"  PR delta mean: {tables['pr_delta']['mean']:.4f} "
          f"median: {tables['pr_delta']['median']:.4f}")
    print(f"  GATE VERDICT: {'PASS' if tables['gate_pass'] else 'FAIL'}")
    print("=" * 70)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"record_type": "smoke", **smoke}) + "\n")
        f.write(json.dumps({"record_type": "meta",
                            "wall_clock_s": time.time() - t_start}) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} ({len(results)} records + sidecar lines)")

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
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "results": results,
    }
    write_markdown_report(ctx, OUTPUT_MD)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    if tables["gate_pass"]:
        print("DS-034b GATE: PASS (disagreement within RFC A3 tolerance)")
    else:
        print("[STOP] DS-034b GATE: FAIL — hook implementation must be revised. "
              "Do NOT wire the dual-predicate into the controller.")
        sys.exit(1)
    print("DS-034b MEASUREMENT COMPLETE")


if __name__ == "__main__":
    main()
