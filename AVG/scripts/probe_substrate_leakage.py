#!/usr/bin/env python3
"""
DS-011: Substrate-leakage measurement probe (OVERNIGHT COMPUTE TASK).

MEASUREMENT probe: no thresholds, no gate logic, no asserts on outcomes.
It produces data (docs/substrate_leakage_results.jsonl) and a report
(docs/SUBSTRATE_LEAKAGE_PROBE.md). human review and humans interpret.

Hypothesis under test (parked from the Jaynesian/adversarial critic exchange):
    "Blind orthogonal kicks at long rollout lengths (S>=128) cause measurable
     semantic-coherence degradation (substrate leakage) compared to dormant
     generation."

Model: distilgpt2. Device: cuda if available else cpu (GPU-enabled night
amendment). SEED = 42 everywhere; explicit torch.manual_seed + random.Random.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

# Standard root-as-package header. From scripts/, parents[2] resolves to '/';
# `import AVG.*` resolves via the /AVG mount. (Protocol import convention.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from AVG.core.metrics import (  # noqa: E402
    compute_coherent_token_ratio,
    is_code_syntax_context,
)
from AVG.governor.controller import (  # noqa: E402
    ActiveVarietyGovernor,
    compute_token_distinct_2_fast,
)

# ---------------------------------------------------------------------------
# Seed provenance
# ---------------------------------------------------------------------------
SEED = 42
MAX_NEW_TOKENS = 128
PROMPT_WHITESPACE_TOKENS = 40
VALID_SUBSET_SELECT_N = 100
DETERMINISM_SMOKE_N = 2

# Fixed trusted texts for active-condition calibration (3 texts).
# Text 1 reuses the trusted text from tests/test_shadow_hook_cleanup.py;
# texts 2-3 are drawn from the repo's existing calibration pool
# (scripts/evaluate_avg.py).
TRUSTED_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "In physics, spacetime is any mathematical model which fuses the three "
    "dimensions of space.",
    "Photosynthesis is a process used by plants and other organisms to convert "
    "light energy into chemical energy.",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
T2S_DEGENERATE_PATH = REPO_ROOT / "tests" / "fixtures" / "t2s_degenerate.jsonl"
VALID_SUBSET_200_PATH = REPO_ROOT / "data" / "t2s_bench" / "valid_subset_200.jsonl"
RESULTS_PATH = REPO_ROOT / "docs" / "substrate_leakage_results.jsonl"
REPORT_PATH = REPO_ROOT / "docs" / "SUBSTRATE_LEAKAGE_PROBE.md"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL corpus and return a list of record dicts."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def first_n_whitespace_tokens(text: str, n: int = PROMPT_WHITESPACE_TOKENS) -> str:
    """Return the first ``n`` whitespace-delimited tokens of ``text``."""
    tokens = text.split()
    if len(tokens) <= n:
        return " ".join(tokens)
    return " ".join(tokens[:n])


def seed_all(seed: int = SEED) -> None:
    """Deterministic seeding for torch and the global RNG (SEED=42)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def make_governor(
    model,
    tokenizer,
    device: torch.device,
    use_dual_resolution: bool = True,
) -> ActiveVarietyGovernor:
    """Build a calibrated governor.

    Reuses the calibrate() pattern from tests/test_shadow_hook_cleanup.py:
    a single governor constructed with an explicit device and calibrated on
    fixed trusted texts before any generation. Here we calibrate on three
    fixed trusted texts per the DS-011 spec.
    """
    seed_all(SEED)
    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        device=device,
        use_dual_resolution=use_dual_resolution,
    )
    trusted_ids = [
        tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        for text in TRUSTED_TEXTS
    ]
    governor.calibrate(trusted_ids)
    return governor


def run_condition(
    governor: ActiveVarietyGovernor,
    tokenizer,
    prompt: str,
    device: torch.device,
    condition: str,
) -> Dict[str, Any]:
    """Run one governed generation for a single condition.

    ``intervene`` is True for condition "active" and False for "dormant".
    Seeding happens immediately before generate() so the intervention noise
    (torch.randn_like in the blind orthogonal kick) is reproducible.
    """
    seed_all(SEED)
    intervene = condition == "active"
    with torch.no_grad():
        out = governor.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            intervene=intervene,
        )

    seq = out["sequences"]
    prompt_len = out["prompt_len"]
    gen_only_text = out.get("gen_only_text", "")

    distinct_2, repeated_ids = compute_token_distinct_2_fast(
        seq, prompt_len=prompt_len, window_len=24
    )
    ctr = compute_coherent_token_ratio(gen_only_text)
    code_flag = is_code_syntax_context(gen_only_text)
    intervention_log = out.get("intervention_log", [])
    intervention_summary = out.get("intervention_summary", {})

    return {
        "corpus": None,  # filled by caller
        "sample_id": None,  # filled by caller
        "condition": condition,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt": prompt,
        "prompt_whitespace_tokens": len(prompt.split()),
        "prompt_token_len": int(prompt_len),
        "continuation": gen_only_text,
        "continuation_token_len": int(seq.shape[-1] - prompt_len),
        "distinct_2": float(distinct_2),
        "repeated_token_count": len(repeated_ids),
        "coherent_token_ratio": float(ctr),
        "is_code_syntax_context": bool(code_flag),
        "n_interventions": int(len(intervention_log)),
        "intervention_summary": intervention_summary,
        "device": str(device),
    }


def run_pipeline(
    governor: ActiveVarietyGovernor,
    tokenizer,
    prompts: Sequence[Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run dormant + active conditions for each prompt in the list.

    ``prompts`` is a list of dicts with at least {"corpus", "sample_id",
    "prompt"}. Returns a flat list of per-condition result dicts, one per
    (prompt, condition), in stable order: for each prompt, dormant first then
    active.
    """
    results: List[Dict[str, Any]] = []
    for item in prompts:
        for condition in ("dormant", "active"):
            rec = run_condition(governor, tokenizer, item["prompt"], device, condition)
            rec["corpus"] = item["corpus"]
            rec["sample_id"] = item["sample_id"]
            results.append(rec)
    return results


def canonical_jsonl(results: List[Dict[str, Any]]) -> str:
    """Serialize results to canonical JSONL (sorted keys, no spaces)."""
    lines = [
        json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for rec in results
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def run_determinism_smoke(
    model,
    tokenizer,
    device: torch.device,
    smoke_prompts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the pipeline twice on a 2-sample subset and assert identical output.

    Scaffolding determinism only: proves the full pipeline (calibrate ->
    generate -> metrics) is byte-identical across two runs. A fresh governor
    is calibrated for each run so the proof covers calibration too.
    """
    run_a: List[Dict[str, Any]] = []
    run_b: List[Dict[str, Any]] = []

    gov_a = make_governor(model, tokenizer, device)
    run_a = run_pipeline(gov_a, tokenizer, smoke_prompts, device)
    text_a = canonical_jsonl(run_a)

    gov_b = make_governor(model, tokenizer, device)
    run_b = run_pipeline(gov_b, tokenizer, smoke_prompts, device)
    text_b = canonical_jsonl(run_b)

    sha_a = hashlib.sha256(text_a.encode("utf-8")).hexdigest()
    sha_b = hashlib.sha256(text_b.encode("utf-8")).hexdigest()
    identical = text_a == text_b

    return {
        "identical": identical,
        "sha256_run_a": sha_a,
        "sha256_run_b": sha_b,
        "n_results": len(run_a),
        "smoke_prompts": [
            {"corpus": p["corpus"], "sample_id": p["sample_id"]} for p in smoke_prompts
        ],
    }


# ---------------------------------------------------------------------------
# Aggregation / report helpers
# ---------------------------------------------------------------------------
def p90(data: Sequence[float]) -> float:
    """90th percentile via the nearest-rank method.

    Always returns a value actually present in the sample, so it stays within
    [min, max] even for tiny samples (no extrapolation beyond the data range).
    """
    vals = sorted(float(x) for x in data)
    if not vals:
        return float("nan")
    k = max(1, min(len(vals), math.ceil(0.90 * len(vals))))
    return vals[k - 1]


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    """Return mean/median/p90 for a numeric metric."""
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p90": p90(vals),
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-corpus per-condition aggregate statistics."""
    corpus_names = ["t2s_degenerate", "valid_subset_200"]
    conditions = ["dormant", "active"]
    agg: Dict[str, Any] = {}

    for corpus in corpus_names:
        agg[corpus] = {}
        for cond in conditions:
            subset = [
                r for r in results if r["corpus"] == corpus and r["condition"] == cond
            ]
            distinct_2 = [r["distinct_2"] for r in subset]
            ctr = [r["coherent_token_ratio"] for r in subset]
            n_interv = [r["n_interventions"] for r in subset]
            code_true = sum(1 for r in subset if r["is_code_syntax_context"])

            agg[corpus][cond] = {
                "n_samples": len(subset),
                "distinct_2": summarize_metric(distinct_2),
                "coherent_token_ratio": summarize_metric(ctr),
                "n_interventions": {
                    **summarize_metric(n_interv),
                    "total": sum(n_interv),
                    "max": max(n_interv) if n_interv else 0,
                },
                "code_syntax_context_true": code_true,
                "code_syntax_context_fraction": (
                    code_true / len(subset) if subset else float("nan")
                ),
            }
    return agg


def markdown_table(
    agg: Dict[str, Any],
    corpus: str,
    metric_key: str,
    metric_label: str,
) -> str:
    """Render a mean/median/p90 table for one metric across conditions."""
    rows = []
    for cond in ("dormant", "active"):
        m = agg[corpus][cond][metric_key]
        rows.append(
            f"| {cond} | {m['mean']:.4f} | {m['median']:.4f} | {m['p90']:.4f} |"
        )
    return (
        f"### {metric_label} — {corpus}\n\n"
        f"| Condition | Mean | Median | p90 |\n"
        f"|-----------|------|--------|-----|\n"
        + "\n".join(rows)
        + "\n"
    )


def build_report(
    results: List[Dict[str, Any]],
    agg: Dict[str, Any],
    smoke: Dict[str, Any],
    device: str,
    runtime_sec: float,
    selected_valid_ids: List[int],
    model_id: str,
    dtype: str = "torch.float32",
    report_title: str = "DS-011 Substrate-Leakage Measurement Probe",
    results_path_label: str = "docs/substrate_leakage_results.jsonl",
) -> str:
    """Assemble the strictly-factual markdown report (no verdict)."""
    n_dormant = sum(1 for r in results if r["condition"] == "dormant")
    n_active = sum(1 for r in results if r["condition"] == "active")
    n_total = len(results)

    lines: List[str] = []
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append("> MEASUREMENT probe — no thresholds, no gate logic, no asserts on")
    lines.append("> outcomes. Data and report only; human review and humans interpret.")
    lines.append("")
    lines.append("## Hypothesis under test (parked from the Jaynesian/adversarial critic exchange)")
    lines.append("")
    lines.append('> "Blind orthogonal kicks at long rollout lengths (S>=128) cause')
    lines.append('> measurable semantic-coherence degradation (substrate leakage)')
    lines.append('> compared to dormant generation."')
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Model | `{model_id}` |")
    lines.append(f"| Device | `{device}` |")
    lines.append(f"| dtype | `{dtype}` |")
    lines.append(f"| SEED | `{SEED}` (torch.manual_seed + random.Random; cuda.manual_seed_all when cuda) |")
    lines.append(f"| max_new_tokens | `{MAX_NEW_TOKENS}` |")
    lines.append(f"| Prompt truncation | first `{PROMPT_WHITESPACE_TOKENS}` whitespace tokens |")
    lines.append(f"| do_sample | `False` (greedy) |")
    lines.append(f"| t2s_degenerate samples | 100/100 |")
    lines.append(f"| valid_subset_200 samples | {len(selected_valid_ids)}/200 (seeded select, see below) |")
    lines.append(f"| Conditions per prompt | dormant (`intervene=False`), active (`intervene=True`) |")
    lines.append(f"| Result records | {n_total} ({n_dormant} dormant + {n_active} active) |")
    lines.append(f"| Runtime (wall clock) | {runtime_sec:.1f} s ({runtime_sec / 60:.1f} min) |")
    lines.append("")
    lines.append("### Calibration trusted texts (active condition)")
    lines.append("")
    for i, t in enumerate(TRUSTED_TEXTS, 1):
        lines.append(f"{i}. `{t}`")
    lines.append("")
    lines.append("### valid_subset_200 selection (seeded)")
    lines.append("")
    lines.append(
        "100 of the 200 certified valid_subset_200 records were selected "
        f"deterministically with `random.Random({SEED}).sample(records, {VALID_SUBSET_SELECT_N})`, "
        "then sorted by corpus `id` for processing order."
    )
    lines.append("")
    lines.append(f"Selected ids: `{selected_valid_ids}`")
    lines.append("")
    lines.append("## Determinism smoke (scaffolding)")
    lines.append("")
    smoke_ok = smoke["identical"]
    lines.append(
        f"- Subset: {len(smoke['smoke_prompts'])} prompt(s) — "
        + ", ".join(
            f"{p['corpus']}#{p['sample_id']}" for p in smoke["smoke_prompts"]
        )
    )
    lines.append(f"- Run A SHA-256: `{smoke['sha256_run_a']}`")
    lines.append(f"- Run B SHA-256: `{smoke['sha256_run_b']}`")
    lines.append(
        f"- Identical: **{'YES' if smoke_ok else 'NO — RED GATE'}"
        f"** ({smoke['n_results']} result records compared)"
    )
    lines.append("")
    if not smoke_ok:
        lines.append("**RED GATE: determinism smoke failed.** The pipeline is not")
        lines.append("byte-reproducible under SEED=42. See agent report.")
        lines.append("")
    lines.append("## Aggregate metric tables")
    lines.append("")
    for corpus in ("t2s_degenerate", "valid_subset_200"):
        lines.append(markdown_table(agg, corpus, "distinct_2", "Distinct-2 (last 24 generated tokens)"))
        lines.append("")
        lines.append(markdown_table(agg, corpus, "coherent_token_ratio", "Coherent-token ratio (CTR)"))
        lines.append("")
        lines.append("### Code-syntax-context flag")
        lines.append("")
        lines.append(f"| Condition | True count | Fraction |")
        lines.append(f"|-----------|-----------|----------|")
        for cond in ("dormant", "active"):
            c = agg[corpus][cond]
            lines.append(
                f"| {cond} | {c['code_syntax_context_true']} | "
                f"{c['code_syntax_context_fraction']:.4f} |"
            )
        lines.append("")
        lines.append("### Intervention counts")
        lines.append("")
        lines.append(f"| Condition | Mean | Median | p90 | Total | Max |")
        lines.append(f"|-----------|------|--------|-----|-------|-----|")
        for cond in ("dormant", "active"):
            ni = agg[corpus][cond]["n_interventions"]
            lines.append(
                f"| {cond} | {ni['mean']:.4f} | {ni['median']:.4f} | "
                f"{ni['p90']:.4f} | {ni['total']} | {ni['max']} |"
            )
        lines.append("")

    # Byte-identical active-vs-dormant pair count (directly derivable from
    # the result records; a factual statement, not a verdict).
    by_sample: Dict[tuple, Dict[str, Any]] = {}
    for r in results:
        by_sample[(r["corpus"], r["sample_id"], r["condition"])] = r
    identical_pairs = 0
    total_pairs = 0
    for corpus in ("t2s_degenerate", "valid_subset_200"):
        sample_ids = sorted({r["sample_id"] for r in results if r["corpus"] == corpus})
        for sid in sample_ids:
            d = by_sample.get((corpus, sid, "dormant"))
            a = by_sample.get((corpus, sid, "active"))
            if d is not None and a is not None:
                total_pairs += 1
                if d["continuation"] == a["continuation"]:
                    identical_pairs += 1

    lines.append("## Observations (strictly factual)")
    lines.append("")
    lines.append("The following statements report measured values only. No verdict on")
    lines.append("the hypothesis is offered; that is for human review and human")
    lines.append("ratification.")
    lines.append("")
    total_active_interventions = sum(
        agg[c]["active"]["n_interventions"]["total"]
        for c in ("t2s_degenerate", "valid_subset_200")
    )
    if total_active_interventions == 0:
        lines.append(
            f"- Across both corpora, {identical_pairs}/{total_pairs} active continuations "
            "were byte-identical to their dormant counterpart on the same prompt "
            "(0 interventions were logged in the active condition)."
        )
    else:
        lines.append(
            f"- Across both corpora, {identical_pairs}/{total_pairs} active continuations "
            "were byte-identical to their dormant counterpart on the same prompt; "
            f"the active condition logged {total_active_interventions} total interventions "
            "across continuations (per-corpus counts in the intervention tables below)."
        )
    for corpus in ("t2s_degenerate", "valid_subset_200"):
        d = agg[corpus]["dormant"]
        a = agg[corpus]["active"]
        lines.append(f"- In `{corpus}`, dormant mean Distinct-2 was "
                     f"{d['distinct_2']['mean']:.4f} "
                     f"(median {d['distinct_2']['median']:.4f}, "
                     f"p90 {d['distinct_2']['p90']:.4f}); "
                     f"active mean Distinct-2 was {a['distinct_2']['mean']:.4f} "
                     f"(median {a['distinct_2']['median']:.4f}, "
                     f"p90 {a['distinct_2']['p90']:.4f}).")
        lines.append(f"- In `{corpus}`, dormant mean CTR was "
                     f"{d['coherent_token_ratio']['mean']:.4f} "
                     f"(median {d['coherent_token_ratio']['median']:.4f}, "
                     f"p90 {d['coherent_token_ratio']['p90']:.4f}); "
                     f"active mean CTR was {a['coherent_token_ratio']['mean']:.4f} "
                     f"(median {a['coherent_token_ratio']['median']:.4f}, "
                     f"p90 {a['coherent_token_ratio']['p90']:.4f}).")
        lines.append(f"- In `{corpus}`, the active condition logged "
                     f"{a['n_interventions']['total']} total interventions "
                     f"(mean {a['n_interventions']['mean']:.4f} per continuation, "
                     f"median {a['n_interventions']['median']:.4f}, "
                     f"p90 {a['n_interventions']['p90']:.4f}, "
                     f"max {a['n_interventions']['max']}). "
                     f"The dormant condition logged {d['n_interventions']['total']} "
                     f"interventions by construction (`intervene=False`).")
        lines.append(f"- In `{corpus}`, the code-syntax-context flag was True on "
                     f"{d['code_syntax_context_true']}/{d['n_samples']} "
                     f"dormant continuations and "
                     f"{a['code_syntax_context_true']}/{a['n_samples']} "
                     f"active continuations.")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Results JSONL: `{results_path_label}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="DS-011/DS-020 substrate-leakage probe")
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="HF model id (default: distilgpt2).",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Output JSONL path (default: docs/substrate_leakage_results.jsonl).",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Output report path (default: docs/SUBSTRATE_LEAKAGE_PROBE.md).",
    )
    parser.add_argument(
        "--report-title",
        type=str,
        default="DS-011 Substrate-Leakage Measurement Probe",
        help="Report H1 title (default: DS-011).",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the determinism smoke on a 2-sample subset and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N prompts of the full pipeline (dev/testing).",
    )
    args = parser.parse_args()

    results_path = Path(args.results_path) if args.results_path else RESULTS_PATH
    report_path = Path(args.report_path) if args.report_path else REPORT_PATH
    try:
        results_path_label = str(results_path.relative_to(REPO_ROOT))
    except ValueError:
        results_path_label = str(results_path)

    t_start = time.time()
    print(f"[ds-011] SEED={SEED} device=auto max_new_tokens={MAX_NEW_TOKENS}", flush=True)

    # ------------------------------------------------------------------
    # Model load. GPU-enabled night amendment; fp16 on cuda, fp32 on cpu.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[ds-011] Loading {args.model} on {device} with {torch_dtype} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    print(f"[ds-011] Model loaded. hidden_size={model.config.hidden_size}, "
          f"dtype={model.dtype}", flush=True)

    # ------------------------------------------------------------------
    # Build prompt lists
    # ------------------------------------------------------------------
    t2s_records = load_jsonl(T2S_DEGENERATE_PATH)
    valid_records = load_jsonl(VALID_SUBSET_200_PATH)
    print(f"[ds-011] Loaded {len(t2s_records)} t2s_degenerate, "
          f"{len(valid_records)} valid_subset_200", flush=True)

    # All 100 t2s_degenerate samples.
    t2s_prompts = [
        {
            "corpus": "t2s_degenerate",
            "sample_id": int(rec["id"]),
            "prompt": first_n_whitespace_tokens(rec["text"]),
        }
        for rec in sorted(t2s_records, key=lambda r: int(r["id"]))
    ]

    # Seeded 100-sample subset of valid_subset_200.
    rng = random.Random(SEED)
    selected_valid = sorted(
        rng.sample(valid_records, VALID_SUBSET_SELECT_N), key=lambda r: int(r["id"])
    )
    selected_valid_ids = [int(r["id"]) for r in selected_valid]
    valid_prompts = [
        {
            "corpus": "valid_subset_200",
            "sample_id": int(rec["id"]),
            "prompt": first_n_whitespace_tokens(rec["text"]),
        }
        for rec in selected_valid
    ]

    all_prompts: List[Dict[str, Any]] = t2s_prompts + valid_prompts
    if args.limit is not None:
        all_prompts = all_prompts[: args.limit]
    print(f"[ds-011] Prompt list: {len(all_prompts)} prompts "
          f"(t2s_degenerate {len(t2s_prompts)}, "
          f"valid_subset_200 {len(valid_prompts)})", flush=True)

    # ------------------------------------------------------------------
    # Determinism smoke (DETERMINISM_SMOKE_N-sample subset, run twice).
    # ------------------------------------------------------------------
    smoke_prompts = [
        {"corpus": "t2s_degenerate", "sample_id": int(t2s_prompts[i]["sample_id"]),
         "prompt": t2s_prompts[i]["prompt"]}
        for i in range(DETERMINISM_SMOKE_N)
    ]
    print(f"[ds-011] Determinism smoke on {len(smoke_prompts)} prompts ...", flush=True)
    smoke = run_determinism_smoke(model, tokenizer, device, smoke_prompts)
    print(f"[ds-011] Smoke run A SHA-256: {smoke['sha256_run_a']}", flush=True)
    print(f"[ds-011] Smoke run B SHA-256: {smoke['sha256_run_b']}", flush=True)
    print(f"[ds-011] Smoke identical: {smoke['identical']}", flush=True)
    if not smoke["identical"]:
        print("[ds-011] RED GATE: determinism smoke failed.", file=sys.stderr, flush=True)
        return 1

    if args.smoke_only:
        print("[ds-011] --smoke-only: skipping full pipeline.", flush=True)
        return 0

    # ------------------------------------------------------------------
    # Full pipeline.
    # ------------------------------------------------------------------
    print("[ds-011] Calibrating governor on 3 fixed trusted texts ...", flush=True)
    governor = make_governor(model, tokenizer, device)
    print(f"[ds-011] Governor calibrated. Baseline layers: "
          f"{sorted(governor.baseline.mean_attenuation.keys())}", flush=True)

    print(f"[ds-011] Running full pipeline over {len(all_prompts)} prompts "
          f"x 2 conditions ...", flush=True)
    results = run_pipeline(governor, tokenizer, all_prompts, device)
    print(f"[ds-011] Pipeline complete: {len(results)} result records.", flush=True)

    # ------------------------------------------------------------------
    # Write results JSONL.
    # ------------------------------------------------------------------
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(
                json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    print(f"[ds-011] Wrote {results_path}", flush=True)

    # ------------------------------------------------------------------
    # Aggregate + report.
    # ------------------------------------------------------------------
    agg = aggregate_results(results)
    runtime_sec = time.time() - t_start
    report = build_report(
        results,
        agg,
        smoke,
        str(device),
        runtime_sec,
        selected_valid_ids,
        model_id=args.model,
        dtype=str(torch_dtype),
        report_title=args.report_title,
        results_path_label=results_path_label,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[ds-011] Wrote {report_path}", flush=True)
    print(f"[ds-011] Total runtime: {runtime_sec:.1f} s ({runtime_sec / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
