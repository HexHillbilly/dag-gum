#!/usr/bin/env python3
"""DS-023: Qwen degeneracy-induction stimulus corpus (Gate 2.2 support).

Builds and characterizes a Qwen2.5-1.5B degeneracy-induction corpus.

Gate 2.2 needs a degenerate corpus that ACTUALLY induces degeneracy on
Qwen2.5-1.5B.  ds-020 proved t2s_degenerate does not under greedy
(median Distinct-2 1.0000).  This builder instead *induces* degeneracy via
a small decoding-parameter grid (do_sample=True, temperature in {0.9, 1.2},
top_p in {0.95, 1.0}, repetition_penalty=1.0) over the 100 t2s_degenerate
PROMPT texts (first 40 whitespace tokens each, matching the ds-011/ds-020
prompt-truncation convention), then curates the 100 grid continuations with
the LOWEST last-24-token Distinct-2 (computed by
``AVG.governor.controller.compute_token_distinct_2_fast``).  Corpus curation
is a measurement/selection step, not a gate.

A prose control (same seeded 100-of-200 valid_subset_200 selection as
ds-011/ds-020, greedy decoding) is generated in the same run so its
Distinct-2 distribution can be reported for comparison.  The control is not a
fixture deliverable; only its aggregate distribution is reported.

Seed provenance: SEED = 4243 (DS-023).  Every grid generation uses a unique
deterministic seed derived from SEED, the t2s sample index, and the grid-cell
index (SEED*1000 + sample_idx*10 + cell_idx), so the full 400-record grid,
the 100-record curated fixture, and the control are byte-identical across
runs and reviewers.

Model: Qwen/Qwen2.5-1.5B (public, no token).  fp16 on cuda, fp32 on cpu;
device and dtype are recorded in the report.  HF cache is read-only here
(model is already cached).  The container has no C compiler, so the Triton
native-DSL path is disabled via TORCH_DISABLE_NATIVE_JIT=1 (torch falls back
to the eager aten implementation); this is a runtime-only workaround, not a
repo change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Container workaround: no C compiler, so disable torch's Triton native-DSL
# ops (they would need to JIT a CUDA-driver wrapper with a C compiler).
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch  # noqa: E402

# Standard root-as-package header. From scripts/, parents[2] resolves to '/';
# `import AVG.*` resolves via the /AVG mount. (Protocol import convention.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from AVG.governor.controller import compute_token_distinct_2_fast  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 4243
CONTROL_SEED = 42  # valid_subset_200 selection seed, same as ds-011/ds-020
MODEL_ID = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 128
PROMPT_WHITESPACE_TOKENS = 40
NUM_SELECT = 100
CONTROL_SELECT_N = 100

# Degeneracy-inducing decoding grid (do_sample=True throughout;
# repetition_penalty=1.0 means no anti-repetition pressure).
GRID: List[Dict[str, Any]] = [
    {"temperature": 0.9, "top_p": 0.95},
    {"temperature": 0.9, "top_p": 1.0},
    {"temperature": 1.2, "top_p": 0.95},
    {"temperature": 1.2, "top_p": 1.0},
]
GRID_SAMPLE = True
GRID_REPETITION_PENALTY = 1.0

# Determinism smoke: regenerate these (sample_idx, cell_idx) records and
# assert byte-identical text.
SMOKE_POINTS = [(0, 0), (1, 3)]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
T2S_DEGENERATE_PATH = ROOT / "tests" / "fixtures" / "t2s_degenerate.jsonl"
VALID_SUBSET_200_PATH = ROOT / "data" / "t2s_bench" / "valid_subset_200.jsonl"
OUTPUT_PATH = ROOT / "tests" / "fixtures" / "qwen_degenerate.jsonl"
REPORT_PATH = ROOT / "docs" / "QWEN_DEGENERATE_PROFILE.md"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    """Deterministic seeding for torch, cuda, and the global RNG."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


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


def sha256_bytes(data: bytes) -> str:
    """Return the hex sha256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def p10(values: Sequence[float]) -> float:
    """10th percentile via the nearest-rank method (value actually present)."""
    vals = sorted(float(x) for x in values)
    if not vals:
        return float("nan")
    k = max(1, min(len(vals), math.ceil(0.10 * len(vals))))
    return vals[k - 1]


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Return mean/median/p10 for a numeric metric."""
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan")}
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p10": p10(vals),
    }


def markdown_table(corpus: Dict[str, float], control: Dict[str, float]) -> str:
    """Render the corpus-vs-control mean/median/p10 comparison table."""
    return (
        f"| Corpus | Mean | Median | p10 |\n"
        f"|--------|------|--------|-----|\n"
        f"| qwen_degenerate (curated) | {corpus['mean']:.4f} | "
        f"{corpus['median']:.4f} | {corpus['p10']:.4f} |\n"
        f"| valid_subset_200 control (greedy) | {control['mean']:.4f} | "
        f"{control['median']:.4f} | {control['p10']:.4f} |\n"
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def run_generation(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    top_p: float,
    do_sample: bool,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one seeded generation and return the continuation + Distinct-2.

    ``seed`` is applied immediately before ``generate()`` so the sampling RNG
    path is fully reproducible (verified by the determinism smoke).
    ``min_new_tokens`` is set to ``MAX_NEW_TOKENS`` so every record carries
    exactly ``MAX_NEW_TOKENS`` generated tokens (a sampled EOS token does not
    truncate the continuation).
    """
    seed_all(seed)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(enc["input_ids"].shape[-1])
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=GRID_REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    seq = out[0]
    gen_ids = seq[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    distinct_2, _repeated = compute_token_distinct_2_fast(
        seq.unsqueeze(0), prompt_len=prompt_len, window_len=24
    )
    return {
        "prompt": prompt,
        "text": text,
        "distinct_2": float(distinct_2),
        "prompt_token_len": prompt_len,
        "continuation_token_len": int(gen_ids.shape[-1]),
    }


def run_determinism_smoke(
    model, tokenizer, prompts: List[str], device: torch.device
) -> Dict[str, Any]:
    """Regenerate two grid records and assert byte-identical continuations."""
    results: Dict[str, Any] = {"identical": True, "pairs": []}
    for sample_idx, cell_idx in SMOKE_POINTS:
        cell = GRID[cell_idx]
        seed = _grid_seed(sample_idx, cell_idx)
        prompt = prompts[sample_idx]
        a = run_generation(
            model, tokenizer, prompt, cell["temperature"], cell["top_p"],
            GRID_SAMPLE, seed, device,
        )
        b = run_generation(
            model, tokenizer, prompt, cell["temperature"], cell["top_p"],
            GRID_SAMPLE, seed, device,
        )
        same = a["text"] == b["text"]
        results["pairs"].append(
            {
                "sample_idx": sample_idx,
                "cell_idx": cell_idx,
                "seed": seed,
                "identical": same,
                "sha256_run_a": sha256_bytes(a["text"].encode("utf-8")),
                "sha256_run_b": sha256_bytes(b["text"].encode("utf-8")),
            }
        )
        if not same:
            results["identical"] = False
    return results


def _grid_seed(sample_idx: int, cell_idx: int) -> int:
    """Deterministic per-record seed derived from SEED, sample, and cell."""
    return SEED * 1000 + sample_idx * 10 + cell_idx


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_grid(
    model, tokenizer, t2s_prompts: List[str], device: torch.device
) -> List[Dict[str, Any]]:
    """Generate the full 100x4 grid and return 400 flat records."""
    records: List[Dict[str, Any]] = []
    total = len(t2s_prompts) * len(GRID)
    n = 0
    for sample_idx, prompt in enumerate(t2s_prompts):
        for cell_idx, cell in enumerate(GRID):
            seed = _grid_seed(sample_idx, cell_idx)
            gen = run_generation(
                model, tokenizer, prompt, cell["temperature"], cell["top_p"],
                GRID_SAMPLE, seed, device,
            )
            records.append(
                {
                    # Top-level keys match the DS-023 fixture spec exactly:
                    # id, prompt, decoding, text, distinct_2.  All provenance
                    # (seed, source t2s id, grid coordinates) lives inside the
                    # ``decoding`` dict.
                    "id": None,  # assigned by curate()
                    "prompt": prompt,
                    "decoding": {
                        "do_sample": GRID_SAMPLE,
                        "temperature": cell["temperature"],
                        "top_p": cell["top_p"],
                        "repetition_penalty": GRID_REPETITION_PENALTY,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "min_new_tokens": MAX_NEW_TOKENS,
                        "seed": seed,
                        "source_id": sample_idx,
                        "sample_idx": sample_idx,
                        "cell_idx": cell_idx,
                    },
                    "text": gen["text"],
                    "distinct_2": gen["distinct_2"],
                }
            )
            n += 1
            if n % 50 == 0 or n == total:
                print(f"[ds-023] grid generation {n}/{total}", flush=True)
    return records


def build_control(
    model, tokenizer, valid_records: List[Dict[str, Any]], device: torch.device
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Seeded 100-of-200 valid_subset_200 selection + greedy generation.

    Returns (control records, selected ids).  Mirrors ds-011/ds-020:
    ``random.Random(CONTROL_SEED).sample(records, 100)`` then sorted by id.
    """
    rng = random.Random(CONTROL_SEED)
    selected = sorted(
        rng.sample(valid_records, CONTROL_SELECT_N), key=lambda r: int(r["id"])
    )
    selected_ids = [int(r["id"]) for r in selected]
    records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(selected):
        prompt = first_n_whitespace_tokens(rec["text"])
        gen = run_generation(
            model, tokenizer, prompt, 1.0, 1.0, False, CONTROL_SEED, device
        )
        records.append(
            {
                "id": int(rec["id"]),
                "prompt": prompt,
                "text": gen["text"],
                "distinct_2": gen["distinct_2"],
                "seed": CONTROL_SEED,
            }
        )
        if (idx + 1) % 25 == 0 or (idx + 1) == len(selected):
            print(f"[ds-023] control generation {idx + 1}/{len(selected)}", flush=True)
    return records, selected_ids


def curate(grid_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select the 100 records with the LOWEST Distinct-2.

    Stable tie-break by generation order (sample_idx, cell_idx), so selection
    is deterministic.  Corpus curation only; not a gate.
    """
    ordered = sorted(
        grid_records,
        key=lambda r: (
            r["distinct_2"],
            r["decoding"]["sample_idx"],
            r["decoding"]["cell_idx"],
        ),
    )
    selected = ordered[:NUM_SELECT]
    # Re-id as 0..99 in curation order.
    for i, rec in enumerate(selected):
        rec["id"] = i
    return selected


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> str:
    """Write records as canonical JSONL and return the file's sha256."""
    lines = [
        json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for rec in records
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode("utf-8"))


def build_report(
    *,
    grid: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    control: List[Dict[str, Any]],
    smoke: Dict[str, Any],
    selected_ids: List[int],
    device: str,
    dtype: str,
    fixture_sha256: str,
    runtime_sec: float,
    limit: Any,
) -> str:
    """Assemble docs/QWEN_DEGENERATE_PROFILE.md (strictly factual)."""
    corpus_d2 = [r["distinct_2"] for r in corpus]
    control_d2 = [r["distinct_2"] for r in control]
    corpus_summary = summarize(corpus_d2)
    control_summary = summarize(control_d2)
    gap = corpus_summary["mean"] - control_summary["mean"]

    lines: List[str] = []
    lines.append("# DS-023 — Qwen Degeneracy-Induction Stimulus Corpus")
    lines.append("")
    lines.append("> Corpus curation — no gate thresholds, no governor changes.")
    lines.append("> The gap below is a REPORTED measurement, never a forced one;")
    lines.append("> if induction had failed, that would be a valid negative result.")
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Model | `{MODEL_ID}` |")
    lines.append(f"| Device | `{device}` |")
    lines.append(f"| dtype | `{dtype}` |")
    lines.append(f"| SEED | `{SEED}` (grid seeds `SEED*1000 + sample_idx*10 + cell_idx`) |")
    lines.append(f"| control selection seed | `{CONTROL_SEED}` (`random.Random(42).sample`, sorted by id) |")
    lines.append(f"| max_new_tokens / min_new_tokens | `{MAX_NEW_TOKENS}` / `{MAX_NEW_TOKENS}` (every record is exactly 128 generated tokens) |")
    lines.append(f"| Prompt truncation | first `{PROMPT_WHITESPACE_TOKENS}` whitespace tokens |")
    lines.append(f"| Runtime (wall clock) | {runtime_sec:.1f} s ({runtime_sec / 60:.1f} min) |")
    lines.append("")
    lines.append("## Decoding grid (degeneracy-induction)")
    lines.append("")
    lines.append("All grid cells use `do_sample=True` and")
    lines.append(f"`repetition_penalty={GRID_REPETITION_PENALTY}` (no anti-repetition pressure).")
    lines.append("")
    lines.append("| # | temperature | top_p |")
    lines.append("|---|------------|-------|")
    for i, cell in enumerate(GRID):
        lines.append(f"| {i} | {cell['temperature']} | {cell['top_p']} |")
    lines.append("")
    lines.append(f"The grid yields {len(t2s_prompts_cache) * len(GRID)} records "
                 f"(100 prompts x {len(GRID)} cells); the 100 with the LOWEST "
                 "last-24-token Distinct-2 are curated into the fixture.")
    lines.append("")
    lines.append("## Determinism smoke (scaffolding)")
    lines.append("")
    lines.append("Two grid records were regenerated with identical seeds and compared:")
    lines.append("")
    lines.append("| sample_idx | cell_idx | seed | Run A sha256 | Run B sha256 | Identical |")
    lines.append("|---|---|---|---|---|---|")
    for p in smoke["pairs"]:
        lines.append(
            f"| {p['sample_idx']} | {p['cell_idx']} | {p['seed']} | "
            f"`{p['sha256_run_a']}` | `{p['sha256_run_b']}` | "
            f"{'YES' if p['identical'] else 'NO — RED GATE'} |"
        )
    lines.append("")
    if not smoke["identical"]:
        lines.append("**RED GATE: determinism smoke failed.** Grid generation is")
        lines.append("not byte-reproducible under the documented seeds.")
        lines.append("")
    lines.append("## Distinct-2 (last 24 generated tokens)")
    lines.append("")
    lines.append(markdown_table(corpus_summary, control_summary))
    lines.append("")
    lines.append(f"Curated corpus mean Distinct-2 is {abs(gap):.4f} "
                 f"{'BELOW' if gap < 0 else 'ABOVE/equal to'} the greedy prose control mean "
                 f"({corpus_summary['mean']:.4f} vs {control_summary['mean']:.4f}).")
    lines.append("")
    lines.append("### valid_subset_200 control selection (seeded)")
    lines.append("")
    lines.append(
        "100 of the 200 certified valid_subset_200 records were selected "
        f"deterministically with `random.Random({CONTROL_SEED}).sample(records, "
        f"{CONTROL_SELECT_N})`, then sorted by corpus `id` for processing order."
    )
    lines.append("")
    lines.append(f"Selected ids: `{selected_ids}`")
    lines.append("")
    lines.append("### Corpus curation provenance")
    lines.append("")
    lines.append("Grid records were sorted by `(distinct_2, sample_idx, cell_idx)` and")
    lines.append(f"the first {NUM_SELECT} taken; stable tie-break keeps selection")
    lines.append("byte-identical across runs.  The 100 curated records draw from")
    lines.append(f"{len(set(r['decoding']['sample_idx'] for r in corpus))} "
                 "distinct t2s_degenerate prompts (multiple grid cells of the same "
                 "prompt are allowed).")
    lines.append("")
    lines.append("## Fixture")
    lines.append("")
    lines.append("- JSONL: `tests/fixtures/qwen_degenerate.jsonl`")
    lines.append(f"- sha256: `{fixture_sha256}`")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append("- Grid records (all 400, unsorted): not persisted as a fixture;")
    lines.append("  the curated 100 are the deliverable.")
    lines.append("")
    return "\n".join(lines)


# Cache of t2s prompts for the report (populated in main).
t2s_prompts_cache: List[str] = []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DS-023 Qwen degeneracy-induction corpus builder"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only build the grid for the first N t2s prompts (dev/testing).",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the determinism smoke on 2 grid records and exit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Fixture output path (default: tests/fixtures/qwen_degenerate.jsonl).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Report output path (default: docs/QWEN_DEGENERATE_PROFILE.md).",
    )
    args = parser.parse_args()

    global t2s_prompts_cache
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    report_path = Path(args.report) if args.report else REPORT_PATH

    t_start = time.time()
    print(f"[ds-023] SEED={SEED} model={MODEL_ID} max_new_tokens={MAX_NEW_TOKENS}",
          flush=True)

    # ------------------------------------------------------------------
    # Model load. fp16 on cuda, fp32 on cpu.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[ds-023] Loading {MODEL_ID} on {device} with {torch_dtype} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch_dtype)
    model.to(device)
    model.eval()
    print(f"[ds-023] Model loaded. hidden_size={model.config.hidden_size}, "
          f"dtype={model.dtype}", flush=True)

    # ------------------------------------------------------------------
    # Load corpora and build prompt lists.
    # ------------------------------------------------------------------
    t2s_records = load_jsonl(T2S_DEGENERATE_PATH)
    valid_records = load_jsonl(VALID_SUBSET_200_PATH)
    t2s_records = sorted(t2s_records, key=lambda r: int(r["id"]))
    t2s_prompts_cache = [
        first_n_whitespace_tokens(rec["text"]) for rec in t2s_records
    ]
    if args.limit is not None:
        t2s_prompts_cache = t2s_prompts_cache[: args.limit]
    print(f"[ds-023] Loaded {len(t2s_records)} t2s_degenerate, "
          f"{len(valid_records)} valid_subset_200; "
          f"grid prompts={len(t2s_prompts_cache)}", flush=True)

    # ------------------------------------------------------------------
    # Determinism smoke.
    # ------------------------------------------------------------------
    print(f"[ds-023] Determinism smoke on {len(SMOKE_POINTS)} grid records ...",
          flush=True)
    smoke = run_determinism_smoke(model, tokenizer, t2s_prompts_cache, device)
    for p in smoke["pairs"]:
        print(f"[ds-023] smoke sample_idx={p['sample_idx']} cell_idx={p['cell_idx']} "
              f"identical={p['identical']}", flush=True)
    if not smoke["identical"]:
        print("[ds-023] RED GATE: determinism smoke failed.", file=sys.stderr, flush=True)
        return 1

    if args.smoke_only:
        print("[ds-023] --smoke-only: skipping corpus build.", flush=True)
        return 0

    # ------------------------------------------------------------------
    # Full grid + curation.
    # ------------------------------------------------------------------
    print(f"[ds-023] Building grid ({len(t2s_prompts_cache)} prompts x "
          f"{len(GRID)} cells) ...", flush=True)
    grid_records = build_grid(model, tokenizer, t2s_prompts_cache, device)
    if args.limit is None:
        corpus = curate(grid_records)
    else:
        # --limit is a dev mode: curate from whatever subset exists, but do not
        # claim it is the certified fixture.
        corpus = curate(grid_records)
    print(f"[ds-023] Curated {len(corpus)} records with LOWEST Distinct-2.", flush=True)

    # ------------------------------------------------------------------
    # Prose control (greedy, seeded selection).
    # ------------------------------------------------------------------
    print(f"[ds-023] Building prose control ({CONTROL_SELECT_N} valid_subset_200 "
          f"greedy) ...", flush=True)
    control_records, selected_ids = build_control(
        model, tokenizer, valid_records, device
    )
    print(f"[ds-023] Control complete: {len(control_records)} records.", flush=True)

    # ------------------------------------------------------------------
    # Write fixture + report.
    # ------------------------------------------------------------------
    if args.limit is None:
        fixture_sha256 = write_jsonl(corpus, output_path)
        print(f"[ds-023] Wrote {output_path} (sha256 {fixture_sha256})", flush=True)
    else:
        fixture_sha256 = "not-written (--limit dev mode)"

    runtime_sec = time.time() - t_start
    report = build_report(
        grid=grid_records,
        corpus=corpus,
        control=control_records,
        smoke=smoke,
        selected_ids=selected_ids,
        device=str(device),
        dtype=str(torch_dtype),
        fixture_sha256=fixture_sha256,
        runtime_sec=runtime_sec,
        limit=args.limit,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[ds-023] Wrote {report_path}", flush=True)

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    corpus_d2 = [r["distinct_2"] for r in corpus]
    control_d2 = [r["distinct_2"] for r in control_records]
    cs = summarize(corpus_d2)
    ctrl = summarize(control_d2)
    print(f"[ds-023] Corpus Distinct-2 mean={cs['mean']:.4f} median={cs['median']:.4f} "
          f"p10={cs['p10']:.4f}", flush=True)
    print(f"[ds-023] Control Distinct-2 mean={ctrl['mean']:.4f} "
          f"median={ctrl['median']:.4f} p10={ctrl['p10']:.4f}", flush=True)
    print(f"[ds-023] Total runtime: {runtime_sec:.1f} s ({runtime_sec / 60:.1f} min)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
