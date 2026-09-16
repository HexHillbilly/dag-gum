#!/usr/bin/env python3
"""DS-027: Gate 2.3b rebuilt held-out hard-degenerate slice (Part A).

MEASUREMENT / FIXTURE CURATION ONLY. No thresholds are derived here, no gate
scripts are touched, no governor/controller.py edits, no asserts on outcomes,
and no green/red verdict is produced. The frozen thresholds live in
``docs/gate23/FROZEN_THRESHOLDS.md`` (read-only) and are consumed by
``scripts/score_gate23.py`` (Part B, unchanged frozen thresholds).

Method (per the DS-027 task file / gate-2.3b greenlight construction, exactly):
  1. Held-out prose prompts = the COMPLEMENT of the Gate 2.2 prose control 100
     within the 200-record ``valid_subset_200`` fixture (same seeded
     ``random.Random(42).sample(records, 100)`` control selection as ds-011 /
     ds-020 / ds-023 / ds-024 / ds-025; the heldout slice is the other 100).
  2. For each of the 100 held-out prose records build ONE mutated prompt:
         first3 = first 3 whitespace tokens of the prose text
         mutated_prompt = (first3 x6) + " " + original prompt
     where the original prompt is the unmutated first-3-token prompt, i.e.
     mutated_prompt is the first-3-token phrase repeated 7 times in total
     (pure repetition, no prose continuation).
  3. Run the proven ds-023 degeneracy-induction decoding grid on those 100
     prompts (do_sample=True, temperature in {0.9, 1.2}, top_p in {0.95, 1.0},
     repetition_penalty=1.0, max_new_tokens=min_new_tokens=128). Grid size =
     100 prompts x 4 cells = 400 records.
  4. Curate the 100 grid records with the LOWEST last-24-token Distinct-2
     (``AVG.governor.controller.compute_token_distinct_2_fast``); stable
     tie-break sorts by (distinct_2, sample_idx, cell_idx) so selection is
     byte-identical across runs.
  5. Determinism smoke FIRST: two grid records regenerated with identical
     seeds must produce byte-identical continuations, otherwise STOP and
     report.
  6. SLICE-QUALITY STOP per greenlight: curated mean Distinct-2 must be
     <= 0.5 * heldout-100 prose mean (ds-025 measured 0.9700). If not: write
     the profile, STOP, and do not score.

Seed provenance: SEED = 4245 (DS-027). Every grid generation uses a unique
deterministic seed derived from SEED, the heldout sample index, and the
grid-cell index (SEED*1000 + sample_idx*10 + cell_idx), so the full 400-record
grid and the 100-record curated fixture are byte-identical across runs and
reviewers.

Model: Qwen/Qwen2.5-1.5B @ 8faed761 (public, no token). bf16 on cuda, fp32 on
cpu; device and dtype are recorded in the report. HF cache is read-only here
(model is already cached); set HF_HOME to a writable fallback when the system
HF cache is mounted read-only. The container has no C compiler, so the torch
2.13 CUDA ``bmm_outer_product`` Triton override is deregistered (env-only) and
``TORCH_DISABLE_NATIVE_JIT=1`` is set (runtime-only workaround); neither
modifies any repo file.
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
from typing import Any, Dict, List, Sequence

# Container workaround: no C compiler, so disable torch's Triton native-DSL
# ops (they would need to JIT a CUDA-driver wrapper with a C compiler).
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch  # noqa: E402

# Standard root-as-package header. From scripts/, parents[2] resolves to '/';
# `import AVG.*` resolves via the /AVG mount. (Protocol import convention.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Env-only adaptation: deregister torch 2.13's CUDA bmm Triton override
# (requires a Triton JIT C compiler absent from this container).
try:
    from torch._native import triton_utils as _triton_utils

    _triton_utils.deregister_op_overrides()
    print("[env] torch bmm Triton override deregistered")
except Exception as _env_e:  # pragma: no cover - env dependent
    print(f"[env] bmm override deregistration skipped: {_env_e}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from AVG.governor.controller import compute_token_distinct_2_fast  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 4245  # DS-027
CONTROL_SEED = 42  # valid_subset_200 control-selection seed (ds-011/ds-020)
MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128
MUTATE_WHITESPACE_TOKENS = 3  # first 3 whitespace tokens
MUTATE_REPEAT = 6  # x6
NUM_SELECT = 100
CONTROL_SELECT_N = 100
ANALYSIS_WINDOW = 24

# Proven ds-023 degeneracy-induction decoding grid (do_sample=True throughout;
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
VALID_SUBSET_200_PATH = ROOT / "data" / "t2s_bench" / "valid_subset_200.jsonl"
OUTPUT_PATH = ROOT / "tests" / "fixtures" / "heldout_degenerate_v2.jsonl"
REPORT_PATH = ROOT / "docs" / "gate23" / "HELDOUT_V2_PROFILE.md"

# The Gate 2.2 prose-control ids (printed in docs/SUBSTRATE_LEAKAGE_PROBE.md
# and docs/gate22/GATE22_AUROC_RESULTS.md). The heldout slice is the
# complement within valid_subset_200; if the complement split fails (any of
# these ids is not found, or the complement is not exactly 100), STOP.
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


def build_mutated_prompt(prose_text: str) -> str:
    """Greenlight mutation: (first 3 whitespace tokens) x6 + original prompt.

    ``original prompt`` is the unmutated first-3-whitespace-token prompt of
    the prose text. The mutated prompt is therefore the first-3-token phrase
    repeated 6 times followed by the original 3-token prompt, i.e. the phrase
    repeated 7 times in total. The prompt stays purely repetitive (no prose
    continuation) so the model has no coherent context to escape to; this is
    the degeneracy-induction mechanism the greenlight construction relies on.
    """
    tokens = prose_text.split()
    if not tokens:
        return ""
    first3 = " ".join(tokens[:MUTATE_WHITESPACE_TOKENS])
    original_prompt = first3
    return " ".join([first3] * MUTATE_REPEAT) + " " + original_prompt


def sha256_bytes(data: bytes) -> str:
    """Return the hex sha256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Return the hex sha256 digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


def select_heldout_prose(valid_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heldout-100 = complement of the Gate 2.2 prose-control 100 within
    ``valid_subset_200``. Returns records sorted by corpus id.

    STOPs if the control selection does not match the documented ids or the
    complement is not exactly 100 records (task boundary).
    """
    rng = random.Random(CONTROL_SEED)
    selected = sorted(
        rng.sample(valid_records, CONTROL_SELECT_N), key=lambda r: int(r["id"])
    )
    control_ids = [int(r["id"]) for r in selected]
    if control_ids != EXPECTED_CONTROL_IDS:
        raise SystemExit(
            "[STOP] Prose control ids do NOT match docs/SUBSTRATE_LEAKAGE_PROBE.md. "
            f"got {control_ids}"
        )
    control_set = set(control_ids)
    heldout = [r for r in valid_records if int(r["id"]) not in control_set]
    heldout = sorted(heldout, key=lambda r: int(r["id"]))
    if len(heldout) != NUM_SELECT:
        raise SystemExit(
            "[STOP] Complement split failed: heldout prose slice has "
            f"{len(heldout)} records, expected {NUM_SELECT}."
        )
    return heldout


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def run_generation(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    top_p: float,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run one seeded generation and return the continuation + Distinct-2.

    ``seed`` is applied immediately before ``generate()`` so the sampling RNG
    path is fully reproducible (verified by the determinism smoke).
    ``min_new_tokens`` is set to ``MAX_NEW_TOKENS`` so every record carries
    exactly ``MAX_NEW_TOKENS`` generated tokens (a sampled EOS token does not
    truncate the continuation). The generate() call mirrors the proven ds-023
    builder exactly (do_sample, temperature, top_p, repetition_penalty;
    no top_k override so the HF default applies).
    """
    seed_all(seed)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(enc["input_ids"].shape[-1])
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MAX_NEW_TOKENS,
            do_sample=GRID_SAMPLE,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=GRID_REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    seq = out[0]
    gen_ids = seq[prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    distinct_2, _repeated = compute_token_distinct_2_fast(
        seq.unsqueeze(0), prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
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
            seed, device,
        )
        b = run_generation(
            model, tokenizer, prompt, cell["temperature"], cell["top_p"],
            seed, device,
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
    model, tokenizer, heldout_prompts: List[Dict[str, Any]], device: torch.device
) -> List[Dict[str, Any]]:
    """Generate the full 100x4 grid and return 400 flat records.

    Each record carries the top-level fixture keys: id (assigned by curate),
    source_prose_id, mutated_prompt, params, text, distinct_2.
    """
    records: List[Dict[str, Any]] = []
    total = len(heldout_prompts) * len(GRID)
    n = 0
    for sample_idx, rec in enumerate(heldout_prompts):
        mutated_prompt = build_mutated_prompt(rec["text"])
        source_prose_id = int(rec["id"])
        for cell_idx, cell in enumerate(GRID):
            seed = _grid_seed(sample_idx, cell_idx)
            gen = run_generation(
                model, tokenizer, mutated_prompt, cell["temperature"],
                cell["top_p"], seed, device,
            )
            records.append(
                {
                    "id": None,  # assigned by curate()
                    "source_prose_id": source_prose_id,
                    "mutated_prompt": mutated_prompt,
                    "params": {
                        "do_sample": GRID_SAMPLE,
                        "temperature": cell["temperature"],
                        "top_p": cell["top_p"],
                        "repetition_penalty": GRID_REPETITION_PENALTY,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "min_new_tokens": MAX_NEW_TOKENS,
                        "seed": seed,
                        "sample_idx": sample_idx,
                        "cell_idx": cell_idx,
                    },
                    "text": gen["text"],
                    "distinct_2": gen["distinct_2"],
                }
            )
            n += 1
            if n % 50 == 0 or n == total:
                print(f"[ds-027] grid generation {n}/{total}", flush=True)
    return records


def curate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select the 100 lowest-distinct-2 grid records.

    Stable tie-break: sort by (distinct_2, sample_idx, cell_idx) and take the
    first 100; then renumber id 0..99 preserving that order. The tie-break is
    byte-identical across runs because the grid seeds are deterministic.
    """
    ordered = sorted(
        records,
        key=lambda r: (
            float(r["distinct_2"]),
            int(r["params"]["sample_idx"]),
            int(r["params"]["cell_idx"]),
        ),
    )
    selected = ordered[:NUM_SELECT]
    return selected


def prose_distinct_2(
    tokenizer, rec: Dict[str, Any], device: torch.device
) -> float:
    """Teacher-forced Distinct-2 of a prose text over the last-24 window."""
    enc = tokenizer(rec["text"], return_tensors="pt").to(device)
    d2, _repeated = compute_token_distinct_2_fast(
        enc["input_ids"], prompt_len=0, window_len=ANALYSIS_WINDOW
    )
    return float(d2)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def markdown_table(corpus: Dict[str, float], prose: Dict[str, float]) -> str:
    """Render the curated-vs-prose mean/median/p10 comparison table."""
    return (
        f"| Corpus | Mean | Median | p10 |\n"
        f"|--------|------|--------|-----|\n"
        f"| heldout_degenerate_v2 (curated) | {corpus['mean']:.4f} | "
        f"{corpus['median']:.4f} | {corpus['p10']:.4f} |\n"
        f"| heldout-100 prose (teacher-forced text) | {prose['mean']:.4f} | "
        f"{prose['median']:.4f} | {prose['p10']:.4f} |\n"
    )


def write_profile(
    ctx: Dict[str, Any], smoke: Dict[str, Any],
    curated: List[Dict[str, Any]], prose_summary: Dict[str, float],
    heldout_ids: List[int], out_path: Path,
) -> None:
    """Write docs/gate23/HELDOUT_V2_PROFILE.md (strictly factual)."""
    md: List[str] = []
    md.append("# DS-027 — Rebuilt Held-Out Hard-Degenerate Slice (Part A, v2)")
    md.append("")
    md.append("> Fixture curation — no gate thresholds, no governor changes, no")
    md.append("> green/red verdict. The gap below is a REPORTED measurement; if")
    md.append("> induction had failed, that would be a valid negative result.")
    md.append("> The slice-quality stop is applied per the gate-2.3b greenlight.")
    md.append("")

    md.append("## Run metadata")
    md.append("")
    meta = ctx["metadata"]
    md.append("| Field | Value |")
    md.append("|---|---|")
    for k, v in meta.items():
        md.append(f"| {k} | {v} |")
    md.append("")

    md.append("## Mutation spec (greenlight construction)")
    md.append("")
    md.append("For each of the 100 held-out prose records, ONE mutated prompt is")
    md.append("built from the prose text:")
    md.append("")
    md.append("```")
    md.append("first3 = first 3 whitespace tokens of the prose text")
    md.append("original_prompt = first3 (the unmutated 3-token prompt)")
    md.append("mutated_prompt = (first3 repeated 6 times) + \" \" + original_prompt")
    md.append("              = first3 repeated 7 times in total")
    md.append("```")
    md.append("")
    md.append(f"The mutated prompt is `{MUTATE_REPEAT + 1} x 3` = "
             f"`{(MUTATE_REPEAT + 1) * MUTATE_WHITESPACE_TOKENS}` whitespace tokens of "
             f"pure repetition with no prose continuation, so the model has no "
             f"coherent context to escape to.")
    md.append("")

    md.append("## Decoding grid (degeneracy-induction)")
    md.append("")
    md.append("All grid cells use `do_sample=True` and")
    md.append(f"`repetition_penalty={GRID_REPETITION_PENALTY}` (no anti-repetition ")
    md.append("pressure). This is the proven ds-023 grid.")
    md.append("")
    md.append("| # | temperature | top_p |")
    md.append("|---|------------|-------|")
    for cell_idx, cell in enumerate(GRID):
        md.append(f"| {cell_idx} | {cell['temperature']} | {cell['top_p']} |")
    md.append("")
    md.append(f"The grid yields {len(heldout_ids)} x {len(GRID)} = "
             f"{len(heldout_ids) * len(GRID)} records; the "
             f"{NUM_SELECT} with the LOWEST last-24-token Distinct-2 are "
             f"curated into the fixture.")
    md.append("")

    md.append("## Determinism smoke (scaffolding)")
    md.append("")
    md.append("Two grid records were regenerated with identical seeds and compared:")
    md.append("")
    md.append("| sample_idx | cell_idx | seed | Run A sha256 | Run B sha256 | Identical |")
    md.append("|---|---|---|---|---|---|")
    for p in smoke["pairs"]:
        md.append(f"| {p['sample_idx']} | {p['cell_idx']} | {p['seed']} | "
                  f"`{p['sha256_run_a']}` | `{p['sha256_run_b']}` | "
                  f"{'YES' if p['identical'] else 'NO'} |")
    md.append("")
    md.append(f"All identical: **{smoke['identical']}**")
    md.append("")

    md.append("## Distinct-2 (last 24 generated tokens)")
    md.append("")
    md.append("Percentile convention: p10 uses the nearest-rank method (value actually")
    md.append("present), matching the DS-023 builder; NumPy-linear percentile (used in")
    md.append("Gate 2.2/2.3 reports) may differ slightly.")
    md.append("")
    curated_summary = summarize([r["distinct_2"] for r in curated])
    md.append(markdown_table(curated_summary, prose_summary))
    gap = prose_summary["mean"] - curated_summary["mean"]
    md.append("")
    md.append(f"Curated corpus mean Distinct-2 is {gap:.4f} BELOW the heldout-100 "
             f"prose mean ({curated_summary['mean']:.4f} vs "
             f"{prose_summary['mean']:.4f}).")
    md.append("")

    md.append("## Slice-quality stop (gate-2.3b greenlight)")
    md.append("")
    md.append("Stop condition: curated mean Distinct-2 must be <= 0.5 * heldout-100")
    md.append("prose mean (ds-025 measured 0.9700).")
    md.append("")
    md.append(f"- heldout-100 prose mean (this run): {prose_summary['mean']:.4f}")
    md.append(f"- 0.5 * prose mean (this run): {0.5 * prose_summary['mean']:.4f}")
    md.append(f"- curated mean Distinct-2: {curated_summary['mean']:.4f}")
    md.append(f"- PASS: {curated_summary['mean']:.4f} <= "
             f"{0.5 * prose_summary['mean']:.4f} -> "
             f"{'YES' if curated_summary['mean'] <= 0.5 * prose_summary['mean'] else 'NO'}")
    md.append("")
    if curated_summary["mean"] > 0.5 * prose_summary["mean"]:
        md.append("**SLICE-QUALITY STOP TRIGGERED.** The curated slice does not meet")
        md.append("the greenlight quality bound; scoring is NOT performed.")
        md.append("")
    else:
        md.append("Slice-quality stop PASSED; scoring proceeds under the frozen")
        md.append("thresholds in `docs/gate23/FROZEN_THRESHOLDS.md`.")
        md.append("")

    md.append("## Held-out prose slice selection (seeded complement)")
    md.append("")
    md.append("100 of the 200 certified valid_subset_200 records were selected as")
    md.append("the Gate 2.2 prose control with "
             "`random.Random(42).sample(records, 100)` sorted by corpus `id`. "
             "The heldout-100 slice is the COMPLEMENT of that control within "
             "`valid_subset_200` (ids below).")
    md.append("")
    md.append(f"Heldout ids: `{heldout_ids}`")
    md.append("")

    md.append("## Corpus curation provenance")
    md.append("")
    md.append("Grid records were sorted by `(distinct_2, sample_idx, cell_idx)` ")
    md.append(f"and the first {NUM_SELECT} taken; stable tie-break keeps selection ")
    md.append("byte-identical across runs.")
    md.append("")

    md.append("## Fixture")
    md.append("")
    md.append("- JSONL: `tests/fixtures/heldout_degenerate_v2.jsonl`")
    md.append(f"- sha256: `{ctx['fixture_sha256']}`")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="DS-027 held-out degenerate fixture v2")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Cap on heldout prompts (for smoke/dev runs).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip the determinism smoke (dev only).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-027 Gate 2.3b held-out degenerate slice v2 (Part A)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {args.model}@{args.revision}")
    print(f"torch: {torch.__version__}")

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
    # Held-out slice construction
    # ------------------------------------------------------------------
    valid_records = load_jsonl(VALID_SUBSET_200_PATH)
    heldout = select_heldout_prose(valid_records)
    heldout_ids = [int(r["id"]) for r in heldout]
    print(f"valid_subset_200: {len(valid_records)} | heldout-100 prose: {len(heldout)}")
    print("heldout ids match complement of docs/SUBSTRATE_LEAKAGE_PROBE.md: TRUE")

    if args.max_prompts is not None:
        heldout = heldout[: args.max_prompts]
        print(f"[dev] capped heldout prompts at {args.max_prompts}")

    # ------------------------------------------------------------------
    # Determinism smoke FIRST
    # ------------------------------------------------------------------
    print("\n--- Determinism smoke ---")
    if args.skip_smoke:
        smoke = {"identical": "SKIPPED (dev)", "pairs": []}
        print("  skipped")
    else:
        prompts = [build_mutated_prompt(r["text"]) for r in heldout]
        smoke = run_determinism_smoke(model, tokenizer, prompts, device)
        for p in smoke["pairs"]:
            print(f"  sample_idx={p['sample_idx']} cell_idx={p['cell_idx']} "
                  f"seed={p['seed']} identical={p['identical']}")
        if not smoke["identical"]:
            print("[STOP] Determinism smoke FAILED: continuations differ across "
                  "identical-seed replays.")
            sys.exit(1)
        print(f"  determinism smoke: ALL IDENTICAL ({smoke['identical']})")

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------
    print("\n--- Grid generation ---")
    grid_records = build_grid(model, tokenizer, heldout, device)
    print(f"grid records: {len(grid_records)}")

    # ------------------------------------------------------------------
    # Curation
    # ------------------------------------------------------------------
    curated = curate(grid_records)
    curated_sample_idx = [int(r["params"]["sample_idx"]) for r in curated]
    n_distinct_prompts = len(set(curated_sample_idx))
    print(f"curated: {len(curated)} records from {n_distinct_prompts} distinct prompts")

    # ------------------------------------------------------------------
    # Heldout-100 prose Distinct-2 (teacher-forced, last-24 window)
    # ------------------------------------------------------------------
    print("\n--- Heldout-100 prose Distinct-2 (teacher-forced) ---")
    prose_d2 = [prose_distinct_2(tokenizer, r, device) for r in heldout]
    prose_summary = summarize(prose_d2)
    print(f"  prose distinct_2: mean={prose_summary['mean']:.4f} "
          f"median={prose_summary['median']:.4f} p10={prose_summary['p10']:.4f}")

    # ------------------------------------------------------------------
    # Slice-quality stop (greenlight). Write profile FIRST, then STOP if failed.
    # ------------------------------------------------------------------
    for idx, rec in enumerate(curated):
        rec["id"] = idx
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for rec in curated:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
    fixture_sha = sha256_of_file(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH} ({len(curated)} records)")

    metadata = {
        "Model": f"`{args.model}`",
        "Revision": f"`{args.revision}`",
        "Device": device,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "SEED": str(SEED),
        "grid seeds": "SEED*1000 + sample_idx*10 + cell_idx",
        "control selection seed": str(CONTROL_SEED),
        "max_new_tokens / min_new_tokens": f"{MAX_NEW_TOKENS} / {MAX_NEW_TOKENS}",
        "Mutation": f"first {MUTATE_WHITESPACE_TOKENS} whitespace tokens x{MUTATE_REPEAT} + original prompt (= first-3-token phrase x{MUTATE_REPEAT + 1})",
        "Runtime (wall clock)": f"{time.time() - t_start:.1f} s",
    }
    ctx = {
        "metadata": metadata,
        "fixture_sha256": fixture_sha,
    }
    write_profile(ctx, smoke, curated, prose_summary, heldout_ids, REPORT_PATH)
    print(f"wrote {REPORT_PATH}")

    curated_sum = summarize([r["distinct_2"] for r in curated])
    if curated_sum["mean"] > 0.5 * prose_summary["mean"]:
        print(f"\n[STOP] SLICE-QUALITY STOP: curated mean Distinct-2 "
              f"{curated_sum['mean']:.4f} > 0.5 * prose mean "
              f"{0.5 * prose_summary['mean']:.4f}. Scoring NOT performed.")
        sys.exit(2)

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("FIXTURE COMPLETE (no verdict offered)")


if __name__ == "__main__":
    main()
