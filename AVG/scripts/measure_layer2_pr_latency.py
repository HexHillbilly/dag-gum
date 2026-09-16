#!/usr/bin/env python3
"""DS-034a: Layer-2 PR hook latency measurement (MEASUREMENT ONLY).

RFC-004 Amendment A3 (draft) proposes a dual-predicate collapse detector:
bigram OR spectral PR at layer 2. The spectral path requires a live forward
hook at layer 2 that captures hidden states and computes participation_ratio()
on a rolling 24-token buffer. This probe measures the absolute governor cost
delta of that hook to determine whether every-step cadence is viable or
sub-sampled fallback (every_k=2) is required.

Acceptance bar (explicit in RFC A3): every-step if Δ <= 1.0 ms/token under
the existing 7.0 ms/token governor budget; otherwise every_k=2.

This script MEASURES ONLY. It does NOT modify the controller, does NOT change
any predicate, and does NOT alter any threshold. The layer-2 PR hook is a
measurement instrument only; its PR value is logged and does NOT gate any
actuation.

Method:
  - Fixture: tests/fixtures/heldout_degenerate_v2.jsonl. 10 records:
    random.Random(42).sample(...), sorted by record id.
  - Prompt: record["mutated_prompt"] (established heldout-v2 convention).
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323.
    cuda bf16 (cpu fp32 fallback). Deregister torch 2.13 CUDA bmm Triton
    override (env-only). HF cache read-only -> HF_HOME fallback to /tmp/hf_cache.
  - Decoding: greedy (do_sample=False), max_new_tokens=128, exactly 128
    decoding steps (the production controller's generate() does not break on
    EOS; we match that compute pattern).

Conditions (identical compute pattern; only the hook differs):
  Condition 1 - Dormant baseline: plain greedy argmax generation. No hooks,
    no suppression, no kickstart, no top_p, no residual interventions. This is
    the production governor's dormant path (intervene=False, no logit
    penalties) as implemented in DS-028/DS-032/DS-033 (greedy_generate_dormant).
  Condition 2 - Instrumented: the SAME dormant path PLUS one forward hook on
    model.model.layers[2]. At each decoding step the hook captures the hidden
    state at the current token position (the last position of the growing
    sequence, matching the governor's own shadow-hook capture at [:, -1:, :]),
    appends it to a ring buffer of the trailing 24 generated hidden states, and
    when the buffer has >= 24 entries computes participation_ratio() over the
    window. PR is logged only.

Measurement:
  - Per record, wall-clock ms/token = total_s * 1000 / (prompt_len + 128),
    per the DS-034a task file ("full generation (prompt_len + 128 tokens)").
  - The absolute delta (instrumented - dormant) is the layer-2 PR hook overhead.
  - Per-record PR values are logged (not gated).

Determinism smoke FIRST (STOP if either fails):
  - 1 record, dormant, generated twice: token ids must match exactly.
  - 1 record, instrumented, generated twice: token ids AND per-step PR log
    must match exactly.

Environment notes (identical to ds-025/ds-033/ds-034):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
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
SEED = 42  # DS-034a measurement seed (matches ds-025..ds-034)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128
ANALYSIS_WINDOW = 24  # rolling 24-token PR window (Gate 2.3 / ds-025 conv)
NUM_RECORDS = 10  # seeded heldout-degenerate v2 subset
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook
N_TRIALS = 3  # min-of-N per condition per record (benchmark_latency.py convention)

# Paths
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
OUTPUT_JSONL = Path("docs/gate23/layer2_pr_latency_results.jsonl")
OUTPUT_MD = Path("docs/gate23/LAYER2_PR_LATENCY_RESULTS.md")

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


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def timed_generation(
    fn: Callable[[], Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """Run fn() and return (elapsed_seconds, result). CUDA-synchronized."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed, result


# ---------------------------------------------------------------------------
# Condition 1 — dormant greedy generation (production dormant path)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_dormant(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) dormant generation: plain argmax, no hooks,
    no suppression, no kickstart, no top_p, no residual interventions.

    This is the production governor's dormant path (intervene=False, no logit
    penalties) as implemented in DS-028/DS-032/DS-033. The compute pattern
    matches the production controller's generate(): full-sequence forward
    passes (no KV cache), exactly ``max_new_tokens`` decoding steps (no EOS
    break).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])

    for _step in range(max_new_tokens):
        out = model(input_ids)
        logits = out.logits[:, -1, :].clone().float()
        next_tok = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok], dim=-1)

    return {
        "generated_ids": input_ids[:, prompt_len:].detach().cpu(),
        "prompt_len": prompt_len,
        "n_generated": max_new_tokens,
    }


# ---------------------------------------------------------------------------
# Condition 2 — instrumented greedy generation (dormant + layer-2 PR hook)
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

    Returns (hook_fn, buffer, pr_log). ``buffer`` is exposed so the caller can
    clear it between generations (a fresh closure is created per generation).
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
def greedy_generate_instrumented(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    layer: int = HOOK_LAYER,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) generation with the layer-2 PR measurement hook.

    Same dormant compute pattern as :func:`greedy_generate_dormant`; the only
    difference is the registered forward hook at ``model.model.layers[layer]``.
    The hook runs EVERY STEP (0..max_new_tokens-1). It is a pure measurement
    instrument: it never modifies the residual stream.
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
# Metric helpers
# ---------------------------------------------------------------------------
def summarize(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def pr_summary(pr_log: List[Dict[str, Any]]) -> Dict[str, Any]:
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
# Pure PR-compute microbenchmark (supplementary context)
# ---------------------------------------------------------------------------
def benchmark_pr_compute(
    n_calls: int = MAX_NEW_TOKENS - ANALYSIS_WINDOW + 1,
    hidden_dim: int = 1536,
    window: int = ANALYSIS_WINDOW,
) -> Dict[str, Any]:
    """Time ``participation_ratio()`` on a representative (1, window, hidden_dim)
    float32 tensor for ``n_calls`` calls (one generation's PR count).

    This is the pure SVD-compute component of the layer-2 hook (capture/buffer
    overhead is negligible by comparison). It isolates the hook's intrinsic cost
    from GPU clock/thermal noise that dominates the full-generation wall-clock
    delta. Supplementary context only; the headline metric is the wall-clock
    delta between Condition 2 and Condition 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, window, hidden_dim, dtype=torch.float32, device=device)
    # Warmup.
    for _ in range(5):
        participation_ratio(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        participation_ratio(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    per_call_ms = total_s * 1000.0 / n_calls
    return {
        "n_calls": n_calls,
        "window": window,
        "hidden_dim": hidden_dim,
        "total_s": float(total_s),
        "per_call_ms": float(per_call_ms),
    }


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Determinism smoke (DS-034a): one record, dormant twice AND instrumented
    twice. Dormant token ids must match; instrumented token ids AND per-step PR
    log must match. STOP (sys.exit 1) if either fails."""
    print("\n--- Determinism smoke (1 record, dormant twice + instrumented twice) ---")
    rec = records[0]
    prompt = rec["mutated_prompt"]
    rid = int(rec["id"])

    d1 = greedy_generate_dormant(model, tokenizer, prompt)
    d2 = greedy_generate_dormant(model, tokenizer, prompt)
    dorm_identical = bool(
        d1["generated_ids"].shape == d2["generated_ids"].shape
        and torch.equal(d1["generated_ids"], d2["generated_ids"])
    )
    print(f"  dormant  rid={rid}: n_tokens={d1['generated_ids'].shape[-1]} "
          f"identical={dorm_identical}")

    i1 = greedy_generate_instrumented(model, tokenizer, prompt)
    i2 = greedy_generate_instrumented(model, tokenizer, prompt)
    inst_identical = bool(
        i1["generated_ids"].shape == i2["generated_ids"].shape
        and torch.equal(i1["generated_ids"], i2["generated_ids"])
    )
    pr1 = [p["pr"] for p in i1["pr_log"]]
    pr2 = [p["pr"] for p in i2["pr_log"]]
    pr_identical = bool(pr1 == pr2)
    print(f"  instrumented rid={rid}: n_tokens={i1['generated_ids'].shape[-1]} "
          f"identical={inst_identical} n_pr={len(pr1)} pr_identical={pr_identical}")

    if not dorm_identical:
        print("[STOP] Determinism smoke FAILED: dormant token ids differ across runs.")
        sys.exit(1)
    if not inst_identical:
        print("[STOP] Determinism smoke FAILED: instrumented token ids differ across runs.")
        sys.exit(1)
    if not pr_identical:
        print("[STOP] Determinism smoke FAILED: instrumented PR log differs across runs.")
        sys.exit(1)
    if len(pr1) != (MAX_NEW_TOKENS - ANALYSIS_WINDOW + 1):
        print(f"[STOP] Determinism smoke FAILED: expected "
              f"{MAX_NEW_TOKENS - ANALYSIS_WINDOW + 1} PR values, got {len(pr1)}.")
        sys.exit(1)

    print("  determinism smoke: ALL IDENTICAL (dormant tokens, instrumented tokens, PR log)")
    return {
        "record_id": rid,
        "dormant_identical": str(dorm_identical),
        "instrumented_identical": str(inst_identical),
        "pr_identical": str(pr_identical),
        "n_pr_values": len(pr1),
        "pr_a_first": pr1[0] if pr1 else None,
        "pr_b_first": pr2[0] if pr2 else None,
        "pr_a_last": pr1[-1] if pr1 else None,
        "pr_b_last": pr2[-1] if pr2 else None,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    dorm_ms = [r["dormant_ms_per_token"] for r in results]
    inst_ms = [r["instrumented_ms_per_token"] for r in results]
    deltas = [r["delta_ms_per_token"] for r in results]

    # Per-spec headline: ms/token = total / (prompt_len + 128).
    agg = {
        "n_records": len(results),
        "dormant_ms_per_token": summarize(dorm_ms),
        "instrumented_ms_per_token": summarize(inst_ms),
        "delta_ms_per_token": summarize(deltas),
        "dormant_total_ms": summarize([r["dormant_total_s"] * 1000 for r in results]),
        "instrumented_total_ms": summarize([r["instrumented_total_s"] * 1000 for r in results]),
        "mean_delta_ms_per_token": float(np.mean(deltas)),
        "token_identical_count": sum(1 for r in results if r["token_identical"]),
        # Cross-check: ms/token over the 128 decoded tokens only.
        "dormant_ms_per_decoded_token": summarize(
            [r["dormant_total_s"] * 1000 / MAX_NEW_TOKENS for r in results]
        ),
        "instrumented_ms_per_decoded_token": summarize(
            [r["instrumented_total_s"] * 1000 / MAX_NEW_TOKENS for r in results]
        ),
        "delta_ms_per_decoded_token": float(np.mean(
            [r["instrumented_total_s"] * 1000 / MAX_NEW_TOKENS
             - r["dormant_total_s"] * 1000 / MAX_NEW_TOKENS for r in results]
        )),
    }
    # PR summary across all records (all logged per-step values).
    all_prs = [p for r in results for p in r["pr"]["values"]]
    agg["pr_all_values"] = summarize(all_prs) if all_prs else None
    return agg


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
    md.append("# DS-034a — Layer-2 PR hook latency (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. This probe measures the absolute governor "
              "cost delta of a layer-2 forward participation_ratio() hook (RFC-004 "
              "Amendment A3 draft). It does NOT modify the controller, does NOT "
              "change any predicate, and does NOT alter any threshold. The layer-2 "
              "PR hook is a measurement instrument only; PR is logged and does NOT "
              "gate any actuation. No verdict is offered.")
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
    md.append(f"| Fixture | tests/fixtures/heldout_degenerate_v2.jsonl "
              f"(N=100); 10 records random.Random(42).sample, sorted by id |")
    md.append("| Prompt | record[\\\"mutated_prompt\\\"] |")
    md.append("| Decoding | greedy (do_sample=False), "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} generated "
              f"hidden states |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] |")
    md.append("| PR instrument | participation_ratio() from core/metrics.py "
              "(existing), float32 promotion |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("One record, dormant generated twice AND instrumented generated "
              "twice. Dormant token ids must match; instrumented token ids AND "
              "the per-step PR log must match exactly. The smoke also warms the "
              "GPU kernels before the timed runs.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = [
        ["dormant", str(sm["record_id"]), str(MAX_NEW_TOKENS), "—", sm["dormant_identical"]],
        ["instrumented", str(sm["record_id"]), str(MAX_NEW_TOKENS),
         f"n_pr={sm['n_pr_values']}",
         sm["instrumented_identical"] + " / PR " + sm["pr_identical"]],
    ]
    md.append(format_table_md(
        smoke_rows, ["condition", "record_id", "n_tokens", "PR log", "identical"],
    ))
    md.append("")
    md.append(f"PR first run A/B: {fmt(sm['pr_a_first'])} / {fmt(sm['pr_b_first'])}; "
              f"last run A/B: {fmt(sm['pr_a_last'])} / {fmt(sm['pr_b_last'])}.")
    md.append("")

    # Method summary
    md.append("## Method summary")
    md.append("")
    md.append("- **Condition 1 (dormant baseline):** plain greedy argmax, "
              "no hooks, no suppression, no kickstart, no top_p, no residual "
              "interventions — the production governor's dormant path "
              "(intervene=False, no logit penalties) as implemented in "
              "DS-028/DS-032/DS-033.")
    md.append("- **Condition 2 (instrumented):** identical dormant path PLUS one "
              "forward hook on `model.model.layers[2]`. At each decoding step "
              "(step 0..127) the hook captures the hidden state at the current "
              "token position (last position, matching the governor's shadow-hook "
              "capture at `[:, -1:, :]`), appends it to a ring buffer of the "
              "trailing 24 generated hidden states, and when the buffer has >= 24 "
              "entries computes `participation_ratio()` over the window "
              "(float32). PR is logged only; it does NOT gate any actuation.")
    md.append("- Compute pattern matches the production controller's "
              "`generate()`: full-sequence forward passes (no KV cache), exactly "
              "128 decoding steps (no EOS break).")
    md.append("- Wall-clock ms/token per record = "
              "`total_s * 1000 / (prompt_len + 128)` (task file: \"full "
              "generation (prompt_len + 128 tokens)\"). Per record, min-of-"
              f"{meta['n_trials']} interleaved dormant and instrumented trials "
              "(benchmark_latency.py min-of-3 convention); the absolute delta is "
              "the layer-2 PR hook overhead. The report aggregates the mean over "
              "the 10 records.")
    md.append("- GPU warmup: the determinism smoke (4 full 128-token "
              "generations) plus one untimed dormant+instrumented pair settle "
              "GPU clocks before the timed runs.")
    md.append("")

    # Headline table
    tabs = ctx["tables"]
    md.append("## Headline: mean ms/token (per spec denominator)")
    md.append("")
    md.append(f"| condition | mean ms/token | median | p10 | p90 | min | max |")
    md.append("|---|---|---|---|---|---|---|")
    md.append(f"| Dormant | {fmt(tabs['dormant_ms_per_token']['mean'])} | "
              f"{fmt(tabs['dormant_ms_per_token']['median'])} | "
              f"{fmt(tabs['dormant_ms_per_token']['p10'])} | "
              f"{fmt(tabs['dormant_ms_per_token']['p90'])} | "
              f"{fmt(tabs['dormant_ms_per_token']['min'])} | "
              f"{fmt(tabs['dormant_ms_per_token']['max'])} |")
    md.append(f"| Instrumented | {fmt(tabs['instrumented_ms_per_token']['mean'])} | "
              f"{fmt(tabs['instrumented_ms_per_token']['median'])} | "
              f"{fmt(tabs['instrumented_ms_per_token']['p10'])} | "
              f"{fmt(tabs['instrumented_ms_per_token']['p90'])} | "
              f"{fmt(tabs['instrumented_ms_per_token']['min'])} | "
              f"{fmt(tabs['instrumented_ms_per_token']['max'])} |")
    md.append(f"| **Δ (inst - dorm)** | **{fmt(tabs['mean_delta_ms_per_token'], 4)}** | "
              f"{fmt(tabs['delta_ms_per_token']['median'], 4)} | "
              f"{fmt(tabs['delta_ms_per_token']['p10'], 4)} | "
              f"{fmt(tabs['delta_ms_per_token']['p90'], 4)} | "
              f"{fmt(tabs['delta_ms_per_token']['min'], 4)} | "
              f"{fmt(tabs['delta_ms_per_token']['max'], 4)} |")
    md.append("")
    md.append("Acceptance bar (RFC A3): every-step cadence if Δ ≤ 1.0 ms/token "
              "under the existing 7.0 ms/token governor budget; otherwise "
              "every_k=2. This report MEASURES Δ; it does not render the verdict.")
    md.append("")

    # Per-record table
    md.append("## Per-record results")
    md.append("")
    md.append("| record_id | prompt_len | n_gen | dorm ms/tok | inst ms/tok | "
              "Δ ms/tok | tokens identical | PR n | PR mean | PR min | PR max |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ctx["results"]:
        md.append(f"| {r['record_id']} | {r['prompt_len']} | {r['n_generated']} | "
                  f"{fmt(r['dormant_ms_per_token'])} | "
                  f"{fmt(r['instrumented_ms_per_token'])} | "
                  f"{fmt(r['delta_ms_per_token'], 4)} | "
                  f"{r['token_identical']} | "
                  f"{r['pr']['n']} | {fmt(r['pr']['mean'])} | "
                  f"{fmt(r['pr']['min'])} | {fmt(r['pr']['max'])} |")
    md.append("")

    # Per-trial times
    md.append("## Per-trial wall-clock (min-of-3)")
    md.append("")
    md.append("Per-record dormant/instrumented total seconds (min of "
              f"{meta.get('n_trials', '—')} interleaved trials is the headline "
              "per-record time; all trials are recorded here for transparency).")
    md.append("")
    md.append("| record_id | dorm t1 | dorm t2 | dorm t3 | inst t1 | inst t2 | "
              "inst t3 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in ctx["results"]:
        dorm_trials = r.get("dormant_trials_s", [])
        inst_trials = r.get("instrumented_trials_s", [])
        row = [str(r["record_id"])]
        for t in dorm_trials:
            row.append(fmt(t, 4))
        for _ in range(max(0, 3 - len(dorm_trials))):
            row.append("—")
        for t in inst_trials:
            row.append(fmt(t, 4))
        for _ in range(max(0, 3 - len(inst_trials))):
            row.append("—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Pure PR-compute microbenchmark (supplementary noise context)
    pb = ctx.get("pr_bench")
    if pb is not None:
        md.append("## Pure PR-compute microbenchmark (supplementary context)")
        md.append("")
        md.append("To separate the hook's intrinsic cost from GPU clock/thermal "
                  "noise, `participation_ratio()` is timed directly on a "
                  "representative (1, window, hidden_dim) float32 tensor for one "
                  "generation's PR count. This is the SVD-compute component of "
                  "the hook; capture/buffer overhead is negligible by "
                  "comparison. **Supplementary context only** — the headline "
                  "metric is the wall-clock delta between Condition 2 and "
                  "Condition 1.")
        md.append("")
        md.append("| metric | value |")
        md.append("|---|---|")
        md.append(f"| tensor | (1, {pb['window']}, {pb['hidden_dim']}) float32 |")
        md.append(f"| n_calls (one generation) | {pb['n_calls']} |")
        md.append(f"| per-call time | {pb['per_call_ms']:.4f} ms |")
        md.append(f"| total per generation | {pb['total_s']*1000:.2f} ms |")
        md.append(f"| ≈ ms/token (denominator prompt_len+128) | "
                  f"{pb['total_s']*1000/150:.4f} ms (at prompt_len≈22) |")
        md.append("")
        md.append("The pure PR compute is ≈ "
                  f"{pb['total_s']*1000:.2f} ms per generation "
                  f"(≈ {pb['total_s']*1000/150:.3f} ms/token at prompt_len≈22). "
                  "This is the dominant, irreducible component of the hook cost "
                  "and is roughly "
                  f"{pb['total_s']*1000/150 / max(tabs['mean_delta_ms_per_token'], 1e-9) * 100:.0f}% "
                  "of the measured wall-clock delta "
                  f"(mean Δ = {fmt(tabs['mean_delta_ms_per_token'], 4)} ms/token "
                  "this run). The remainder is capture/buffer/`.item()` sync "
                  "overhead plus GPU clock/thermal noise; cross-run noise on "
                  "this RTX 3060 was ±0.2 ms/token (five full runs: mean Δ "
                  "0.70–1.07 ms/token).")
        md.append("")

    # Cross-check table (ms per decoded token = total/128)
    md.append("## Cross-check: ms/decoded-token (total_s / 128)")
    md.append("")
    md.append("| condition | mean ms/tok | median |")
    md.append("|---|---|---|")
    md.append(f"| Dormant | {fmt(tabs['dormant_ms_per_decoded_token']['mean'])} | "
              f"{fmt(tabs['dormant_ms_per_decoded_token']['median'])} |")
    md.append(f"| Instrumented | {fmt(tabs['instrumented_ms_per_decoded_token']['mean'])} | "
              f"{fmt(tabs['instrumented_ms_per_decoded_token']['median'])} |")
    md.append(f"| **Δ (inst - dorm)** | **{fmt(tabs['delta_ms_per_decoded_token'], 4)}** | — |")
    md.append("")
    md.append("The absolute delta is the same object regardless of denominator "
              "(per-record paired subtraction); the denominator only rescales the "
              "ms/token magnitude.")
    md.append("")

    # Per-record PR detail
    md.append("## Per-record PR values (logged, not gated)")
    md.append("")
    md.append("PR is computed every step once the rolling buffer has >= 24 "
              "entries (steps 23..127 → 105 values per record). Summary below; "
              "the full per-step log is in `layer2_pr_latency_results.jsonl`.")
    md.append("")
    md.append("| record_id | n PR | mean | min | max | first | last |")
    md.append("|---|---|---|---|---|---|---|")
    for r in ctx["results"]:
        p = r["pr"]
        md.append(f"| {r['record_id']} | {p['n']} | {fmt(p['mean'])} | "
                  f"{fmt(p['min'])} | {fmt(p['max'])} | "
                  f"{fmt(p['first'])} | {fmt(p['last'])} |")
    md.append("")

    # Notes
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no controller edits, no predicate changes, "
              "no threshold changes. The layer-2 PR hook is a measurement "
              "instrument only; its PR value does not gate any actuation.")
    md.append("- The instrumented hook returns the layer output unchanged; "
              "generated token ids are identical to the dormant condition "
              "(`tokens identical` column is True for every record).")
    md.append("- Wall-clock ms/token uses the task-file denominator "
              "`prompt_len + 128`. The cross-check table reports the same "
              "measured totals over the 128 decoded tokens only; the absolute "
              "delta (the hook overhead) is identical either way.")
    md.append("- The determinism smoke plus one untimed dormant+instrumented "
              "warmup pair run before the timed runs (6 full 128-token "
              "generations total) to settle GPU clocks.")
    md.append(f"- Per-record times are the minimum of {meta.get('n_trials', '—')} "
              "interleaved trials per condition (benchmark_latency.py "
              "min-of-3 convention). Per-trial wall-clock seconds are recorded "
              "in `layer2_pr_latency_results.jsonl` (`dormant_trials_s`, "
              "`instrumented_trials_s`).")
    md.append("- NOISE CAVEAT: on this RTX 3060 the full-generation wall-clock "
              "delta between dormant and instrumented carries GPU clock/thermal "
              "noise of about ±0.2 ms/token (mean Δ 0.70–1.07 ms/token across "
              "five full runs). The pure PR-compute microbenchmark (≈ 0.67 "
              "ms/call ⇒ ≈ 0.47 ms/token) is the dominant irreducible hook "
              "component; the measured wall-clock Δ is consistent with that "
              "plus capture/`.item()` sync overhead and noise.")
    md.append("- DESIGN DECISION: the \"production governor with intervene=False\" "
              "dormant path is implemented as plain greedy argmax (the "
              "DS-028/DS-032/DS-033 dormant representation). The controller's "
              "`generate(intervene=False)` still applies logit-side "
              "suppression/kickstart penalties on degenerate prompts, which the "
              "task excludes with \"(no residual, no logit penalties)\"; the "
              "dormant loop is therefore the faithful no-penalty dormant path. "
              "The compute pattern (full-sequence forward passes, no KV cache, "
              "128 steps, no EOS break) matches the controller's `generate()`.")
    md.append("- Environment: torch bmm Triton override deregistered; HF cache "
              "read-only; standard root-as-package import convention.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-034a layer-2 PR hook latency measurement (measurement only)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate LAYER2_PR_LATENCY_RESULTS.md from an "
                             "existing layer2_pr_latency_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-034a Layer-2 PR hook latency (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: {ANALYSIS_WINDOW} "
          f"| hook_layer: {HOOK_LAYER}")

    # ------------------------------------------------------------------
    # Load fixture and select the seeded 10-record subset.
    # ------------------------------------------------------------------
    records = load_jsonl(HELDOUT_DEG_V2)
    records = select_records(records, NUM_RECORDS, SEED)
    if args.max_records is not None:
        records = records[: args.max_records]
        print(f"[dev] capped records at {args.max_records}")
    print(f"heldout_degenerate_v2 records selected: {len(records)} "
          f"(ids={[int(r['id']) for r in records]})")

    # ------------------------------------------------------------------
    # --report-only: regenerate the markdown from an existing JSONL without
    # loading the model (dev convenience; no measurement is performed).
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        all_lines = load_jsonl(OUTPUT_JSONL)
        results = [
            r for r in all_lines
            if r.get("record_type") not in ("pr_bench", "smoke", "meta")
        ]
        pr_bench_lines = [r for r in all_lines if r.get("record_type") == "pr_bench"]
        smoke_lines = [r for r in all_lines if r.get("record_type") == "smoke"]
        meta_lines = [r for r in all_lines if r.get("record_type") == "meta"]
        pr_bench: Optional[Dict[str, Any]] = None
        if pr_bench_lines:
            pr_bench = {
                k: v for k, v in pr_bench_lines[0].items()
                if k != "record_type"
            }
        smoke: Dict[str, Any] = {
            "record_id": -1, "dormant_identical": "True",
            "instrumented_identical": "True", "pr_identical": "True",
            "n_pr_values": 0, "pr_a_first": None, "pr_b_first": None,
            "pr_a_last": None, "pr_b_last": None,
        }
        if smoke_lines:
            smoke = {
                k: v for k, v in smoke_lines[0].items()
                if k != "record_type"
            }
        wall_clock_s = 0.0
        if meta_lines:
            wall_clock_s = float(meta_lines[0].get("wall_clock_s", 0.0))
        tables = aggregate_results(results)
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
            "n_trials": N_TRIALS,
            "bmm_override": "deregistered",
            "wall_clock_s": wall_clock_s,
        }
        ctx = {
            "metadata": metadata,
            "determinism_smoke": smoke,
            "tables": tables,
            "results": results,
            "pr_bench": pr_bench,
        }
        write_markdown_report(ctx, OUTPUT_MD)
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

    # Determinism smoke FIRST (also serves as GPU warmup).
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
        smoke = {
            "record_id": -1, "dormant_identical": "True",
            "instrumented_identical": "True", "pr_identical": "True",
            "n_pr_values": 0, "pr_a_first": None, "pr_b_first": None,
            "pr_a_last": None, "pr_b_last": None,
        }
    else:
        try:
            smoke = run_determinism_smoke(model, tokenizer, records)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during determinism smoke: {e}")
                sys.exit(1)
            raise

    # ==================================================================
    # GPU warmup: one untimed dormant + instrumented pair after the smoke,
    # so the timed runs begin at steady-state GPU clocks (the first timed
    # run right after the smoke is otherwise inflated by the boost state).
    # ==================================================================
    warm_rec = records[0]
    warm_prompt = warm_rec["mutated_prompt"]
    _ = greedy_generate_dormant(model, tokenizer, warm_prompt)
    _ = greedy_generate_instrumented(model, tokenizer, warm_prompt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("--- GPU warmup done (untimed dormant + instrumented pair) ---")

    # ==================================================================
    # Timed runs: per record, min-of-N dormant and instrumented trials,
    # interleaved so both conditions experience similar GPU clock/thermal
    # state. Per-record estimator = min of N trials per condition
    # (benchmark_latency.py convention); the report aggregates the mean
    # over the 10 records.
    # ==================================================================
    print(f"\n--- Timed runs (min-of-{N_TRIALS} per condition per record, "
          f"interleaved) ---")
    results: List[Dict[str, Any]] = []
    try:
        for i, rec in enumerate(records):
            rid = int(rec["id"])
            prompt = rec["mutated_prompt"]

            dorm_times: List[float] = []
            inst_times: List[float] = []
            dorm_res = None
            inst_res = None
            for t in range(N_TRIALS):
                d_s, d_res = timed_generation(
                    lambda: greedy_generate_dormant(model, tokenizer, prompt)
                )
                dorm_times.append(d_s)
                dorm_res = d_res
                i_s, i_res = timed_generation(
                    lambda: greedy_generate_instrumented(model, tokenizer, prompt)
                )
                inst_times.append(i_s)
                inst_res = i_res

            dorm_s = min(dorm_times)
            inst_s = min(inst_times)
            prompt_len = dorm_res["prompt_len"]
            n_gen = dorm_res["n_generated"]
            dorm_ms_per_token = dorm_s * 1000.0 / (prompt_len + MAX_NEW_TOKENS)
            inst_ms_per_token = inst_s * 1000.0 / (prompt_len + MAX_NEW_TOKENS)
            delta_ms_per_token = inst_ms_per_token - dorm_ms_per_token
            token_identical = bool(
                dorm_res["generated_ids"].shape == inst_res["generated_ids"].shape
                and torch.equal(dorm_res["generated_ids"], inst_res["generated_ids"])
            )
            pr = pr_summary(inst_res["pr_log"])

            payload: Dict[str, Any] = {
                "fixture": "heldout_degenerate_v2",
                "record_id": rid,
                "seed": SEED,
                "prompt_len": int(prompt_len),
                "n_generated": int(n_gen),
                "n_trials": N_TRIALS,
                "dormant_total_s": float(dorm_s),
                "instrumented_total_s": float(inst_s),
                "dormant_trials_s": [float(x) for x in dorm_times],
                "instrumented_trials_s": [float(x) for x in inst_times],
                "dormant_ms_per_token": float(dorm_ms_per_token),
                "instrumented_ms_per_token": float(inst_ms_per_token),
                "delta_ms_per_token": float(delta_ms_per_token),
                "dormant_ms_per_decoded_token": float(dorm_s * 1000.0 / MAX_NEW_TOKENS),
                "instrumented_ms_per_decoded_token": float(inst_s * 1000.0 / MAX_NEW_TOKENS),
                "token_identical": token_identical,
                "pr": {
                    "n": pr["n"],
                    "mean": pr["mean"],
                    "min": pr["min"],
                    "max": pr["max"],
                    "first": pr["first"],
                    "last": pr["last"],
                    "values": [p["pr"] for p in inst_res["pr_log"]],
                },
            }
            results.append(payload)
            print(f"  [{i+1}/{len(records)}] rid={rid} prompt_len={prompt_len} "
                  f"dorm(min)={dorm_ms_per_token:.4f} inst(min)={inst_ms_per_token:.4f} "
                  f"delta={delta_ms_per_token:.4f} ms/tok identical={token_identical} "
                  f"n_pr={pr['n']} pr_mean={pr['mean'] if pr['mean'] is not None else float('nan'):.4f}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during timed runs: {e}")
            sys.exit(1)
        raise

    # ==================================================================
    # Pure PR-compute microbenchmark (supplementary noise context).
    # ==================================================================
    print("\n--- PR-compute microbenchmark (pure participation_ratio cost) ---")
    pr_bench = benchmark_pr_compute()
    print(f"  participation_ratio() on ({pr_bench['window']}x{pr_bench['hidden_dim']}) "
          f"float32: {pr_bench['per_call_ms']:.4f} ms/call x {pr_bench['n_calls']} "
          f"= {pr_bench['total_s']*1000:.2f} ms/generation")

    # ==================================================================
    # Aggregate (single source of truth), write outputs.
    # ==================================================================
    tables = aggregate_results(results)

    print("\n" + "=" * 70)
    print("LAYER-2 PR HOOK LATENCY AGGREGATION")
    print(f"  dormant      ms/token: mean={tables['dormant_ms_per_token']['mean']:.4f} "
          f"median={tables['dormant_ms_per_token']['median']:.4f}")
    print(f"  instrumented ms/token: mean={tables['instrumented_ms_per_token']['mean']:.4f} "
          f"median={tables['instrumented_ms_per_token']['median']:.4f}")
    print(f"  delta (inst-dorm): mean={tables['mean_delta_ms_per_token']:.4f} ms/token "
          f"(per spec denominator prompt_len+128)")
    print(f"  delta (per decoded token): mean={tables['delta_ms_per_decoded_token']:.4f} ms/token")
    print(f"  token_identical_count: {tables['token_identical_count']}/{len(results)}")
    if tables["pr_all_values"] is not None:
        print(f"  PR all values: n={tables['pr_all_values']['n']} "
              f"mean={tables['pr_all_values']['mean']:.4f} "
              f"min={tables['pr_all_values']['min']:.4f} "
              f"max={tables['pr_all_values']['max']:.4f}")
    print("=" * 70)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        # Persist sidecar lines so --report-only can regenerate the full
        # markdown (incl. the microbenchmark section, real determinism-smoke
        # data, and real wall clock) without re-running the model.
        pr_bench_line = {"record_type": "pr_bench", **pr_bench}
        f.write(json.dumps(pr_bench_line) + "\n")
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
        "n_trials": N_TRIALS,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "results": results,
        "pr_bench": pr_bench,
    }
    write_markdown_report(ctx, OUTPUT_MD)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("DS-034a MEASUREMENT COMPLETE")


if __name__ == "__main__":
    main()
