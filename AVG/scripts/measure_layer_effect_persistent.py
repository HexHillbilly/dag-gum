#!/usr/bin/env python3
"""DS-028: Persistent-regime layer effect sweep (MEASUREMENT ONLY).

RFC-004 §3 requires the detection layer to be chosen by JOINT measurement:
detection quality AND downstream intervention effect at the SAME layer.
ds-027 (Gate 2.3b) scored detection per candidate layer {2, 12, 26} on the
heldout-degenerate v2 fixture; ds-026 measured single-fire effect and found
0.0000 Δ at every layer (single kicks do nothing). Probe B proved persistent
multi-step injection works. This probe closes the gap: per-layer effect under
the PRODUCTION persistent regime with the controller's actual force ramp.

MEASUREMENT ONLY. No threshold is created or modified beyond the ds-025
frozen PR values, no gate script is touched, no governor/controller.py edit,
no assert on outcomes, and no green/red verdict and no layer recommendation
are produced. Joint layer selection is the human's act per RFC-004 §3.

Method (per the DS-028 task file):
  1. Select 50 heldout-degenerate v2 records with random.Random(42).sample,
     sorted by record id. Prompt = record["mutated_prompt"].
  2. Per candidate layer L in {2, 12, 26}, per record, TWO greedy
     generations (do_sample=False, max_new_tokens=128):
     (a) dormant: no hooks;
     (b) hooked: a persistent forward hook on layer L that:
         1. maintains a ring buffer of the trailing 24 generated hidden states
            at layer L;
         2. when the buffer is >= 24, computes PR; if PR < T_PR(L) it
            increments below_ctr, else it resets below_ctr AND
            consecutive_interventions to 0;
         3. when below_ctr >= 2 (persistence hysteresis) it fires;
         4. strength = clamp(min(0.75, 0.45 + (consecutive-1)*0.05),
            [0.35, 0.75]); first fire in a run uses 0.45, the second
            consecutive uses 0.50, etc.;
         5. applies ONE bounded orthogonal-style perturbation at that
            strength;
         6. records fire_step, PR@fire, strength, below_ctr,
            consecutive_interventions;
         7. the hook stays alive for the entire generation (does NOT clear
            after firing).
  3. Measure per record per layer: continuation Distinct-2 (trailing 24
     generated tokens) and CTR (decoded continuation) for both conditions,
     Δ = hooked - dormant, and per-fire mean next-token KL(hooked || dormant)
     over the 8 generation steps following each fire.
  4. Determinism smoke FIRST: 1 record, dormant condition, twice — token ids
     must match exactly; else STOP.

Production force budget (ground truth, controller.py:120-146, 566):
  base_strength=0.45, max_strength=0.75, applied L2 clamp [0.35, 0.75].
  Persistence hysteresis below_ctr >= 2 (controller.py:497).
  This probe implements its OWN measurement hooks; it does not modify
  governor/controller.py and does not use the production diagnose() path.

Environment notes (identical to ds-025/ds-026):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; HF_HOME fallback is set when a writable cache
    is required.
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
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np
import torch
import torch.nn.functional as F

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
from AVG.core.metrics import (  # noqa: E402
    compute_coherent_token_ratio,
    participation_ratio,
)
from AVG.governor.controller import compute_token_distinct_2_fast  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-028 measurement seed (matches ds-025 / ds-026 / ds-027)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128
ANALYSIS_WINDOW = 24  # trailing generated positions for PR / distinct-2
CANDIDATE_LAYERS = ["2", "12", "26"]
NUM_RECORDS = 50

# Production force budget (controller.py:120-146, 566).
BASE_STRENGTH = 0.45
MAX_STRENGTH = 0.75
STRENGTH_RAMP = 0.05  # +0.05 per consecutive intervention
FORCE_CLAMP_LOW = 0.35
FORCE_CLAMP_HIGH = 0.75

# Persistence hysteresis (controller.py:497): below_ctr >= 2.
PERSISTENCE_THRESHOLD = 2

# KL measurement horizon: 8 generation steps following each fire (ds-026).
KL_STEPS = 8

# Paths
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/layer_effect_persistent_results.jsonl")
OUTPUT_MD = Path("docs/gate23/LAYER_EFFECT_PERSISTENT_RESULTS.md")

# HF cache fallback: if the system cache is not writable, use /tmp.
_DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
if not os.access(_DEFAULT_HF_CACHE, os.W_OK):
    _fallback = "/tmp/hf_cache"
    os.makedirs(_fallback, exist_ok=True)
    os.environ.setdefault("HF_HOME", _fallback)
    print(f"[env] HF_HOME fallback -> {_fallback}")


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
# Frozen thresholds (read from the ds-025 Part A freeze file; never re-derived)
# ---------------------------------------------------------------------------
def load_frozen_pr_thresholds(path: Path) -> Dict[str, float]:
    """Return {layer: PR_T} from the machine-parseable freeze block."""
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md has no ```json freeze block.")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md freeze block is unterminated.")
    freeze = json.loads(text[start:end])
    out: Dict[str, float] = {}
    for layer in CANDIDATE_LAYERS:
        if layer not in freeze["layers"]:
            raise SystemExit(f"[STOP] FROZEN_THRESHOLDS.md missing layer {layer}.")
        if "T" not in freeze["layers"][layer]["pr"]:
            raise SystemExit(f"[STOP] FROZEN_THRESHOLDS.md missing {layer}.pr.T.")
        out[layer] = float(freeze["layers"][layer]["pr"]["T"])
    return out


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
# Greedy generation with optional persistent measurement hook
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_persistent(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    layer: Optional[int] = None,
    pr_threshold: Optional[float] = None,
    record_id: int = -1,
    perturbation_seed: int = SEED,
    capture_logits: bool = True,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) generation from ``prompt``.

    When ``layer``/``pr_threshold`` are given, registers a PERSISTENT forward
    hook on ``model.model.layers[layer]``. The hook:
      1. maintains a ring buffer of the trailing 24 generated hidden states;
      2. when buffer >= 24, computes PR; PR < T -> below_ctr += 1; PR >= T ->
         resets below_ctr AND consecutive_interventions to 0;
      3. when below_ctr >= 2 (persistence hysteresis), fires;
      4. strength = clamp(min(0.75, 0.45 + (consecutive-1)*0.05), [0.35,0.75]);
      5. applies ONE bounded orthogonal-style perturbation at that strength;
      6. records fire_step, PR@fire, strength, below_ctr, consecutive;
      7. the hook stays alive for the entire generation (does NOT clear).

    Returns a dict with generated token ids, logits per step, and intervention
    state. Logits list index 0 is the prefill forward pass (predicts the first
    generated token); index t (t>=1) is the (t-1)-th decoding forward pass.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])

    generated: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    buffer: List[torch.Tensor] = []  # trailing generated hidden states at layer L
    counter = {"decoding_step": -1}  # incremented per decoding forward pass
    state: Dict[str, Any] = {
        "below_ctr": 0,
        "consecutive_interventions": 0,
        "total_count": 0,
        "interventions": [],
    }

    handle = None
    if layer is not None and pr_threshold is not None:
        def hook_fn(module, args, output):
            hs = output if isinstance(output, torch.Tensor) else output[0]
            # Decoding steps have seq_len == 1 (one new token position).
            if hs.shape[1] == 1:
                counter["decoding_step"] += 1
                buffer.append(hs[:, -1:, :].detach().to(dtype=torch.float32))
                if len(buffer) >= ANALYSIS_WINDOW:
                    win = torch.cat(buffer[-ANALYSIS_WINDOW:], dim=1)
                    pr = participation_ratio(win).item()
                    if pr < pr_threshold:
                        state["below_ctr"] += 1
                        if state["below_ctr"] >= PERSISTENCE_THRESHOLD:
                            # ---- Persistent fire: ONE bounded perturbation ----
                            state["consecutive_interventions"] += 1
                            # Production force ramp
                            # (controller.py:491-494): first fire uses the base
                            # strength, each consecutive fire adds +0.05, capped
                            # at max_strength, then clamped to [0.35, 0.75].
                            raw_strength = min(
                                MAX_STRENGTH,
                                BASE_STRENGTH
                                + (state["consecutive_interventions"] - 1)
                                * STRENGTH_RAMP,
                            )
                            clamped_force = max(
                                FORCE_CLAMP_LOW,
                                min(FORCE_CLAMP_HIGH, raw_strength),
                            )
                            original = hs.detach()
                            orig_dtype = original.dtype
                            res_flat = original[:, -1, :].float()
                            res_norm = res_flat.norm(dim=-1, keepdim=True) + 1e-8
                            # Deterministic noise seeded per (record, layer,
                            # decoding_step) — same scheme as ds-026.
                            fire_seed = (
                                perturbation_seed * 1_000_000
                                + max(record_id, 0) * 10_000
                                + (layer or 0) * 100
                                + int(counter["decoding_step"])
                            )
                            gen = torch.Generator(
                                device=res_flat.device
                            ).manual_seed(fire_seed)
                            raw_noise = torch.randn(
                                res_flat.shape, generator=gen, device=res_flat.device
                            )
                            # Production fallback orthogonalisation
                            # (controller.py _make_sae_guided_reset_fn, no SAE):
                            dot_prod = torch.sum(
                                raw_noise * res_flat, dim=-1, keepdim=True
                            )
                            proj = (dot_prod / (res_norm ** 2)) * res_flat
                            guided_direction = raw_noise - proj
                            dot_p = torch.sum(
                                guided_direction * res_flat, dim=-1, keepdim=True
                            )
                            ortho_vec = guided_direction - (
                                dot_p / (res_norm ** 2)
                            ) * res_flat
                            ortho_unit = ortho_vec / (
                                ortho_vec.norm(dim=-1, keepdim=True) + 1e-8
                            )
                            intervened = original.clone()
                            intervened[:, -1, :] = (
                                res_flat + clamped_force * ortho_unit
                            ).to(dtype=orig_dtype)
                            state["total_count"] += 1
                            state["interventions"].append({
                                "step": int(counter["decoding_step"]),
                                "pr_at_fire": float(pr),
                                "strength": float(clamped_force),
                                "below_ctr": int(state["below_ctr"]),
                                "consecutive_interventions": int(
                                    state["consecutive_interventions"]
                                ),
                            })
                            return intervened
                    else:
                        # PR recovered above the frozen threshold: the collapse
                        # run is over; reset the persistence counters.
                        state["below_ctr"] = 0
                        state["consecutive_interventions"] = 0
            return output

        handle = model.model.layers[int(layer)].register_forward_hook(hook_fn)

    try:
        # Prefill pass: processes the prompt, predicts the first generated token.
        out = model(input_ids)
        logits = out.logits[:, -1, :].float()
        past_key_values = out.past_key_values
        if capture_logits:
            logits_list.append(logits)
        next_tok = logits.argmax(dim=-1, keepdim=True)
        generated.append(next_tok)

        # Decoding loop: up to max_new_tokens, stopping at EOS (matches HF
        # generate's default; the v2 fixture's degenerate records do not hit
        # EOS before 128 tokens, so both conditions always produce 128).
        eos_id = tokenizer.eos_token_id
        for _step in range(max_new_tokens - 1):
            if int(next_tok.item()) == eos_id:
                break
            out = model(next_tok, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :].float()
            if capture_logits:
                logits_list.append(logits)
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
    finally:
        if handle is not None:
            handle.remove()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
        "logits": logits_list if capture_logits else [],
        "state": state,
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


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P || Q) between two next-token logit distributions (float32)."""
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    eps = 1e-12
    kl = (p * (p + eps).log() - p * (q + eps).log()).sum(dim=-1)
    # Small negative values are floating-point noise (KL is non-negative in
    # expectation); clamp for reporting cleanliness.
    return float(max(0.0, kl.mean().item()))


def per_fire_kl(
    fire: Dict[str, Any],
    hooked_logits: Sequence[torch.Tensor],
    dormant_logits: Sequence[torch.Tensor],
    kl_steps: int = KL_STEPS,
) -> Dict[str, Any]:
    """Mean next-token KL over the 8 steps following a fire.

    The hook fires during decoding step ``S`` (0-based); the logits from that
    forward pass sit at index ``S+1`` in the logits list (index 0 is the
    prefill). The 8 following steps are indices S+1 .. S+8. When fewer than 8
    steps remain, KL is averaged over the available steps.
    """
    S = fire.get("step")
    if S is None:
        return {"measured": False, "reason": "no_intervention"}
    start = S + 1
    end = min(start + kl_steps, len(hooked_logits))
    if start >= len(hooked_logits) or start >= len(dormant_logits):
        return {"measured": False, "reason": "insufficient_steps"}
    vals = []
    for t in range(start, end):
        vals.append(kl_divergence(hooked_logits[t], dormant_logits[t]))
    return {
        "measured": True,
        "n_steps": len(vals),
        "start_logits_index": start,
        "per_step": vals,
        "mean": float(statistics.mean(vals)),
    }


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module, tokenizer: Any, prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS, record_id: int = -1,
) -> Dict[str, Any]:
    """1 record, dormant condition, twice — token ids must match exactly."""
    print("\n--- Determinism smoke (1 record, dormant, twice) ---")
    a = greedy_generate_persistent(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
        capture_logits=False,
    )
    b = greedy_generate_persistent(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
        capture_logits=False,
    )
    ids_a = a["generated_ids"]
    ids_b = b["generated_ids"]
    identical = bool(torch.equal(ids_a, ids_b))
    print(f"  run A tokens: {ids_a.shape[-1]}, run B tokens: {ids_b.shape[-1]}")
    print(f"  identical: {identical}")
    if not identical:
        print("[STOP] Determinism smoke FAILED: token ids differ across replays.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL")
    return {
        "record_id": record_id,
        "n_tokens_a": int(ids_a.shape[-1]),
        "n_tokens_b": int(ids_b.shape[-1]),
        "identical": str(identical),
        "all_identical": "True",
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(values: Sequence[float]) -> Dict[str, float]:
    """mean / median / p10 / p90 / min / max over a distribution."""
    arr = [float(v) for v in values]
    if not arr:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
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


def fmt_sci(v: float, nd: int = 2) -> str:
    """Scientific notation for very small values (e.g. KL ~ 1e-5)."""
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{nd}e}"


def write_markdown_report(ctx: Dict[str, Any], pr_thr: Dict[str, float]) -> None:
    md: List[str] = []
    md.append("# DS-028 — Persistent-regime layer effect sweep (per-layer effect)")
    md.append("")
    md.append("> MEASUREMENT REPORT. This document reports measured values only.")
    md.append("> No threshold is created or modified beyond the ds-025 frozen PR")
    md.append("> values, no gate script is touched, no governor/controller.py edit,")
    md.append("> and no green/red verdict and no layer recommendation are offered.")
    md.append("> Joint detection x effect layer selection is the human's act per")
    md.append("> RFC-004 §3.")
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
    md.append("| Frozen PR thresholds | docs/gate23/FROZEN_THRESHOLDS.md (ds-025) |")
    md.append("| Persistence hysteresis | "
              f"below_ctr >= {meta['persistence_threshold']} "
              "(controller.py:497) |")
    md.append("| Force ramp | "
              f"base={meta['force_budget']['base_strength']}, "
              f"+{meta['force_budget']['strength_ramp']}/consecutive, "
              f"max={meta['force_budget']['max_strength']}, "
              f"clamp={meta['force_budget']['clamp']} (controller.py:120-146,566) |")
    md.append("| Perturbation | orthogonal-style, "
              "(controller.py fallback path, no SAE) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| KL horizon | {meta['kl_steps']} steps after each fire |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    md.append("## Determinism smoke")
    md.append("")
    md.append("One heldout-degenerate record, dormant condition (no hooks),")
    md.append("generated twice with the same seed; the generated token ids must")
    md.append("match exactly.")
    md.append("")
    sm = ctx["determinism_smoke"]
    md.append(format_table_md(
        [[str(sm["record_id"]), str(sm["n_tokens_a"]),
          str(sm["n_tokens_b"]), sm["identical"]],
         ["", "", "ALL", str(sm["all_identical"])]],
        ["record id", "run A tokens", "run B tokens", "identical"],
    ))
    md.append("")

    md.append("## Method summary")
    md.append("")
    md.append("- Per record, per candidate layer {2, 12, 26}: two greedy")
    md.append("  generations from the record's mutated_prompt.")
    md.append("- Dormant: no hooks. Hooked: a PERSISTENT forward hook on layer L")
    md.append("  computes PR over the trailing 24 generated positions at each")
    md.append("  decoding step; when PR < T_PR(L) (frozen) it increments")
    md.append("  below_ctr (else resets below_ctr AND consecutive_interventions")
    md.append("  to 0); when below_ctr >= 2 it fires, applying ONE bounded")
    md.append("  orthogonal-style perturbation at")
    md.append("  clamp(min(0.75, 0.45 + (consecutive-1)*0.05), [0.35, 0.75]).")
    md.append("  The hook stays alive for the entire generation (does NOT clear).")
    md.append("- Distinct-2 is measured on the trailing 24 generated tokens")
    md.append("  (ds-025 convention); CTR is the coherent-token ratio of the")
    md.append("  decoded continuation.")
    md.append("- Per-fire KL is the mean next-token KL(hooked || dormant) over")
    md.append("  the 8 generation steps following each fire (the fire step's")
    md.append("  forward pass and the next 7), same horizon as ds-026.")
    md.append("")

    md.append("## Frozen PR thresholds used (from ds-025, read-only)")
    md.append("")
    md.append("| layer | PR T |")
    md.append("|---|---|")
    for layer in CANDIDATE_LAYERS:
        md.append(f"| {layer} | {fmt(pr_thr[layer], 6)} |")
    md.append("")
    md.append("The hook fires when PR < T_PR(L) (the frozen PR primary threshold)")
    md.append("and below_ctr >= 2.")
    md.append("")

    by_layer = ctx["by_layer"]
    # ---- distinct_2 table ----
    md.append("## Per-layer effect on continuation distinct_2")
    md.append("")
    md.append("Distinct-2 over the trailing 24 generated tokens. Dormant is")
    md.append("layer-independent (no hooks) and is the same across layers for a")
    md.append("given record. delta = hooked - dormant.")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        s = by_layer[L]
        rows.append([
            L,
            str(s["n"]),
            fmt(s["d2_dormant"]["mean"]), fmt(s["d2_dormant"]["median"]),
            fmt(s["d2_dormant"]["p10"]), fmt(s["d2_dormant"]["p90"]),
            fmt(s["d2_hooked"]["mean"]), fmt(s["d2_hooked"]["median"]),
            fmt(s["d2_hooked"]["p10"]), fmt(s["d2_hooked"]["p90"]),
            fmt(s["d2_delta"]["mean"]), fmt(s["d2_delta"]["median"]),
            fmt(s["d2_delta"]["p10"]), fmt(s["d2_delta"]["p90"]),
            str(s["d2_improve_n"]), str(s["n_fired"]),
        ])
    md.append(format_table_md(
        rows, ["layer", "n", "dorm mean", "dorm med", "dorm p10", "dorm p90",
               "hook mean", "hook med", "hook p10", "hook p90",
               "Δ mean", "Δ med", "Δ p10", "Δ p90", "#Δ>0", "#fired≥1"],
    ))
    md.append("")
    md.append("- #Δ>0 = number of records where hooked distinct-2 > dormant")
    md.append("  distinct-2; #fired≥1 = number of records with at least one")
    md.append("  persistent-regime fire.")
    md.append("")

    # ---- CTR table ----
    md.append("## Per-layer effect on continuation CTR")
    md.append("")
    md.append("Coherent-token ratio of the decoded continuation. delta = hooked - dormant.")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        s = by_layer[L]
        rows.append([
            L,
            str(s["n"]),
            fmt(s["ctr_dormant"]["mean"]), fmt(s["ctr_dormant"]["median"]),
            fmt(s["ctr_dormant"]["p10"]), fmt(s["ctr_dormant"]["p90"]),
            fmt(s["ctr_hooked"]["mean"]), fmt(s["ctr_hooked"]["median"]),
            fmt(s["ctr_hooked"]["p10"]), fmt(s["ctr_hooked"]["p90"]),
            fmt(s["ctr_delta"]["mean"]), fmt(s["ctr_delta"]["median"]),
            fmt(s["ctr_delta"]["p10"]), fmt(s["ctr_delta"]["p90"]),
            str(s["ctr_improve_n"]), str(s["n_fired"]),
        ])
    md.append(format_table_md(
        rows, ["layer", "n", "dorm mean", "dorm med", "dorm p10", "dorm p90",
               "hook mean", "hook med", "hook p10", "hook p90",
               "Δ mean", "Δ med", "Δ p10", "Δ p90", "#Δ>0", "#fired≥1"],
    ))
    md.append("")

    # ---- intervention count distribution ----
    md.append("## Intervention count distribution (per layer, per record)")
    md.append("")
    md.append("Number of persistent-regime fires per record (0 if a record never")
    md.append("satisfied below_ctr >= 2 with PR below the frozen threshold).")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        s = by_layer[L]
        ic = s["intervention_count"]
        rows.append([
            L, str(ic["n"]), str(ic["total_fires"]), str(s["n_fired"]),
            fmt(ic["mean"]), fmt(ic["median"]), fmt(ic["p10"]), fmt(ic["p90"]),
            str(int(ic["min"])), str(int(ic["max"])),
        ])
    md.append(format_table_md(
        rows, ["layer", "n_records", "total_fires", "#records≥1 fire",
               "mean", "median", "p10", "p90", "min", "max"],
    ))
    md.append("")

    # ---- applied strength distribution ----
    md.append("## Applied strength distribution (per fire)")
    md.append("")
    md.append("Strength = clamp(min(0.75, 0.45 + (consecutive-1)*0.05),")
    md.append("[0.35, 0.75]) — the production force ramp. First fire in a run")
    md.append("uses 0.45; each consecutive fire adds +0.05 up to 0.75.")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        s = by_layer[L]
        st = s["strength"]
        if st["n"] == 0:
            rows.append([L, "0", "—", "—", "—", "—", "—", "—"])
        else:
            rows.append([
                L, str(st["n"]), fmt(st["mean"]), fmt(st["median"]),
                fmt(st["p10"]), fmt(st["p90"]), fmt(st["min"]), fmt(st["max"]),
            ])
    md.append(format_table_md(
        rows, ["layer", "n_fires", "mean", "median", "p10", "p90", "min", "max"],
    ))
    md.append("")

    # ---- fire step distribution ----
    md.append("## Fire step distribution (per fire)")
    md.append("")
    md.append("0-based decoding step at which each persistent-regime fire was")
    md.append("applied (the step's forward pass predicted the next generated")
    md.append("token).")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        s = by_layer[L]
        fs = s["fire_step"]
        if fs["n"] == 0:
            rows.append([L, "0", "—", "—", "—", "—", "—", "—"])
        else:
            rows.append([
                L, str(fs["n"]), fmt(fs["mean"]), fmt(fs["median"]),
                fmt(fs["p10"]), fmt(fs["p90"]), fmt(fs["min"]), fmt(fs["max"]),
            ])
    md.append(format_table_md(
        rows, ["layer", "n_fires", "mean", "median", "p10", "p90", "min", "max"],
    ))
    md.append("")

    # ---- PR@fire distribution ----
    md.append("## PR@fire distribution (per fire)")
    md.append("")
    md.append("Participation ratio over the trailing 24 generated hidden states")
    md.append("at the step that triggered the fire (always below the frozen")
    md.append("T_PR(L)).")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        s = by_layer[L]
        pf = s["pr_at_fire"]
        if pf["n"] == 0:
            rows.append([L, "0", "—", "—", "—", "—", "—", "—"])
        else:
            rows.append([
                L, str(pf["n"]), fmt(pf["mean"]), fmt(pf["median"]),
                fmt(pf["p10"]), fmt(pf["p90"]), fmt(pf["min"]), fmt(pf["max"]),
            ])
    md.append(format_table_md(
        rows, ["layer", "n_fires", "mean", "median", "p10", "p90", "min", "max"],
    ))
    md.append("")

    # ---- per-fire KL table ----
    md.append("## Per-fire mean next-token KL (hooked vs dormant, 8 steps after each fire)")
    md.append("")
    md.append("KL(hooked || dormant) over the 8 generation steps following each")
    md.append("fire (the fire step's forward pass plus the next 7). Per-fire KL")
    md.append("is averaged over those steps; the table aggregates per layer over")
    md.append("all fires where KL was measured.")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        s = by_layer[L]
        k = s["kl"]
        if k["n"] == 0:
            rows.append([L, "0", "NOT MEASURED", "—", "—", "—", "—", "—"])
        else:
            rows.append([
                L, str(k["n"]), "",
                fmt_sci(k["mean"]), fmt_sci(k["median"]),
                fmt_sci(k["p10"]), fmt_sci(k["p90"]),
                fmt_sci(k["min"]), fmt_sci(k["max"]),
            ])
    md.append(format_table_md(
        rows, ["layer", "n_fires+KL", "note", "mean", "median",
               "p10", "p90", "min", "max"],
    ))
    md.append("")
    md.append("KL is a measurement of how much the bounded perturbation changed")
    md.append("the model's next-token distribution; it is not a decision signal")
    md.append("and it does not by itself imply a layer choice.")
    md.append("")

    md.append("## Per-record JSONL")
    md.append("")
    md.append("Per-record per-layer results (dormant and hooked continuations,")
    md.append("intervention count, per-fire step/PR@fire/strength/below_ctr/")
    md.append("consecutive_interventions, per-fire KL) are in")
    md.append("`layer_effect_persistent_results.jsonl`.")
    md.append("")

    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY. No verdict, no layer recommendation.")
    md.append("- The hook is PERSISTENT: it stays alive for the entire")
    md.append("  generation and may fire multiple times when PR stays below the")
    md.append("  frozen T_PR(L) with below_ctr >= 2.")
    md.append("- The frozen T_PR(L) is the ds-025 PR primary threshold; no band")
    md.append("  or SVSE fusion is used in this effect measurement.")
    md.append("- The perturbation uses the production fallback orthogonalisation")
    md.append("  (random noise projected orthogonal to the residual, no SAE),")
    md.append("  same as ds-026 and the controller.py SAE-fallback path.")
    md.append("- Per-fire KL windows may overlap when fires are consecutive; each")
    md.append("  fire is measured independently over the same 8-step horizon.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-028 persistent-regime layer effect sweep (measurement only)"
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
    print("DS-028 Persistent-regime layer effect sweep (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"force budget: base={BASE_STRENGTH} +{STRENGTH_RAMP}/consec "
          f"max={MAX_STRENGTH} clamp=[{FORCE_CLAMP_LOW},{FORCE_CLAMP_HIGH}]")
    print(f"persistence hysteresis: below_ctr >= {PERSISTENCE_THRESHOLD}")
    print(f"KL horizon: {KL_STEPS} steps after each fire")

    # Frozen thresholds FIRST (ds-025 Part A freeze; read-only).
    pr_thr = load_frozen_pr_thresholds(FROZEN_THRESHOLDS_PATH)
    print("frozen PR thresholds loaded from docs/gate23/FROZEN_THRESHOLDS.md")
    for layer in CANDIDATE_LAYERS:
        print(f"  layer {layer}: PR T={pr_thr[layer]:.6f}")

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
    print(f"selected {len(records)} heldout-degenerate v2 records (seeded, sorted)")

    # Determinism smoke FIRST.
    smoke = None
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        smoke = run_determinism_smoke(
            model, tokenizer, records[0]["mutated_prompt"],
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
        print(f"  record {rid} ({rec_i+1}/{len(records)})")

        # Dormant condition is layer-independent (no hooks); run once per
        # record. Logits are kept only for this record (needed for KL vs the
        # three hooked runs), then freed.
        dorm = greedy_generate_persistent(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
            capture_logits=True,
        )
        dorm_meta = {
            "n_generated": dorm["n_generated"],
            "distinct_2": continuation_distinct2(dorm["generated_ids"]),
            "ctr": continuation_ctr(tokenizer, dorm["generated_ids"]),
            "generated_text": tokenizer.decode(
                dorm["generated_ids"][0], skip_special_tokens=True
            ),
            "generated_ids": dorm["generated_ids"][0].tolist(),
        }

        for layer_str in CANDIDATE_LAYERS:
            layer = int(layer_str)
            h = greedy_generate_persistent(
                model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
                layer=layer, pr_threshold=pr_thr[layer_str],
                record_id=rid, perturbation_seed=SEED, capture_logits=True,
            )
            st = h["state"]
            h_d2 = continuation_distinct2(h["generated_ids"])
            h_ctr = continuation_ctr(tokenizer, h["generated_ids"])

            interventions_out = []
            for fire in st["interventions"]:
                kl = per_fire_kl(fire, h["logits"], dorm["logits"])
                interventions_out.append({
                    "step": fire["step"],
                    "pr_at_fire": fire["pr_at_fire"],
                    "strength": fire["strength"],
                    "below_ctr": fire["below_ctr"],
                    "consecutive_interventions": fire["consecutive_interventions"],
                    "kl_8step_mean": kl.get("mean"),
                    "kl_per_step": kl.get("per_step", []),
                    "kl_not_measured": not kl.get("measured", False),
                })

            rec_out = {
                "record_id": rid,
                "source_prose_id": int(rec.get("source_prose_id", -1)),
                "layer": layer,
                "prompt": prompt,
                "prompt_len_tokens": int(h["prompt_len"]),
                "dormant": dict(dorm_meta),
                "hooked": {
                    "n_generated": h["n_generated"],
                    "distinct_2": h_d2,
                    "ctr": h_ctr,
                    "generated_text": tokenizer.decode(
                        h["generated_ids"][0], skip_special_tokens=True
                    ),
                    "generated_ids": h["generated_ids"][0].tolist(),
                    "intervention_count": st["total_count"],
                    "interventions": interventions_out,
                },
            }
            results.append(rec_out)

            if len(results) % 30 == 0 or rec_i == len(records) - 1:
                n_fired = sum(
                    1 for r in results if r["hooked"]["intervention_count"] > 0
                )
                total_fires = sum(
                    r["hooked"]["intervention_count"] for r in results
                )
                print(f"    ... {len(results)} (record x layer) pairs; "
                      f"{n_fired} fired≥1, {total_fires} total fires so far")

        # Free the record's dormant logits before the next record.
        del dorm

    # ------------------------------------------------------------------
    # Per-layer aggregation.
    # ------------------------------------------------------------------
    print("\n--- Aggregating per layer ---")
    by_layer: Dict[str, Dict[str, Any]] = {}
    for L in CANDIDATE_LAYERS:
        recs = [r for r in results if r["layer"] == int(L)]
        d2_dorm = [r["dormant"]["distinct_2"] for r in recs]
        d2_hook = [r["hooked"]["distinct_2"] for r in recs]
        ctr_dorm = [r["dormant"]["ctr"] for r in recs]
        ctr_hook = [r["hooked"]["ctr"] for r in recs]
        d2_delta = [h - d for d, h in zip(d2_dorm, d2_hook)]
        ctr_delta = [h - d for d, h in zip(ctr_dorm, ctr_hook)]
        fired = [r for r in recs if r["hooked"]["intervention_count"] > 0]
        fire_counts = [r["hooked"]["intervention_count"] for r in recs]
        all_fires = [
            i for r in recs for i in r["hooked"]["interventions"]
        ]
        strengths = [i["strength"] for i in all_fires]
        fire_steps = [i["step"] for i in all_fires]
        pr_at_fire = [i["pr_at_fire"] for i in all_fires]
        kl_means = [
            i["kl_8step_mean"] for i in all_fires
            if i.get("kl_8step_mean") is not None
        ]
        ic = summarize(fire_counts)
        ic["total_fires"] = int(sum(fire_counts))
        by_layer[L] = {
            "n": len(recs),
            "d2_dormant": summarize(d2_dorm),
            "d2_hooked": summarize(d2_hook),
            "d2_delta": summarize(d2_delta),
            "ctr_dormant": summarize(ctr_dorm),
            "ctr_hooked": summarize(ctr_hook),
            "ctr_delta": summarize(ctr_delta),
            "d2_improve_n": int(sum(1 for d in d2_delta if d > 0)),
            "ctr_improve_n": int(sum(1 for d in ctr_delta if d > 0)),
            "n_fired": len(fired),
            "intervention_count": ic,
            "strength": summarize(strengths),
            "fire_step": summarize(fire_steps),
            "pr_at_fire": summarize(pr_at_fire),
            "kl": summarize(kl_means),
        }
        s = by_layer[L]
        ic = s["intervention_count"]
        print(f"  layer {L}: n={s['n']} fired={s['n_fired']} "
              f"total_fires={int(ic['total_fires'])} "
              f"d2 dormant={s['d2_dormant']['mean']:.4f} "
              f"hooked={s['d2_hooked']['mean']:.4f} "
              f"delta={s['d2_delta']['mean']:.4f} | "
              f"ctr dormant={s['ctr_dormant']['mean']:.4f} "
              f"hooked={s['ctr_hooked']['mean']:.4f} "
              f"delta={s['ctr_delta']['mean']:.4f} | "
              f"strength n={s['strength']['n']} mean={s['strength']['mean']:.4f} | "
              f"KL n={s['kl']['n']} mean={s['kl']['mean']:.4f}")

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
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
        "persistence_threshold": PERSISTENCE_THRESHOLD,
        "force_budget": {
            "base_strength": BASE_STRENGTH,
            "max_strength": MAX_STRENGTH,
            "strength_ramp": STRENGTH_RAMP,
            "clamp": [FORCE_CLAMP_LOW, FORCE_CLAMP_HIGH],
        },
        "kl_steps": KL_STEPS,
        "max_new_tokens": args.max_new_tokens,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "by_layer": by_layer,
    }

    write_markdown_report(ctx, pr_thr)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("MEASUREMENT COMPLETE (no verdict, no layer recommendation offered)")


if __name__ == "__main__":
    main()
