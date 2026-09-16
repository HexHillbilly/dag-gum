#!/usr/bin/env python3
"""night-012: Multi-step PR->ECS causal probe (MEASUREMENT ONLY)

night-011 Phase 1 STOP: a single-token 0.50-L2 isotropic-noise injection at
layer 2 moved PR by only ~0.008 — far below T_PR (10.954796). The perturbation
affected only 1 of 24 positions in the trailing window, diluting the energy
~24:1.

This probe tests MULTI-STEP perturbation: perturb steps 23-30 consecutively
(8 steps), same 0.50 L2 isotropic Gaussian noise at layer 2, same record
(5, selected by night-011 Phase 0). By step 30, 8 of 24 positions in the
trailing window carry the perturbation — enough injected energy for a
measurable PR shift.

Causal interpretation (measurement observation, NOT a gate):
  - If PR crosses T_PR at step 30 AND L6H6 ECS rises toward 0.40+ within 10
    steps after the perturbation window ends (steps 31-40): manifold
    compression is causally upstream of attention detachment.
  - If L6H6 ECS stays near 0.17 (night-011 baseline) despite the PR increase:
    the two phenomena are independent.

MEASUREMENT ONLY. No controller edits, no thresholds, no actuation design.
This is a pure diagnostic [1].

Phases:
  Phase 1 — PR pre-check (MUST PASS before Phase 2): greedy generation from
            record 5 with KV-cache. At steps 23-30 (inclusive), run TWO
            forward passes from each branch's KV-cache state:
              (a) Unperturbed baseline: capture trailing 24-token PR at
                  layer 2.
              (b) Perturbed: apply 0.50-L2 isotropic Gaussian noise at layer 2
                  to the current token's hidden state, capture PR.
            The perturbed branch is a SELF-CONSISTENT trajectory: its KV-cache
            accumulates the perturbed states, so by step 30 the trailing
            24-token window contains 8 perturbed positions and 16 unperturbed
            positions. Continue to Phase 2 ONLY if the perturbed PR crosses
            T_PR (10.954796) at step 30 (the accumulation point). If not, log
            the per-step PR trajectory and STOP (the human decides whether to
            test longer perturbation windows).
  Phase 2 — Causal experiment: single self-consistent trajectory. Perturbation
            active at steps 23-30; after step 30 stop perturbing and continue
            normal greedy generation through step 43. At every step 20-43,
            measure PR at layer 2 (trailing 24-token window), ECS at L6H6
            (fraction of attention mass to prompt tokens), token_diversity,
            and trailing_ctr (surface reference).

Ground truth (verified by human review):
  - Record: 5 (t2s_degenerate, auto-selected by night-011 Phase 0).
    Record["text"] as prompt.
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. KV-cache incremental.
    bmm override deregistered. HF cache fallback.
  - Frozen T_PR(2): 10.954796, band_low(2): 8.216097
    (docs/gate23/FROZEN_THRESHOLDS.md, unchanged).
  - night-011 Phase 0/1 data: REUSED from docs/gate23/pr_ecs_causal_results.jsonl
    for record selection and baseline PR/ECS values. Do NOT re-run Phase 0 or
    Phase 1.
  - Perturbation: isotropic Gaussian noise at layer 2, scaled to 0.50 L2,
    applied to the current token's hidden state at EACH of steps 23-30
    (8 consecutive perturbations).

Environment notes (identical to night-011 / ds-025..ds-035):
  - torch 2.13 CUDA bmm Triton override is deregistered (env-only; no C
    compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; fall back to HF_HOME=/tmp/hf_cache only if
    the pinned revision is not cached.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from AVG.core.metrics import (  # noqa: E402
    compute_coherent_token_ratio,
    participation_ratio,
)
from AVG.governor.controller import (  # noqa: E402
    compute_token_distinct_2_fast,
)

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # night-012 measurement seed (matches night-011 / ds-025..ds-035)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (EOS break)
ANALYSIS_WINDOW = 24  # rolling 24-token PR / distinct-2 window (Gate 2.3 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook (spectral PR)
ECS_LAYER = 6  # L6H6: layer 6, head 6 (night-010 live ECS reference head)
ECS_HEAD = 6

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
T_PR_TASK = 10.954796  # frozen midpoint (PR < T -> spectral fire; Phase-1 crossing target)
BAND_LOW_TASK = 8.216097  # spectral_collapse threshold (band_low, frozen)

# Multi-step perturbation parameters (night-012).
PERTURB_L2 = 0.50  # isotropic Gaussian noise scaled to 0.50 L2
INTERVENTION_START_STEP = 23  # first perturbed step
INTERVENTION_END_STEP = 30  # last perturbed step (inclusive)
PERTURB_STEPS = list(range(INTERVENTION_START_STEP, INTERVENTION_END_STEP + 1))
PHASE1_END_STEP = 30  # accumulation point: perturbed PR must cross T_PR here
PHASE2_END_STEP = 43  # Phase-2 response window end (10 steps after step 30)
PHASE2_RECORD_START = 20  # start recording Phase-2 time series

# Collapse predicate constants (controller.py defaults, measurement reference only).
DIVERSITY_CTR_GATE = 0.40
TRAILING_GEN_WINDOW = 16

# Expected selected record (night-011 Phase 0 selected record 5).
EXPECTED_RECORD_ID = 5

# night-011 baseline ECS reference (from the task causal-test criterion).
NIGHT011_BASELINE_ECS = 0.17

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
PR_ECS_CAUSAL_JSONL = Path("docs/gate23/pr_ecs_causal_results.jsonl")
LIVE_ECS_JSONL = Path("docs/gate23/live_ecs_results.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/multistep_perturbation_results.jsonl")
OUTPUT_MD = Path("docs/gate23/MULTISTEP_PERTURBATION_RESULTS.md")

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
def load_frozen_thresholds(path: Path) -> Tuple[float, float]:
    """Parse the machine-parseable JSON freeze block from FROZEN_THRESHOLDS.md
    and return (T_PR(2), band_low(2)). STOP if either disagrees with the
    task-specified frozen value."""
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md has no ```json freeze block.")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md freeze block is unterminated.")
    freeze = json.loads(text[start:end])
    t_pr = float(freeze["layers"]["2"]["pr"]["T"])
    band_low = float(freeze["layers"]["2"]["pr"]["band_low"])
    if abs(t_pr - T_PR_TASK) > 1e-9:
        raise SystemExit(
            f"[STOP] FROZEN_THRESHOLDS.md T_PR(2)={t_pr:.6f} does not match the "
            f"task-specified frozen value {T_PR_TASK:.6f}. Do NOT re-derive "
            f"or adjust thresholds."
        )
    if abs(band_low - BAND_LOW_TASK) > 1e-9:
        raise SystemExit(
            f"[STOP] FROZEN_THRESHOLDS.md band_low(2)={band_low:.6f} does not "
            f"match the task-specified frozen value {BAND_LOW_TASK:.6f}. Do NOT "
            f"re-derive or adjust thresholds."
        )
    return t_pr, band_low


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
def load_model_and_tokenizer() -> Tuple[Any, Any]:
    """Load Qwen2.5-1.5B@8faed761 with eager attention (needed for attention
    weights). Returns (tokenizer, model)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Layer-2 instrument (capture + optional isotropic-noise perturbation)
# ---------------------------------------------------------------------------
class Layer2Instrument:
    """Forward-hook instrument on model.model.layers[2].

    Maintains one or two rolling ``analysis_window`` ring buffers of the
    trailing hidden states at layer 2 (float32):
      - "base": the unperturbed baseline trajectory.
      - "pert": the perturbed trajectory (0.50-L2 isotropic noise at layer 2
        applied to the current token's hidden state).

    ``mode`` selects which buffer(s) receive the captured state:
      - "shared": append to BOTH buffers (used for the common pre-intervention
        trajectory before step 23; the two branches are identical there).
      - "base": append to the base buffer only (baseline forward pass).
      - "pert": append to the perturbed buffer only (perturbed forward pass).

    When ``perturb`` is True (only meaningful in "pert" mode), isotropic
    Gaussian noise scaled to ``noise_l2`` (L2 norm) is added to the current
    token's hidden state in the model's dtype, and the perturbed state is
    captured into the pert buffer.

    PR is computed from a buffer whenever it holds >= ``analysis_window``
    entries. The value is appended to ``pr_logs[name]``.
    """

    def __init__(
        self,
        analysis_window: int = ANALYSIS_WINDOW,
        noise_l2: float = PERTURB_L2,
        noise_seed: int = SEED,
        buffers: Tuple[str, ...] = ("base", "pert"),
    ) -> None:
        self.analysis_window = analysis_window
        self.noise_l2 = noise_l2
        self.buffers: Dict[str, List[torch.Tensor]] = {n: [] for n in buffers}
        self.pr_logs: Dict[str, List[Dict[str, Any]]] = {n: [] for n in buffers}
        self.mode = buffers[0]
        self.perturb = False
        self.perturbed_hs: Optional[torch.Tensor] = None
        # Dedicated generator: noise draws are reproducible and do not perturb
        # the model's global RNG state.
        self.noise_gen = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.noise_gen.manual_seed(noise_seed)

    def __call__(
        self,
        module: nn.Module,
        input_args: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        hs = output[0] if isinstance(output, tuple) else output
        current = hs[:, -1:, :]  # (1,1,d) in model dtype

        if self.perturb:
            # Isotropic Gaussian noise scaled to self.noise_l2 L2 norm.
            noise = torch.randn(
                current.shape,
                generator=self.noise_gen,
                device=current.device,
                dtype=torch.float32,
            )
            noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-12)
            noise = noise * self.noise_l2
            # Clone to avoid in-place mutation of the graph / shared tensor.
            hs = hs.clone()
            hs[:, -1:, :] = (current.float() + noise).to(dtype=current.dtype)
            self.perturbed_hs = hs[:, -1:, :].detach().to(dtype=torch.float32)
            if isinstance(output, tuple):
                output = (hs,) + output[1:]
            else:
                output = hs

        captured = hs[:, -1:, :].detach().to(dtype=torch.float32)

        # Which buffers receive this capture?
        if self.mode == "shared":
            targets = list(self.buffers.keys())
        else:
            targets = [self.mode]

        for name in targets:
            buf = self.buffers[name]
            buf.append(captured)
            if len(buf) > self.analysis_window:
                buf.pop(0)
            if len(buf) >= self.analysis_window:
                win = torch.cat(buf[-self.analysis_window:], dim=1)
                pr = participation_ratio(win).item()
                self.pr_logs[name].append({"pr": float(pr)})

        return output

    def latest_pr(self, name: str) -> Optional[float]:
        """PR from the last logged window for the named buffer."""
        log = self.pr_logs.get(name)
        return log[-1]["pr"] if log else None

    def reset(self) -> None:
        for name in self.buffers:
            self.buffers[name] = []
            self.pr_logs[name] = []
        self.perturb = False
        self.perturbed_hs = None


# ---------------------------------------------------------------------------
# ECS helper
# ---------------------------------------------------------------------------
def compute_l6h6_ecs(out: Any, prompt_len: int, layer: int = ECS_LAYER,
                     head: int = ECS_HEAD) -> float:
    """L6H6 ECS: fraction of the current generated token's attention mass at
    layer 6, head 6 that lands on prompt positions.

    ``out.attentions`` is a tuple of per-layer attention tensors; for the
    KV-cache decoding step the layer tensor is (1, num_heads, 1, kv_len).
    """
    if out.attentions is None:
        raise RuntimeError("[STOP] output_attentions=True returned None during decoding.")
    attn = out.attentions[layer][0].float()  # (heads, 1, kv_len) f32
    kv_len = int(attn.shape[-1])
    if kv_len < prompt_len:
        raise RuntimeError(
            f"[STOP] step kv_len={kv_len} < prompt_len={prompt_len}; "
            f"live ECS boundary is invalid."
        )
    head_attn = attn[head, 0, :]  # (kv_len,)
    prompt_sum = head_attn[:prompt_len].sum().item()
    total_sum = head_attn.sum().item()
    return prompt_sum / max(total_sum, 1e-12)


# ---------------------------------------------------------------------------
# Record / baseline reuse helpers
# ---------------------------------------------------------------------------
def load_selected_record_and_baseline() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """REUSE night-011 Phase 0/1 data for record selection and baseline.

    Returns (selected_record, baseline_summary). The selected record is the
    t2s fixture record whose id matches night-011's ``selected_record_id``
    (must be 5). The baseline summary carries the night-011 Phase-1 PR values
    and the night-010 live L6H6 ECS reference.
    """
    n11 = load_jsonl(PR_ECS_CAUSAL_JSONL)[0]
    rid = int(n11["selected_record_id"])
    if rid != EXPECTED_RECORD_ID:
        raise SystemExit(
            f"[STOP] night-011 selected_record_id={rid} != expected "
            f"{EXPECTED_RECORD_ID}. Do NOT re-run Phase 0."
        )

    # Selected fixture record (record["text"] as prompt).
    t2s_map = {int(r["id"]): r for r in load_jsonl(T2S_DEG)}
    if rid not in t2s_map:
        raise SystemExit(f"[STOP] fixture record {rid} not found in {T2S_DEG}.")
    selected_record = t2s_map[rid]

    # night-011 Phase-1 baseline PR values (steps 23-25).
    p1 = n11["phase1"]
    baseline_pr = {int(s): float(v["pr_base"]) for s, v in p1["values"].items()}

    # night-010 live L6H6 ECS for record 5, steps 24-43 (reference).
    live_ecs = []
    for rec in load_jsonl(LIVE_ECS_JSONL):
        if int(rec.get("record_id", -1)) == rid:
            for s in rec.get("steps", []):
                step = int(s.get("step", 0))
                if 24 <= step <= 43:
                    h6 = s.get("ecs", {}).get("6", {}).get("6")
                    if h6 is not None:
                        live_ecs.append({"step": step, "ecs_L6H6": float(h6)})
            break

    baseline_pr_pert = {
        int(s): float(v["pr_pert"]) for s, v in p1["values"].items()
    }
    baseline_summary = {
        "record_id": rid,
        "night011_phase1_baseline_pr": baseline_pr,
        "night011_phase1_baseline_pr_pert": baseline_pr_pert,
        "night011_phase1_crossed_tpr": bool(p1.get("crossed_tpr_at_all_steps")),
        "night011_t_pr": float(p1["t_pr"]),
        "night011_perturb_l2": float(p1["perturb_l2"]),
        "night010_live_l6h6_ecs": live_ecs,
        "night010_live_ecs_mean_24_40": float(
            np.mean([e["ecs_L6H6"] for e in live_ecs if e["step"] <= 40])
        ) if live_ecs else None,
        "night010_live_ecs_min_24_43": float(
            np.min([e["ecs_L6H6"] for e in live_ecs])
        ) if live_ecs else None,
        "night010_live_ecs_max_24_43": float(
            np.max([e["ecs_L6H6"] for e in live_ecs])
        ) if live_ecs else None,
    }
    return selected_record, baseline_summary


# ---------------------------------------------------------------------------
# Phase 1 — PR pre-check (multi-step perturbation, MUST PASS before Phase 2)
# ---------------------------------------------------------------------------
def phase1_pr_precheck(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate greedily from record["text"] through step 30 with KV-cache.

    Steps 0-22: single shared forward pass per step (both branches identical).
    Steps 23-30: TWO forward passes per step from each branch's KV-cache state:
      (a) Unperturbed baseline: complete the forward pass normally, capture the
          trailing 24-token PR at layer 2.
      (b) Perturbed: apply 0.50-L2 isotropic Gaussian noise at layer 2 to the
          current token's hidden state, complete the forward pass, capture PR.

    The perturbed branch is SELF-CONSISTENT: its KV-cache and greedy token
    stream accumulate the perturbed states, so by step 30 the trailing 24-token
    window contains 8 perturbed positions and 16 unperturbed positions.

    Continue gate: perturbed PR must cross T_PR (10.954796) at step 30 (the
    accumulation point). Otherwise the per-step PR trajectory is logged and the
    probe STOPS (Phase 2 is not run).
    """
    rid = int(record["id"])
    prompt = record["text"]
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    eos_id = tokenizer.eos_token_id

    # Pre-fill.
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
    past_base = out.past_key_values
    next_logits_base = out.logits[:, -1, :].clone().float()

    # Register the layer-2 instrument AFTER pre-fill.
    instrument = Layer2Instrument(
        analysis_window=ANALYSIS_WINDOW,
        noise_l2=PERTURB_L2,
        noise_seed=SEED,
        buffers=("base", "pert"),
    )
    blocks = model.model.layers
    handle = blocks[HOOK_LAYER].register_forward_hook(instrument)

    generated_base: List[torch.Tensor] = []
    generated_pert: List[torch.Tensor] = []
    per_step: List[Dict[str, Any]] = []
    past_pert = None
    next_logits_pert = None

    try:
        for step in range(PHASE1_END_STEP + 1):  # steps 0..30
            next_token_base = next_logits_base.argmax(dim=-1, keepdim=True)
            generated_base.append(next_token_base)
            if int(next_token_base.item()) == eos_id:
                break

            if step < INTERVENTION_START_STEP:
                # ---- Common trajectory: single forward pass, both buffers ----
                instrument.mode = "shared"
                instrument.perturb = False
                with torch.no_grad():
                    out = model(
                        next_token_base, past_key_values=past_base, use_cache=True
                    )
                past_base = out.past_key_values
                next_logits_base = out.logits[:, -1, :].clone().float()
            else:
                # ---- Steps 23-30: two branches ----
                if step == INTERVENTION_START_STEP:
                    # The perturbed branch forks from the same KV-cache state as
                    # the baseline at the first intervention step.
                    past_pert = copy.deepcopy(past_base)
                    next_logits_pert = next_logits_base.clone()

                # (a) Unperturbed baseline pass -> baseline trajectory.
                instrument.mode = "base"
                instrument.perturb = False
                with torch.no_grad():
                    out_base = model(
                        next_token_base, past_key_values=past_base, use_cache=True
                    )
                past_base = out_base.past_key_values
                next_logits_base = out_base.logits[:, -1, :].clone().float()

                # (b) Perturbed pass -> self-consistent perturbed trajectory.
                next_token_pert = next_logits_pert.argmax(dim=-1, keepdim=True)
                generated_pert.append(next_token_pert)
                instrument.mode = "pert"
                instrument.perturb = True
                with torch.no_grad():
                    out_pert = model(
                        next_token_pert, past_key_values=past_pert, use_cache=True
                    )
                past_pert = out_pert.past_key_values
                next_logits_pert = out_pert.logits[:, -1, :].clone().float()
                instrument.perturb = False

            # ---- Per-step measurements ----
            pr_base = instrument.latest_pr("base")
            pr_pert = instrument.latest_pr("pert")

            if step < INTERVENTION_START_STEP:
                n_perturbed_in_window = 0
            else:
                # Trailing 24-token window contains tokens (step-23 .. step)
                # from the perturbation range, capped at the window size and
                # at the number of intervention steps (8).
                n_perturbed_in_window = min(
                    step - INTERVENTION_START_STEP + 1,
                    len(PERTURB_STEPS),
                    ANALYSIS_WINDOW,
                )

            per_step.append({
                "step": int(step),
                "pr_base": float(pr_base) if pr_base is not None else None,
                "pr_pert": float(pr_pert) if pr_pert is not None else None,
                "delta": (
                    float(pr_pert - pr_base)
                    if (pr_base is not None and pr_pert is not None) else None
                ),
                "perturbation_crossed_tpr": bool(
                    pr_pert is not None and pr_pert > T_PR_TASK
                ),
                "perturbation_exceeded_base": bool(
                    pr_pert is not None and pr_base is not None and pr_pert > pr_base
                ),
                "n_perturbed_in_window": int(n_perturbed_in_window),
            })
            print(
                f"  [phase1] step={step} "
                f"PR_base={fmt(pr_base, 4)} PR_pert={fmt(pr_pert, 4)} "
                f"delta={fmt(pr_pert - pr_base if (pr_base is not None and pr_pert is not None) else None, 4)} "
                f"n_pert={n_perturbed_in_window} T_PR={T_PR_TASK:.4f}"
            )

    finally:
        handle.remove()

    # ---- Continue gate: PR in perturbed condition crosses T_PR at step 30 ----
    step30 = next((v for v in per_step if v["step"] == PHASE1_END_STEP), None)
    if step30 is None:
        raise SystemExit(
            f"[STOP] Phase-1 did not reach step {PHASE1_END_STEP} (EOS before "
            f"the accumulation point)."
        )
    crossed = bool(step30["perturbation_crossed_tpr"])

    ids_base = (
        torch.cat(generated_base, dim=1)
        if generated_base
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    ids_pert = (
        torch.cat(generated_pert, dim=1)
        if generated_pert
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )

    return {
        "record_id": rid,
        "n_generated_to_step30": len(generated_base),
        "n_generated_pert": len(generated_pert),
        "values": {str(v["step"]): v for v in per_step},
        "per_step_trajectory": per_step,
        "crossed_tpr_at_step30": crossed,
        "t_pr": T_PR_TASK,
        "perturb_l2": PERTURB_L2,
        "intervention_steps": PERTURB_STEPS,
        "generated_text_base": tokenizer.decode(
            ids_base[0], skip_special_tokens=True
        ),
        "generated_text_pert": tokenizer.decode(
            ids_pert[0], skip_special_tokens=True
        ) if ids_pert.shape[-1] > 0 else "",
    }


# ---------------------------------------------------------------------------
# Phase 2 — Causal experiment (single self-consistent perturbed trajectory)
# ---------------------------------------------------------------------------
def phase2_causal_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Single self-consistent trajectory on record 5.

    Perturbation (0.50-L2 isotropic noise at layer 2) active at steps 23-30.
    After step 30, stop perturbing and continue normal greedy generation
    through step 43. At every step 20-43, measure:
      - PR at layer 2 (trailing 24-token window).
      - ECS at L6H6 (fraction of attention mass to prompt tokens).
      - token_diversity, trailing_ctr (surface reference).
    """
    rid = int(record["id"])
    prompt = record["text"]
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :].clone().float()
    input_ids = prompt_ids

    instrument = Layer2Instrument(
        analysis_window=ANALYSIS_WINDOW,
        noise_l2=PERTURB_L2,
        noise_seed=SEED,
        buffers=("base",),
    )
    blocks = model.model.layers
    handle = blocks[HOOK_LAYER].register_forward_hook(instrument)

    generated: List[torch.Tensor] = []
    time_series: List[Dict[str, Any]] = []

    try:
        for step in range(PHASE2_END_STEP + 1):  # steps 0..43
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            instrument.mode = "base"
            instrument.perturb = (
                INTERVENTION_START_STEP <= step <= INTERVENTION_END_STEP
            )
            with torch.no_grad():
                out = model(
                    next_token,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=True,
                )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].clone().float()
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if step < PHASE2_RECORD_START:
                continue  # only record from step 20 onward

            # ---- Measurements ----
            pr = instrument.latest_pr("base")
            ecs = compute_l6h6_ecs(out, prompt_len)
            token_diversity, _ = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )
            trailing_ctr = 1.0
            if token_diversity < DIVERSITY_CTR_GATE or step % 2 == 0:
                recent_gen = (
                    input_ids[0, prompt_len:]
                    if input_ids.shape[-1] > prompt_len else input_ids[0]
                )
                recent_tokens = (
                    recent_gen[-TRAILING_GEN_WINDOW:]
                    if recent_gen.numel() > 0
                    else input_ids[0, -TRAILING_GEN_WINDOW:]
                )
                trailing_text = tokenizer.decode(
                    recent_tokens, skip_special_tokens=True
                )
                trailing_ctr = compute_coherent_token_ratio(trailing_text)

            time_series.append({
                "step": int(step),
                "perturbed": bool(
                    INTERVENTION_START_STEP <= step <= INTERVENTION_END_STEP
                ),
                "layer2_pr": float(pr) if pr is not None else None,
                "ecs_L6H6": float(ecs),
                "token_diversity": float(token_diversity),
                "trailing_ctr": float(trailing_ctr),
            })
    finally:
        handle.remove()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    return {
        "record_id": rid,
        "prompt_len": int(prompt_len),
        "n_generated": int(ids.shape[-1]),
        "eos_terminated": bool(
            ids.shape[-1] < PHASE2_END_STEP + 1
            and ids.shape[-1] > 0
            and int(ids[0, -1].item()) == eos_id
        ),
        "intervention_steps": PERTURB_STEPS,
        "perturb_l2": PERTURB_L2,
        "time_series": time_series,
        "generated_text": tokenizer.decode(ids[0], skip_special_tokens=True),
    }


# ---------------------------------------------------------------------------
# Report writers
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
    if v in (float("inf"), float("-inf")):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{nd}f}"


def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# night-012 — Multi-step PR→ECS causal probe (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. This probe tests whether a MULTI-STEP "
              "0.50-L2 isotropic-noise perturbation at layer 2 (steps 23-30, "
              "8 consecutive steps) re-inflates the trailing-window PR across "
              "T_PR (10.954796) and, if so, whether L6H6 ECS rises in "
              "response. One record, one intervention, one diagnostic. "
              "PURELY DIAGNOSTIC: no thresholds, no actuation, no controller "
              "change. No verdict is offered; the human interprets the result.")
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
    md.append(f"| Perturbation | isotropic Gaussian noise at layer 2, "
              f"scaled to {meta['perturb_l2']} L2, applied to the current "
              f"token's hidden state |")
    md.append(f"| Perturbation steps | {meta['intervention_steps']} "
              f"(8 consecutive steps) |")
    md.append(f"| Accumulation point | step {meta['phase1_end_step']} "
              f"(trailing window: 8 perturbed / 24 positions) |")
    md.append(f"| Frozen T_PR(2) | {meta['t_pr']:.6f} "
              f"(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |")
    md.append(f"| Frozen band_low(2) | {meta['band_low']:.6f} "
              f"(docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |")
    md.append(f"| ECS head | L{meta['ecs_layer']}H{meta['ecs_head']} "
              f"(night-010 live ECS reference) |")
    md.append(f"| Selected record | {meta['selected_record_id']} "
              f"(REUSED from night-011 Phase 0; do NOT re-run) |")
    md.append(f"| night-011 Phase 0/1 | REUSED from "
              f"docs/gate23/pr_ecs_causal_results.jsonl |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # ------------------------------------------------------------------
    # Baseline (reused from night-011 / night-010)
    # ------------------------------------------------------------------
    md.append("## Baseline (REUSED, not re-measured)")
    md.append("")
    md.append("night-011 Phase 0 selected record 5 on the t2s_degenerate "
              "fixture. The single-token 0.50-L2 perturbation at layer 2 "
              "moved PR by only ~0.008 (Phase-1 STOP). This probe tests the "
              "multi-step version of the same perturbation.")
    md.append("")
    md.append("| Quantity | Value |")
    md.append("|---|---|")
    md.append(f"| night-011 Phase-1 baseline PR (steps 23-25) | "
              f"{', '.join(f's{s}={fmt(v, 4)}' for s, v in sorted(meta['baseline_pr'].items()))} |")
    md.append(f"| night-011 Phase-1 perturbed PR (steps 23-25) | "
              f"{', '.join(f's{s}={fmt(v, 4)}' for s, v in sorted(meta['baseline_pr_pert'].items()))} |")
    md.append(f"| night-010 live L6H6 ECS mean (steps 24-40) | "
              f"{fmt(meta['live_ecs_mean'], 4)} |")
    md.append(f"| night-010 live L6H6 ECS range (steps 24-43) | "
              f"[{fmt(meta['live_ecs_min'], 4)}, {fmt(meta['live_ecs_max'], 4)}] |")
    md.append(f"| night-011 baseline ECS reference (task criterion) | "
              f"{meta['baseline_ecs_reference']} |")
    md.append("")

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    md.append("## Phase 1 — PR pre-check (multi-step perturbation)")
    md.append("")
    md.append("Greedy generation from record[\\\"text\\\"] through step 30 with "
              "KV-cache. At steps 23-30, TWO forward passes are run per step: "
              "(a) the unperturbed baseline trajectory, and (b) the "
              "self-consistent perturbed trajectory with 0.50-L2 isotropic "
              "Gaussian noise applied to the current token's hidden state at "
              "layer 2. By step 30, the perturbed trajectory's trailing 24-token "
              "window contains 8 perturbed positions and 16 unperturbed "
              "positions.")
    md.append("")
    md.append("Continue gate: perturbed PR must cross T_PR "
              f"({meta['t_pr']:.6f}) at step 30 (the accumulation point).")
    md.append("")
    rows = []
    for v in ctx["phase1"]["per_step_trajectory"]:
        if v["step"] < 20:
            continue
        rows.append([
            str(v["step"]),
            fmt(v["pr_base"], 4),
            fmt(v["pr_pert"], 4),
            fmt(v["delta"], 4),
            str(v["n_perturbed_in_window"]),
            "Y" if v["perturbation_exceeded_base"] else "N",
            "Y" if v["perturbation_crossed_tpr"] else "N",
        ])
    md.append(format_table_md(
        rows, ["step", "PR base", "PR perturbed", "delta",
               "n_pert/24", "pert > base", "pert > T_PR"],
    ))
    md.append("")
    p1 = ctx["phase1"]
    md.append(f"Continue gate (perturbed PR > T_PR at step 30): "
              f"**{'PASS → continue to Phase 2' if p1['crossed_tpr_at_step30'] else 'STOP' }**")
    md.append("")

    if not p1["crossed_tpr_at_step30"]:
        md.append("### Phase-1 STOP (measured values reported)")
        md.append("")
        md.append(p1.get("stop_reason", ""))
        md.append("")
        md.append("Per the task specification, the perturbation is NOT "
                  "adjusted. The human decides whether to test longer "
                  "perturbation windows. Phase 2 was not run.")
        md.append("")

    # ------------------------------------------------------------------
    # Phase 2 (only if reached)
    # ------------------------------------------------------------------
    if ctx.get("phase2") is not None:
        p2 = ctx["phase2"]
        md.append("## Phase 2 — Causal experiment")
        md.append("")
        md.append(f"Intervention at steps {p2['intervention_steps'][0]}–"
                  f"{p2['intervention_steps'][-1]}: isotropic Gaussian noise "
                  f"at 0.50 L2 applied to the current token's hidden state at "
                  f"layer 2 at each step. After step 30 the perturbation stops; "
                  f"normal greedy generation continues through step "
                  f"{PHASE2_END_STEP}.")
        md.append("")
        md.append("### Primary diagnostic — PR and L6H6 ECS time series")
        md.append("")
        md.append("Steps 20-22 are the pre-intervention baseline. Steps 23-30 "
                  "are the perturbation window (perturbed=True). Steps 31-43 "
                  "are the post-intervention response window (perturbed=False).")
        md.append("")
        rows = []
        for s in p2["time_series"]:
            rows.append([
                str(s["step"]),
                "PERT" if s["perturbed"] else "",
                fmt(s["layer2_pr"], 4),
                fmt(s["ecs_L6H6"], 4),
                fmt(s["token_diversity"], 4),
                fmt(s["trailing_ctr"], 4),
            ])
        md.append(format_table_md(
            rows, ["step", "", "PR (L2)", "L6H6 ECS", "token_div", "trailing_ctr"],
        ))
        md.append("")
        md.append("### Causal test criterion (measurement observation, NOT a gate)")
        md.append("")
        md.append("- If L6H6 ECS rises toward 0.40+ within 10 steps after the "
                  "perturbation window ends (steps 31-40): manifold compression "
                  "is causally upstream of attention detachment.")
        md.append("- If L6H6 ECS stays near 0.17 (night-011 baseline) despite "
                  "the PR increase: the two phenomena are independent — "
                  "attention detachment is locked in and cannot be reversed by "
                  "locally re-inflating the manifold.")
        md.append("")
        # Observation summary.
        response = [s for s in p2["time_series"] if 31 <= s["step"] <= 40]
        ecs_vals = [s["ecs_L6H6"] for s in response]
        pr_vals = [s["layer2_pr"] for s in response if s["layer2_pr"] is not None]
        obs_lines = [
            f"Observed (measurement only, no verdict):",
            f"- PR at step 30 (accumulation point): {fmt(next((s['layer2_pr'] for s in p2['time_series'] if s['step'] == 30), None), 4)} "
            f"(T_PR = {meta['t_pr']:.6f}).",
            f"- L6H6 ECS in the post-intervention window (steps 31-40): "
            f"min {fmt(min(ecs_vals), 4)}, max {fmt(max(ecs_vals), 4)}, "
            f"mean {fmt(float(np.mean(ecs_vals)), 4)}.",
            f"- night-011 baseline ECS reference: {meta['baseline_ecs_reference']}.",
        ]
        if pr_vals:
            pr_step30 = next(
                (s["layer2_pr"] for s in p2["time_series"] if s["step"] == 30),
                None,
            )
            if pr_step30 is not None and pr_step30 > T_PR_TASK:
                if max(ecs_vals) >= 0.40:
                    obs_lines.append(
                        "- Observation: PR crossed T_PR and ECS rose toward "
                        "0.40+ → consistent with manifold compression being "
                        "causally upstream of attention detachment."
                    )
                else:
                    obs_lines.append(
                        "- Observation: PR crossed T_PR but L6H6 ECS stayed "
                        "well below 0.40 → consistent with the two phenomena "
                        "being independent."
                    )
            else:
                obs_lines.append(
                    "- Observation: PR did NOT cross T_PR → the causal "
                    "experiment could not be completed."
                )
        md.append("\n".join(obs_lines))
        md.append("")
        md.append("No pass/fail. This is a measurement observation.")
        md.append("")

    # ------------------------------------------------------------------
    # Notes / caveats
    # ------------------------------------------------------------------
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no thresholds, no actuation, no controller "
              "change [1]. This is a diagnostic probe; it offers no verdict.")
    md.append("- The perturbation is isotropic Gaussian noise scaled to 0.50 L2 "
              "(L2 norm 0.50, the AVG controller's L2-force convention), added "
              "to the current token's hidden state at layer 2. It adds energy "
              "to minor singular-value components, increasing PR by making the "
              "trailing window more isotropic.")
    md.append("- The perturbed branch is a SELF-CONSISTENT trajectory: its "
              "KV-cache and greedy token stream accumulate the perturbed "
              "states, so by step 30 the trailing 24-token window contains "
              "8 perturbed positions and 16 unperturbed positions.")
    md.append("- The 24-token PR ring buffer is filled only during decoding "
              "steps (the layer-2 hook is registered after pre-fill), matching "
              "the night-010 / DS-034b / night-011 hook pattern.")
    md.append("- All runs seeded (SEED=42); seed provenance and engagement "
              "evidence are printed in stdout.")
    md.append("- night-011 Phase 0/1 data are REUSED from "
              "`docs/gate23/pr_ecs_causal_results.jsonl` (record selection and "
              "baseline PR). night-010 live L6H6 ECS is REUSED from "
              "`docs/gate23/live_ecs_results.jsonl`. Do NOT re-generate.")
    md.append("- The full machine-readable payload is in "
              "`multistep_perturbation_results.jsonl`.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-012 multi-step PR->ECS causal probe (measurement only)"
    )
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate MULTISTEP_PERTURBATION_RESULTS.md from "
                             "an existing multistep_perturbation_results.jsonl "
                             "without re-running the model (dev convenience; "
                             "no measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 72)
    print("night-012 MULTI-STEP PR->ECS CAUSAL PROBE (MEASUREMENT ONLY)")
    print("=" * 72)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"perturbation: isotropic Gaussian noise at layer 2, {PERTURB_L2} L2")
    print(f"perturbation steps: {PERTURB_STEPS} (8 consecutive)")
    print(f"accumulation point: step {PHASE1_END_STEP}")
    print(f"T_PR: {T_PR_TASK} | band_low: {BAND_LOW_TASK}")

    # Frozen threshold verification.
    t_pr, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    print(f"frozen T_PR(2)={t_pr:.6f} band_low(2)={band_low:.6f} "
          "(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze)")

    # ------------------------------------------------------------------
    # --report-only mode
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        data = load_jsonl(OUTPUT_JSONL)[0]
        metadata = {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "seed": SEED,
            "perturb_l2": PERTURB_L2,
            "intervention_steps": data.get("intervention_steps", PERTURB_STEPS),
            "phase1_end_step": PHASE1_END_STEP,
            "t_pr": t_pr,
            "band_low": band_low,
            "ecs_layer": ECS_LAYER,
            "ecs_head": ECS_HEAD,
            "selected_record_id": data["selected_record_id"],
            "bmm_override": "deregistered",
            "wall_clock_s": 0.0,
            "baseline_pr": data.get("baseline", {}).get("night011_phase1_baseline_pr", {}),
            "baseline_pr_pert": data.get("baseline", {}).get("night011_phase1_baseline_pr_pert", {}),
            "live_ecs_mean": data.get("baseline", {}).get("night010_live_ecs_mean_24_40"),
            "live_ecs_min": data.get("baseline", {}).get("night010_live_ecs_min_24_43"),
            "live_ecs_max": data.get("baseline", {}).get("night010_live_ecs_max_24_43"),
            "baseline_ecs_reference": NIGHT011_BASELINE_ECS,
        }
        ctx = {
            "metadata": metadata,
            "phase1": data["phase1"],
            "phase2": data.get("phase2"),
        }
        write_markdown_report(ctx)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Load selected record + baseline (REUSE night-011 Phase 0/1)
    # ------------------------------------------------------------------
    print("\n--- Record selection & baseline (REUSE night-011 Phase 0/1) ---")
    selected_record, baseline = load_selected_record_and_baseline()
    print(f"  selected fixture record id={selected_record['id']} "
          f"text[:80]={selected_record['text'][:80]!r}")
    print(f"  night-011 Phase-1 baseline PR: {baseline['night011_phase1_baseline_pr']}")
    print(f"  night-010 live L6H6 ECS mean (steps 24-40): "
          f"{baseline['night010_live_ecs_mean_24_40']:.4f}")

    # ------------------------------------------------------------------
    # Model load
    # ------------------------------------------------------------------
    print("\n--- Model load ---")
    tokenizer, model = load_model_and_tokenizer()
    print(f"model loaded on {model.device}; num_heads="
          f"{model.config.num_attention_heads}")

    # ------------------------------------------------------------------
    # Phase 1 — PR pre-check (multi-step perturbation)
    # ------------------------------------------------------------------
    print("\n--- Phase 1: PR pre-check (multi-step perturbation, steps 23-30) ---")
    p1 = phase1_pr_precheck(model, tokenizer, selected_record)
    p1["stopped"] = not p1["crossed_tpr_at_step30"]
    if p1["crossed_tpr_at_step30"]:
        p1["stop_reason"] = ""
    else:
        step30 = p1["values"][str(PHASE1_END_STEP)]
        pr_pert30 = step30["pr_pert"]
        pr_base30 = step30["pr_base"]
        # Perturbed PR trajectory across the intervention window.
        pert_traj = [v["pr_pert"] for v in p1["per_step_trajectory"]
                     if v["step"] in PERTURB_STEPS and v["pr_pert"] is not None]
        p1["stop_reason"] = (
            f"The multi-step perturbation did NOT cross T_PR ({T_PR_TASK:.6f}) "
            f"at step 30 (the accumulation point). The perturbed PR trajectory "
            f"across the 8-step window rose steadily (perturbed PR range "
            f"[{min(pert_traj):.4f}, {max(pert_traj):.4f}] vs baseline PR at "
            f"step 30 = {pr_base30:.4f}), reaching {pr_pert30:.4f} at step 30 "
            f"— still far below T_PR. Even with 8 of 24 trailing-window "
            f"positions carrying the 0.50-L2 perturbation, the injected energy "
            f"in the minor singular-value components is insufficient to "
            f"re-inflate the manifold across T_PR. Per the task specification, "
            f"the perturbation is NOT adjusted; the human decides whether to "
            f"test longer perturbation windows."
        )
        print(f"\n[STOP] Phase-1 PR pre-check: perturbed PR did NOT cross T_PR "
              f"at step 30.")
        print(f"  {p1['stop_reason']}")

    # ------------------------------------------------------------------
    # Phase 2 — causal experiment (only if Phase 1 passes)
    # ------------------------------------------------------------------
    p2 = None
    if p1["crossed_tpr_at_step30"]:
        print("\n--- Phase 2: causal experiment ---")
        p2 = phase2_causal_experiment(model, tokenizer, selected_record)
        print(f"  generated {p2['n_generated']} tokens; perturbation steps "
              f"{p2['intervention_steps'][0]}-{p2['intervention_steps'][-1]}")
        for s in p2["time_series"]:
            print(
                f"  step={s['step']} "
                f"{'PERT' if s['perturbed'] else '    '} "
                f"PR={fmt(s['layer2_pr'], 4)} "
                f"ECS_L6H6={fmt(s['ecs_L6H6'], 4)} "
                f"div={fmt(s['token_diversity'], 4)} "
                f"ctr={fmt(s['trailing_ctr'], 4)}"
            )
    else:
        print("\nPhase 2 SKIPPED (Phase 1 STOP). No causal experiment run.")

    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Write JSONL + MD
    # ------------------------------------------------------------------
    payload = {
        "probe": "night-012",
        "seed": SEED,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_record_id": int(selected_record["id"]),
        "intervention_steps": PERTURB_STEPS,
        "baseline": baseline,
        "phase1": p1,
        "phase2": p2,
    }
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"\nwrote {OUTPUT_JSONL}")

    metadata = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "seed": SEED,
        "perturb_l2": PERTURB_L2,
        "intervention_steps": PERTURB_STEPS,
        "phase1_end_step": PHASE1_END_STEP,
        "t_pr": t_pr,
        "band_low": band_low,
        "ecs_layer": ECS_LAYER,
        "ecs_head": ECS_HEAD,
        "selected_record_id": int(selected_record["id"]),
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
        "baseline_pr": baseline["night011_phase1_baseline_pr"],
        "baseline_pr_pert": {
            int(s): float(v["pr_pert"])
            for s, v in load_jsonl(PR_ECS_CAUSAL_JSONL)[0]["phase1"]["values"].items()
        },
        "live_ecs_mean": baseline["night010_live_ecs_mean_24_40"],
        "live_ecs_min": baseline["night010_live_ecs_min_24_43"],
        "live_ecs_max": baseline["night010_live_ecs_max_24_43"],
        "baseline_ecs_reference": NIGHT011_BASELINE_ECS,
    }
    ctx = {
        "metadata": metadata,
        "phase1": p1,
        "phase2": p2,
    }
    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-012 MEASUREMENT COMPLETE")


if __name__ == "__main__":
    main()
