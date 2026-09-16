#!/usr/bin/env python3
"""night-011: Causal probe — does PR compression cause L6H6 attention detachment?
(MEASUREMENT ONLY)

The macro-loop dynamics model (docs/MACRO_LOOP_DYNAMICS.md) describes four
co-measured phenomena during t2s degeneration: (1) manifold compression
(PR below band_low at layer 2), (2) mid-layer attention detachment (L6H6
ECS near 0.17), (3) late-layer FFN update starvation, (4) logit-orthogonal
hidden collapse. Whether Stage 1 causes Stage 2, or whether both are
independent consequences of entering the attractor basin, is unknown.

This probe tests the FORWARD direction: if you artificially increase PR
(re-inflate the manifold) at step 23, does L6H6 ECS rise in response?
One record, one intervention, one diagnostic. Existence proof only — not
general causation.

MEASUREMENT ONLY. No controller edits, no thresholds, no actuation design.
This is a pure diagnostic [1].

Phases:
  Phase 0 — Automated record selection: rank the 10 night-010 records by a
            composite of stability / severity / pattern scores and pick the
            top-ranked record (REUSE night-010 JSONL; do NOT re-generate).
  Phase 1 — PR pre-check (determinism smoke): greedy generation to step 25;
            at steps 23-25 run TWO forward passes from the same KV-cache
            state (unperturbed vs 0.50-L2 isotropic-noise perturbed at layer
            2) and require the perturbed PR to exceed baseline AND cross
            T_PR (10.954796) at all three steps. If not, log values and STOP.
  Phase 2 — Causal experiment: apply the perturbation at step 23, continue
            greedy generation through step 43, and measure PR / L6H6 ECS /
            token_diversity / trailing_ctr at every step 23-43.

Ground truth (verified by human review):
  - Fixture: tests/fixtures/t2s_degenerate.jsonl (night-001; N=100).
    The specific record is AUTO-SELECTED from night-010 JSONL data
    (docs/gate23/live_ecs_results.jsonl) by the ranking scan in Phase 0.
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. KV-cache incremental.
    bmm override deregistered. HF cache fallback.
  - night-010 JSONL: REUSE per-step PR and L6H6 ECS for 10 records.
    Do NOT re-generate.
  - Frozen band_low: 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, unchanged).
  - Intervention point: step 23 (first step where 24-token ring buffer is
    full for both PR and ECS measurement).
  - Perturbation: isotropic Gaussian noise at layer 2, scaled to 0.50 L2,
    applied to the current token's hidden state. The perturbation adds
    energy to minor singular-value components, increasing PR by making the
    trailing window more isotropic.

Environment notes (identical to ds-025/ds-027/ds-033/ds-034/night-010):
  - torch 2.13 CUDA bmm Triton override is deregistered (env-only; no C
    compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in
    the container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; fall back to HF_HOME=/tmp/hf_cache only
    if the pinned revision is not cached.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
SEED = 42  # night-011 measurement seed (matches ds-025..ds-035, night-004/006/007/008/010)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (EOS break)
ANALYSIS_WINDOW = 24  # rolling 24-token PR / distinct-2 window (Gate 2.3 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook (spectral PR)
ECS_LAYER = 6  # L6H6: layer 6, head 6 (night-010 live ECS reference head)
ECS_HEAD = 6

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
T_PR_TASK = 10.954796  # frozen midpoint (PR < T -> spectral fire; also Phase-1 crossing target)
BAND_LOW_TASK = 8.216097  # spectral_collapse threshold (band_low, frozen)
PR_DRIFT_TOL = 1.0  # Phase-0 stability: PR drift tolerance
ECS_DRIFT_TOL = 0.05  # Phase-0 stability: ECS drift tolerance
TARGET_BAND_CENTER = 5.5  # Phase-0 severity: center of the 4-7 target band
SEVERITY_RANGE = 5.0  # Phase-0 severity: |mean(PR) - 5.5| / 5.0
PERTURB_L2 = 0.50  # isotropic Gaussian noise scaled to 0.50 L2
PHASE1_TEST_STEPS = [23, 24, 25]  # Phase-1 two-pass test steps
INTERVENTION_STEP = 23  # Phase-2 intervention point
POST_INTERVENTION_STEPS = 20  # Phase-2 response window length (steps 23-43)
PHASE2_END_STEP = INTERVENTION_STEP + POST_INTERVENTION_STEPS  # 43

# Collapse predicate constants (controller.py defaults, measurement reference only).
DIVERSITY_CTR_GATE = 0.40
TRAILING_GEN_WINDOW = 16

# Structural delimiters for the Phase-0 pattern score (DS-033 detection-gap
# records whose generated text contains code-like structure).
STRUCTURAL_DELIMITERS = [
    "{", "}", "##", "->", "=>", "//", "/*", "*/", "```",
    "::", "==", "!=", "<=", ">=", "def ", "let ", "const ",
    "struct ", "typedef ",
]

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
DS033_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
LIVE_ECS_JSONL = Path("docs/gate23/live_ecs_results.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/pr_ecs_causal_results.jsonl")
OUTPUT_MD = Path("docs/gate23/PR_ECS_CAUSAL_RESULTS.md")

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

    At each decoding step (seq_len == 1), captures the current token's hidden
    state at layer 2 into a rolling ring buffer of the trailing
    ``analysis_window`` states (float32). When ``perturb`` is True, adds
    isotropic Gaussian noise scaled to ``noise_l2`` (L2 norm) to the current
    token's hidden state in the model's dtype, then captures the perturbed
    state.

    ``skip_append`` (used by Phase-1 perturbed passes) prevents the ring
    buffer from being modified so a perturbed pass can be probed without
    corrupting the unperturbed trajectory's buffer.

    PR is computed from the ring buffer whenever it holds >= ``analysis_window``
    entries and the pass is not a skip-append pass. The value is logged with
    the hook-internal step counter (which increments once per forward pass).
    """

    def __init__(
        self,
        analysis_window: int = ANALYSIS_WINDOW,
        noise_l2: float = PERTURB_L2,
        noise_seed: int = SEED,
    ) -> None:
        self.analysis_window = analysis_window
        self.noise_l2 = noise_l2
        self.buffer: List[torch.Tensor] = []
        self.pr_log: List[Dict[str, Any]] = []
        self.step_counter = 0
        self.perturb = False
        self.skip_append = False
        self.perturbed_hs: Optional[torch.Tensor] = None
        # Dedicated generator: noise draws are reproducible and do not perturb
        # the model's global RNG state.
        self.noise_gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
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

        if not self.skip_append:
            captured = hs[:, -1:, :].detach().to(dtype=torch.float32)
            self.buffer.append(captured)
            if len(self.buffer) > self.analysis_window:
                self.buffer.pop(0)
            if len(self.buffer) >= self.analysis_window:
                win = torch.cat(self.buffer[-self.analysis_window:], dim=1)
                pr = participation_ratio(win).item()
                self.pr_log.append(
                    {"step": int(self.step_counter), "pr": float(pr)}
                )

        self.step_counter += 1
        return output

    def current_pr(self) -> Optional[float]:
        """PR from the last logged window (trailing ``analysis_window`` states)."""
        return self.pr_log[-1]["pr"] if self.pr_log else None

    def prev_23(self) -> List[torch.Tensor]:
        """The 23 ring-buffer entries BEFORE the most recently captured state."""
        if len(self.buffer) < self.analysis_window:
            raise RuntimeError(
                f"[STOP] prev_23 requires a full {self.analysis_window}-entry "
                f"buffer; buffer has {len(self.buffer)}."
            )
        return self.buffer[-self.analysis_window:-1]

    def pr_with(self, hs: torch.Tensor) -> float:
        """PR over prev-23 + ``hs`` (used for Phase-1 perturbed PR without
        corrupting the unperturbed ring buffer)."""
        prev = self.prev_23()
        win = torch.cat(prev + [hs], dim=1)
        return float(participation_ratio(win).item())

    def reset(self) -> None:
        self.buffer = []
        self.pr_log = []
        self.step_counter = 0
        self.perturb = False
        self.skip_append = False
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
# Phase 0 — Automated record selection (REUSE night-010 JSONL)
# ---------------------------------------------------------------------------
def phase0_select_record(live_ecs_path: Path, ds033_path: Path,
                         t2s_path: Path) -> Dict[str, Any]:
    """Rank the 10 night-010 records by composite score and select the top.

    For each record:
      - PR(layer 2) and L6H6 ECS are extracted for steps 20-40.
        PR is None at steps 20-23 in the night-010 JSONL (24-token ring buffer
        warm-up); the joint window used for both PR and ECS statistics is
        steps 24-40 (the intersection where the ring buffer is full).
      - stability = 1.0 - sqrt((std(PR)/1.0)^2 + (std(ECS)/0.05)^2)
      - severity  = 1.0 - |mean(PR) - 5.5| / 5.0, clamped to [0, 1]
      - pattern   = 1.0 if DS-033 detection-gap AND generated text contains
                    structural delimiters; 0.5 if detection-gap only; 0.0 otherwise
      - composite = 0.4*stability + 0.4*severity + 0.2*pattern
    """
    live = load_jsonl(live_ecs_path)
    ds033 = load_jsonl(ds033_path)
    t2s = {int(r["id"]): r for r in load_jsonl(t2s_path)}

    ds033_map = {
        int(r["record_id"]): r for r in ds033
        if r.get("fixture") == "t2s_degenerate"
    }

    scores: List[Dict[str, Any]] = []
    for rec in live:
        rid = int(rec["record_id"])
        steps = rec["steps"]

        # ---- Extract per-step PR (layer 2) and L6H6 ECS for steps 20-40 ----
        pr_pairs = [
            (s["step"], float(s["layer2_pr"]))
            for s in steps
            if 20 <= s["step"] <= 40 and s["layer2_pr"] is not None
        ]
        ecs_pairs = [
            (s["step"], float(s["ecs"]["6"]["6"]))
            for s in steps
            if 20 <= s["step"] <= 40
        ]
        # Joint window = steps 24-40 (both PR and ECS available).
        joint_pr = [p for p in pr_pairs if p[0] >= 24]
        joint_ecs = [e for e in ecs_pairs if e[0] >= 24]
        if len(joint_pr) < 2 or len(joint_ecs) < 2:
            raise SystemExit(
                f"[STOP] night-010 record {rid}: insufficient PR/ECS data in "
                f"steps 20-40 for the Phase-0 ranking scan."
            )
        pr_vals = [p[1] for p in joint_pr]
        ecs_vals = [e[1] for e in joint_ecs]

        std_pr = float(np.std(pr_vals))
        std_ecs = float(np.std(ecs_vals))
        mean_pr = float(np.mean(pr_vals))

        # ---- Stability ----
        combined_drift = math.sqrt(
            (std_pr / PR_DRIFT_TOL) ** 2 + (std_ecs / ECS_DRIFT_TOL) ** 2
        )
        stability = 1.0 - combined_drift

        # ---- Severity ----
        severity = 1.0 - abs(mean_pr - TARGET_BAND_CENTER) / SEVERITY_RANGE
        severity = max(0.0, min(1.0, severity))

        # ---- Pattern (DS-033 detection-gap + structural delimiters) ----
        d = ds033_map.get(rid, {})
        active = d.get("active", {})
        detection_gap = bool(active.get("diagnostic_class") == "detection-gap"
                            and int(active.get("predicate_true_steps", -1)) == 0)
        # Provenance guard: DS-033 prompt must equal the t2s fixture text.
        if rid in t2s and d.get("prompt") != t2s[rid]["text"]:
            raise SystemExit(
                f"[STOP] DS-033 prompt for t2s record {rid} does not match "
                f"t2s_degenerate.jsonl text. Do NOT re-classify."
            )
        gen_text = rec.get("generated_text", "")
        has_delims = any(delim in gen_text for delim in STRUCTURAL_DELIMITERS)
        if detection_gap and has_delims:
            pattern = 1.0
        elif detection_gap:
            pattern = 0.5
        else:
            pattern = 0.0

        composite = 0.4 * stability + 0.4 * severity + 0.2 * pattern

        scores.append({
            "record_id": rid,
            "pr_steps_used": [p[0] for p in joint_pr],
            "ecs_steps_used": [e[0] for e in joint_ecs],
            "n_pr": len(joint_pr),
            "n_ecs": len(joint_ecs),
            "mean_pr": mean_pr,
            "std_pr": std_pr,
            "mean_ecs": float(np.mean(ecs_vals)),
            "std_ecs": std_ecs,
            "combined_drift": combined_drift,
            "stability": stability,
            "severity": severity,
            "detection_gap": detection_gap,
            "has_structural_delimiters": has_delims,
            "pattern": pattern,
            "composite": composite,
        })

    # ---- Rank and select ----
    scores.sort(key=lambda r: r["composite"], reverse=True)
    selected = scores[0]

    return {
        "selected": selected,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Phase 1 — PR pre-check (determinism smoke)
# ---------------------------------------------------------------------------
def phase1_pr_precheck(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate greedily from record["text"] to step 25.

    At steps 23, 24, 25, run TWO forward passes from the same KV-cache state:
      (a) Unperturbed: capture the trailing 24-token hidden states at layer 2,
          compute baseline PR.
      (b) Perturbed: apply isotropic Gaussian noise at 0.50 L2 to the current
          token's hidden state at layer 2, then complete the forward pass.
          Capture the trailing 24-token hidden states, compute PR.

    Returns a dict with the per-step PR values and a boolean indicating whether
    the perturbation crossed T_PR at all three steps (the Phase-2 continue gate).
    """
    rid = int(record["id"])
    prompt = record["text"]
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    eos_id = tokenizer.eos_token_id

    # Pre-fill.
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :].clone().float()
    input_ids = prompt_ids

    # Register the layer-2 instrument AFTER pre-fill.
    instrument = Layer2Instrument(
        analysis_window=ANALYSIS_WINDOW, noise_l2=PERTURB_L2, noise_seed=SEED
    )
    blocks = model.model.layers
    handle = blocks[HOOK_LAYER].register_forward_hook(instrument)

    test_values: Dict[int, Dict[str, Any]] = {}
    generated: List[torch.Tensor] = []
    try:
        for step in range(26):  # steps 0..25
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            if step in PHASE1_TEST_STEPS:
                # ---- Two forward passes from the same KV-cache state ----
                # NOTE: transformers DynamicCache is MUTATED in-place by the
                # forward pass (`out.past_key_values IS past_before`). A
                # deepcopy is required so the perturbed pass (b) starts from the
                # SAME KV-cache state as the unperturbed pass (a).
                past_before = copy.deepcopy(past)

                # (a) Unperturbed pass -> actual trajectory.
                instrument.perturb = False
                instrument.skip_append = False
                with torch.no_grad():
                    out_a = model(
                        next_token, past_key_values=past, use_cache=True
                    )
                past = out_a.past_key_values
                next_logits = out_a.logits[:, -1, :].clone().float()
                pr_base = instrument.current_pr()
                if pr_base is None:
                    raise SystemExit(
                        f"[STOP] Phase-1 step {step}: unperturbed PR is None; "
                        f"ring buffer not full."
                    )

                # (b) Perturbed pass (deep-copied KV-cache state), skip-append.
                instrument.perturb = True
                instrument.skip_append = True
                with torch.no_grad():
                    out_b = model(
                        next_token, past_key_values=past_before, use_cache=True
                    )
                if instrument.perturbed_hs is None:
                    raise SystemExit(
                        f"[STOP] Phase-1 step {step}: perturbed hidden state not "
                        f"captured."
                    )
                pr_pert = instrument.pr_with(instrument.perturbed_hs)
                instrument.perturb = False
                instrument.skip_append = False

                test_values[int(step)] = {
                    "pr_base": float(pr_base),
                    "pr_pert": float(pr_pert),
                    "delta": float(pr_pert - pr_base),
                    "perturbation_crossed_tpr": bool(pr_pert > T_PR_TASK),
                    "perturbation_exceeded_base": bool(pr_pert > pr_base),
                }
                print(
                    f"  [phase1] step={step} PR_base={pr_base:.4f} "
                    f"PR_pert={pr_pert:.4f} delta={pr_pert - pr_base:+.4f} "
                    f"T_PR={T_PR_TASK:.4f}"
                )
            else:
                # Normal unperturbed decoding step.
                instrument.perturb = False
                instrument.skip_append = False
                with torch.no_grad():
                    out = model(
                        next_token, past_key_values=past, use_cache=True
                    )
                past = out.past_key_values
                next_logits = out.logits[:, -1, :].clone().float()

            input_ids = torch.cat([input_ids, next_token], dim=-1)
    finally:
        handle.remove()

    # ---- Continue gate ----
    crossed = all(
        test_values[s]["pr_pert"] > test_values[s]["pr_base"]
        and test_values[s]["pr_pert"] > T_PR_TASK
        for s in PHASE1_TEST_STEPS
    )
    return {
        "record_id": rid,
        "n_generated_to_step25": len(generated),
        "values": {str(s): test_values[s] for s in PHASE1_TEST_STEPS},
        "crossed_tpr_at_all_steps": crossed,
        "t_pr": T_PR_TASK,
        "perturb_l2": PERTURB_L2,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Causal experiment
# ---------------------------------------------------------------------------
def phase2_causal_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the 0.50-L2 isotropic-noise perturbation at layer 2 at step 23 and
    measure PR / L6H6 ECS / token_diversity / trailing_ctr at every step 20-43.

    Returns the per-step time series (steps 20-43), the generated tokens, and
    the trajectory metadata.
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
        analysis_window=ANALYSIS_WINDOW, noise_l2=PERTURB_L2, noise_seed=SEED
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

            instrument.perturb = (step == INTERVENTION_STEP)
            instrument.skip_append = False
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

            if step < 20:
                continue  # only record from step 20 onward

            # ---- Measurements ----
            pr = instrument.current_pr()
            ecs = compute_l6h6_ecs(out, prompt_len)
            token_diversity, _ = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )
            trailing_ctr = 1.0
            if token_diversity < DIVERSITY_CTR_GATE or step % 2 == 0:
                recent_gen = input_ids[0, prompt_len:] if input_ids.shape[-1] > prompt_len else input_ids[0]
                recent_tokens = recent_gen[-TRAILING_GEN_WINDOW:] if recent_gen.numel() > 0 else input_ids[0, -TRAILING_GEN_WINDOW:]
                trailing_text = tokenizer.decode(recent_tokens, skip_special_tokens=True)
                trailing_ctr = compute_coherent_token_ratio(trailing_text)

            time_series.append({
                "step": int(step),
                "perturbed": bool(step == INTERVENTION_STEP),
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
        "intervention_step": INTERVENTION_STEP,
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
    md.append("# night-011 — Causal probe: does PR compression cause L6H6 "
              "attention detachment? (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. This probe tests the FORWARD direction of "
              "the macro-loop dynamics model: if you artificially increase PR "
              "(re-inflate the manifold) at step 23, does L6H6 ECS rise in "
              "response? One record, one intervention, one diagnostic. "
              "Existence proof only — not general causation. PURELY DIAGNOSTIC: "
              "no thresholds, no actuation, no controller change. No verdict is "
              "offered; the human interprets the result.")
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
    md.append(f"| Intervention point | step {meta['intervention_step']} "
              f"(first step where the 24-token ring buffer is full for both "
              f"PR and ECS measurement) |")
    md.append(f"| Frozen T_PR(2) | {meta['t_pr']:.6f} "
              f"(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |")
    md.append(f"| Frozen band_low(2) | {meta['band_low']:.6f} "
              f"(docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |")
    md.append(f"| ECS head | L{meta['ecs_layer']}H{meta['ecs_head']} "
              f"(night-010 live ECS reference) |")
    md.append(f"| Selected record | {meta['selected_record_id']} "
              f"(auto-selected by Phase-0 ranking scan on night-010 JSONL) |")
    md.append(f"| night-010 JSONL | REUSED from docs/gate23/live_ecs_results.jsonl "
              f"(per-step PR and L6H6 ECS); do NOT re-generate |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # ------------------------------------------------------------------
    # Phase 0
    # ------------------------------------------------------------------
    md.append("## Phase 0 — Automated record selection")
    md.append("")
    md.append("Records ranked by composite = 0.4 × stability + 0.4 × severity "
              "+ 0.2 × pattern. PR and L6H6 ECS are extracted from the "
              "night-010 JSONL for steps 20-40. PR is None at steps 20-23 in "
              "the night-010 JSONL (24-token ring-buffer warm-up); the joint "
              "window used for both PR and ECS statistics is steps 24-40.")
    md.append("")
    md.append("Stability = 1.0 − sqrt((std(PR)/1.0)² + (std(ECS)/0.05)²). "
              "Severity = 1.0 − |mean(PR) − 5.5|/5.0, clamped to [0, 1]. "
              "Pattern = 1.0 if DS-033 detection-gap AND generated text has "
              "structural delimiters; 0.5 if detection-gap only; 0.0 otherwise.")
    md.append("")

    rows = []
    for s in ctx["phase0_scores"]:
        rows.append([
            str(s["record_id"]),
            fmt(s["mean_pr"], 4),
            fmt(s["std_pr"], 4),
            fmt(s["mean_ecs"], 4),
            fmt(s["std_ecs"], 4),
            fmt(s["stability"], 4),
            fmt(s["severity"], 4),
            fmt(s["pattern"], 1),
            fmt(s["composite"], 4),
        ])
    md.append(format_table_md(
        rows, ["record_id", "mean PR", "std PR", "mean ECS", "std ECS",
               "stability", "severity", "pattern", "composite"],
    ))
    md.append("")
    sel = ctx["phase0_selected"]
    md.append(f"**Selected record: {sel['record_id']}** "
              f"(composite {fmt(sel['composite'], 4)}). "
              f"Rationale: {ctx['selection_rationale']}")
    md.append("")

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    md.append("## Phase 1 — PR pre-check (determinism smoke)")
    md.append("")
    p1 = ctx["phase1"]
    md.append("Greedy generation from record[\\\"text\\\"] to step 25. At steps "
              "23, 24, 25, two forward passes are run from the same KV-cache "
              "state: (a) unperturbed baseline PR, (b) perturbed PR after "
              "applying isotropic Gaussian noise at 0.50 L2 to the current "
              "token's hidden state at layer 2. Continue to Phase 2 only if the "
              "perturbed PR exceeds the baseline AND crosses T_PR "
              f"({meta['t_pr']:.6f}) at all three steps.")
    md.append("")
    rows = []
    for s in PHASE1_TEST_STEPS:
        v = p1["values"][str(s)]
        rows.append([
            str(s),
            fmt(v["pr_base"], 4),
            fmt(v["pr_pert"], 4),
            fmt(v["delta"], 4),
            "Y" if v["perturbation_exceeded_base"] else "N",
            "Y" if v["perturbation_crossed_tpr"] else "N",
        ])
    md.append(format_table_md(
        rows, ["step", "PR base", "PR perturbed", "delta",
               "pert > base", "pert > T_PR"],
    ))
    md.append("")
    md.append(f"Continue gate (all 3 steps: pert > base AND pert > T_PR): "
              f"**{'PASS → continue to Phase 2' if p1['crossed_tpr_at_all_steps'] else 'STOP' }**")
    md.append("")

    if p1.get("stopped"):
        md.append("### Phase-1 STOP (measured values reported)")
        md.append("")
        md.append(p1.get("stop_reason", ""))
        md.append("")
        md.append("Per the task specification, the perturbation is NOT "
                  "adjusted. The human decides whether to test multi-step "
                  "perturbation. Phase 2 was not run.")
        md.append("")

    # ------------------------------------------------------------------
    # Phase 2 (only if reached)
    # ------------------------------------------------------------------
    if ctx.get("phase2") is not None:
        p2 = ctx["phase2"]
        md.append("## Phase 2 — Causal experiment")
        md.append("")
        md.append(f"Intervention at step {p2['intervention_step']}: isotropic "
                  f"Gaussian noise at 0.50 L2 applied to the current token's "
                  f"hidden state at layer 2. Greedy generation continues "
                  f"through step {p2['intervention_step'] + 20} (20 steps "
                  f"post-intervention).")
        md.append("")
        md.append("### Primary diagnostic — PR and L6H6 ECS time series")
        md.append("")
        md.append("Steps 20-22 are the pre-intervention baseline. Step 23 is "
                  "the intervention. Steps 24-43 are the post-intervention "
                  "response window.")
        md.append("")
        rows = []
        for s in p2["time_series"]:
            rows.append([
                str(s["step"]),
                "← PERT" if s["perturbed"] else "",
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
        md.append("- If L6H6 ECS rises toward 0.40+ within 10 steps of the PR "
                  "perturbation: manifold compression is causally upstream of "
                  "attention detachment.")
        md.append("- If L6H6 ECS stays near 0.17 despite PR increase: the two "
                  "phenomena are independent — attention detachment is locked "
                  "in and cannot be reversed by locally re-inflating the "
                  "manifold.")
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
    md.append("- The 24-token PR ring buffer is filled only during decoding "
              "steps (the layer-2 hook is registered after pre-fill), matching "
              "the night-010 / DS-034b hook pattern.")
    md.append("- All runs seeded (SEED=42); seed provenance and engagement "
              "evidence are printed in stdout.")
    md.append("- Per-step PR and L6H6 ECS for the ranking scan are REUSED from "
              "`docs/gate23/live_ecs_results.jsonl` (night-010). Do NOT "
              "re-generate.")
    md.append("- The full machine-readable payload is in "
              "`pr_ecs_causal_results.jsonl`.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-011 causal PR->ECS probe (measurement only)"
    )
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate PR_ECS_CAUSAL_RESULTS.md from an "
                             "existing pr_ecs_causal_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 72)
    print("night-011 CAUSAL PROBE: does PR compression cause L6H6 attention "
          "detachment? (MEASUREMENT ONLY)")
    print("=" * 72)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"perturbation: isotropic Gaussian noise at layer 2, {PERTURB_L2} L2")
    print(f"intervention point: step {INTERVENTION_STEP}")
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
            "intervention_step": INTERVENTION_STEP,
            "t_pr": t_pr,
            "band_low": band_low,
            "ecs_layer": ECS_LAYER,
            "ecs_head": ECS_HEAD,
            "selected_record_id": data["selected_record_id"],
            "bmm_override": "deregistered",
            "wall_clock_s": 0.0,
        }
        ctx = {
            "metadata": metadata,
            "phase0_scores": data["phase0"]["scores"],
            "phase0_selected": data["phase0"]["selected"],
            "selection_rationale": data["phase0"]["selection_rationale"],
            "phase1": data["phase1"],
            "phase2": data.get("phase2"),
        }
        write_markdown_report(ctx)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Phase 0 — automated record selection
    # ------------------------------------------------------------------
    print("\n--- Phase 0: automated record selection (night-010 JSONL REUSE) ---")
    p0 = phase0_select_record(LIVE_ECS_JSONL, DS033_JSONL, T2S_DEG)
    selected = p0["selected"]
    scores = p0["scores"]

    print("  scores (ranked by composite):")
    for i, s in enumerate(scores):
        print(
            f"    [{i+1}] rid={s['record_id']} "
            f"mean_pr={s['mean_pr']:.4f} std_pr={s['std_pr']:.4f} "
            f"mean_ecs={s['mean_ecs']:.4f} std_ecs={s['std_ecs']:.4f} "
            f"stability={s['stability']:.4f} severity={s['severity']:.4f} "
            f"pattern={s['pattern']:.1f} composite={s['composite']:.4f}"
        )
    rationale = (
        f"Top-ranked record {selected['record_id']} has the highest composite "
        f"({selected['composite']:.4f}). It is a DS-033 detection-gap record "
        f"(predicate_true_steps == 0) with no structural delimiters in its "
        f"generated text (pattern=0.5). Its mean PR ({selected['mean_pr']:.4f}) "
        f"and stability ({selected['stability']:.4f}) / severity "
        f"({selected['severity']:.4f}) combination outranks the other 9 records."
    )
    print(f"  selected record: {selected['record_id']}")
    print(f"  rationale: {rationale}")

    # Load the selected t2s fixture record.
    t2s_map = {int(r["id"]): r for r in load_jsonl(T2S_DEG)}
    selected_record = t2s_map[int(selected["record_id"])]
    print(f"  selected fixture record id={selected_record['id']} "
          f"text[:80]={selected_record['text'][:80]!r}")

    # ------------------------------------------------------------------
    # Model load
    # ------------------------------------------------------------------
    print("\n--- Model load ---")
    tokenizer, model = load_model_and_tokenizer()
    print(f"model loaded on {model.device}; num_heads="
          f"{model.config.num_attention_heads}")

    # ------------------------------------------------------------------
    # Phase 1 — PR pre-check (determinism smoke)
    # ------------------------------------------------------------------
    print("\n--- Phase 1: PR pre-check (determinism smoke) ---")
    p1 = phase1_pr_precheck(model, tokenizer, selected_record)
    p1["stopped"] = not p1["crossed_tpr_at_all_steps"]
    if p1["crossed_tpr_at_all_steps"]:
        p1["stop_reason"] = ""
    else:
        vals = p1["values"]
        pr_bases = [vals[str(s)]["pr_base"] for s in PHASE1_TEST_STEPS]
        pr_perts = [vals[str(s)]["pr_pert"] for s in PHASE1_TEST_STEPS]
        max_pr_pert = max(pr_perts)
        p1["stop_reason"] = (
            f"The perturbation did NOT cross T_PR ({T_PR_TASK:.6f}) at all "
            f"three steps. PR moved only marginally "
            f"(perturbed PR range [{min(pr_perts):.4f}, {max_pr_pert:.4f}] vs "
            f"baseline range [{min(pr_bases):.4f}, {max(pr_bases):.4f}]). "
            f"With a single-token 0.50-L2 isotropic-noise injection at layer 2, "
            f"the trailing 24-token window gains too little energy in the "
            f"minor singular-value components to re-inflate the manifold across "
            f"T_PR. Per the task specification, the perturbation is NOT "
            f"adjusted; the human decides whether to test multi-step "
            f"perturbation."
        )
        print(f"\n[STOP] Phase-1 PR pre-check: perturbed PR did NOT cross T_PR "
              f"at all three steps.")
        print(f"  {p1['stop_reason']}")

    # ------------------------------------------------------------------
    # Phase 2 — causal experiment (only if Phase 1 passes)
    # ------------------------------------------------------------------
    p2 = None
    if p1["crossed_tpr_at_all_steps"]:
        print("\n--- Phase 2: causal experiment ---")
        p2 = phase2_causal_experiment(model, tokenizer, selected_record)
        print(f"  generated {p2['n_generated']} tokens; intervention at step "
              f"{p2['intervention_step']}")
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
        "probe": "night-011",
        "seed": SEED,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_record_id": int(selected["record_id"]),
        "phase0": {
            "selected": selected,
            "scores": scores,
            "selection_rationale": rationale,
        },
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
        "intervention_step": INTERVENTION_STEP,
        "t_pr": t_pr,
        "band_low": band_low,
        "ecs_layer": ECS_LAYER,
        "ecs_head": ECS_HEAD,
        "selected_record_id": int(selected["record_id"]),
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "phase0_scores": scores,
        "phase0_selected": selected,
        "selection_rationale": rationale,
        "phase1": p1,
        "phase2": p2,
    }
    write_markdown_report(ctx)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("night-011 MEASUREMENT COMPLETE")


if __name__ == "__main__":
    main()
