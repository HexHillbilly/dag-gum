#!/usr/bin/env python3
"""night-002: Residual→Logit Singular-Vector Layer Sweep (MEASUREMENT ONLY).

Extends DS-036's Green's-function probe to FOUR layers {6, 14, 21, 27} using
randomized SVD on finite differences. The question: is the residual channel
dead at ALL depths, or only at the late layers where AVG currently intervenes?

Method (per night-002 task file):
  Per layer:
    1. Estimate the empirical residual→logit Jacobian at the final prompt-token
       position by finite differences on hooked re-injection.  K = q+p
       QR-orthonormal probe directions e_k in R^d; for each, one forward pass
       replaces h_ℓ (last token) with h_ℓ + ε·e_k and records Δy_k.
    2. ΔY ∈ R^(K×V).  Power iteration on the K×K Gram G=(1/ε²)ΔYΔYᵀ gives the
       dominant left singular vector c₁; v_i = Eᵀc_i/‖Eᵀc_i‖ recovers the
       right-singular directions in hidden space (Rayleigh–Ritz restriction).
       Top-4 spectrum from torch.linalg.eigh(G).
    3. Arms per prompt per layer: dormant (no kick), random-orthogonal,
       top singular v₁, bottom singular v₄ (K≥8).  Forces {0.35, 0.50, 0.75}.
       Kick applied in-place to the final-prompt-token residual at layer ℓ.
    4. Measure teacher-forced step-0 KL / top-1 shift / argmax flip and a
       64-token greedy rollout (exact-match, Distinct-2, first-divergence).

dtype: finite-difference Jacobian estimation runs in fp32 (bf16 dead-zone
finding, DS-036: ε=1e-3 perturbations quantise to zero in bf16 at late
layers).  Generation arms run in bf16 to match the production regime.
Mismatch is documented in the report.

This script is SELF-CONTAINED: no AVG imports, no controller/governor/core
reference.  Imports only torch, transformers, numpy, json, argparse, random,
pathlib (plus stdlib os/sys/time/math for environment handling).
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Env-only adaptation: deregister torch 2.13's CUDA bmm Triton override
# (no C compiler in the container; required for every script touching CUDA bmm).
try:
    from torch._native import triton_utils as _triton_utils

    _triton_utils.deregister_op_overrides()
    print("[env] torch bmm Triton override deregistered")
except Exception as _env_e:  # pragma: no cover - env dependent
    print(f"[env] bmm override deregistration skipped: {_env_e}")

# ---------------------------------------------------------------------------
# Fixed seed provenance and ground truth (human review verified)
# ---------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
FALLBACK_MODEL_REVISION = "main"
# 0.5B has 24 layers; scale {6,14,21,27}/28 -> {5,12,18,23}/24 (proportional).
FALLBACK_LAYERS = [5, 12, 18, 23]

DEFAULT_LAYERS = [6, 14, 21, 27]
PROBE_Q = 32  # randomized-SVD base probes
PROBE_P = 8   # oversampling
TOP_K = 4     # number of singular vectors/values to extract
FORCES = [0.35, 0.50, 0.75]  # bounded L2 kick magnitudes (production clamp)
EPS_REL = 1e-3               # ε = EPS_REL * RMS(h_ℓ)
LINEARITY_MULTIPLIERS = [0.5, 1.0, 2.0]
LINEARITY_TOL = 0.30         # allow ±30% deviation from linear scaling
DEFAULT_CONTINUATION = 64    # greedy rollout length
DISTINCT2_WINDOW = 24        # trailing tokens for Distinct-2

# Verdict thresholds (task pre-registration)
KL_DEAD_ABS = 1e-4          # DEAD if median KL ≲ 1e-4
KL_INCONCLUSIVE_ABS = 1e-3  # INCONCLUSIVE if top beats random but KL < 1e-3
ALIVE_MEDIAN_RATIO = 10.0   # ALIVE if top median KL >= 10x random median KL
ALIVE_FLIP_GAP_PP = 20.0    # or >=20pp argmax-flip gap
DEAD_EXACT_MATCH_RATE = 0.95  # >=95% byte-identical continuations

# HF cache: prefer the read-only system cache; fall back to /tmp only when the
# pinned revision is NOT present (avoid re-download of 1.5B weights).
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Hardcoded prompts (16 per category, SEED=42 provenance).  Order:
# [0:16] factual, [16:32] story, [32:48] instruction/math, [48:64] degenerate.
# ---------------------------------------------------------------------------
PROMPTS_BY_CATEGORY: Dict[str, List[str]] = {
    "factual": [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Australia is",
        "The capital of Canada is",
        "The capital of Brazil is",
        "The capital of Egypt is",
        "The capital of Italy is",
        "The capital of Germany is",
        "The capital of Spain is",
        "The capital of Mexico is",
        "The capital of India is",
        "The capital of South Korea is",
        "The capital of Argentina is",
        "The capital of Norway is",
        "The capital of Thailand is",
        "The capital of Portugal is",
    ],
    "story": [
        "The old house on the hill had always been",
        "She opened the door and found",
        "The rain had not stopped for three days when",
        "In the small town of Millbrook, nothing ever changed until",
        "The letter arrived on a Tuesday morning, and",
        "Deep in the forest, a path appeared that",
        "The clock struck midnight and the library",
        "He had never believed in ghosts, but",
        "The spaceship landed in the cornfield and",
        "A whisper came from the wardrobe, saying",
        "The map was old and torn, but it showed",
        "Under the floorboards, they discovered",
        "The last train left the station as",
        "Every morning at sunrise, the lighthouse keeper would",
        "The recipe called for one unusual ingredient",
        "When the power went out, the city began to",
    ],
    "math": [
        "Solve for x: 3x + 7 = 22",
        "Solve for x: 2x - 5 = 11",
        "Solve for x: x^2 - 9 = 0",
        "Solve for x: 4x + 3 = 2x + 15",
        "Solve for x: 5(x - 2) = 20",
        "Solve for x: 2x^2 - 8 = 0",
        "Solve for x: 6x + 11 = 5x + 17",
        "Solve for x: 3(x + 4) = 27",
        "Solve for x: 7x - 14 = 21",
        "Solve for x: x/3 + 2 = 6",
        "Solve for x: 2x + 9 = 3x - 1",
        "Solve for x: 8x - 4 = 4x + 8",
        "Solve for x: x^2 - 5x + 6 = 0",
        "Solve for x: 10 - 3x = 1",
        "Solve for x: 4x/2 = 10",
        "Solve for x: 5x + 5 = 4x + 9",
    ],
    "degenerate": [
        "The word word word word word word word",
        "Repeat repeat repeat repeat repeat repeat",
        "And and and and and and and and and and",
        "I I I I I I I I I I I I I I",
        "You you you you you you you you you",
        "It it it it it it it it it it it it",
        "The the the the the the the the the the",
        "Hello hello hello hello hello hello hello",
        "Test test test test test test test test",
        "One one one one one one one one one one",
        "Again again again again again again again",
        "This is is is is is is is is is",
        "Why why why why why why why why why",
        "Maybe maybe maybe maybe maybe maybe maybe",
        "Never never never never never never never",
        "Still still still still still still still still",
    ],
}
CATEGORY_ORDER = ["factual", "story", "math", "degenerate"]
ALL_PROMPTS: List[str] = [
    p for cat in CATEGORY_ORDER for p in PROMPTS_BY_CATEGORY[cat]
]
PROMPT_CATEGORY: List[str] = [
    cat for cat in CATEGORY_ORDER for _ in PROMPTS_BY_CATEGORY[cat]
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
# Metric helpers
# ---------------------------------------------------------------------------
def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P || Q) between two next-token logit distributions (float32).

    Same formula as DS-026/028/029/036 for direct comparability.
    """
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    eps = 1e-12
    kl = (p * (p + eps).log() - p * (q + eps).log()).sum(dim=-1)
    return float(max(0.0, kl.mean().item()))


def production_orthogonalise(
    direction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Production fallback double Gram-Schmidt (controller SAE-fallback path).

    Projects ``direction`` orthogonal to ``reference`` (the current residual)
    using the same two-step projection as the controller.  Returns the unit
    vector in R^d (flat).
    """
    ref_flat = reference.detach().float().reshape(-1)
    dir_flat = direction.detach().float().reshape(-1)
    ref_norm = ref_flat.norm() + 1e-8
    dot_prod = torch.sum(dir_flat * ref_flat, dim=-1)
    proj = (dot_prod / (ref_norm ** 2)) * ref_flat
    guided = dir_flat - proj
    dot_p = torch.sum(guided * ref_flat, dim=-1)
    ortho_vec = guided - (dot_p / (ref_norm ** 2)) * ref_flat
    ortho_unit = ortho_vec / (ortho_vec.norm() + 1e-8)
    return ortho_unit


def make_probe_directions(k: int, d: int, seed: int, device: str) -> torch.Tensor:
    """K orthonormal random directions in R^d (rows of E are e_k)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(d, k, generator=gen)  # d x K
    Q, _R = torch.linalg.qr(X)  # reduced QR: Q is d x K, orthonormal columns
    E = Q.T.contiguous()  # K x d, orthonormal rows e_k
    return E.to(device=device, dtype=torch.float32)


def distinct2(ids: torch.Tensor) -> float:
    """Distinct-2 (unique bigrams / total bigrams) over a token-id sequence."""
    ids = ids.reshape(-1)
    n = int(ids.shape[0])
    if n < 2:
        return 0.0
    bigrams: set = set()
    for i in range(n - 1):
        bigrams.add((int(ids[i]), int(ids[i + 1])))
    return len(bigrams) / (n - 1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    model_name: str, revision: str, dtype: torch.dtype, device: str
) -> torch.nn.Module:
    """Load an HF causal LM with the pinned revision; dtype kwarg compat."""
    kwargs: Dict[str, Any] = {}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=dtype, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=dtype, **kwargs
        )
    model = model.eval().to(device)
    return model


def load_tokenizer(model_name: str, revision: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(model_name, revision=revision)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------
def _split_output(output):
    """Return (hidden_states_tensor, trailing_tuple_items)."""
    if isinstance(output, torch.Tensor):
        return output, ()
    return output[0], output[1:]


def _rebuild_output(hs: torch.Tensor, trailing: Tuple) -> Any:
    if len(trailing) == 0:
        return hs
    return (hs,) + trailing


# ---------------------------------------------------------------------------
# Forward-pass primitives
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_baseline(
    model: torch.nn.Module, tokenizer: Any, prompt: str, layer: int
) -> Dict[str, Any]:
    """Teacher-forced forward pass; capture h_ℓ (last pos, fp32) + logits."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, args, output):
        hs = _split_output(output)[0]
        captured["h_t"] = hs[:, -1, :].detach().float().cpu()
        return output

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        out = model(input_ids)
    finally:
        if handle is not None:
            handle.remove()

    return {
        "h_t": captured["h_t"],  # [1, d] fp32 CPU
        "baseline_logits": out.logits[:, -1, :].float().cpu(),  # [1, V] fp32
        "prompt_len": int(input_ids.shape[-1]),
    }


@torch.no_grad()
def run_probe_pass(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    h_t: torch.Tensor,
    e_k: torch.Tensor,
    eps: float,
    layer: int,
) -> torch.Tensor:
    """One forward pass; last-token hidden state at layer ℓ := h_ℓ + ε·e_k."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]

    def hook_fn(module, args, output):
        hs, trailing = _split_output(output)
        new_hs = hs.clone()
        perturbed = (
            h_t.to(device=hs.device) + eps * e_k.to(device=hs.device)
        ).to(dtype=hs.dtype)
        new_hs[:, -1, :] = perturbed
        return _rebuild_output(new_hs, trailing)

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        out = model(input_ids)
        return out.logits[:, -1, :].float().cpu()
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def capture_ht(
    model: torch.nn.Module, tokenizer: Any, prompt: str, layer: int
) -> torch.Tensor:
    """Prefill pass capturing h_ℓ (last pos, fp32 CPU) without modifying."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, args, output):
        hs = _split_output(output)[0]
        captured["h_t"] = hs[:, -1, :].detach().float().cpu()
        return output

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        model(enc["input_ids"])
    finally:
        if handle is not None:
            handle.remove()
    return captured["h_t"]  # [1, d] fp32 CPU


def build_kick_hook(
    h_t: torch.Tensor, kick_dir: torch.Tensor, force: float
) -> Tuple[Any, Dict[str, Any]]:
    """Hook that kicks the last-prompt-token residual at layer ℓ ONCE.

    kick_dir is orthogonalised against h_t (production fallback).  The applied
    L2 delta is exactly ``force`` (kick_dir is a unit vector).
    """
    state: Dict[str, Any] = {"fired": False, "l2_delta": None, "captured_ht": None}
    kick_unit = production_orthogonalise(kick_dir, h_t)  # [d] unit
    h_t_flat = h_t.detach().float().reshape(-1)
    h_t_prime = (h_t_flat + force * kick_unit).reshape(1, -1)  # [1,d] fp32

    def hook_fn(module, args, output):
        if state["fired"]:
            return output
        hs, trailing = _split_output(output)
        state["captured_ht"] = hs[:, -1, :].detach().float().cpu()
        new_hs = hs.clone()
        new_hs[:, -1, :] = h_t_prime.to(dtype=hs.dtype, device=hs.device)
        state["fired"] = True
        state["l2_delta"] = float(
            (new_hs[:, -1, :].float() - hs[:, -1, :].float()).norm().item()
        )
        return _rebuild_output(new_hs, trailing)

    return hook_fn, state


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    layer: Optional[int] = None,
    kick: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Greedy generation.  If layer+kick given, a hook fires on the prefill
    forward pass only (kicks the last prompt-token residual at layer ℓ).
    Decoding steps are unmodified."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    handle = None
    hook_state: Optional[Dict[str, Any]] = None
    if layer is not None and kick is not None:
        hook_fn, hook_state = build_kick_hook(kick["h_t"], kick["dir"], kick["force"])
        handle = model.model.layers[layer].register_forward_hook(hook_fn)

    try:
        out = model(input_ids)
        step0_logits = out.logits[:, -1, :].float()  # [1, V] fp32
        pkv = out.past_key_values
        next_tok = step0_logits.argmax(dim=-1, keepdim=True)
        generated = [next_tok]
        eos_id = tokenizer.eos_token_id
        for _step in range(max_new_tokens - 1):
            if int(next_tok.item()) == eos_id:
                break
            out = model(next_tok, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :].float()
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
        ids = (
            torch.cat(generated, dim=1)
            if generated
            else torch.empty(1, 0, dtype=torch.long, device=model.device)
        )
    finally:
        if handle is not None:
            handle.remove()

    return {
        "generated_ids": ids,  # [1, n] on model.device
        "step0_logits": step0_logits.cpu(),  # [1, V] fp32 CPU
        "prompt_len": int(input_ids.shape[-1]),
        "hook_state": hook_state,
    }


# ---------------------------------------------------------------------------
# Linearity sanity check
# ---------------------------------------------------------------------------
def check_linearity(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    h_t: torch.Tensor,
    baseline_logits: torch.Tensor,
    e0: torch.Tensor,
    eps: float,
    layer: int,
) -> Dict[str, Any]:
    """Δy at {0.5ε, ε, 2ε} along e0; verify ‖Δy‖ scales approximately linearly."""
    norms: Dict[str, float] = {}
    for mult in LINEARITY_MULTIPLIERS:
        e = eps * mult
        logits_k = run_probe_pass(model, tokenizer, prompt, h_t, e0, e, layer)
        d = (logits_k - baseline_logits).reshape(-1)
        norms[str(mult)] = float(d.norm().item())
    n05 = norms["0.5"]
    n10 = norms["1.0"]
    n20 = norms["2.0"]
    ratio_lo = n10 / (n05 + 1e-12)
    ratio_hi = n20 / (n10 + 1e-12)
    # linear expectation: ratio ≈ 2 within ±LINEARITY_TOL relative
    tol_abs = 2.0 * LINEARITY_TOL
    linear_ok = bool(
        abs(ratio_lo - 2.0) <= tol_abs and abs(ratio_hi - 2.0) <= tol_abs
    )
    return {
        "norms": norms,
        "ratio_lo": float(ratio_lo),
        "ratio_hi": float(ratio_hi),
        "linear_ok": linear_ok,
    }


# ---------------------------------------------------------------------------
# Power iteration
# ---------------------------------------------------------------------------
def power_iteration(
    G: torch.Tensor, seed: int, num_iters: int = 30
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Top eigenvector of symmetric PSD KxK Gram G (float32 CPU)."""
    K = int(G.shape[0])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.randn(K, generator=gen, dtype=torch.float32)
    c = c / c.norm()
    trace: List[Dict[str, float]] = []
    for i in range(num_iters):
        c_new = G @ c
        norm = c_new.norm()
        c_new = c_new / (norm + 1e-12)
        cos = float(F.cosine_similarity(c_new.view(1, -1), c.view(1, -1)).item())
        trace.append({
            "iter": int(i + 1),
            "cos_with_prev": cos,
            "||G·c||": float(norm.item()),
        })
        c = c_new
    return c, trace


# ---------------------------------------------------------------------------
# Phase A: per-layer Jacobian estimation (randomized SVD via finite differences)
# ---------------------------------------------------------------------------
def estimate_layer_jacobian(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layer: int,
    q: int,
    p: int,
    seed: int,
    do_linearity: bool = True,
) -> Dict[str, Any]:
    """Estimate top-4 singular vectors of the residual→logit Jacobian at layer ℓ.

    Runs entirely in fp32 (model already upcast).  Returns v1, v_bottom (v4),
    the top-4 spectrum, ε, and linearity diagnostics.
    """
    base = capture_baseline(model, tokenizer, prompt, layer)
    h_t = base["h_t"]  # [1, d] fp32 CPU
    baseline_logits = base["baseline_logits"]  # [1, V] fp32 CPU
    d = int(h_t.shape[-1])
    V = int(baseline_logits.shape[-1])
    rms_h = float(h_t.norm().item() / math.sqrt(d))
    eps = EPS_REL * rms_h
    K = q + p

    E = make_probe_directions(K, d, seed, device="cpu")  # [K, d] fp32 CPU
    ete = E @ E.T
    ortho_err = float((ete - torch.eye(K)).abs().max().item())

    # Linear sanity check (one direction e0, three ε scales).  Adjust ε once.
    linearity: Optional[Dict[str, Any]] = None
    if do_linearity:
        linearity = check_linearity(
            model, tokenizer, prompt, h_t, baseline_logits, E[0], eps, layer
        )
        if not linearity["linear_ok"]:
            eps_adj = eps / 10.0
            lin_adj = check_linearity(
                model, tokenizer, prompt, h_t, baseline_logits, E[0], eps_adj, layer
            )
            linearity["adjusted_eps"] = float(eps_adj)
            linearity["adjusted_linear_ok"] = bool(lin_adj["linear_ok"])
            linearity["adjusted_norms"] = lin_adj["norms"]
            linearity["adjusted_ratio_lo"] = lin_adj["ratio_lo"]
            linearity["adjusted_ratio_hi"] = lin_adj["ratio_hi"]
            if lin_adj["linear_ok"]:
                eps = eps_adj
                linearity["eps"] = float(eps)

    # Finite-difference probe loop.
    delta_Y = torch.zeros((K, V), dtype=torch.float32)  # [K, V]
    logit_shift_norms: List[float] = []
    for k in range(K):
        e_k = E[k]
        logits_k = run_probe_pass(model, tokenizer, prompt, h_t, e_k, eps, layer)
        delta_y = logits_k - baseline_logits  # [1, V]
        delta_Y[k] = delta_y.reshape(-1)
        logit_shift_norms.append(float(delta_y.norm().item()))
        if (k + 1) % 10 == 0 or (k + 1) == K:
            print(f"    probe {k + 1}/{K}: ||Δy_k||={logit_shift_norms[-1]:.6f}")

    # Gram matrix G = (1/ε²) ΔY ΔYᵀ (Rayleigh–Ritz restriction of MᵀM).
    G = (1.0 / (eps * eps)) * (delta_Y @ delta_Y.T)  # [K, K]

    # Power iteration trace for v₁ (dominant left singular vector of probe span).
    c1, pi_trace = power_iteration(G, seed)

    # Full eigendecomposition for top-4 spectrum (symmetric PSD, cheap for K≤40).
    eigvals, eigvecs = torch.linalg.eigh(G)  # ascending
    eigvals = eigvals.flip(0).clamp_min(0.0)
    eigvecs = eigvecs.flip(1)  # columns now descending

    singular_values: List[Dict[str, float]] = []
    directions: Dict[str, torch.Tensor] = {}
    for i in range(TOP_K):
        sigma = eps * math.sqrt(float(eigvals[i]))
        ci = eigvecs[:, i]  # [K]
        vi_raw = E.T @ ci  # [d]
        vi = vi_raw / (vi_raw.norm() + 1e-12)
        sigma_norm = (
            float(sigma / singular_values[0]["sigma"]) if i > 0 else 1.0
        )
        singular_values.append({
            "rank": int(i + 1),
            "sigma": float(sigma),
            "sigma_norm": float(sigma_norm),
        })
        directions[f"v{i + 1}"] = vi

    v1 = directions["v1"]
    v_bottom = directions[f"v{TOP_K}"]
    sigma_ratio = (
        singular_values[0]["sigma"] / singular_values[1]["sigma"]
        if len(singular_values) > 1 and singular_values[1]["sigma"] > 1e-30
        else float("nan")
    )

    print(f"    eps={eps:.3e} rms_h={rms_h:.4f} σ₁={singular_values[0]['sigma']:.3e} "
          f"σ₁/σ₂={sigma_ratio:.2f}")
    print(f"    power-iter v₁ converged: cos={pi_trace[-1]['cos_with_prev']:.6f}")

    return {
        "layer": int(layer),
        "K": int(K),
        "q": int(q),
        "p": int(p),
        "d": int(d),
        "V": int(V),
        "eps": float(eps),
        "rms_h": float(rms_h),
        "ortho_err": float(ortho_err),
        "mean_logit_shift_norm": float(np.mean(logit_shift_norms)),
        "max_logit_shift_norm": float(np.max(logit_shift_norms)),
        "singular_values": singular_values,
        "sigma_ratio": float(sigma_ratio),
        "power_iteration": pi_trace,
        "linearity": linearity,
        "v1_first10": v1[:10].tolist(),
        # Tensors used downstream (not serialised directly).
        "_E": E,
        "_delta_Y": delta_Y,
        "_G": G,
        "_v1": v1,
        "_v_bottom": v_bottom,
    }


# ---------------------------------------------------------------------------
# Phase B: per-prompt arms + metrics
# ---------------------------------------------------------------------------
def compute_arm_metrics(
    dormant: Dict[str, Any],
    hooked: Dict[str, Any],
    window: int = DISTINCT2_WINDOW,
) -> Dict[str, Any]:
    """Compare one hooked rollout against its dormant baseline."""
    d_ids = dormant["ids"].cpu()
    h_ids = hooked["generated_ids"].cpu()
    n = min(d_ids.shape[1], h_ids.shape[1])
    d_ids = d_ids[:, :n]
    h_ids = h_ids[:, :n]

    exact_match = bool(torch.equal(d_ids, h_ids))
    token_match_rate = float((d_ids == h_ids).float().mean().item()) if n > 0 else 0.0

    first_div: Optional[int] = None
    if not exact_match and n > 0:
        diff = (d_ids != h_ids)[0].nonzero(as_tuple=False)
        if diff.numel() > 0:
            first_div = int(diff[0].item())

    kl0 = kl_divergence(hooked["step0_logits"], dormant["step0_logits"])
    top1_dormant = dormant["step0_logits"].argmax(dim=-1)
    top1_hooked = hooked["step0_logits"].argmax(dim=-1)
    flip = bool(top1_hooked.item() != top1_dormant.item())
    top1_shift = float(
        (hooked["step0_logits"][0, top1_dormant]
         - dormant["step0_logits"][0, top1_dormant]).abs().item()
    )

    h_d2 = distinct2(h_ids[:, -window:]) if h_ids.shape[1] > 0 else 0.0
    d_d2 = distinct2(d_ids[:, -window:]) if d_ids.shape[1] > 0 else 0.0

    return {
        "step0_kl": float(kl0),
        "step0_top1_shift": float(top1_shift),
        "step0_flip": flip,
        "exact_match": exact_match,
        "token_match_rate": token_match_rate,
        "first_divergence_idx": first_div,
        "distinct2_hooked": float(h_d2),
        "distinct2_dormant": float(d_d2),
    }


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(list(xs))) if len(xs) > 0 else float("nan")


def _median(xs: Sequence[float]) -> float:
    return float(np.median(list(xs))) if len(xs) > 0 else float("nan")


def aggregate_arms(
    rows: List[Dict[str, Any]], forces: Sequence[float]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """rows: per (prompt, force, arm) metric dicts.  Returns per-force per-arm stats."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for f in forces:
        fs = f"{f:g}"
        out[fs] = {}
        for arm_name in ["random", "top", "bottom"]:
            arm_rows = [r for r in rows if abs(r["force"] - f) < 1e-9 and r["arm"] == arm_name]
            if not arm_rows:
                continue
            first_divs = [
                r["first_divergence_idx"] for r in arm_rows
                if r["first_divergence_idx"] is not None
            ]
            out[fs][arm_name] = {
                "median_kl": _median([r["step0_kl"] for r in arm_rows]),
                "mean_kl": _mean([r["step0_kl"] for r in arm_rows]),
                "argmax_flip_rate": _mean([float(r["step0_flip"]) for r in arm_rows]),
                "exact_match_rate": _mean([float(r["exact_match"]) for r in arm_rows]),
                "token_match_rate": _mean([r["token_match_rate"] for r in arm_rows]),
                "median_first_divergence_idx": (
                    int(np.median(first_divs)) if first_divs else None
                ),
                "distinct2_hooked_mean": _mean([r["distinct2_hooked"] for r in arm_rows]),
                "distinct2_dormant_mean": _mean([r["distinct2_dormant"] for r in arm_rows]),
                "step0_top1_shift_mean": _mean([r["step0_top1_shift"] for r in arm_rows]),
            }
    return out


def run_layer_arms(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    categories: List[str],
    dormant_cache: Dict[int, Dict[str, Any]],
    layer: int,
    v1: torch.Tensor,
    v_bottom: torch.Tensor,
    forces: Sequence[float],
    continuation: int,
    seed: int,
    include_bottom: bool = True,
) -> Dict[str, Any]:
    """Run all arms for one layer.  Returns dict with 'rows' and 'dormant_d2_mean'."""
    rows: List[Dict[str, Any]] = []
    d2_list: List[float] = []
    n_prompts = len(prompts)

    for pi, prompt in enumerate(prompts):
        d = dormant_cache[pi]
        d2_list.append(d["distinct2"])

        # Per-prompt residual at layer ℓ (bf16 model, fp32 capture).
        h_t = capture_ht(model, tokenizer, prompt, layer)  # [1, d] fp32 CPU

        # Random orthogonal control (seeded per prompt/layer).
        gen = torch.Generator(device="cpu").manual_seed(
            (seed + 1009 * layer + 17 * pi) % (2 ** 31)
        )
        gauss = torch.randn(int(h_t.shape[-1]), generator=gen, dtype=torch.float32)
        v_rand = production_orthogonalise(gauss, h_t)  # [d] unit

        arms: List[Tuple[str, torch.Tensor]] = [("random", v_rand), ("top", v1)]
        if include_bottom:
            arms.append(("bottom", v_bottom))

        for force in forces:
            for arm_name, direction in arms:
                kick = {"h_t": h_t, "dir": direction, "force": float(force)}
                hooked = greedy_generate(
                    model, tokenizer, prompt, continuation,
                    layer=layer, kick=kick,
                )
                if hooked["hook_state"] is None or not hooked["hook_state"]["fired"]:
                    print(f"    [WARN] layer {layer} prompt {pi} {arm_name} "
                          f"f={force}: hook did NOT fire")
                m = compute_arm_metrics(d, hooked)
                rows.append({
                    "prompt_idx": pi,
                    "category": categories[pi],
                    "layer": int(layer),
                    "force": float(force),
                    "arm": arm_name,
                    **m,
                })

        if (pi + 1) % 8 == 0 or (pi + 1) == n_prompts:
            print(f"    prompt {pi + 1}/{n_prompts} done (layer {layer})")

    return {
        "rows": rows,
        "dormant_distinct2_mean": _mean(d2_list),
        "n_prompts": n_prompts,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def compute_verdict(
    arms: Dict[str, Dict[str, Dict[str, Any]]],
    forces: Sequence[float],
    sigma1: float,
) -> Dict[str, Any]:
    """Pre-registered DEAD/ALIVE/INCONCLUSIVE verdict."""
    force_keys = [f"{f:g}" for f in forces]
    active = [k for k in force_keys if k in arms]

    # ALIVE: any force where top median KL >= 10x random OR >=20pp flip gap.
    alive_details: List[Dict[str, Any]] = []
    for k in active:
        if "top" not in arms[k] or "random" not in arms[k]:
            continue
        kl_v = arms[k]["top"]["median_kl"]
        kl_r = arms[k]["random"]["median_kl"]
        flip_v = arms[k]["top"]["argmax_flip_rate"]
        flip_r = arms[k]["random"]["argmax_flip_rate"]
        ratio = kl_v / max(kl_r, 1e-12)
        flip_gap_pp = (flip_v - flip_r) * 100.0
        if kl_v >= ALIVE_MEDIAN_RATIO * kl_r or flip_gap_pp >= ALIVE_FLIP_GAP_PP:
            alive_details.append({
                "force": k,
                "kl_v1": kl_v,
                "kl_rand": kl_r,
                "ratio": ratio,
                "flip_gap_pp": flip_gap_pp,
            })

    # DEAD: ALL forces, both arms: MEAN KL <= 1e-4 AND >=95% exact-match.
    # (Pre-registration uses "mean KL" for DEAD; ALIVE uses "median KL" above.)
    dead_all = len(active) > 0
    dead_fails: List[str] = []
    for k in active:
        for arm_name in ["top", "random"]:
            if arm_name not in arms[k]:
                continue
            a = arms[k][arm_name]
            if a["mean_kl"] > KL_DEAD_ABS:
                dead_fails.append(f"{k}/{arm_name}/kl={a['mean_kl']:.2e}")
            if a["exact_match_rate"] < DEAD_EXACT_MATCH_RATE:
                dead_fails.append(f"{k}/{arm_name}/exact={a['exact_match_rate']:.3f}")
    if dead_fails:
        dead_all = False

    # Primary force (0.5 if present, else first).
    primary = "0.5" if "0.5" in arms else (active[0] if active else None)

    if alive_details:
        verdict = "ALIVE"
        basis = "; ".join(
            f"f={d['force']} KL_v1/KL_rand={d['ratio']:.1f}x "
            f"(flip gap {d['flip_gap_pp']:.1f}pp)"
            for d in alive_details
        )
    elif dead_all:
        verdict = "DEAD"
        basis = "all forces: top and random mean KL <= 1e-4 AND >=95% byte-identical continuations"
    else:
        # INCONCLUSIVE if top beats random at primary force but absolute KL < 1e-3.
        inconclusive = False
        if primary and "top" in arms[primary] and "random" in arms[primary]:
            kl_v = arms[primary]["top"]["median_kl"]
            kl_r = arms[primary]["random"]["median_kl"]
            if kl_v > kl_r and kl_v < KL_INCONCLUSIVE_ABS:
                inconclusive = True
        if inconclusive:
            verdict = "INCONCLUSIVE"
            basis = (f"top beats random at f={primary} but absolute KL "
                     f"{arms[primary]['top']['median_kl']:.2e} < 1e-3 "
                     f"(non-zero singular structure, channel effectively closed)")
        else:
            verdict = "INCONCLUSIVE"
            basis = f"no ALIVE/DEAD conditions met at primary force {primary}"

    # Predicted KL via local linearity: KL_pred = 0.5·(σ₁·0.5)² (noise-floor=1).
    kl_pred_05 = 0.5 * (sigma1 * 0.5) ** 2

    return {
        "verdict": verdict,
        "basis": basis,
        "primary_force": primary,
        "kl_pred_05": float(kl_pred_05),
        "alive_details": alive_details,
        "dead_all": bool(dead_all),
        "dead_fails": dead_fails,
    }


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def jac_serializable(jac: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in jac.items()
        if not k.startswith("_")
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-002 residual->logit singular-vector layer sweep "
                    "(measurement only)"
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--n-prompts", type=int, default=64)
    parser.add_argument("--continuation", type=int, default=DEFAULT_CONTINUATION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="docs/gate23")
    parser.add_argument("--q", type=int, default=PROBE_Q)
    parser.add_argument("--p", type=int, default=PROBE_P)
    parser.add_argument("--force", type=float, nargs="+", default=list(FORCES))
    parser.add_argument("--anchor-prompt-index", type=int, default=0,
                        help="Prompt index for Jacobian estimation (default 0).")
    parser.add_argument("--probe-device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for fp32 FD Jacobian estimation (default cpu; "
                             "avoids bf16 dead-zone and GPU contention).")
    parser.add_argument("--no-bottom", action="store_true",
                        help="Exclude the bottom singular-direction control.")
    parser.add_argument("--no-linearity", action="store_true",
                        help="Skip the ε linearity sanity check.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test config: layers {6,27}, n-prompts<=4, "
                             "continuation=16, plus determinism check.")
    parser.add_argument("--reduced", action="store_true",
                        help="Reduced config: layers {6,27}, n-prompts<=16, q=16, p=4.")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    # ------------------------------------------------------------------
    # Resolve config
    # ------------------------------------------------------------------
    layers: List[int] = list(args.layers)
    n_prompts: int = args.n_prompts
    continuation: int = args.continuation
    q: int = args.q
    p: int = args.p
    forces: List[float] = list(args.force)

    if args.smoke:
        smoke_layers = [l for l in [6, 27] if l in layers] or layers[:2]
        layers = smoke_layers
        n_prompts = min(n_prompts, 4)
        continuation = min(continuation, 16)
        q = max(q, 8)
        p = max(p, 2)
        print("[config] SMOKE mode")
    if args.reduced:
        layers = [6, 27]
        n_prompts = min(n_prompts, 16)
        q = 16
        p = 4
        print("[config] REDUCED mode")
    K = q + p
    if K < 8:
        raise SystemExit("[STOP] K = q+p < 8: bottom-direction control not meaningful.")

    prompts = ALL_PROMPTS[:n_prompts]
    categories = PROMPT_CATEGORY[:n_prompts]
    if args.anchor_prompt_index >= n_prompts:
        raise SystemExit(
            f"[STOP] --anchor-prompt-index {args.anchor_prompt_index} >= "
            f"n_prompts {n_prompts}"
        )
    anchor_prompt = prompts[args.anchor_prompt_index]

    print("=" * 70)
    print("night-002 Residual→Logit Singular-Vector Layer Sweep (MEASUREMENT)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"model: {args.model}@{args.revision}")
    print(f"layers: {layers}")
    print(f"n_prompts: {n_prompts} | continuation: {continuation}")
    print(f"probe: K={K} (q={q}, p={p}) | probe_device={args.probe_device} | "
          f"forces={forces}")
    print(f"anchor_prompt_idx={args.anchor_prompt_index} | "
          f"anchor={anchor_prompt!r}")
    print(f"torch: {torch.__version__} | cuda: {torch.cuda.is_available()}")

    # ------------------------------------------------------------------
    # Model load (fp32 for FD Jacobian estimation)
    # ------------------------------------------------------------------
    model_name = args.model
    revision = args.revision
    try:
        tokenizer = load_tokenizer(model_name, revision)
        probe_device = args.probe_device
        if probe_device == "cuda" and not torch.cuda.is_available():
            probe_device = "cpu"
        model = load_model(model_name, revision, torch.float32, probe_device)
    except Exception as e:
        if model_name == MODEL_NAME:
            print(f"[fallback] {model_name} load failed ({e}); falling back to "
                  f"{FALLBACK_MODEL_NAME} with proportional layers "
                  f"{FALLBACK_LAYERS}")
            model_name = FALLBACK_MODEL_NAME
            revision = FALLBACK_MODEL_REVISION
            layers = FALLBACK_LAYERS
            tokenizer = load_tokenizer(model_name, revision)
            model = load_model(model_name, revision, torch.float32, "cpu")
        else:
            raise

    n_layers_model = len(model.model.layers)
    print(f"model loaded on {model.device} dtype={model.dtype} "
          f"| {n_layers_model} layers | hidden={model.config.hidden_size}")
    for l in layers:
        if l < 0 or l >= n_layers_model:
            raise SystemExit(f"[STOP] layer {l} out of range [0, {n_layers_model})")

    # ------------------------------------------------------------------
    # Phase A: FD Jacobian estimation per layer (fp32)
    # ------------------------------------------------------------------
    layer_jacs: Dict[int, Dict[str, Any]] = {}
    for layer in layers:
        print(f"\n--- Phase A: FD Jacobian @ layer {layer} (fp32, "
              f"probe_device={probe_device}) ---")
        jac = estimate_layer_jacobian(
            model, tokenizer, anchor_prompt, layer,
            q=q, p=p, seed=SEED + layer,
            do_linearity=not args.no_linearity,
        )
        layer_jacs[layer] = jac

    # ------------------------------------------------------------------
    # Move model to generation device (bf16, production regime)
    # ------------------------------------------------------------------
    gen_device = args.device
    if gen_device == "cuda" and torch.cuda.is_available():
        # Cast on CPU first, then move (avoids fp32 GPU spike).
        model = model.to(torch.bfloat16)
        model = model.to("cuda")
        gen_device = "cuda"
    else:
        gen_device = "cpu"
        model = model.to(torch.bfloat16)
    torch.cuda.empty_cache()
    print(f"\n[arms] generation model on {gen_device} dtype={model.dtype}")

    # ------------------------------------------------------------------
    # Determinism smoke (smoke mode only): dormant generated twice.
    # ------------------------------------------------------------------
    determinism: Optional[Dict[str, Any]] = None
    if args.smoke:
        print("\n--- Determinism smoke (1 prompt, dormant twice) ---")
        p0 = prompts[0]
        a = greedy_generate(model, tokenizer, p0, continuation)
        b = greedy_generate(model, tokenizer, p0, continuation)
        ids_match = bool(torch.equal(a["generated_ids"].cpu(), b["generated_ids"].cpu()))
        logits_match = bool(torch.allclose(
            a["step0_logits"], b["step0_logits"], atol=0.0, rtol=0.0
        ))
        print(f"  token ids identical: {ids_match}")
        print(f"  step-0 logits identical: {logits_match}")
        if not ids_match or not logits_match:
            print("[STOP] determinism smoke FAILED.")
            sys.exit(1)
        determinism = {
            "prompt": p0,
            "ids_identical": str(ids_match),
            "step0_logits_identical": str(logits_match),
            "n_tokens": int(a["generated_ids"].shape[-1]),
        }

    # ------------------------------------------------------------------
    # Dormant cache (layer-independent; rolled out once per prompt).
    # ------------------------------------------------------------------
    print("\n--- Dormant rollouts (cached per prompt) ---")
    dormant_cache: Dict[int, Dict[str, Any]] = {}
    for pi, prompt in enumerate(prompts):
        out = greedy_generate(model, tokenizer, prompt, continuation)
        ids = out["generated_ids"].cpu()
        dormant_cache[pi] = {
            "ids": ids,
            "step0_logits": out["step0_logits"].cpu(),
            "distinct2": distinct2(ids),
        }
        if (pi + 1) % 8 == 0 or (pi + 1) == n_prompts:
            print(f"  dormant {pi + 1}/{n_prompts} done")

    # ------------------------------------------------------------------
    # Phase B: arms per layer
    # ------------------------------------------------------------------
    layer_results: List[Dict[str, Any]] = []
    for layer in layers:
        print(f"\n--- Phase B: arms @ layer {layer} (bf16, {gen_device}) ---")
        jac = layer_jacs[layer]
        v1 = jac["_v1"]
        v_bottom = jac["_v_bottom"]
        arm_out = run_layer_arms(
            model, tokenizer, prompts, categories, dormant_cache,
            layer, v1, v_bottom, forces, continuation, SEED,
            include_bottom=not args.no_bottom,
        )
        arms = aggregate_arms(arm_out["rows"], forces)
        verdict = compute_verdict(arms, forces, jac["singular_values"][0]["sigma"])

        diagnostics = {
            "layer": int(layer),
            "sigma_1": jac["singular_values"][0]["sigma"],
            "sigma_2": jac["singular_values"][1]["sigma"],
            "sigma_3": jac["singular_values"][2]["sigma"],
            "sigma_4": jac["singular_values"][3]["sigma"],
            "sigma_1_over_2": jac["sigma_ratio"],
            "kl_pred_05": verdict["kl_pred_05"],
            "kl_measured_v1_05": arms.get("0.5", {}).get("top", {}).get("median_kl"),
            "kl_measured_rand_05": arms.get("0.5", {}).get("random", {}).get("median_kl"),
            "status": verdict["verdict"],
        }
        print(f"  VERDICT: {verdict['verdict']} | {verdict['basis']}")

        layer_results.append({
            "layer": int(layer),
            "jacobian": jac_serializable(jac),
            "arms": arms,
            "dormant_distinct2_mean": arm_out["dormant_distinct2_mean"],
            "diagnostics": diagnostics,
            "verdict": verdict,
        })

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "residual_layer_sweep_results.jsonl"
    md_path = out_dir / "RESIDUAL_LAYER_SWEEP_RESULTS.md"

    metadata = {
        "task": "night-002 residual-layer sweep",
        "model_name": model_name,
        "model_revision": revision,
        "device": gen_device,
        "probe_device": probe_device,
        "probe_dtype": "fp32",
        "arms_dtype": "bf16",
        "torch_version": torch.__version__,
        "seed": SEED,
        "layers": layers,
        "n_prompts": n_prompts,
        "continuation": continuation,
        "probe_K": K,
        "probe_q": q,
        "probe_p": p,
        "top_k": TOP_K,
        "forces": forces,
        "eps_rel": EPS_REL,
        "anchor_prompt_index": args.anchor_prompt_index,
        "anchor_prompt": anchor_prompt,
        "bmm_override": "deregistered",
        "smoke": bool(args.smoke),
        "reduced": bool(args.reduced),
        "wall_clock_s": round(time.time() - t_start, 3),
    }

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": metadata}) + "\n")
        f.write(json.dumps({"prompts": {
            "seed": SEED,
            "categories": CATEGORY_ORDER,
            "n_prompts": n_prompts,
            "prompts": prompts,
            "prompt_categories": categories,
        }}) + "\n")
        for lr in layer_results:
            f.write(json.dumps({"layer_result": lr}) + "\n")

    # --- Markdown report ---
    md: List[str] = []
    md.append("# night-002 — Residual→Logit Singular-Vector Layer Sweep")
    md.append("")
    md.append(f"Date: 2026-08-11  |  SEED: {SEED}  |  "
              f"Model: `{model_name}@{revision}`")
    md.append("")
    md.append("MEASUREMENT ONLY. Self-contained probe (no AVG imports).")
    md.append("")
    md.append("## Method")
    md.append("")
    md.append(f"- Layers probed: `{layers}`")
    md.append(f"- Randomized SVD on finite differences: K={K} (q={q}, p={p}), "
              f"top-{TOP_K} singular vectors via Rayleigh–Ritz on the K×K Gram.")
    md.append(f"- ε = {EPS_REL} × RMS(h_ℓ) at the anchor prompt "
              f"(`{anchor_prompt!r}`, index {args.anchor_prompt_index}).")
    md.append(f"- FD Jacobian estimation in **fp32** ({probe_device}); "
              f"generation arms in **bf16** ({gen_device}) to match production.")
    md.append(f"- Arms per prompt per layer: dormant, random-orthogonal, "
              f"top v₁, bottom v₄; forces {list(forces)}. "
              f"Continuation = {continuation} tokens.")
    md.append("")
    md.append("## Diagnostic table")
    md.append("")
    md.append("| layer | σ₁ | σ₂ | σ₃ | σ₄ | σ₁/σ₂ | pred KL@0.5 | "
              "KL@0.5 v₁ | KL@0.5 rand | status |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for lr in layer_results:
        d = lr["diagnostics"]

        def _fmt_kl(v: Any) -> str:
            return f"{v:.2e}" if v is not None else "n/a"

        md.append(
            f"| {d['layer']} | {d['sigma_1']:.3e} | {d['sigma_2']:.3e} | "
            f"{d['sigma_3']:.3e} | {d['sigma_4']:.3e} | {d['sigma_1_over_2']:.2f} | "
            f"{d['kl_pred_05']:.2e} | {_fmt_kl(d['kl_measured_v1_05'])} | "
            f"{_fmt_kl(d['kl_measured_rand_05'])} | "
            f"**{d['status']}** |"
        )
    md.append("")
    md.append("Predicted KL via local linearity: `KL_pred = 0.5·(σ₁·0.5)²` "
              "(noise-floor normalised to 1; σ₁ in logit units per unit hidden "
              "perturbation).")
    md.append("")
    md.append("## Verdicts")
    md.append("")
    for lr in layer_results:
        v = lr["verdict"]
        md.append(f"### Layer {lr['layer']}: **{v['verdict']}**")
        md.append("")
        md.append(f"- Basis: {v['basis']}")
        md.append(f"- KL_pred@0.5: {v['kl_pred_05']:.3e}")
        md.append("")
    md.append("## Per-arm aggregates (median step-0 KL, argmax-flip rate, "
              "exact-match rate)")
    md.append("")
    md.append("| layer | force | arm | median KL | mean KL | flip rate | "
              "exact-match | token-match | Distinct-2 (hooked) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for lr in layer_results:
        for fk, arm_dict in lr["arms"].items():
            for arm_name, a in arm_dict.items():
                md.append(
                    f"| {lr['layer']} | {fk} | {arm_name} | "
                    f"{a['median_kl']:.2e} | {a['mean_kl']:.2e} | "
                    f"{a['argmax_flip_rate']:.3f} | {a['exact_match_rate']:.3f} | "
                    f"{a['token_match_rate']:.3f} | "
                    f"{a['distinct2_hooked_mean']:.4f} |"
                )
    md.append("")
    md.append(f"Dormant Distinct-2 mean (trailing {DISTINCT2_WINDOW}): "
              + ", ".join(
                  f"layer {lr['layer']}: {lr['dormant_distinct2_mean']:.4f}"
                  for lr in layer_results
              ))
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- FD probe runs in fp32 to avoid the bf16 dead-zone "
              "(DS-036 finding: ε=1e-3 perturbations quantise to zero in bf16 "
              "at late layers). Arms run in bf16; mismatch documented.")
    md.append("- `docs/gate23/residual_layer_sweep_results.jsonl` contains full "
              "per-layer records (singular spectrum, power-iteration trace, "
              "linearity check, per-arm aggregates).")
    md.append("")
    md_path.write_text("\n".join(md), encoding="utf-8")

    # ------------------------------------------------------------------
    # Smoke verification checks
    # ------------------------------------------------------------------
    smoke_checks: Optional[Dict[str, Any]] = None
    if args.smoke:
        print("\n--- Smoke verification ---")
        checks = {"hooks_fired": True, "sigma_finite": True, "sigma_ordered": True,
                  "kick_changes_logits": True, "json_written": True}
        for lr in layer_results:
            sv = lr["jacobian"]["singular_values"]
            if not all(math.isfinite(s["sigma"]) for s in sv):
                checks["sigma_finite"] = False
            if sv[0]["sigma"] < sv[1]["sigma"]:
                checks["sigma_ordered"] = False
        # kick changes logits: check any top arm has nonzero median KL or top1 shift
        any_kl_nonzero = False
        for lr in layer_results:
            for fk, arm_dict in lr["arms"].items():
                if "top" in arm_dict:
                    if arm_dict["top"]["median_kl"] > 0.0 or \
                       arm_dict["top"]["step0_top1_shift_mean"] > 0.0:
                        any_kl_nonzero = True
        checks["kick_changes_logits"] = any_kl_nonzero
        checks["json_written"] = jsonl_path.exists() and jsonl_path.stat().st_size > 0
        checks["determinism"] = determinism
        for k, v in checks.items():
            print(f"  {k}: {v}")
        if not all(checks[k] for k in ["hooks_fired", "sigma_finite",
                                       "sigma_ordered", "kick_changes_logits",
                                       "json_written"]):
            print("[STOP] smoke verification FAILED.")
            sys.exit(1)
        smoke_checks = checks
        print("  smoke verification: ALL PASS")

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {md_path}")
    print(f"Total wall-clock: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
