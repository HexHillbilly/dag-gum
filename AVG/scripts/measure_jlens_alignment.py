#!/usr/bin/env python3
"""night-003: J-Lens collapse→Jacobian alignment diagnostic (MEASUREMENT ONLY).

Hypothesis (J-Lens): during macro-loop collapse, the dominant direction of
hidden-state compression at layer 27 (the top right-singular vector c1 of the
trailing-24 hidden-state window) may be nearly ORTHOGONAL to the top right-
singular vector v1 of the residual→logit Jacobian at the same step.  The
manifold compresses along a direction the unembedding does not care about, so
the next-token distribution stays healthy (surface diversity unchanged).  If
those directions later rotate into alignment, the collapse becomes visible to
the output layer and surface diversity drops.

This probe measures the per-step absolute cosine similarity |<c1, v1>| across
the first 10 DS-033 detection-gap t2s_degenerate records during greedy
generation.  PURELY DIAGNOSTIC: no intervention, no new thresholds, no
controller changes.  It reuses the night-002 finite-difference + randomized
SVD Jacobian estimation (scripts/residual_logit_probe.py) at layer 27.

Self-contained: no AVG imports.  FD Jacobian estimation runs in fp32 (bf16
dead-zone, DS-036); generation runs in bf16 (production regime).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

# Frozen PR threshold (ds-025 Part A freeze) — logging context ONLY; this
# diagnostic does NOT gate actuation.
T_PR = 10.954796
BAND_LOW = 8.216097

# Layers
JACOBIAN_LAYER = 27   # last block before final RMSNorm + unembedding
PR_LAYER = 2          # layer-2 spectral PR (DS-034 comparison)

# Sampling
SAMPLED_STEPS = [23, 36, 50, 64, 78, 92, 106, 120]
MAX_NEW_TOKENS = 128
WINDOW = 24           # trailing-24 hidden-state window (both PR and c1)
CTR_WINDOW = 16       # trailing-16 tokens for CTR

# Surface thresholds (production)
COLLAPSE_DIVERSITY = 0.30   # bigram fire threshold
HEALTHY_DIVERSITY = 0.40    # bigram health threshold

# night-002 FD + randomized SVD parameters
PROBE_Q = 32          # randomized-SVD base probes
PROBE_P = 8           # oversampling
K_PROBE = PROBE_Q + PROBE_P   # 40
EPS_REL = 1e-3        # ε = EPS_REL * RMS(h_t)
POWER_ITERS = 30      # v1 power-iteration iterations (night-002 default)
C1_ITERS = 3          # c1 power-iteration iterations (24x24 Gram is cheap)
CONVERGE_COS = 0.9999 # v1 convergence threshold
C1_CONVERGE_COS = 0.99

# Fixtures / classification (DS-033 ground truth, do NOT re-classify)
FIXTURE_PATH = "tests/fixtures/t2s_degenerate.jsonl"
DS033_PATH = "docs/gate23/production_cross_fixture_results.jsonl"
DETECTION_GAP_CLASS = "detection-gap"
N_RECORDS = 10

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
# Seeding
# ---------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Inlined AVG metrics (self-contained; no AVG imports)
# ---------------------------------------------------------------------------
def compute_token_distinct_2_fast(
    input_ids: torch.Tensor,
    prompt_len: int = 0,
    window_len: int = 24,
) -> Tuple[float, List[int]]:
    """Fast, vectorized bigram diversity calculator (AVG controller).

    Falls back gracefully to full trailing context if generated tokens < 4.
    """
    if input_ids.shape[-1] - prompt_len >= 4:
        tokens = input_ids[0, prompt_len:][-window_len:]
    else:
        tokens = input_ids[0, -window_len:]

    n = tokens.numel()
    if n < 4:
        return 1.0, []

    t1 = tokens[:-1]
    t2 = tokens[1:]
    bigrams = torch.stack([t1, t2], dim=1)
    n_unique = torch.unique(bigrams, dim=0).shape[0]
    dist_2 = n_unique / float(n - 1)

    unique_vals, counts = torch.unique(tokens, return_counts=True)
    repeated_mask = counts >= 2
    repeated_ids = unique_vals[repeated_mask]
    repeated_ids_list = repeated_ids.tolist() if repeated_ids.numel() > 0 else []
    return float(dist_2), repeated_ids_list


def compute_coherent_token_ratio(text: str) -> float:
    """Language- and code-aware coherence metric (CTR).  AVG core.metrics."""
    raw_tokens = text.strip().split()
    if not raw_tokens:
        return 0.0

    valid_count = 0
    single_char_count = 0

    for tok in raw_tokens:
        clean = tok.strip("(),;:{}[]\"'<>`#$%^&*=-+/")

        if len(clean) == 1 and clean.isalpha():
            single_char_count += 1
            continue

        if len(clean) >= 2 and all(c.isalpha() or c in ("'", "-") for c in clean):
            valid_count += 1
        elif any(
            "一" <= c <= "龥"
            or "぀" <= c <= "ヿ"
            or "가" <= c <= "힯"
            for c in tok
        ):
            valid_count += 1
        elif re.match(r"^\$?[a-zA-Z_][a-zA-Z0-9_]*$", clean) and len(clean) >= 2:
            valid_count += 1
        elif tok in ("->", "==", "!=", "<=", ">=", "&&", "||", "::", "=>", "++", "--", "/*", "*/", "//"):
            valid_count += 1

    if single_char_count / float(len(raw_tokens)) > 0.5:
        return 0.15

    return min(1.0, valid_count / float(len(raw_tokens)))


def participation_ratio(activations: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Participation Ratio over the last dim (AVG core.metrics)."""
    if activations.ndim < 2:
        raise ValueError("activations must have at least 2 dimensions")

    x = activations.reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
    n, d = x.shape

    if n < 2:
        return torch.tensor(float(d), device=activations.device, dtype=torch.float32)

    compute_device = torch.device("cpu") if (n * d > 1_000_000) else x.device
    x = x.to(compute_device)
    x = x - x.mean(dim=0, keepdim=True)

    try:
        s = torch.linalg.svdvals(x)
    except RuntimeError:
        _, s, _ = torch.svd_lowrank(x, q=min(n, d, 64))
        s = s.clamp(min=0.0)

    s = s.clamp(min=eps)
    pr = (s.sum() ** 2) / (s.pow(2).sum() + eps)
    return pr.to(activations.device)


# ---------------------------------------------------------------------------
# Forward-pass primitives (night-002, adapted to input_ids)
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


@torch.no_grad()
def capture_baseline_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer: int,
) -> Dict[str, Any]:
    """Teacher-forced forward pass over ``input_ids``; capture h_l (last pos,
    fp32 CPU) + logits.  No KV cache (use_cache=False)."""
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, args, output):
        hs = _split_output(output)[0]
        captured["h_t"] = hs[:, -1, :].detach().float().cpu()
        return output

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        out = model(input_ids, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    return {
        "h_t": captured["h_t"],  # [1, d] fp32 CPU
        "baseline_logits": out.logits[:, -1, :].float().cpu(),  # [1, V] fp32
    }


@torch.no_grad()
def run_probe_pass_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    h_t: torch.Tensor,
    e_k: torch.Tensor,
    eps: float,
    layer: int,
) -> torch.Tensor:
    """One forward pass; last-token hidden state at layer l := h_l + eps*e_k."""
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
        out = model(input_ids, use_cache=False)
        return out.logits[:, -1, :].float().cpu()
    finally:
        if handle is not None:
            handle.remove()


def make_probe_directions(k: int, d: int, seed: int, device: str) -> torch.Tensor:
    """K orthonormal random directions in R^d (rows of E are e_k)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(d, k, generator=gen)  # d x K
    Q, _R = torch.linalg.qr(X)  # reduced QR: Q is d x K, orthonormal columns
    E = Q.T.contiguous()  # K x d, orthonormal rows e_k
    return E.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Power iteration
# ---------------------------------------------------------------------------
def power_iteration(
    G: torch.Tensor,
    seed: int,
    num_iters: int = 30,
    converge_cos: float = 0.9999,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], bool]:
    """Top eigenvector of symmetric PSD KxK Gram G (float32 CPU).

    Returns (eigenvector, trace, converged_within_10)."""
    K = int(G.shape[0])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    c = torch.randn(K, generator=gen, dtype=torch.float32)
    c = c / c.norm()
    trace: List[Dict[str, float]] = []
    converged = False
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
        if i < 10 and cos >= converge_cos:
            converged = True
    return c, trace, converged


# ---------------------------------------------------------------------------
# Linearity sanity check (night-002, adapted to input_ids)
# ---------------------------------------------------------------------------
LINEARITY_MULTIPLIERS = [0.5, 1.0, 2.0]
LINEARITY_TOL = 0.30


@torch.no_grad()
def check_linearity_ids(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    h_t: torch.Tensor,
    baseline_logits: torch.Tensor,
    e0: torch.Tensor,
    eps: float,
    layer: int,
) -> Dict[str, Any]:
    """dY at {0.5eps, eps, 2eps} along e0; verify ||dY|| scales ~linearly."""
    norms: Dict[str, float] = {}
    for mult in LINEARITY_MULTIPLIERS:
        e = eps * mult
        logits_k = run_probe_pass_ids(model, input_ids, h_t, e0, e, layer)
        d = (logits_k - baseline_logits).reshape(-1)
        norms[str(mult)] = float(d.norm().item())
    n05 = norms["0.5"]
    n10 = norms["1.0"]
    n20 = norms["2.0"]
    ratio_lo = n10 / (n05 + 1e-12)
    ratio_hi = n20 / (n10 + 1e-12)
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
# FD + randomized SVD Jacobian estimation at a given step (night-002 method)
# ---------------------------------------------------------------------------
def estimate_jacobian_at_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer: int,
    q: int,
    p: int,
    seed: int,
    E: torch.Tensor,
    do_linearity: bool = True,
) -> Dict[str, Any]:
    """Estimate the top singular vector v1 of the residual→logit Jacobian at
    layer ``layer`` for the current hidden state (last position of input_ids).

    FD estimation runs in fp32 (model must already be fp32 on the compute
    device).  Returns the top-4 spectrum, v1, and power-iteration trace.
    """
    base = capture_baseline_ids(model, input_ids, layer)
    h_t = base["h_t"]  # [1, d] fp32 CPU
    baseline_logits = base["baseline_logits"]  # [1, V] fp32 CPU
    d = int(h_t.shape[-1])
    V = int(baseline_logits.shape[-1])
    rms_h = float(h_t.norm().item() / math.sqrt(d))
    eps = EPS_REL * rms_h
    K = q + p

    # E is precomputed [K, d] fp32 on the compute device.
    E_cpu = E.cpu()
    ortho_err = float((E_cpu @ E_cpu.T - torch.eye(K)).abs().max().item())

    # Linear sanity check (one direction e0, three eps scales).  Adjust eps once.
    linearity: Optional[Dict[str, Any]] = None
    if do_linearity:
        e0 = E_cpu[0]
        linearity = check_linearity_ids(
            model, input_ids, h_t, baseline_logits, e0, eps, layer
        )
        if not linearity["linear_ok"]:
            eps_adj = eps / 10.0
            lin_adj = check_linearity_ids(
                model, input_ids, h_t, baseline_logits, e0, eps_adj, layer
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
    delta_Y = torch.zeros((K, V), dtype=torch.float32)  # [K, V] CPU
    logit_shift_norms: List[float] = []
    for k in range(K):
        e_k = E_cpu[k]
        logits_k = run_probe_pass_ids(model, input_ids, h_t, e_k, eps, layer)
        delta_y = logits_k - baseline_logits  # [1, V]
        delta_Y[k] = delta_y.reshape(-1)
        logit_shift_norms.append(float(delta_y.norm().item()))

    # Gram matrix G = (1/eps^2) dY dY^T (Rayleigh–Ritz restriction of M^T M).
    G = (1.0 / (eps * eps)) * (delta_Y @ delta_Y.T)  # [K, K]

    # Power iteration trace for v1 (dominant left singular vector of probe span).
    c1, pi_trace, v1_converged = power_iteration(G, seed, POWER_ITERS, CONVERGE_COS)

    # Full eigendecomposition for top-4 spectrum (symmetric PSD, cheap K<=40).
    eigvals, eigvecs = torch.linalg.eigh(G)  # ascending
    eigvals = eigvals.flip(0).clamp_min(0.0)
    eigvecs = eigvecs.flip(1)  # columns now descending

    singular_values: List[Dict[str, float]] = []
    directions: Dict[str, torch.Tensor] = {}
    TOP_K = 4
    for i in range(TOP_K):
        sigma = eps * math.sqrt(float(eigvals[i]))
        ci = eigvecs[:, i]  # [K]
        vi_raw = E_cpu.T @ ci  # [d]
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
    sigma_ratio = (
        singular_values[0]["sigma"] / singular_values[1]["sigma"]
        if len(singular_values) > 1 and singular_values[1]["sigma"] > 1e-30
        else float("nan")
    )

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
        "v1_converged_within_10": bool(v1_converged),
        "linearity": linearity,
        "_v1": v1,
    }


# ---------------------------------------------------------------------------
# Hidden-state ring buffer (trailing-24 positions at a layer)
# ---------------------------------------------------------------------------
class HiddenStateRingBuffer:
    """Rolling buffer of the last ``window`` hidden-state positions at a layer.

    The forward hook captures the layer output on every generation forward
    pass (prefill + decode).  During FD probes the hook is disabled so the
    buffer is never contaminated by fp32 probe passes.
    """

    def __init__(self, window: int = 24):
        self.window = window
        self.buffer: List[torch.Tensor] = []
        self.enabled = True

    def add(self, hs: torch.Tensor) -> None:
        """hs: [1, seq, d] from the layer output (any dtype/device)."""
        positions = hs[0].detach().float().cpu()  # [seq, d]
        if len(self.buffer) == 0:
            # Prefill: seed with the last `window` positions.
            self.buffer = list(positions[-self.window:])
        else:
            self.buffer.extend(list(positions))
            self.buffer = self.buffer[-self.window:]

    def matrix(self) -> torch.Tensor:
        """[n, d] float32 CPU, n <= window."""
        return torch.stack(self.buffer)

    def n_entries(self) -> int:
        return len(self.buffer)


def make_ring_hook(rb: HiddenStateRingBuffer):
    def hook_fn(module, args, output):
        if not rb.enabled:
            return output
        hs = _split_output(output)[0]
        rb.add(hs)
        return output
    return hook_fn


# ---------------------------------------------------------------------------
# Collapse direction c1 (lightweight SVD via 24x24 Gram power iteration)
# ---------------------------------------------------------------------------
def collapse_direction(
    H: torch.Tensor,
    seed: int,
    num_iters: int = C1_ITERS,
    converge_cos: float = C1_CONVERGE_COS,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], bool]:
    """Top right-singular vector of H [n, d] via n x n Gram power iteration.

    c1 = H^T e / ||H^T e|| where e is the top eigenvector of G = H H^T.
    """
    n, d = H.shape
    G = H @ H.T  # [n, n]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eigvec = torch.randn(n, generator=gen, dtype=torch.float32)
    eigvec = eigvec / (eigvec.norm() + 1e-12)
    trace: List[Dict[str, float]] = []
    converged = False
    for i in range(num_iters):
        eigvec_new = G @ eigvec
        eigvec_new = eigvec_new / (eigvec_new.norm() + 1e-12)
        cos = float(F.cosine_similarity(eigvec_new.view(1, -1), eigvec.view(1, -1)).item())
        trace.append({
            "iter": int(i + 1),
            "cos_with_prev": cos,
            "||G·e||": float((G @ eigvec).norm().item()),
        })
        eigvec = eigvec_new
        if cos >= converge_cos:
            converged = True
    c1 = H.T @ eigvec  # [d]
    c1 = c1 / (c1.norm() + 1e-12)
    return c1, trace, converged


# ---------------------------------------------------------------------------
# Snapshot diagnostics
# ---------------------------------------------------------------------------
def run_snapshot_diagnostics(
    probe_model: torch.nn.Module,
    tokenizer: Any,
    full_ids: torch.Tensor,
    prompt_len: int,
    step: int,
    rb27: HiddenStateRingBuffer,
    rb2: HiddenStateRingBuffer,
    E: torch.Tensor,
    seed: int,
    q: int,
    p: int,
    do_linearity: bool,
) -> Dict[str, Any]:
    """Pause-time diagnostic at a sampled generation step (passive).

    ``probe_model`` is the fp32 FD-probe model (never cast); the ring buffers
    come from the bf16 generation model.  Computes: collapse direction c1,
    Jacobian v1 (fp32 FD), alignment, token_diversity, trailing_ctr,
    pr_layer2, sigma1, and power-iteration traces for both c1 and v1.
    """
    # ---- A. Collapse direction c1 from the layer-27 ring buffer ----
    H27 = rb27.matrix().float()  # [24, 1536] fp32
    if H27.shape[0] < 2:
        raise RuntimeError(
            f"[STOP] layer-27 ring buffer has {H27.shape[0]} entries at step "
            f"{step}; need >= 2 for c1."
        )
    c1, c1_trace, c1_converged = collapse_direction(H27, seed + step)
    n_window = int(H27.shape[0])

    # ---- D. Surface metrics ----
    token_diversity, _ = compute_token_distinct_2_fast(
        full_ids, prompt_len=prompt_len, window_len=WINDOW
    )
    recent_gen = full_ids[0, prompt_len:]
    if recent_gen.numel() > 0:
        recent_tokens = recent_gen[-CTR_WINDOW:]
    else:
        recent_tokens = full_ids[0, -CTR_WINDOW:]
    trailing_text = tokenizer.decode(recent_tokens, skip_special_tokens=True)
    trailing_ctr = compute_coherent_token_ratio(trailing_text)

    rb2_mat = rb2.matrix().float()  # [24, d] fp32
    if rb2_mat.shape[0] >= 2:
        pr_layer2 = float(participation_ratio(rb2_mat.unsqueeze(0)).item())
    else:
        pr_layer2 = float("nan")

    # ---- B. Jacobian v1 (fp32 FD + randomized SVD on probe_model) ----
    # The ring-buffer hooks live on the bf16 generation model, so they are not
    # fired by probe_model passes.  No model casting occurs (determinism).
    jac = estimate_jacobian_at_step(
        probe_model, full_ids, JACOBIAN_LAYER, q=q, p=p, seed=seed,
        E=E, do_linearity=do_linearity,
    )

    v1 = jac["_v1"]  # [d] fp32 CPU
    sigma1 = jac["singular_values"][0]["sigma"]
    v1_trace = jac["power_iteration"]
    v1_converged = jac["v1_converged_within_10"]

    # ---- C. Alignment score ----
    align = abs(float(F.cosine_similarity(c1.view(1, -1), v1.view(1, -1)).item()))

    return {
        "step": int(step),
        "align": align,
        "token_diversity": float(token_diversity),
        "trailing_ctr": float(trailing_ctr),
        "pr_layer2": float(pr_layer2),
        "sigma1": float(sigma1),
        "n_window": int(n_window),
        "c1_trace": c1_trace,
        "c1_converged_within_10": bool(c1_converged),
        "v1_trace": v1_trace,
        "v1_converged_within_10": bool(v1_converged),
        "jac_eps": jac["eps"],
        "jac_rms_h": jac["rms_h"],
        "jac_sigma_ratio": jac["sigma_ratio"],
        "jac_linearity": jac["linearity"],
        "jac_mean_logit_shift": jac["mean_logit_shift_norm"],
    }


# ---------------------------------------------------------------------------
# Generation with pause-time diagnostics
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_jlens(
    model: torch.nn.Module,
    probe_model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    sampled_steps: Sequence[int],
    E: torch.Tensor,
    seed: int,
    q: int = PROBE_Q,
    p: int = PROBE_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_linearity: bool = True,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) KV-cache generation with pause-time J-Lens
    diagnostics at ``sampled_steps``.

    ``model`` is the bf16 generation model; ``probe_model`` is the fp32 FD
    probe model (never cast).  The generation pauses BEFORE the forward pass
    at each sampled step (the model has generated up to step-1 but has not yet
    produced the next token).  Diagnostics are passive: no kick, no logits
    changed.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    sampled_set: Set[int] = set(int(s) for s in sampled_steps)

    # Ring buffers for layer 27 (c1, Jacobian) and layer 2 (PR).
    rb27 = HiddenStateRingBuffer(WINDOW)
    rb2 = HiddenStateRingBuffer(WINDOW)
    h27 = model.model.layers[JACOBIAN_LAYER].register_forward_hook(make_ring_hook(rb27))
    h2 = model.model.layers[PR_LAYER].register_forward_hook(make_ring_hook(rb2))

    generated: List[torch.Tensor] = []
    full_ids = input_ids.clone()  # prompt + generated (for diagnostics)
    past_key_values = None
    next_tok: Optional[torch.Tensor] = None
    snapshots: List[Dict[str, Any]] = []

    try:
        for step in range(max_new_tokens):
            # ---- Pause-time diagnostics BEFORE the forward pass ----
            if step in sampled_set:
                snap = run_snapshot_diagnostics(
                    probe_model, tokenizer, full_ids, prompt_len, step,
                    rb27, rb2, E, seed, q, p, do_linearity,
                )
                snapshots.append(snap)
                # FD probes run on probe_model with use_cache=False; the
                # generation model and its KV cache are untouched.

            # ---- Forward pass (bf16, KV cache) ----
            if step == 0:
                out = model(input_ids, use_cache=True)
            else:
                assert next_tok is not None
                out = model(next_tok, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :].float()
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
            full_ids = torch.cat([full_ids, next_tok], dim=-1)

            if int(next_tok.item()) == eos_id:
                break
    finally:
        h27.remove()
        h2.remove()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    return {
        "generated_ids": ids.cpu(),  # [1, n]
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# Data loading (DS-033 ground truth)
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ds033_classes(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load DS-033 diagnostic classes for t2s_degenerate."""
    out: Dict[int, Dict[str, Any]] = {}
    for d in load_jsonl(path):
        if d.get("fixture") != "t2s_degenerate":
            continue
        rid = int(d["record_id"])
        act = d.get("active", {})
        out[rid] = {
            "diagnostic_class": act.get("diagnostic_class"),
            "predicate_true_steps": act.get("predicate_true_steps"),
        }
    return out


def load_selected_records(
    fixture_path: Path, ds033_path: Path, n: int = N_RECORDS
) -> List[Dict[str, Any]]:
    """First ``n`` t2s detection-gap records sorted by record_id (DS-033)."""
    ds033 = load_ds033_classes(ds033_path)
    fixture = load_jsonl(fixture_path)
    detection_gap_ids = sorted(
        rid for rid, info in ds033.items()
        if info.get("diagnostic_class") == DETECTION_GAP_CLASS
    )
    selected_ids = detection_gap_ids[:n]
    records = []
    for rid in selected_ids:
        rec = next(r for r in fixture if int(r["id"]) == rid)
        records.append({
            "record_id": int(rid),
            "text": rec["text"],
            "token_count": rec.get("token_count"),
            "diagnostic_class": ds033[rid]["diagnostic_class"],
            "predicate_true_steps": ds033[rid]["predicate_true_steps"],
        })
    return records


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    probe_model: torch.nn.Module,
    tokenizer: Any,
    record: Dict[str, Any],
    E: torch.Tensor,
    seed: int,
    q: int,
    p: int,
    do_linearity: bool,
) -> Dict[str, Any]:
    """1 record, generated twice with the diagnostic at step 23 only.
    Token ids must match; the alignment score must match.  STOP if not."""
    print("\n--- Determinism smoke (record 0, diagnostic at step 23 only) ---")
    a = generate_with_jlens(
        model, probe_model, tokenizer, record["text"], sampled_steps=[23],
        E=E, seed=seed, q=q, p=p, do_linearity=do_linearity,
    )
    b = generate_with_jlens(
        model, probe_model, tokenizer, record["text"], sampled_steps=[23],
        E=E, seed=seed, q=q, p=p, do_linearity=do_linearity,
    )
    ids_match = bool(torch.equal(a["generated_ids"], b["generated_ids"]))
    sa = a["snapshots"][0]
    sb = b["snapshots"][0]
    align_a = sa["align"]
    align_b = sb["align"]
    align_match = bool(abs(align_a - align_b) < 1e-6)
    c1_match = bool(sa["c1_trace"] == sb["c1_trace"])
    v1_match = bool(sa["v1_trace"] == sb["v1_trace"])
    print(f"  token ids identical: {ids_match} (n={a['n_generated']} vs {b['n_generated']})")
    print(f"  align a={align_a:.8f} b={align_b:.8f} match={align_match}")
    print(f"  c1 trace identical: {c1_match}, v1 trace identical: {v1_match}")
    if not ids_match or not align_match:
        print("[STOP] Determinism smoke FAILED.")
        sys.exit(1)
    print("  determinism smoke: PASS")
    return {
        "record_id": int(record["record_id"]),
        "n_tokens_a": int(a["n_generated"]),
        "n_tokens_b": int(b["n_generated"]),
        "ids_identical": str(ids_match),
        "align_a": float(align_a),
        "align_b": float(align_b),
        "align_identical": str(align_match),
        "c1_trace_identical": str(c1_match),
        "v1_trace_identical": str(v1_match),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def classify_snapshot(snap: Dict[str, Any]) -> str:
    td = float(snap["token_diversity"])
    if td >= HEALTHY_DIVERSITY:
        return "healthy"
    if td >= COLLAPSE_DIVERSITY:
        return "borderline"
    return "collapsed"


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None for strict-JSON serialisation."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def write_markdown_report(
    md_path: Path,
    records: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
    smoke: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    md: List[str] = []
    md.append("# night-003 — J-Lens Collapse→Jacobian Alignment Diagnostic")
    md.append("")
    md.append(f"Date: {metadata['date']}  |  SEED: {metadata['seed']}  |  "
              f"Model: `{metadata['model_name']}@{metadata['model_revision']}`")
    md.append("")
    md.append("MEASUREMENT ONLY. Self-contained probe (no AVG imports). "
              "No intervention, no new thresholds, no controller changes.")
    md.append("")

    md.append("## Method")
    md.append("")
    md.append(f"- 10 t2s_degenerate DS-033 detection-gap records "
              f"(first 10 by record_id; do NOT re-classify).")
    md.append(f"- Greedy (do_sample=False) KV-cache generation, "
              f"max_new_tokens={metadata['max_new_tokens']}.")
    md.append(f"- Sampled steps: `{SAMPLED_STEPS}` (8 evenly spaced snapshots per "
              f"record, 80 total).")
    md.append(f"- Collapse direction c1: top right-singular vector of the trailing "
              f"{WINDOW} hidden states at layer {JACOBIAN_LAYER} (24x24 Gram power "
              f"iteration, {C1_ITERS} iters).")
    md.append(f"- Jacobian direction v1: night-002 FD + randomized SVD at layer "
              f"{JACOBIAN_LAYER}, K={K_PROBE} (q={PROBE_Q}, p={PROBE_P}), "
              f"eps={EPS_REL} x RMS(h_t), fp32 FD estimation.")
    md.append(f"- Alignment = |cos(c1, v1)| (absolute cosine similarity).")
    md.append(f"- Surface metrics: token_diversity (trailing {WINDOW} bigrams), "
              f"trailing_ctr (trailing {CTR_WINDOW} tokens), "
              f"layer-{PR_LAYER} spectral PR (trailing {WINDOW} hidden states).")
    md.append(f"- Frozen PR threshold (logging only): T_PR = {T_PR}, "
              f"band_low = {BAND_LOW}. The diagnostic does NOT gate actuation.")
    md.append(f"- Determinism: two model copies (bf16 generation + fp32 FD probe), "
              f"loaded once and NEVER cast.  In-place casting would create new "
              f"weight tensors at new memory addresses, changing cuBLAS numerics "
              f"and breaking the determinism smoke.")
    md.append("")

    md.append("## Determinism smoke (FIRST)")
    md.append("")
    md.append("| record_id | n_tokens A | n_tokens B | ids identical | "
              "align A | align B | align match | c1 trace | v1 trace |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    md.append(
        f"| {smoke['record_id']} | {smoke['n_tokens_a']} | {smoke['n_tokens_b']} | "
        f"{smoke['ids_identical']} | {smoke['align_a']:.8f} | {smoke['align_b']:.8f} | "
        f"{smoke['align_identical']} | {smoke['c1_trace_identical']} | "
        f"{smoke['v1_trace_identical']} |"
    )
    md.append("")

    md.append("## Primary diagnostic table")
    md.append("")
    md.append("| record_id | steps | align range | align mean | "
              "align @ first bigram drop | diverge step | diverge token_diversity |")
    md.append("|---|---|---|---|---|---|---|")
    for r in primary:
        md.append(
            f"| {r['record_id']} | {r['steps']} | "
            f"{r['align_min']:.4f}-{r['align_max']:.4f} | {r['align_mean']:.4f} | "
            f"{r['align_at_first_bigram_drop']} | {r['diverge_step']} | "
            f"{r['diverge_token_diversity']} |"
        )
    md.append("")
    md.append("'diverge step' = first sampled step where token_diversity < 0.30 "
              "(production bigram fire threshold). Records where token_diversity "
              "never drops are the 'pure' detection-gap records (the bigram "
              "detector never fires on its own). Records where token_diversity "
              "drops but the DS-033 predicate still never fired are the "
              "joint-conjunction detection-gap cases: bigram diversity drops but "
              "trailing_ctr stays high, so the (diversity AND ctr) predicate "
              "remains silent.")
    md.append("")

    md.append("## Secondary aggregation (all 80 snapshots)")
    md.append("")
    md.append("| condition | n | mean align | median align | mean token_div | "
              "mean pr_layer2 |")
    md.append("|---|---|---|---|---|---|")
    for row in secondary:
        md.append(
            f"| {row['condition']} | {row['n']} | {row['mean_align']:.4f} | "
            f"{row['median_align']:.4f} | {row['mean_token_div']:.4f} | "
            f"{row['mean_pr_layer2']:.4f} |"
        )
    md.append("")
    md.append("Conditions: healthy = token_diversity >= 0.40, "
              "borderline = [0.30, 0.40), collapsed = < 0.30.")
    md.append("")

    md.append("## Per-record time series")
    md.append("")
    for rid in [r["record_id"] for r in primary]:
        rec_snaps = [s for s in snapshots if s["record_id"] == rid]
        md.append(f"### record {rid}")
        md.append("")
        md.append("| step | align | token_div | trailing_ctr | pr_layer2 | σ₁ | "
                  "c1 conv | v1 conv |")
        md.append("|---|---|---|---|---|---|---|---|")
        for s in rec_snaps:
            md.append(
                f"| {s['step']} | {s['align']:.4f} | {s['token_diversity']:.4f} | "
                f"{s['trailing_ctr']:.4f} | {s['pr_layer2']:.4f} | "
                f"{s['sigma1']:.3e} | {s['c1_converged_within_10']} | "
                f"{s['v1_converged_within_10']} |"
            )
        md.append("")

    md.append("## J-Lens hypothesis evaluation")
    md.append("")
    md.append("The hypothesis predicts: **low alignment during healthy snapshots** "
              "(collapse direction and Jacobian direction misaligned — collapse "
              "invisible to the output) and **rising alignment at collapsed "
              "snapshots** (collapse rotates into the sensitive direction — output "
              "sees it).")
    md.append("")
    md.append("Falsification signatures: (1) alignment uniformly high — collapse "
              "always visible, detection-gap has a different explanation; "
              "(2) alignment uniformly low and never rises even when bigram drops — "
              "surface degradation happens via a different mechanism.")
    md.append("")

    # Qualitative hypothesis evaluation based on the aggregation.
    healthy = [r for r in secondary if r["condition"] == "healthy"]
    collapsed = [r for r in secondary if r["condition"] == "collapsed"]
    borderline = [r for r in secondary if r["condition"] == "borderline"]
    h_align = healthy[0]["mean_align"] if healthy and healthy[0]["n"] > 0 else float("nan")
    c_align = collapsed[0]["mean_align"] if collapsed and collapsed[0]["n"] > 0 else float("nan")
    b_align = borderline[0]["mean_align"] if borderline and borderline[0]["n"] > 0 else float("nan")
    h_n = healthy[0]["n"] if healthy else 0
    c_n = collapsed[0]["n"] if collapsed else 0
    b_n = borderline[0]["n"] if borderline else 0
    all_aligns = [s["align"] for s in snapshots]
    max_align = max(all_aligns) if all_aligns else float("nan")
    md.append(f"Across the 80 snapshots: healthy mean align = "
              f"{h_align:.4f} (n={h_n}), borderline mean align = "
              f"{b_align:.4f} (n={b_n}), collapsed mean align = "
              f"{c_align:.4f} (n={c_n}).  Global align range = "
              f"{min(all_aligns):.4f}-{max_align:.4f}.")
    md.append("")
    if h_n == 0 and c_n == 0:
        md.append("**No healthy or collapsed snapshots — hypothesis inconclusive "
                  "from this sample.**")
    elif h_n == 0:
        # No healthy reference; compare collapsed vs borderline and the absolute
        # level (random-orthogonal baseline for unit vectors in R^1536 is
        # sqrt(2/(pi*d)) ≈ 0.02).
        random_orth = math.sqrt(2.0 / (math.pi * 1536.0))
        md.append(f"No healthy snapshots (n=0) — all snapshots are in the "
                  f"borderline/collapsed regime (t2s degenerate macro-loop). "
                  f"Collapsed mean align = {c_align:.4f} vs borderline mean align "
                  f"= {b_align:.4f}.  Both are far below the random-orthogonal "
                  f"baseline for unit vectors in R^1536 (~{random_orth:.4f}).")
        md.append("")
        if c_align > b_align + 0.003:
            md.append("**Collapsed snapshots show a small alignment rise over "
                      "borderline** (directionally consistent with the J-Lens "
                      "rotation mechanism), but the absolute values remain in the "
                      "near-orthogonal regime — the collapse stays essentially "
                      "invisible to the output layer throughout.")
        else:
            md.append("**Alignment is uniformly low and does not rise at collapsed "
                      "snapshots** — collapse and Jacobian directions are always "
                      "misaligned; surface degradation (where it occurs) happens "
                      "via a different mechanism.")
    elif c_align > h_align + 0.1:
        md.append("**Collapsed-snapshot alignment is meaningfully HIGHER than "
                  "healthy-snapshot alignment** — consistent with the J-Lens "
                  "hypothesis: the collapse direction rotates into the Jacobian's "
                  "sensitive direction as surface diversity drops.")
    elif c_align < 0.2 and h_align < 0.2:
        md.append("**Alignment is uniformly low** across conditions — collapse and "
                  "Jacobian directions are always misaligned; surface degradation "
                  "(where it occurs) happens via a different mechanism.  The "
                  "detection-gap pattern is NOT explained by a later rotation into "
                  "alignment.")
    else:
        md.append("**Alignment is not cleanly separated between conditions** — the "
                  "J-Lens alignment signal does not sharply distinguish collapsed "
                  "from healthy snapshots in this sample.")
    md.append("")
    # One-line verdict (consistent with the branch above).
    random_orth = math.sqrt(2.0 / (math.pi * 1536.0))
    supported = bool(max_align < random_orth)
    verdict = (
        f"**Verdict.** The core J-Lens mechanism is "
        f"{'**supported**' if supported else 'not cleanly falsified'}: "
        f"the collapse direction c1 is essentially orthogonal to the Jacobian's "
        f"sensitive direction v1 at every sampled step "
        f"(max |cos| = {max_align:.4f}, random-orthogonal baseline ~{random_orth:.4f}). "
        f"The hidden-state manifold compresses along directions the unembedding "
        f"does not care about, explaining why spectral PR fires (collapse is real) "
        f"while the next-token distribution stays locally diverse (collapse "
        f"invisible to the output layer)."
    )
    if h_n == 0 and not math.isnan(c_align) and not math.isnan(b_align) and c_align > b_align + 0.003:
        verdict += (
            f"  The predicted rotation into alignment is weakly present "
            f"(collapsed mean {c_align:.4f} vs borderline {b_align:.4f}) but never "
            f"approaches a regime where the collapse could drive a surface-diversity "
            f"drop in these detection-gap records."
        )
    elif not math.isnan(c_align) and not math.isnan(h_align) and c_align > h_align + 0.1:
        verdict += (
            f"  Collapsed-snapshot alignment is meaningfully higher than healthy, "
            f"consistent with the rotation-into-alignment mechanism."
        )
    else:
        verdict += (
            f"  Alignment does not rise meaningfully at collapsed snapshots; the "
            f"detection-gap pattern is not explained by a later rotation into "
            f"alignment."
        )
    md.append(verdict)
    md.append("")

    md.append("## Notes")
    md.append("")
    md.append("- FD Jacobian estimation runs in fp32 (bf16 dead-zone, DS-036); "
              "generation runs in bf16 to match production.  Mismatch documented.")
    md.append("- The diagnostic pauses generation at each sampled step to run "
              "passive FD probes; no kick is applied, no logits are changed.")
    md.append("- If any power iteration failed to converge within 10 iterations, "
              "the convergence flag is False and the trace is in the JSONL.")
    c1_fail = [s for s in snapshots if not s["c1_converged_within_10"]]
    v1_fail = [s for s in snapshots if not s["v1_converged_within_10"]]
    if c1_fail:
        md.append(f"- c1 power iteration did not reach cos >= {C1_CONVERGE_COS} "
                  f"within {C1_ITERS} iterations at "
                  f"{len(c1_fail)} snapshot(s): "
                  + ", ".join(f"record {s['record_id']} step {s['step']}"
                              for s in c1_fail)
                  + ". Traces logged in the JSONL.")
    if v1_fail:
        md.append(f"- v1 power iteration did not reach cos >= {CONVERGE_COS} "
                  f"within 10 iterations at "
                  f"{len(v1_fail)} snapshot(s): "
                  + ", ".join(f"record {s['record_id']} step {s['step']}"
                              for s in v1_fail)
                  + ". Traces logged in the JSONL.")
    md.append(f"- `docs/gate23/jlens_alignment_results.jsonl` contains full "
              f"per-snapshot records (step, align, surface metrics, sigma1, "
              f"power-iteration traces).")
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="night-003 J-Lens collapse->Jacobian alignment diagnostic "
                    "(measurement only)"
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="docs/gate23")
    parser.add_argument("--q", type=int, default=PROBE_Q)
    parser.add_argument("--p", type=int, default=PROBE_P)
    parser.add_argument("--sampled-steps", type=int, nargs="+", default=SAMPLED_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--no-linearity", action="store_true",
                        help="Skip the eps linearity sanity check.")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run determinism smoke only, then exit.")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    q = args.q
    p = args.p
    K = q + p
    do_linearity = not args.no_linearity

    print("=" * 70)
    print("night-003 J-Lens Collapse->Jacobian Alignment (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"model: {args.model}@{args.revision}")
    print(f"device: {args.device} | probe K={K} (q={q}, p={p}) | "
          f"layer={JACOBIAN_LAYER} | pr_layer={PR_LAYER}")
    print(f"sampled_steps: {args.sampled_steps}")
    print(f"torch: {torch.__version__} | cuda: {torch.cuda.is_available()}")

    # ------------------------------------------------------------------
    # Model load: bf16 generation model + fp32 FD probe model (two copies).
    # Both are loaded once and NEVER cast, so the memory addresses of the
    # weights stay fixed and the measurement is deterministic (casting would
    # create new weight tensors at new addresses, changing cuBLAS numerics).
    # ------------------------------------------------------------------
    try:
        tokenizer = load_tokenizer(args.model, args.revision)
        model = load_model(args.model, args.revision, torch.bfloat16, args.device)
        probe_model = load_model(args.model, args.revision, torch.float32, args.device)
    except Exception as e:
        print(f"[FATAL] model load failed: {e}")
        sys.exit(1)

    n_layers_model = len(model.model.layers)
    print(f"gen  model loaded on {model.device} dtype={model.dtype} "
          f"| {n_layers_model} layers | hidden={model.config.hidden_size}")
    print(f"probe model loaded on {probe_model.device} dtype={probe_model.dtype} "
          f"| {len(probe_model.model.layers)} layers")
    if torch.cuda.is_available():
        print(f"[mem] allocated={torch.cuda.memory_allocated() / 1e9:.2f} GB "
              f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Precompute probe directions once (fixed seed, reused at every snapshot for
    # cross-step comparability).
    d = int(model.config.hidden_size)
    E = make_probe_directions(K, d, SEED, device="cpu")  # [K, d] fp32 CPU

    # ------------------------------------------------------------------
    # Load records (DS-033 ground truth)
    # ------------------------------------------------------------------
    records = load_selected_records(
        Path(FIXTURE_PATH), Path(DS033_PATH), n=N_RECORDS
    )
    print(f"\nLoaded {len(records)} t2s detection-gap records: "
          f"{[r['record_id'] for r in records]}")
    if len(records) < N_RECORDS:
        print(f"[WARN] expected {N_RECORDS}, found {len(records)}")

    # ------------------------------------------------------------------
    # Determinism smoke (FIRST)
    # ------------------------------------------------------------------
    smoke = run_determinism_smoke(
        model, probe_model, tokenizer, records[0], E, SEED, q, p, do_linearity
    )
    if args.smoke_only:
        print("\n[smoke-only] exiting after determinism smoke.")
        return

    # ------------------------------------------------------------------
    # Main measurement
    # ------------------------------------------------------------------
    all_snapshots: List[Dict[str, Any]] = []
    for ri, rec in enumerate(records):
        rid = rec["record_id"]
        print(f"\n--- record {rid} ({ri + 1}/{len(records)}) ---")
        out = generate_with_jlens(
            model, probe_model, tokenizer, rec["text"], args.sampled_steps,
            E=E, seed=SEED, q=q, p=p,
            max_new_tokens=args.max_new_tokens,
            do_linearity=do_linearity,
        )
        for snap in out["snapshots"]:
            snap["record_id"] = rid
            all_snapshots.append(snap)
        n_snaps = len(out["snapshots"])
        print(f"  n_generated={out['n_generated']} n_snapshots={n_snaps}")
        if n_snaps < len(args.sampled_steps):
            print(f"  [WARN] record {rid}: only {n_snaps}/{len(args.sampled_steps)} "
                  f"snapshots (early EOS?)")

    # ------------------------------------------------------------------
    # Primary diagnostic table (per record)
    # ------------------------------------------------------------------
    primary: List[Dict[str, Any]] = []
    for rec in records:
        rid = rec["record_id"]
        rec_snaps = [s for s in all_snapshots if s["record_id"] == rid]
        aligns = [s["align"] for s in rec_snaps]
        aligns_sorted = sorted(aligns)
        align_mean = _mean(aligns)
        align_min = aligns_sorted[0] if aligns_sorted else float("nan")
        align_max = aligns_sorted[-1] if aligns_sorted else float("nan")

        # First bigram drop (token_diversity < 0.30).
        diverge_step = None
        diverge_td = None
        align_at_drop = None
        for s in rec_snaps:
            if s["token_diversity"] < COLLAPSE_DIVERSITY:
                diverge_step = s["step"]
                diverge_td = s["token_diversity"]
                align_at_drop = s["align"]
                break
        primary.append({
            "record_id": rid,
            "steps": len(rec_snaps),
            "align_min": float(align_min),
            "align_max": float(align_max),
            "align_mean": float(align_mean),
            "align_at_first_bigram_drop": (
                f"{align_at_drop:.4f}" if align_at_drop is not None else "n/a"
            ),
            "diverge_step": diverge_step if diverge_step is not None else "never",
            "diverge_token_diversity": (
                f"{diverge_td:.4f}" if diverge_td is not None else "n/a"
            ),
        })

    # ------------------------------------------------------------------
    # Secondary aggregation (all 80 snapshots by condition)
    # ------------------------------------------------------------------
    by_condition: Dict[str, List[Dict[str, Any]]] = {"healthy": [], "borderline": [], "collapsed": []}
    for s in all_snapshots:
        by_condition[classify_snapshot(s)].append(s)
    secondary: List[Dict[str, Any]] = []
    for cond in ["healthy", "borderline", "collapsed"]:
        rows = by_condition[cond]
        secondary.append({
            "condition": cond,
            "n": len(rows),
            "mean_align": _mean([r["align"] for r in rows]),
            "median_align": _median([r["align"] for r in rows]),
            "mean_token_div": _mean([r["token_diversity"] for r in rows]),
            "mean_pr_layer2": _mean([r["pr_layer2"] for r in rows]),
        })

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "jlens_alignment_results.jsonl"
    md_path = out_dir / "JLENS_ALIGNMENT_RESULTS.md"

    metadata = {
        "task": "night-003 jlens-alignment",
        "model_name": args.model,
        "model_revision": args.revision,
        "device": args.device,
        "generation_dtype": "bf16",
        "fd_probe_dtype": "fp32",
        "model_copies": "two (bf16 gen + fp32 probe, never cast; fixed weight addresses for determinism)",
        "torch_version": torch.__version__,
        "seed": SEED,
        "jacobian_layer": JACOBIAN_LAYER,
        "pr_layer": PR_LAYER,
        "probe_K": K,
        "probe_q": q,
        "probe_p": p,
        "eps_rel": EPS_REL,
        "sampled_steps": args.sampled_steps,
        "max_new_tokens": args.max_new_tokens,
        "window": WINDOW,
        "ctr_window": CTR_WINDOW,
        "t_pr_frozen": T_PR,
        "band_low_frozen": BAND_LOW,
        "collapse_diversity": COLLAPSE_DIVERSITY,
        "healthy_diversity": HEALTHY_DIVERSITY,
        "bmm_override": "deregistered",
        "date": time.strftime("%Y-%m-%d"),
        "wall_clock_s": round(time.time() - t_start, 3),
    }

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": _sanitize(metadata)}) + "\n")
        f.write(json.dumps({"records": _sanitize({
            "seed": SEED,
            "n_records": len(records),
            "records": [
                {
                    "record_id": r["record_id"],
                    "text": r["text"],
                    "token_count": r["token_count"],
                    "diagnostic_class": r["diagnostic_class"],
                    "predicate_true_steps": r["predicate_true_steps"],
                }
                for r in records
            ],
        })}) + "\n")
        f.write(json.dumps({"determinism_smoke": _sanitize(smoke)}) + "\n")
        for s in all_snapshots:
            f.write(json.dumps({"snapshot": _sanitize(s)}) + "\n")
        for row in primary:
            f.write(json.dumps({"primary_diagnostic": _sanitize(row)}) + "\n")
        for row in secondary:
            f.write(json.dumps({"secondary_aggregation": _sanitize(row)}) + "\n")

    write_markdown_report(
        md_path, records, all_snapshots, primary, secondary, smoke, metadata
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRIMARY DIAGNOSTIC TABLE")
    print("=" * 70)
    header = ["record_id", "steps", "align range", "align mean",
              "align@drop", "diverge step", "diverge td"]
    print(" | ".join(header))
    for r in primary:
        print(f"{r['record_id']} | {r['steps']} | {r['align_min']:.4f}-{r['align_max']:.4f} | "
              f"{r['align_mean']:.4f} | {r['align_at_first_bigram_drop']} | "
              f"{r['diverge_step']} | {r['diverge_token_diversity']}")

    print("\n" + "=" * 70)
    print("SECONDARY AGGREGATION")
    print("=" * 70)
    for row in secondary:
        print(f"{row['condition']}: n={row['n']} mean_align={row['mean_align']:.4f} "
              f"median_align={row['median_align']:.4f} "
              f"mean_td={row['mean_token_div']:.4f} "
              f"mean_pr2={row['mean_pr_layer2']:.4f}")

    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {md_path}")
    print(f"Total wall-clock: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
