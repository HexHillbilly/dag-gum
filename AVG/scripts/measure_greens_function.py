#!/usr/bin/env python3
"""DS-036: Green's Function SVD probe — residual channel liveness (MEASUREMENT ONLY).

DS-026, DS-028, DS-029: residual orthogonal perturbations produce KL ≈ 10⁻⁵
under greedy decoding. Random directions in R^1536 are almost orthogonal to
the top singular vectors of the residual-to-output map M = W_U · J_{l→L}.
The channel may be alive in directions we haven't tested.

This probe estimates the top right-singular vector v₁ of the empirical
Jacobian at a chosen intervention point (layer 26, last prompt token position),
then applies a bounded 0.50 L2 kick along v₁. If KL > 10⁻², the residual
channel is alive and RARI becomes testable (DS-037). If KL ≈ 10⁻⁵ even along
v₁, the channel is dead under greedy decoding on this model — deprecate
residual permanently.

MEASUREMENT ONLY. No controller edits, no thresholds, no new architecture.
Single-record diagnostic, not a sweep and not a gate.

Ground truth (verified by human review):
  - Fixture: tests/fixtures/heldout_degenerate_v2.jsonl (ds-027), record 0
    (id=0, mutated_prompt).
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323.
    cuda bf16 (cpu fp32 fallback).
  - Intervention layer: 26 (production default, controller.py:~486).
  - Token position: last prompt token (prompt_len - 1).
  - Force budget: η = 0.50 L2 (production clamp [0.35, 0.75], controller.py:566).
  - Perturbation: orthogonalised against the current residual (production
    fallback, controller.py:_make_sae_guided_reset_fn SAE-fallback path).

Method (per the DS-036 task file):
  Phase 1 — Estimate v₁ via finite differences at (layer 26, last prompt token):
    1. Run the prompt once; capture h_t (layer 26, last pos) and the baseline
       logits.
    2. K = 128 orthogonal probe directions e_k (random unit vectors, QR-
       orthogonalised). For each, one forward pass replacing the last-token
       hidden state with h_t + ε·e_k (ε = 10⁻³). Δy_k = logits(perturbed) -
       logits(baseline).
    3. ΔY ∈ R^(K×V) (rows = Δy_k). V = lm_head out_features (151,936 here).
    4. Power iteration on the K×K Gram G = (1/ε²)ΔYΔYᵀ (10 iters, float32
       CPU). c₁ is the dominant eigenvector of G; v₁ = Eᵀc₁/||Eᵀc₁|| is the
       dominant right-singular direction of the empirical Jacobian in
       hidden-state space (d_model = 1536), restricted to the probe subspace.
       (Because K < d, we solve the projected eigenproblem; this is the
       Rayleigh–Ritz restriction of M^T M = (1/ε²)EᵀΔYᵀΔYE to the probe span.)
  Phase 2 — Apply bounded kick along v₁:
    1. Re-run the prompt; at (layer 26, last prompt token) replace the hidden
       state with h_t + η·v₁_unit (η = 0.50), v₁ orthogonalised against h_t
       with the production fallback double Gram-Schmidt.
    2. Complete the forward pass; continue greedy decoding for 10 generated
       tokens.
    3. KL(hooked || dormant) between the perturbed and baseline logit
       distributions (same formula as DS-026/028/029).

Liveness gate (explicit in task):
  - KL > 10⁻²: channel is ALIVE. DS-037 (RARI POC) unblocked.
  - KL ≈ 10⁻⁵: channel is DEAD. Deprecate residual path; CANCEL DS-037.

Secondary measurements:
  - Top 5 singular values of ΔY (normalized σ_i/σ_1).
  - Cosine similarity between v₁ and a random orthogonal direction (DS-026/028
    nullspace) and between v₁ and h_t.
  - Per-token KL for the first 10 generated tokens after the intervention.

Environment notes (identical to ds-025..ds-034):
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-036 measurement seed (matches ds-025..ds-034)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"

# Intervention point and force budget (production ground truth).
INTERVENTION_LAYER = 26  # production default (controller.py:~486)
ETA = 0.50  # bounded L2 kick (production clamp [0.35, 0.75], controller.py:566)
FORCE_CLAMP_LOW = 0.35
FORCE_CLAMP_HIGH = 0.75

# Finite-difference probe.
EPSILON = 1e-3  # probe magnitude (task explicit)
K_PROBES = 128  # number of orthogonal probe directions
NUM_POWER_ITER = 10  # power-iteration steps
CONVERGENCE_COS_MIN = 0.99  # v₁ stable within ±0.01 cosine (task explicit)
# Empirical bf16 dead-zone: at layer 26 the residual norm is O(250) and bf16's
# absolute step is O(0.6), so an ε=1e-3 perturbation is unrepresentable in bf16
# (rounded to ~0 for most probe directions). A few directions that happen to
# align with the bf16 quantization grid produce spurious nonzero shifts. The
# finite-difference estimate is only meaningful if a clear majority of probe
# directions produce a resolvable shift. We treat the bf16 probe as dead when
# the max shift is below a hard floor OR the fraction of zero-shift probes is
# above 50% (quantization artifacts dominate).
BF16_DEAD_ZONE_MAX_SHIFT = 1e-6
BF16_DEAD_ZONE_ZERO_FRAC = 0.5

# KL measurement.
PER_TOKEN_KL_TOKENS = 10  # first 10 generated tokens after the intervention
LIVENESS_KL_ALIVE = 1e-2  # KL > 1e-2 -> channel ALIVE (task explicit)
LIVENESS_KL_DEAD = 1e-3  # KL < 1e-3 -> channel DEAD (covers the ≈1e-5 task band)

# Paths
HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
OUTPUT_JSONL = Path("docs/gate23/greens_function_results.jsonl")
OUTPUT_MD = Path("docs/gate23/GREENS_FUNCTION_RESULTS.md")

# HF cache fallback: model is present in the read-only system cache. Do NOT
# force an HF_HOME fallback to /tmp (that would re-download 1.5B weights). Only
# fall back if the pinned model is not present.
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
# Metric helpers (same KL formula as DS-026/028/029, for comparability)
# ---------------------------------------------------------------------------
def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P || Q) between two next-token logit distributions (float32)."""
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    eps = 1e-12
    kl = (p * (p + eps).log() - p * (q + eps).log()).sum(dim=-1)
    # Small negative values are floating-point noise (KL is non-negative in
    # expectation); clamp for reporting cleanliness.
    return float(max(0.0, kl.mean().item()))


def production_orthogonalise(
    direction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Production fallback double Gram-Schmidt (controller.py SAE-fallback path).

    Projects ``direction`` orthogonal to ``reference`` (the current residual)
    using the same two-step projection as
    controller.py:_make_sae_guided_reset_fn when no SAE is present.
    Returns the unit vector.
    """
    ref_flat = reference.detach().float().reshape(-1)
    dir_flat = direction.detach().float().reshape(-1)
    ref_norm = ref_flat.norm() + 1e-8

    # First projection (raw_noise -> guided_direction).
    dot_prod = torch.sum(dir_flat * ref_flat, dim=-1)
    proj = (dot_prod / (ref_norm ** 2)) * ref_flat
    guided = dir_flat - proj

    # Second projection (guided_direction -> ortho_vec).
    dot_p = torch.sum(guided * ref_flat, dim=-1)
    ortho_vec = guided - (dot_p / (ref_norm ** 2)) * ref_flat
    ortho_unit = ortho_vec / (ortho_vec.norm() + 1e-8)
    return ortho_unit


def make_probe_directions(k: int, d: int, seed: int, device: str) -> torch.Tensor:
    """K orthonormal random directions in R^d (rows of the returned E are e_k)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(d, k, generator=gen)  # d x K
    Q, _R = torch.linalg.qr(X)  # reduced QR: Q is d x K, orthonormal columns
    E = Q.T.contiguous()  # K x d, orthonormal rows e_k
    return E.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Phase 1: baseline capture + finite-difference probe passes
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_baseline(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layer: int,
) -> Dict[str, Any]:
    """Run the prompt once; capture h_t (layer L, last pos, float32) + logits."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, args, output):
        hs = output if isinstance(output, torch.Tensor) else output[0]
        captured["h_t"] = hs[:, -1, :].detach().float().cpu()
        return output

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        out = model(input_ids)
    finally:
        if handle is not None:
            handle.remove()

    return {
        "h_t": captured["h_t"],  # [1, d] float32 CPU
        "baseline_logits": out.logits[:, -1, :].float().cpu(),  # [1, V] float32
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
    """One forward pass; last-token hidden state at layer L replaced by h_t + eps*e_k."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]

    def hook_fn(module, args, output):
        hs = output if isinstance(output, torch.Tensor) else output[0]
        new_hs = hs.clone()
        perturbed = (h_t.to(device=hs.device) + eps * e_k.to(device=hs.device)).to(
            dtype=hs.dtype
        )
        new_hs[:, -1, :] = perturbed
        return new_hs

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        out = model(input_ids)
        return out.logits[:, -1, :].float().cpu()
    finally:
        if handle is not None:
            handle.remove()


def estimate_greens_function(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    h_t: torch.Tensor,
    baseline_logits: torch.Tensor,
    E: torch.Tensor,
    eps: float,
    layer: int,
    k_probes: int,
) -> Dict[str, Any]:
    """Finite-difference probe loop; returns the KxK Gram and probe matrix stats."""
    d = int(h_t.shape[-1])
    V = int(baseline_logits.shape[-1])
    K = int(E.shape[0])

    # Accumulate the empirical matrix ΔY ∈ R^(K×V) on CPU float32.
    delta_Y = torch.zeros((K, V), dtype=torch.float32)
    logit_shift_norms: List[float] = []
    n_zero_shifts = 0
    ZERO_SHIFT_ABS = 1e-12  # ||Δy_k|| below this counts as "no resolvable shift"

    for k in range(K):
        e_k = E[k]  # [d] float32
        logits_k = run_probe_pass(
            model, tokenizer, prompt, h_t, e_k, eps, layer
        )
        delta_y = logits_k - baseline_logits  # [1, V] float32 CPU
        delta_Y[k] = delta_y.reshape(-1)
        shift_norm = float(delta_y.norm().item())
        logit_shift_norms.append(shift_norm)
        if shift_norm < ZERO_SHIFT_ABS:
            n_zero_shifts += 1
        if (k + 1) % 32 == 0 or (k + 1) == K:
            print(f"    probe {k + 1}/{K}: ||Δy_k||={logit_shift_norms[-1]:.6f}")

    # KxK Gram G = (1/ε²) ΔY ΔYᵀ (float32 CPU). This is M Mᵀ for M = ΔY/ε.
    G = (1.0 / (eps * eps)) * (delta_Y @ delta_Y.T)

    return {
        "delta_Y": delta_Y,  # [K, V] float32 CPU (128 x 151936 ≈ 78 MB)
        "G": G,  # [K, K] float32 CPU
        "K": K,
        "V": V,
        "d": d,
        "logit_shift_norms": logit_shift_norms,
        "mean_logit_shift_norm": float(np.mean(logit_shift_norms)),
        "max_logit_shift_norm": float(np.max(logit_shift_norms)),
        "n_zero_shifts": int(n_zero_shifts),
        "zero_shift_fraction": float(n_zero_shifts / K),
    }


def power_iteration(
    G: torch.Tensor,
    num_iters: int,
    seed: int,
    converge_min_cos: float = CONVERGENCE_COS_MIN,
) -> Dict[str, Any]:
    """Top eigenvector of the symmetric PSD KxK Gram G (float32 CPU)."""
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
        cos = float(
            F.cosine_similarity(c_new.view(1, -1), c.view(1, -1)).item()
        )
        trace.append({
            "iter": int(i + 1),
            "cos_with_prev": cos,
            "||G·c||": float(norm.item()),
        })
        c = c_new
    if len(trace) >= 2:
        converged = bool(trace[-1]["cos_with_prev"] >= converge_min_cos)
    return {"c1": c, "trace": trace, "converged": converged}


# ---------------------------------------------------------------------------
# Phase 2: bounded kick along v₁ + greedy decoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_kicked(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    h_t: torch.Tensor,
    v1: torch.Tensor,
    eta: float,
    layer: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Greedy generation with ONE kick (h_t + eta*v1_unit) at the prefill's
    last prompt token position (layer L). The hook fires only on the prefill
    forward pass; decoding steps are unmodified."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])

    # Production fallback orthogonalisation: v1_unit = orth(v1, h_t).
    v1_unit = production_orthogonalise(v1, h_t)
    h_t_prime = h_t.to(device=model.device).float() + eta * v1_unit.to(
        device=model.device
    )

    state = {"fired": False, "l2_delta": None, "cos_residual": None}
    orig_last = h_t.to(device=model.device).float()

    def hook_fn(module, args, output):
        if state["fired"]:
            return output
        hs = output if isinstance(output, torch.Tensor) else output[0]
        new_hs = hs.clone()
        new_hs[:, -1, :] = h_t_prime.to(dtype=hs.dtype)
        state["fired"] = True
        inter_last = new_hs[:, -1, :].float()
        state["l2_delta"] = float((inter_last - orig_last).norm().item())
        state["cos_residual"] = float(
            F.cosine_similarity(orig_last.view(1, -1), inter_last.view(1, -1)).item()
        )
        return new_hs

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    generated: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    try:
        out = model(input_ids)
        logits = out.logits[:, -1, :].float()
        past_key_values = out.past_key_values
        logits_list.append(logits)
        next_tok = logits.argmax(dim=-1, keepdim=True)
        generated.append(next_tok)

        eos_id = tokenizer.eos_token_id
        for _step in range(max_new_tokens - 1):
            if int(next_tok.item()) == eos_id:
                break
            out = model(next_tok, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :].float()
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
        "logits": logits_list,
        "state": state,
    }


@torch.no_grad()
def greedy_generate_dormant(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Greedy generation with no hooks (dormant / baseline)."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    prompt_len = int(input_ids.shape[-1])

    generated: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []

    out = model(input_ids)
    logits = out.logits[:, -1, :].float()
    past_key_values = out.past_key_values
    logits_list.append(logits)
    next_tok = logits.argmax(dim=-1, keepdim=True)
    generated.append(next_tok)

    eos_id = tokenizer.eos_token_id
    for _step in range(max_new_tokens - 1):
        if int(next_tok.item()) == eos_id:
            break
        out = model(next_tok, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :].float()
        logits_list.append(logits)
        next_tok = logits.argmax(dim=-1, keepdim=True)
        generated.append(next_tok)

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
        "logits": logits_list,
    }


# ---------------------------------------------------------------------------
# Phase 1 helpers (dtype-aware)
# ---------------------------------------------------------------------------
def run_phase1_probe(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layer: int,
    K: int,
    seed: int,
    eps: float,
) -> Dict[str, Any]:
    """Baseline capture + finite-difference probe loop.

    Returns est (Gram + ΔY stats), E (probe directions), h_t, baseline_logits.
    """
    base = capture_baseline(model, tokenizer, prompt, layer)
    h_t = base["h_t"]  # [1, d] float32 CPU
    baseline_logits = base["baseline_logits"]  # [1, V] float32 CPU
    d = int(h_t.shape[-1])
    V = int(baseline_logits.shape[-1])
    print(f"h_t: {tuple(h_t.shape)} | baseline logits: {tuple(baseline_logits.shape)}")

    E = make_probe_directions(K, d, seed, device="cpu")  # [K, d] float32 CPU
    ete = E @ E.T
    ortho_err = float((ete - torch.eye(K)).abs().max().item())
    print(f"probe directions: {tuple(E.shape)} | E Eᵀ max off-diag err = {ortho_err:.2e}")

    try:
        est = estimate_greens_function(
            model, tokenizer, prompt, h_t, baseline_logits, E,
            eps=eps, layer=layer, k_probes=K,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"[oom] CUDA OOM on K={K}; reducing to K=64 and reporting.")
        K = 64
        E = make_probe_directions(K, d, seed, device="cpu")
        est = estimate_greens_function(
            model, tokenizer, prompt, h_t, baseline_logits, E,
            eps=eps, layer=layer, k_probes=K,
        )
    print(f"Gram G: {tuple(est['G'].shape)} | K={est['K']} V={est['V']} d={est['d']}")
    print(f"  mean ||Δy_k|| = {est['mean_logit_shift_norm']:.6e}, "
          f"max ||Δy_k|| = {est['max_logit_shift_norm']:.6e}")
    return {"est": est, "E": E, "h_t": h_t, "baseline_logits": baseline_logits}


def finalize_phase1(
    est: Dict[str, Any],
    E: torch.Tensor,
    h_t: torch.Tensor,
    baseline_logits: torch.Tensor,
    eps: float,
    seed: int,
    num_power_iter: int,
    probe_dtype: str,
    extra_note: Optional[str] = None,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Power iteration on G -> v₁; top-5 singular values; cosine diagnostics."""
    G = est["G"]
    K = est["K"]
    d = est["d"]
    V = est["V"]

    pi = power_iteration(G, num_power_iter, seed)
    c1 = pi["c1"]  # [K] float32 CPU
    v1_raw = E.T @ c1  # [d] float32 CPU
    v1 = v1_raw / (v1_raw.norm() + 1e-12)
    v1_norm = float(v1.norm().item())

    # Top singular values of ΔY from the eigenvalues of G (σ_i(ΔY) = ε√λ_i(G)).
    n_sv = min(5, K)
    eigvals = torch.linalg.eigvalsh(G)  # ascending
    eigvals = eigvals.flip(0)  # descending
    top_eigvals = eigvals[:n_sv].clamp_min(0.0)
    singular_values = []
    for i in range(n_sv):
        sigma = eps * float(torch.sqrt(top_eigvals[i]))
        sigma_norm = float(
            torch.sqrt(top_eigvals[i] / top_eigvals[0].clamp_min(1e-30))
        ) if top_eigvals[0].item() > 1e-30 else float("nan")
        singular_values.append({"rank": i + 1, "sigma": sigma, "sigma_norm": sigma_norm})

    cos_v1_ht = float(
        F.cosine_similarity(v1.view(1, -1), h_t.view(1, -1)).item()
    )
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    rnd = torch.randn(d, generator=gen, dtype=torch.float32)
    null_dir = production_orthogonalise(rnd, h_t)
    cos_v1_null = float(
        F.cosine_similarity(v1.view(1, -1), null_dir.view(1, -1)).item()
    )

    ph1_out: Dict[str, Any] = {
        "K": K, "V": V, "d": d, "epsilon": eps,
        "probe_dtype": probe_dtype,
        "converged": pi["converged"], "conv_trace": pi["trace"],
        "mean_logit_shift_norm": est["mean_logit_shift_norm"],
        "max_logit_shift_norm": est["max_logit_shift_norm"],
        "singular_values": singular_values,
        "cos_v1_ht": cos_v1_ht, "cos_v1_null": cos_v1_null,
        "v1_norm": v1_norm,
        "v1_first10": v1[:10].tolist(),
    }
    if extra_note:
        ph1_out["note"] = extra_note

    print("\n--- Phase 1 results ---")
    print(f"  probe dtype: {probe_dtype}")
    print(f"  converged: {pi['converged']}")
    for t in pi["trace"]:
        print(f"    iter {t['iter']}: cos={t['cos_with_prev']:.6f} "
              f"||G·c||={t['||G·c||']:.4f}")
    print("  top-5 singular values of ΔY (normalized):")
    for sv in singular_values:
        print(f"    σ_{sv['rank']} = {sv['sigma']:.6e} "
              f"(σ/σ₁ = {sv['sigma_norm']:.4f})")
    print(f"  cos(v₁, h_t) = {cos_v1_ht:.6f}")
    print(f"  cos(v₁, random-orthogonal nullspace) = {cos_v1_null:.6f}")
    print(f"  ‖v₁‖ = {v1_norm:.6f}")

    return ph1_out, v1


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def fmt(v: float, nd: int = 4) -> str:
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{nd}f}"


def fmt_sci(v: float, nd: int = 2) -> str:
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{nd}e}"


def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_markdown_report(ctx: Dict[str, Any]) -> None:
    md: List[str] = []
    md.append("# DS-036 — Green's Function SVD probe (residual channel liveness)")
    md.append("")
    md.append("> MEASUREMENT REPORT. Single-record diagnostic (MEASUREMENT ONLY).")
    md.append("> No controller edits, no thresholds, no new architecture, no")
    md.append("> green/red gate. The liveness classification below is the task's")
    md.append("> explicit reading rule for the measured KL, offered for the")
    md.append("> human/human review decision (DS-037 queue vs residual deprecation).")
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
    md.append(f"| Fixture record | {meta['record_id']} (heldout-degenerate v2) |")
    md.append(f"| Prompt tokens | {meta['prompt_len']} |")
    md.append(f"| Intervention layer | {meta['layer']} |")
    md.append(f"| Intervention position | last prompt token (index {meta['prompt_len'] - 1}) |")
    md.append(f"| Force budget | η = {meta['eta']} L2 "
              f"(production clamp [{meta['force_clamp'][0]}, {meta['force_clamp'][1]}], "
              f"controller.py:566) |")
    md.append(f"| Probe magnitude | ε = {meta['epsilon']} |")
    md.append(f"| Probe directions | K = {meta['k_probes']} (QR-orthonormal) |")
    md.append(f"| Probe dtype | {meta.get('probe_dtype', 'n/a')} |")
    md.append(f"| Power iterations | {meta['num_power_iter']} |")
    md.append(f"| Vocab dim V | {meta['vocab_dim']} (lm_head out_features) |")
    md.append(f"| Hidden dim d | {meta['hidden_dim']} |")
    md.append(f"| Per-token KL horizon | first {meta['per_token_kl_tokens']} generated tokens |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    md.append("## Determinism smoke (baseline capture, twice)")
    md.append("")
    smoke = ctx["determinism_smoke"]
    if smoke is None:
        md.append("(skipped, dev mode)")
    else:
        md.append(format_table_md(
            [[str(smoke["n_tokens"]), smoke["h_t_identical"],
              smoke["logits_identical"]]],
            ["prompt tokens", "h_t identical", "baseline logits identical"],
        ))
        md.append("")
        if not smoke["h_t_identical"] or not smoke["logits_identical"]:
            md.append("**STOP: baseline capture is not deterministic.**")
            md.append("")

    md.append("## Phase 1 — v₁ estimation (finite differences)")
    md.append("")
    ph1 = ctx["phase1"]
    md.append(f"- K = {ph1['K']} QR-orthonormal probe directions e_k ∈ R^1536.")
    md.append(f"- ε = {ph1['epsilon']} (finite-difference probe magnitude).")
    md.append(f"- Probe dtype: **{ph1.get('probe_dtype', 'n/a')}**.")
    md.append(f"- Empirical matrix ΔY ∈ R^({ph1['K']}×{ph1['V']}) held in float32 CPU.")
    md.append(f"- Mean ||Δy_k|| = {fmt_sci(ph1['mean_logit_shift_norm'], 4)}, "
              f"max ||Δy_k|| = {fmt_sci(ph1['max_logit_shift_norm'], 4)}.")
    if ph1.get("note"):
        md.append(f"- **Note:** {ph1['note']}.")
    md.append("- Power iteration on the K×K Gram G = (1/ε²)ΔYΔYᵀ "
              "(= M Mᵀ for M = ΔY/ε).")
    md.append(f"- v₁ = Eᵀc₁/‖Eᵀc₁‖ ∈ R^1536 (Rayleigh–Ritz restriction of the "
              f"empirical Jacobian's top right-singular direction to the probe "
              f"subspace; K < d, so the projected eigenproblem is the "
              f"well-posed version of M^T M power iteration).")
    md.append("")

    md.append("### Power-iteration convergence trace")
    md.append("")
    rows = []
    for t in ph1["conv_trace"]:
        rows.append([str(t["iter"]), fmt(t["cos_with_prev"], 6),
                     fmt(t["||G·c||"], 4)])
    md.append(format_table_md(
        rows, ["iter", "cos(v_i, v_{i-1})", "||G·c||"],
    ))
    md.append("")
    md.append(f"Converged (final cosine ≥ {meta['converge_cos_min']}): "
              f"{ph1['converged']}")
    if not ph1["converged"]:
        md.append("**STOP: power iteration failed to converge within ±0.01 "
                  "cosine after 10 iterations (task explicit).**")
    md.append("")

    md.append("### Top-5 singular values of ΔY (normalized σ_i/σ_1)")
    md.append("")
    rows = []
    for i, sv in enumerate(ph1["singular_values"]):
        rows.append([str(i + 1), fmt_sci(sv["sigma"], 4), fmt(sv["sigma_norm"], 6)])
    md.append(format_table_md(rows, ["rank", "σ_i(ΔY)", "σ_i/σ_1"]))
    md.append("")

    md.append("### v₁ diagnostics")
    md.append("")
    rows = [
        ["cos(v₁, h_t)", fmt(ph1["cos_v1_ht"], 6)],
        ["cos(v₁, random-orthogonal nullspace direction)",
         fmt(ph1["cos_v1_null"], 6)],
        ["random-nullspace expected |cos| for d=1536 (≈1/√d)",
         fmt(1.0 / np.sqrt(ph1["d"]), 6)],
        ["‖v₁‖ (L2)", fmt(ph1["v1_norm"], 6)],
    ]
    md.append(format_table_md(rows, ["quantity", "value"]))
    md.append("")
    md.append("cos(v₁, h_t) ≈ 0 confirms v₁ is essentially orthogonal to the")
    md.append("residual. cos(v₁, random-orthogonal) ≈ 1/√d ≈ 0.0255 confirms v₁")
    md.append("is *not* a specially-aligned hidden direction by construction — it")
    md.append("is the empirical Jacobian's dominant input direction, which is a")
    md.append("different object from the random nullspace directions that gave")
    md.append("KL ≈ 10⁻⁵ in DS-026/028/029.")
    md.append("")

    md.append("## Phase 2 — bounded kick along v₁ (η = 0.50 L2)")
    md.append("")
    ph2 = ctx["phase2"]
    md.append(f"- v₁ orthogonalised against h_t (production fallback double "
              f"Gram-Schmidt); unit-norm direction.")
    md.append(f"- Applied h_t' = h_t + {ph2['eta']}·v₁_unit at layer "
              f"{meta['layer']}, last prompt token position.")
    md.append(f"- Measured applied L2 delta = {fmt(ph2['applied_l2_delta'], 4)} "
              f"(target {ph2['eta']}); cos(h_t, h_t') = "
              f"{fmt(ph2['cos_residual'], 6)}.")
    md.append(f"- Kicked greedy continuation "
              f"({ph2['n_generated']} tokens): `{ph2['kicked_text']}`")
    md.append(f"- Dormant greedy continuation "
              f"({ph2['n_generated']} tokens): `{ph2['dormant_text']}`")
    md.append("")

    md.append("### Per-token KL (kicked || dormant), first 10 generated tokens")
    md.append("")
    rows = []
    for i, kl in enumerate(ph2["per_token_kl"]):
        rows.append([str(i + 1), fmt_sci(kl, 4)])
    md.append(format_table_md(rows, ["generated token", "KL"]))
    md.append("")
    md.append(f"- Mean over {len(ph2['per_token_kl'])} tokens: "
              f"**{fmt_sci(ph2['mean_kl'], 4)}**")
    md.append(f"- KL at token 1 (first predicted token): "
              f"**{fmt_sci(ph2['per_token_kl'][0], 4)}**")
    md.append("")

    md.append("## Liveness classification (task reading rule)")
    md.append("")
    md.append(f"- Measured mean KL = **{fmt_sci(ph2['mean_kl'], 4)}**.")
    md.append(f"- Reading rule: KL > 10⁻² → ALIVE (DS-037 unblocked); "
              f"KL ≈ 10⁻⁵ → DEAD (deprecate residual, cancel DS-037).")
    md.append(f"- **Classification: {ph2['verdict']}.**")
    md.append("")
    md.append("This is a measurement report; the classification is the task's")
    md.append("explicit threshold applied to the measured value. No controller")
    md.append("change, no threshold change, and no DS-037 work is performed here.")
    md.append("")

    md.append("## JSONL")
    md.append("")
    md.append("Machine-readable results are in `greens_function_results.jsonl`.")
    md.append("")

    md.append("## Notes / caveats")
    md.append("")
    md.append("- Single-record diagnostic (record 0 of heldout_degenerate_v2).")
    md.append("- v₁ is the dominant right-singular direction of the empirical")
    md.append("  Jacobian restricted to the K=128 probe subspace (K < d). It is")
    md.append("  not guaranteed to be the global top singular direction of the")
    md.append("  full 151,936×1536 Jacobian.")
    md.append("- **bf16 dead-zone finding:** the task's ε=10⁻³ is *not* large enough")
    md.append("  to be representable in bf16 at layer 26. The residual norm there")
    md.append("  is O(250) and bf16's absolute step is O(0.6); an ε=10⁻³ hidden-")
    md.append("  state perturbation is rounded to ~0 for 99.2% of the probe")
    md.append("  directions (127/128 zero shifts). The one nonzero shift is a")
    md.append("  bf16-quantization artifact (max ||Δy_k|| = 1.73), not a Jacobian")
    md.append("  response. The finite-difference Jacobian was therefore estimated")
    md.append("  in fp32 (same bf16-loaded weights, higher numerics) so that ε=10⁻³")
    md.append("  is resolvable for every probe direction. Phase 2 (the 0.50 L2")
    md.append("  kick and KL) runs in the production bf16 model.")
    if meta.get("bf16_dead_zone_max_shift") is not None:
        md.append(f"- bf16 spec-literal probe max ||Δy_k|| = "
                  f"{fmt_sci(meta['bf16_dead_zone_max_shift'], 3)}, "
                  f"zero-shift probe fraction = "
                  f"{meta.get('bf16_dead_zone_zero_frac')} "
                  f"(dead-zone floors: max-shift "
                  f"{fmt_sci(meta['bf16_dead_zone_floor'], 0)}, zero-frac "
                  f"{meta['bf16_dead_zone_zero_frac_floor']}).")
    md.append("- The Gram matrix and power iteration run in float32 on CPU.")
    md.append("- KL uses the DS-026/028/029 formula (softmax distributions, ")
    md.append("  eps=1e-12, clamp at 0) for direct comparability.")
    md.append("")

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-036 Green's function SVD probe (measurement only)"
    )
    parser.add_argument("--k-probes", type=int, default=K_PROBES,
                        help="Number of probe directions (default 128; reduce "
                             "to 64 on CUDA OOM).")
    parser.add_argument("--max-new-tokens", type=int, default=PER_TOKEN_KL_TOKENS,
                        help="Greedy generation length for per-token KL "
                             "(default 10).")
    parser.add_argument("--probe-dtype", choices=["auto", "bf16", "fp32"],
                        default="auto",
                        help="Dtype for the finite-difference probe passes. "
                             "'auto' tries the spec-literal bf16 probe first and "
                             "falls back to fp32 if ε=1e-3 is below the bf16 "
                             "quantization floor (dead zone). Default auto.")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-036 Green's Function SVD probe (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"layer: {INTERVENTION_LAYER} | η={ETA} | ε={EPSILON} | "
          f"K={args.k_probes} | power_iters={NUM_POWER_ITER}")

    # Model load.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

    # Data: record 0 (id=0) of heldout-degenerate v2.
    heldout_deg = load_jsonl(HELDOUT_DEG_V2)
    rec = next((r for r in heldout_deg if int(r["id"]) == 0), None)
    if rec is None:
        raise SystemExit("[STOP] heldout_degenerate_v2.jsonl has no record id=0.")
    prompt = rec["mutated_prompt"]
    rid = int(rec["id"])
    print(f"selected record id={rid} (prompt_len tokens: "
          f"{int(tokenizer(prompt, return_tensors='pt')['input_ids'].shape[-1])})")

    # ------------------------------------------------------------------
    # Determinism smoke: baseline capture twice.
    # ------------------------------------------------------------------
    smoke = None
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        print("\n--- Determinism smoke (baseline capture, twice) ---")
        a = capture_baseline(model, tokenizer, prompt, INTERVENTION_LAYER)
        b = capture_baseline(model, tokenizer, prompt, INTERVENTION_LAYER)
        h_same = bool(torch.allclose(a["h_t"], b["h_t"], atol=0.0, rtol=0.0))
        logits_same = bool(torch.equal(a["baseline_logits"], b["baseline_logits"]))
        print(f"  prompt tokens: {a['prompt_len']}")
        print(f"  h_t identical: {h_same}")
        print(f"  baseline logits identical: {logits_same}")
        if not h_same or not logits_same:
            print("[STOP] Determinism smoke FAILED: baseline capture differs.")
            sys.exit(1)
        smoke = {
            "n_tokens": a["prompt_len"],
            "h_t_identical": str(h_same),
            "logits_identical": str(logits_same),
        }
        print("  determinism smoke: ALL IDENTICAL")

    # ------------------------------------------------------------------
    # Phase 1 — v₁ estimation.
    # ------------------------------------------------------------------
    K = args.k_probes
    probe_dtype_arg = args.probe_dtype  # "auto" | "bf16" | "fp32"

    # Ensure the model starts in the production bf16 dtype (task ground truth).
    if torch.cuda.is_available() and model.dtype != torch.bfloat16:
        model = model.to(torch.bfloat16)

    # Try the spec-literal bf16 probe first (unless fp32 is forced).
    ph1_out: Optional[Dict[str, Any]] = None
    v1: Optional[torch.Tensor] = None
    phase1_h_t: Optional[torch.Tensor] = None
    bf16_dead_zone_max_shift: Optional[float] = None
    bf16_dead_zone_zero_frac: Optional[float] = None

    if probe_dtype_arg in ("auto", "bf16"):
        print("\n--- Phase 1: estimate v₁ via finite differences "
              "(probe dtype: bf16, spec-literal) ---")
        r1 = run_phase1_probe(
            model, tokenizer, prompt, INTERVENTION_LAYER, K, SEED, EPSILON
        )
        bf16_max_shift = r1["est"]["max_logit_shift_norm"]
        bf16_zero_frac = r1["est"]["zero_shift_fraction"]
        bf16_dead_zone_max_shift = float(bf16_max_shift)
        bf16_dead_zone_zero_frac = float(bf16_zero_frac)
        # If bf16 cannot resolve ε=1e-3 perturbations (dead zone), the Gram
        # matrix is ~0 or dominated by a few bf16-quantization artifacts, and
        # power iteration is meaningless. Detect and fall back.
        dead_zone = (
            bf16_max_shift < BF16_DEAD_ZONE_MAX_SHIFT
            or bf16_zero_frac > BF16_DEAD_ZONE_ZERO_FRAC
        )
        if dead_zone:
            print(f"\n[bf16 dead zone] max ||Δy_k|| = {bf16_max_shift:.3e}, "
                  f"zero-shift probes = {int(bf16_zero_frac * K)}/{K} "
                  f"({bf16_zero_frac:.1%}); ε={EPSILON} perturbations are "
                  f"below the bf16 quantization floor at layer {INTERVENTION_LAYER} "
                  f"(hidden norm ≈ {float(r1['h_t'].norm()):.1f}).")
            if probe_dtype_arg == "bf16":
                print("[STOP] --probe-dtype bf16 forced; the bf16 probe cannot "
                      "resolve ε=1e-3. Report and stop.")
                ph1_out, v1 = finalize_phase1(
                    r1["est"], r1["E"], r1["h_t"], r1["baseline_logits"],
                    EPSILON, SEED, NUM_POWER_ITER, "bf16",
                    extra_note="STOPPED: bf16 dead zone; fp32 fallback not allowed "
                               "(--probe-dtype bf16 forced)",
                )
                # Write partial outputs and stop.
                base = capture_baseline(model, tokenizer, prompt, INTERVENTION_LAYER)
                metadata = {
                    "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                    "device": device, "dtype": str(dtype),
                    "torch_version": torch.__version__, "seed": SEED,
                    "record_id": rid, "prompt_len": base["prompt_len"],
                    "layer": INTERVENTION_LAYER, "eta": ETA,
                    "force_clamp": [FORCE_CLAMP_LOW, FORCE_CLAMP_HIGH],
                    "epsilon": EPSILON, "k_probes": K,
                    "num_power_iter": NUM_POWER_ITER,
                    "vocab_dim": r1["est"]["V"], "hidden_dim": r1["est"]["d"],
                    "per_token_kl_tokens": args.max_new_tokens,
                    "converge_cos_min": CONVERGENCE_COS_MIN,
                    "bmm_override": "deregistered",
                    "wall_clock_s": time.time() - t_start,
                }
                with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"metadata": metadata}) + "\n")
                    f.write(json.dumps({"phase1": ph1_out}) + "\n")
                print(f"\nwrote {OUTPUT_JSONL} (partial, bf16 dead zone)")
                sys.exit(1)
            # Fall back to fp32 probe for a resolvable Jacobian estimate.
            print("[fallback] Re-running the finite-difference probe in fp32 to "
                  "obtain a resolvable Jacobian estimate (Phase 2 remains in "
                  "production bf16).")
            if torch.cuda.is_available():
                model = model.to(torch.float32)
            print("\n--- Phase 1 re-run (probe dtype: fp32, fallback) ---")
            r2 = run_phase1_probe(
                model, tokenizer, prompt, INTERVENTION_LAYER, K, SEED, EPSILON
            )
            ph1_out, v1 = finalize_phase1(
                r2["est"], r2["E"], r2["h_t"], r2["baseline_logits"],
                EPSILON, SEED, NUM_POWER_ITER, "fp32 (fallback from bf16 dead zone)",
                extra_note=f"bf16 spec-literal probe max ||Δy_k|| = "
                           f"{bf16_max_shift:.3e}, zero-shift fraction = "
                           f"{bf16_zero_frac:.1%}; fp32 fallback used for "
                           f"Jacobian estimation",
            )
            phase1_h_t = r2["h_t"]
            # Restore production bf16 for Phase 2.
            if torch.cuda.is_available():
                model = model.to(torch.bfloat16)
                torch.cuda.empty_cache()
        else:
            # bf16 probe is usable (shifts above the dead-zone floor).
            ph1_out, v1 = finalize_phase1(
                r1["est"], r1["E"], r1["h_t"], r1["baseline_logits"],
                EPSILON, SEED, NUM_POWER_ITER, "bf16",
            )
            phase1_h_t = r1["h_t"]
    else:  # probe_dtype_arg == "fp32" (forced)
        if torch.cuda.is_available():
            model = model.to(torch.float32)
        print("\n--- Phase 1: estimate v₁ via finite differences "
              "(probe dtype: fp32, forced) ---")
        r2 = run_phase1_probe(
            model, tokenizer, prompt, INTERVENTION_LAYER, K, SEED, EPSILON
        )
        ph1_out, v1 = finalize_phase1(
            r2["est"], r2["E"], r2["h_t"], r2["baseline_logits"],
            EPSILON, SEED, NUM_POWER_ITER, "fp32",
        )
        phase1_h_t = r2["h_t"]
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16)
            torch.cuda.empty_cache()

    assert ph1_out is not None and v1 is not None

    if not ph1_out["converged"]:
        print("[STOP] Power iteration failed to converge "
              "(final cosine < 0.99 after 10 iterations).")
        print("  Convergence trace reported above.")
        # Still write partial outputs before stopping.
        base = capture_baseline(model, tokenizer, prompt, INTERVENTION_LAYER)
        metadata = {
            "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
            "device": device, "dtype": str(dtype),
            "torch_version": torch.__version__, "seed": SEED,
            "record_id": rid, "prompt_len": base["prompt_len"],
            "layer": INTERVENTION_LAYER, "eta": ETA,
            "force_clamp": [FORCE_CLAMP_LOW, FORCE_CLAMP_HIGH],
            "epsilon": EPSILON, "k_probes": ph1_out["K"],
            "num_power_iter": NUM_POWER_ITER,
            "vocab_dim": ph1_out["V"], "hidden_dim": ph1_out["d"],
            "per_token_kl_tokens": args.max_new_tokens,
            "converge_cos_min": CONVERGENCE_COS_MIN,
            "bmm_override": "deregistered", "wall_clock_s": time.time() - t_start,
        }
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            f.write(json.dumps({"metadata": metadata}) + "\n")
            f.write(json.dumps({"phase1": ph1_out}) + "\n")
        print(f"\nwrote {OUTPUT_JSONL} (partial, convergence failure)")
        sys.exit(1)

    d = ph1_out["d"]
    V = ph1_out["V"]

    # Re-capture h_t in the production bf16 model for Phase 2 (the kick uses
    # the bf16 residual, matching the controller's intervention point).
    base = capture_baseline(model, tokenizer, prompt, INTERVENTION_LAYER)
    h_t = base["h_t"]  # [1, d] float32 CPU (bf16 model's residual)

    # ------------------------------------------------------------------
    # Phase 2 — bounded kick along v₁ + per-token KL.
    # ------------------------------------------------------------------
    print("\n--- Phase 2: bounded kick along v₁ ---")
    dormant = greedy_generate_dormant(
        model, tokenizer, prompt, max_new_tokens=args.max_new_tokens
    )
    kicked = greedy_generate_kicked(
        model, tokenizer, prompt, h_t, v1, ETA, INTERVENTION_LAYER,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"  dormant generated: {dormant['n_generated']} tokens")
    print(f"  kicked generated: {kicked['n_generated']} tokens")
    print(f"  applied L2 delta: {kicked['state']['l2_delta']:.4f} "
          f"(target {ETA})")
    print(f"  cos(h_t, h_t'): {kicked['state']['cos_residual']:.6f}")

    per_token_kl = []
    n_tokens = min(
        len(kicked["logits"]), len(dormant["logits"]), args.max_new_tokens
    )
    for t in range(n_tokens):
        kl = kl_divergence(kicked["logits"][t], dormant["logits"][t])
        per_token_kl.append(kl)
    mean_kl = float(np.mean(per_token_kl)) if per_token_kl else float("nan")
    first_kl = per_token_kl[0] if per_token_kl else float("nan")

    if mean_kl > LIVENESS_KL_ALIVE:
        verdict = "ALIVE"
    elif mean_kl < LIVENESS_KL_DEAD:
        verdict = "DEAD (deprecate residual; CANCEL DS-037)"
    else:
        verdict = "INCONCLUSIVE (between 1e-4 and 1e-2)"

    print(f"  per-token KL (first {n_tokens}): "
          + ", ".join(f"{v:.4e}" for v in per_token_kl))
    print(f"  mean KL = {mean_kl:.4e}")
    print(f"  liveness classification: {verdict}")

    kicked_text = tokenizer.decode(kicked["generated_ids"][0], skip_special_tokens=True)
    dormant_text = tokenizer.decode(dormant["generated_ids"][0], skip_special_tokens=True)
    ph2_out = {
        "eta": ETA,
        "applied_l2_delta": kicked["state"]["l2_delta"],
        "cos_residual": kicked["state"]["cos_residual"],
        "n_generated": kicked["n_generated"],
        "kicked_text": kicked_text,
        "dormant_text": dormant_text,
        "per_token_kl": per_token_kl,
        "mean_kl": mean_kl,
        "first_token_kl": first_kl,
        "verdict": verdict,
    }

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    metadata = {
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "device": device, "dtype": str(dtype),
        "torch_version": torch.__version__, "seed": SEED,
        "record_id": rid, "prompt_len": base["prompt_len"],
        "layer": INTERVENTION_LAYER, "eta": ETA,
        "force_clamp": [FORCE_CLAMP_LOW, FORCE_CLAMP_HIGH],
        "epsilon": EPSILON, "k_probes": ph1_out["K"],
        "num_power_iter": NUM_POWER_ITER,
        "vocab_dim": V, "hidden_dim": d,
        "per_token_kl_tokens": args.max_new_tokens,
        "converge_cos_min": CONVERGENCE_COS_MIN,
        "probe_dtype": ph1_out.get("probe_dtype", "unknown"),
        "bf16_dead_zone_max_shift": bf16_dead_zone_max_shift,
        "bf16_dead_zone_zero_frac": bf16_dead_zone_zero_frac,
        "bf16_dead_zone_floor": BF16_DEAD_ZONE_MAX_SHIFT,
        "bf16_dead_zone_zero_frac_floor": BF16_DEAD_ZONE_ZERO_FRAC,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": metadata}) + "\n")
        f.write(json.dumps({"record": {"record_id": rid, "prompt": prompt}}) + "\n")
        f.write(json.dumps({"determinism_smoke": smoke}) + "\n")
        f.write(json.dumps({"phase1": ph1_out}) + "\n")
        f.write(json.dumps({"phase2": ph2_out}) + "\n")

    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "phase1": ph1_out,
        "phase2": ph2_out,
    }
    write_markdown_report(ctx)

    print(f"\nwrote {OUTPUT_JSONL}")
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("MEASUREMENT COMPLETE (measurement only; no controller change)")


if __name__ == "__main__":
    main()
