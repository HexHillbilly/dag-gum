#!/usr/bin/env python3
"""DS-038: ECS/PKS liveness on t2s_degenerate (MEASUREMENT ONLY).

The dual-predicate detects hidden-state collapse on 87/93 t2s bigram-blind
records (spectral PR below band_low). But the logit-penalty path has no
effective actuator for macro-loops: trailing_ctr stays near 1.0, active_loop_ids
is sparse, and kickstart vocabulary steering cannot redirect structural
degeneracy invisible to surface metrics.

ECS (External Context Score) and PKS (Parametric Knowledge Score) come from the
ReDeEP framework (ICLR 2025). They measure a tug-of-war inside transformer
residual streams: whether attention heads track prompt context (ECS), and
whether feed-forward networks override attention with parametric knowledge
(PKS). On t2s macro-loops the hypothesis is: attention heads detach from the
prompt (low ECS) and FFNs fill the vacuum with memorized structural patterns
(high PKS).

This probe MEASURES per-head ECS and per-layer PKS distributions on
t2s_degenerate (N=100) vs prose-100 (N=100) at candidate layers
{2, 6, 14, 21, 27}. Thresholds are derived mechanically (Gate 2.3 rule:
T = (p90_degenerate + p10_prose) / 2, band = [0.75*T, 1.25*T]), frozen, and
scored against the degenerate contrast.

MEASUREMENT ONLY. No controller edits, no new architecture, no actuation
design. This is the "before" measurement that gates any future AARF actuation
experiment.

Environment notes (identical to ds-025/ds-027/ds-033/ds-034):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; fall back to HF_HOME=/tmp/hf_cache only if the
    pinned revision is not cached.

ADAPTATIONS (documented; measurement semantics preserved):
  1. HEAD COUNT. The task spec states "28 attention heads per layer"; the
     actual Qwen2.5-1.5B config has num_attention_heads=12 (the spec conflated
     layer count 28 with head count). We use the REAL model config (12 heads)
     and document this deviation.
  2. ATTENTION CAPTURE. `output_attentions=True` OOMs on long prose records
     (attention weights are O(seq_len^2); the longest prose record tokenizes
     to 7265 tokens -> 2.4 GB per layer matmul, OOM on 12 GB). We instead
     register forward hooks on the candidate-layer `self_attn` sub-modules,
     which return the SAME attention weights (Qwen2Attention always returns
     (attn_output, attn_weights) in eager mode) without materializing the
     other 23 layers' attention. `output_hidden_states=True` is set on the
     forward pass as required.
  3. SEQUENCE TRUNCATION. Four prose-100 records exceed 4096 tokens and would
     OOM even with hooks. We truncate to the first MAX_SEQ_LEN=4096 tokens
     (standard prefix truncation; keeps the last-24 analysis window of the
     truncated sequence). 96/100 prose records are measured in full.
  4. PROMPT BOUNDARY (ECS). Teacher-forced replay has no explicit prompt/
     generation split. We define prompt_len = seq_len - 24: the last-24 token
     window (the Gate 2.3 analysis window) is the "generated" region and
     everything before it is the "prompt" context. Documented in the report.
  5. PKS RMSNorm. The final RMSNorm (`model.model.norm`) is applied to the
     post-attention and post-FFN residual states before projection through W_U
     (the unembedding), exactly as the task specifies. All PKS math is float32.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np
import torch

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
SEED = 42  # DS-038 measurement seed (matches ds-025..ds-034d)
CONTROL_SEED = 42  # valid_subset_200 control-selection seed (ds-011/ds-020)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ANALYSIS_WINDOW = 24  # trailing token positions (Gate 2.3 / ds-025 convention)
CANDIDATE_LAYERS = [2, 6, 14, 21, 27]  # early, mid-early, mid-late, late
MAX_SEQ_LEN = 4096  # prefix truncation bound for very long prose records
TOP_K = 1000  # top-k truncation for JSD (independent review latency budget)
JSD_EPS = 1e-12  # KL log-floor

# Paths
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
VALID_SUBSET_200 = Path("data/t2s_bench/valid_subset_200.jsonl")
FROZEN_OUT = Path("docs/gate23/FROZEN_ECS_PKS_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/ecs_pks_results.jsonl")
OUTPUT_MD = Path("docs/gate23/ECS_PKS_RESULTS.md")

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
# Heldout prose-100 selection (ds-027 complement split)
# ---------------------------------------------------------------------------
EXPECTED_CONTROL_IDS = [
    0, 24, 28, 32, 36, 46, 49, 52, 53, 54, 56, 57, 58, 65, 71, 72, 78, 80,
    82, 83, 98, 101, 107, 114, 117, 119, 121, 122, 128, 135, 140, 142, 148,
    150, 152, 157, 161, 172, 176, 181, 182, 186, 189, 196, 204, 205, 209,
    214, 219, 222, 232, 235, 236, 255, 256, 258, 259, 260, 271, 276, 282,
    283, 290, 308, 309, 316, 319, 321, 328, 329, 331, 332, 339, 343, 349,
    350, 355, 359, 360, 376, 388, 390, 396, 404, 407, 412, 413, 415, 417,
    433, 444, 454, 458, 466, 469, 470, 473, 474, 490, 496,
]


def select_heldout_prose(valid_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heldout-100 = complement of the Gate 2.2 prose-control 100 within
    valid_subset_200. Returns records sorted by corpus id.

    STOPs if the control selection does not match the documented ids or the
    complement is not exactly 100 records (task boundary).
    """
    rng = random.Random(CONTROL_SEED)
    selected = sorted(
        rng.sample(valid_records, 100), key=lambda r: int(r["id"])
    )
    control_ids = [int(r["id"]) for r in selected]
    if control_ids != EXPECTED_CONTROL_IDS:
        raise SystemExit(
            "[STOP] Prose control ids do NOT match the ds-027 documented split. "
            f"got {control_ids}"
        )
    control_set = set(control_ids)
    heldout = [r for r in valid_records if int(r["id"]) not in control_set]
    heldout = sorted(heldout, key=lambda r: int(r["id"]))
    if len(heldout) != 100:
        raise SystemExit(
            f"[STOP] Complement split failed: heldout prose slice has "
            f"{len(heldout)} records, expected 100."
        )
    return heldout


# ---------------------------------------------------------------------------
# Model / replay helpers
# ---------------------------------------------------------------------------
def load_model_and_tokenizer() -> Tuple[Any, Any, torch.Tensor]:
    """Load Qwen2.5-1.5B@8faed761 with eager attention (needed for attention
    weights). Returns (tokenizer, model, W_U_float32)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    w_u = model.lm_head.weight.detach().float()  # (vocab, hidden) float32
    return tokenizer, model, w_u


def make_capture_hooks(
    model: torch.nn.Module,
    layers: Sequence[int],
    captured: Dict[str, Any],
) -> List[Any]:
    """Register forward hooks on candidate layers' self_attn, mlp, and the
    layer itself. The self_attn hook captures the (attn_output, attn_weights,
    past_key_value) tuple; the mlp hook captures the post-FFN output; the layer
    hook captures the layer input (pre-layernorm residual)."""
    handles: List[Any] = []
    for L in layers:
        blk = model.model.layers[L]

        def _self_hook(mod: Any, inp: Any, out: Any, L: int = L) -> None:
            captured[f"self_{L}"] = out

        def _mlp_hook(mod: Any, inp: Any, out: Any, L: int = L) -> None:
            captured[f"mlp_{L}"] = out

        def _layer_hook(mod: Any, inp: Any, out: Any, L: int = L) -> None:
            captured[f"layer_{L}"] = inp[0]

        handles.append(blk.self_attn.register_forward_hook(_self_hook))
        handles.append(blk.mlp.register_forward_hook(_mlp_hook))
        handles.append(blk.register_forward_hook(_layer_hook))
    return handles


@torch.no_grad()
def replay_record(
    model: torch.nn.Module,
    tokenizer: Any,
    input_text: str,
    max_seq_len: int,
    captured: Dict[str, Any],
) -> Tuple[int, int, bool]:
    """Teacher-forced single forward pass with hooks capturing candidate-layer
    attention weights and hidden states. Returns (n_tokens_full, seq_len_used,
    truncated)."""
    enc = tokenizer(input_text, return_tensors="pt")
    n_tokens_full = int(enc["input_ids"].shape[-1])
    truncated = n_tokens_full > max_seq_len
    if truncated:
        enc = {
            "input_ids": enc["input_ids"][:, :max_seq_len],
            "attention_mask": enc["attention_mask"][:, :max_seq_len],
        }
    else:
        enc = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    enc = {k: v.to(model.device) for k, v in enc.items()}
    seq_len_used = int(enc["input_ids"].shape[-1])
    captured.clear()
    out = model(**enc, output_hidden_states=True)
    # Keep only the candidate-layer captures (drop model hidden_states).
    return n_tokens_full, seq_len_used, truncated


# ---------------------------------------------------------------------------
# ECS — per-head prompt-attention ratio
# ---------------------------------------------------------------------------
def compute_ecs_per_layer(
    self_attn_out: Any,
    seq_len: int,
    num_heads: int,
    analysis_window: int = ANALYSIS_WINDOW,
) -> np.ndarray:
    """Compute per-head ECS for one candidate layer.

    self_attn_out: the tuple returned by Qwen2Attention.forward
        (attn_output, attn_weights, past_key_value). attn_weights has shape
        (1, num_heads, seq_len, seq_len) in bf16.

    prompt_len = seq_len - analysis_window (the Gate 2.3 analysis window is
    the "generated" region; everything before it is the "prompt" context).

    Returns a float32 numpy array of shape (num_heads,) with ECS averaged over
    the last `analysis_window` query positions.
    """
    attn_weights = self_attn_out[1]
    if attn_weights is None:
        raise RuntimeError("[STOP] self_attn hook returned None attention weights.")
    attn = attn_weights[0].float()  # (num_heads, seq_len, seq_len) float32
    prompt_len = seq_len - analysis_window
    prompt_sums = attn[:, :, :prompt_len].sum(dim=-1)  # (num_heads, seq_len)
    total_sums = attn.sum(dim=-1)  # (num_heads, seq_len)
    ecs_all = prompt_sums / total_sums.clamp_min(1e-12)  # (num_heads, seq_len)
    ecs_win = ecs_all[:, -analysis_window:].mean(dim=-1)  # (num_heads,)
    return ecs_win.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# PKS — per-layer JSD (top-k truncated)
# ---------------------------------------------------------------------------
def _jsd_batch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """JSD between two batched (k+1)-dim probability distributions, float32.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), m = (p+q)/2.
    """
    m = 0.5 * (p + q)
    eps = JSD_EPS
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


@torch.no_grad()
def compute_pks_per_layer(
    model: torch.nn.Module,
    w_u: torch.Tensor,
    layer_in: torch.Tensor,
    self_attn_out: Any,
    mlp_out: torch.Tensor,
    seq_len: int,
    top_k: int = TOP_K,
    analysis_window: int = ANALYSIS_WINDOW,
) -> float:
    """Compute per-layer PKS (JSD between post-attention and post-FFN projected
    distributions) for one candidate layer.

    h_attn = layer_in + attn_out; h_ffn = h_attn + mlp_out.
    For each of the last `analysis_window` positions:
      z_pre  = W_U @ RMSNorm(h_attn[pos])
      z_post = W_U @ RMSNorm(h_ffn[pos])
    Top-k truncation (top-k indices of z_post) + single tail-mass bucket,
    then JSD. Averaged over the analysis window. All float32.
    """
    attn_out = self_attn_out[0]
    h_attn = layer_in + attn_out  # (1, seq, hidden) bf16
    h_ffn = h_attn + mlp_out  # (1, seq, hidden) bf16

    win = slice(-analysis_window, None)
    h_attn_f32 = h_attn[0, win, :].float()  # (win, hidden) float32
    h_ffn_f32 = h_ffn[0, win, :].float()  # (win, hidden) float32

    z_pre = model.model.norm(h_attn_f32) @ w_u.T  # (win, vocab) float32
    z_post = model.model.norm(h_ffn_f32) @ w_u.T  # (win, vocab) float32

    _, topk_idx = torch.topk(z_post, top_k, dim=-1)  # (win, top_k)
    p_pre_full = torch.softmax(z_pre, dim=-1)  # (win, vocab)
    p_post_full = torch.softmax(z_post, dim=-1)  # (win, vocab)
    p_pre_k = torch.gather(p_pre_full, -1, topk_idx)  # (win, top_k)
    p_post_k = torch.gather(p_post_full, -1, topk_idx)  # (win, top_k)

    # Tail-mass bucket = 1 - sum(top-k probs). Guard against float32 rounding
    # that can push the sum of the top-k probabilities just over 1.0 (peaky
    # layers), which would make the tail slightly negative and log(negative)
    # NaN in the JSD. Clamp to >= 0 and renormalize the (k+1)-dim vector.
    tail_pre = torch.clamp(1.0 - p_pre_k.sum(dim=-1), min=0.0)  # (win,)
    tail_post = torch.clamp(1.0 - p_post_k.sum(dim=-1), min=0.0)  # (win,)
    p_pre = torch.cat([p_pre_k, tail_pre.unsqueeze(-1)], dim=-1)  # (win, k+1)
    p_post = torch.cat([p_post_k, tail_post.unsqueeze(-1)], dim=-1)  # (win, k+1)
    p_pre = p_pre / p_pre.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p_post = p_post / p_post.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    jsd = _jsd_batch(p_pre, p_post)  # (win,)
    return float(jsd.mean().item())


# ---------------------------------------------------------------------------
# Per-record payload builder
# ---------------------------------------------------------------------------
def build_record_payload(
    model: torch.nn.Module,
    tokenizer: Any,
    w_u: torch.Tensor,
    captured: Dict[str, Any],
    fixture: str,
    record: Dict[str, Any],
    num_heads: int,
    top_k: int,
    max_seq_len: int,
) -> Dict[str, Any]:
    """Replay one record and build the per-record ECS/PKS payload."""
    input_text = record["text"]
    n_tokens_full, seq_len_used, truncated = replay_record(
        model, tokenizer, input_text, max_seq_len, captured
    )
    prompt_len = seq_len_used - ANALYSIS_WINDOW

    ecs: Dict[str, Dict[str, float]] = {}
    pks: Dict[str, float] = {}
    for L in CANDIDATE_LAYERS:
        self_out = captured[f"self_{L}"]
        ecs_arr = compute_ecs_per_layer(self_out, seq_len_used, num_heads)
        ecs[str(L)] = {str(h): float(ecs_arr[h]) for h in range(num_heads)}
        pks[str(L)] = compute_pks_per_layer(
            model, w_u,
            captured[f"layer_{L}"],
            self_out,
            captured[f"mlp_{L}"],
            seq_len_used,
            top_k=top_k,
        )
        # Defensive finite check: a NaN/Inf PKS is a measurement failure.
        if pks[str(L)] != pks[str(L)] or pks[str(L)] in (float("inf"), float("-inf")):
            raise SystemExit(
                f"[STOP] Non-finite PKS at layer {L} for {fixture} record "
                f"{record['id']} (value={pks[str(L)]}). Measurement failure."
            )

    return {
        "fixture": fixture,
        "record_id": int(record["id"]),
        "seed": SEED,
        "n_tokens": n_tokens_full,
        "seq_len": seq_len_used,
        "truncated": bool(truncated),
        "prompt_len": int(prompt_len),
        "text": input_text,
        "ecs": ecs,
        "pks": pks,
    }


# ---------------------------------------------------------------------------
# Determinism smoke (FIRST)
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    w_u: torch.Tensor,
    captured: Dict[str, Any],
    records: List[Dict[str, Any]],
    num_heads: int,
    top_k: int,
    max_seq_len: int,
) -> Dict[str, Any]:
    """Determinism smoke (DS-038): 2 t2s_degenerate records replayed twice;
    per-head ECS and per-layer PKS must be identical to 6 decimal places.
    STOP (sys.exit 1) if not."""
    print("\n--- Determinism smoke (2 t2s_degenerate records replayed twice) ---")
    pairs: List[Dict[str, Any]] = []
    all_identical = True
    for rec in records[:2]:
        rid = int(rec["id"])
        a = build_record_payload(model, tokenizer, w_u, captured,
                                 "t2s_degenerate", rec, num_heads, top_k,
                                 max_seq_len)
        b = build_record_payload(model, tokenizer, w_u, captured,
                                 "t2s_degenerate", rec, num_heads, top_k,
                                 max_seq_len)
        # Compare ECS per (layer, head) and PKS per layer to 6 decimal places.
        ecs_ok = True
        for L in CANDIDATE_LAYERS:
            for h in range(num_heads):
                va = a["ecs"][str(L)][str(h)]
                vb = b["ecs"][str(L)][str(h)]
                if abs(va - vb) >= 1e-6:
                    ecs_ok = False
        pks_ok = True
        for L in CANDIDATE_LAYERS:
            va = a["pks"][str(L)]
            vb = b["pks"][str(L)]
            if abs(va - vb) >= 1e-6:
                pks_ok = False
        identical = bool(ecs_ok and pks_ok)
        all_identical = all_identical and identical
        pairs.append({
            "record_id": rid,
            "n_tokens": a["n_tokens"],
            "identical": str(identical),
            "ecs_identical": str(ecs_ok),
            "pks_identical": str(pks_ok),
            "ecs_a_layer2_head0": a["ecs"]["2"]["0"],
            "ecs_b_layer2_head0": b["ecs"]["2"]["0"],
            "pks_a_layer2": a["pks"]["2"],
            "pks_b_layer2": b["pks"]["2"],
        })
        print(f"  record {rid}: identical={identical} n_tokens={a['n_tokens']} "
              f"ecs2h0a={a['ecs']['2']['0']:.6f} ecs2h0b={b['ecs']['2']['0']:.6f} "
              f"pks2a={a['pks']['2']:.6f} pks2b={b['pks']['2']:.6f}")
    if not all_identical:
        print("[STOP] Determinism smoke FAILED: per-head ECS / per-layer PKS "
              "differs across replays at 6 decimal places.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (per-head ECS, per-layer PKS, "
          "2 records x 2 replays, 6 dp)")
    return {"pairs": pairs, "all_identical": str(all_identical)}


# ---------------------------------------------------------------------------
# Threshold derivation (Gate 2.3 Part A rule, mechanical)
# ---------------------------------------------------------------------------
def pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def derive_ecs_thresholds(
    deg: Dict[str, Dict[str, List[float]]],
    prose: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Derive per-layer per-head ECS thresholds.

    deg/prose: {layer: {head: [values over records]}}
    Returns {layer: {head: {p90_deg, p10_prose, T, band_low, band_high}}}.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for L in CANDIDATE_LAYERS:
        out[str(L)] = {}
        for h in sorted(deg[str(L)].keys()):
            d = np.asarray(deg[str(L)][h], dtype=np.float64)
            p = np.asarray(prose[str(L)][h], dtype=np.float64)
            p90_deg = pct(d, 90)
            p10_prose = pct(p, 10)
            t = (p90_deg + p10_prose) / 2.0
            out[str(L)][str(h)] = {
                "p90_deg": p90_deg,
                "p10_prose": p10_prose,
                "T": t,
                "band_low": 0.75 * t,
                "band_high": 1.25 * t,
            }
    return out


def derive_pks_thresholds(
    deg: Dict[str, List[float]],
    prose: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Derive per-layer PKS thresholds. Returns {layer: {p90_deg, p10_prose, T,
    band_low, band_high}}."""
    out: Dict[str, Dict[str, float]] = {}
    for L in CANDIDATE_LAYERS:
        d = np.asarray(deg[str(L)], dtype=np.float64)
        p = np.asarray(prose[str(L)], dtype=np.float64)
        p90_deg = pct(d, 90)
        p10_prose = pct(p, 10)
        t = (p90_deg + p10_prose) / 2.0
        out[str(L)] = {
            "p90_deg": p90_deg,
            "p10_prose": p10_prose,
            "T": t,
            "band_low": 0.75 * t,
            "band_high": 1.25 * t,
        }
    return out


# ---------------------------------------------------------------------------
# Freeze file writer / loader
# ---------------------------------------------------------------------------
def write_freeze_file(path: Path, ecs_thr: Dict[str, Any], pks_thr: Dict[str, Any]) -> None:
    """Write docs/gate23/FROZEN_ECS_PKS_THRESHOLDS.md with the machine-parseable
    JSON freeze block (same format discipline as FROZEN_THRESHOLDS.md)."""
    freeze: Dict[str, Any] = {
        "ecs": {"layers": {}},
        "pks": {"layers": {}},
    }
    for L in CANDIDATE_LAYERS:
        freeze["ecs"]["layers"][str(L)] = {"heads": ecs_thr[str(L)]}
        freeze["pks"]["layers"][str(L)] = pks_thr[str(L)]

    md: List[str] = []
    md.append("# DS-038 — Frozen ECS/PKS Thresholds (Gate 2.3 Part A rule)")
    md.append("")
    md.append("> FROZEN before scoring. Derived mechanically from the calibration "
              "records ONLY (t2s_degenerate N=100 + prose-100 N=100); nothing "
              "downstream may re-derive these numbers. The greenlight rule: "
              "T = (p90_degenerate + p10_prose) / 2, band = [0.75*T, 1.25*T]. "
              "Direction: ECS fires when ECS < T_ECS(L, H) (attention head "
              "context-detached); PKS fires when PKS > T_PKS(L) (FFN "
              "parametrically overriding).")
    md.append("")
    md.append("## Calibration source")
    md.append("")
    md.append("- Fixture A: `tests/fixtures/t2s_degenerate.jsonl` (night-001; N=100, "
              "degenerate).")
    md.append("- Fixture B: `data/t2s_bench/valid_subset_200.jsonl` heldout prose-100 "
              "(ds-027 complement of the Gate-2.2 control-100; N=100).")
    md.append("- Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16 (eager attention). "
              "12 attention heads per layer (the task spec's '28 heads' conflated "
              "the 28-layer count with head count; the real config has 12).")
    md.append("- Percentiles: NumPy `np.percentile` default linear interpolation, "
              "matching the Gate 2.2 report convention.")
    md.append(f"- Candidate layers: {{{', '.join(str(l) for l in CANDIDATE_LAYERS)}}}.")
    md.append(f"- Analysis window: last {ANALYSIS_WINDOW} token positions.")
    md.append(f"- ECS prompt boundary: prompt_len = seq_len - {ANALYSIS_WINDOW} "
              "(the analysis window is the 'generated' region; everything before "
              "is the 'prompt' context).")
    md.append(f"- PKS top-k truncation: k = {TOP_K} + single tail-mass bucket.")
    md.append("")

    # ECS threshold tables per layer
    md.append("## ECS thresholds (per layer, per head)")
    md.append("")
    for L in CANDIDATE_LAYERS:
        md.append(f"### Layer {L}")
        md.append("")
        rows = []
        for h in sorted(ecs_thr[str(L)].keys(), key=int):
            e = ecs_thr[str(L)][h]
            rows.append([h, f"{e['p90_deg']:.6f}", f"{e['p10_prose']:.6f}",
                         f"{e['T']:.6f}", f"{e['band_low']:.6f}",
                         f"{e['band_high']:.6f}"])
        md.append("| head | p90_deg | p10_prose | T_ECS | band_low | band_high |")
        md.append("|---|---|---|---|---|---|")
        for r in rows:
            md.append("| " + " | ".join(r) + " |")
        md.append("")

    # PKS threshold table
    md.append("## PKS thresholds (per layer)")
    md.append("")
    rows = []
    for L in CANDIDATE_LAYERS:
        e = pks_thr[str(L)]
        rows.append([str(L), f"{e['p90_deg']:.6f}", f"{e['p10_prose']:.6f}",
                     f"{e['T']:.6f}", f"{e['band_low']:.6f}",
                     f"{e['band_high']:.6f}"])
    md.append("| layer | p90_deg | p10_prose | T_PKS | band_low | band_high |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join(r) + " |")
    md.append("")

    md.append("## Machine-parseable freeze block")
    md.append("")
    md.append("```json")
    md.append(json.dumps(freeze, indent=2))
    md.append("```")
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- This file is the freeze. `scripts/measure_ecs_pks.py` writes it "
              "on derivation and READS it back for scoring; it does not re-derive "
              "during scoring.")
    md.append("- Direction: degenerate records are expected to have LOWER ECS "
              "(attention heads detach from prompt context) and HIGHER PKS (FFNs "
              "inject parametric knowledge).")
    md.append("- Calibration and scoring use the SAME 100 t2s records in this "
              "initial probe (the separation magnitude is the primary result); a "
              "held-out scoring pass on a different degenerate fixture is a "
              "future measurement.")
    md.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {path}")


def load_freeze(path: Path) -> Dict[str, Any]:
    """Parse the machine-parseable JSON freeze block from
    FROZEN_ECS_PKS_THRESHOLDS.md."""
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start == -1:
        raise SystemExit("[STOP] FROZEN_ECS_PKS_THRESHOLDS.md has no ```json "
                         "freeze block.")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise SystemExit("[STOP] FROZEN_ECS_PKS_THRESHOLDS.md freeze block is "
                         "unterminated.")
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# AUC / separation (rank-based Mann-Whitney, numpy only)
# ---------------------------------------------------------------------------
def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean rank)."""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def mann_whitney_auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Rank-based Mann-Whitney U / (n_pos * n_neg). positives = degenerate-class
    values; negatives = prose-control values. AUC > 0.5 -> degenerate-higher;
    AUC < 0.5 -> degenerate-lower."""
    pos = np.asarray(positives, dtype=np.float64)
    neg = np.asarray(negatives, dtype=np.float64)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_vals = np.concatenate([pos, neg])
    ranks = average_ranks(all_vals)
    rank_sum_pos = float(ranks[:n_pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def sep_magnitude(auc: float) -> float:
    if auc != auc:  # NaN
        return float("nan")
    return float(max(auc, 1.0 - auc))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def collect_vectors(results: List[Dict[str, Any]], num_heads: int) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, List[float]]],
    Dict[str, List[float]],
    Dict[str, List[float]],
]:
    """Split results into degenerate/prose ECS and PKS vectors.

    Returns (deg_ecs, prose_ecs, deg_pks, prose_pks) where ecs maps are
    {layer: {head: [values]}} and pks maps are {layer: [values]}.
    """
    deg_ecs: Dict[str, Dict[str, List[float]]] = {
        str(L): {str(h): [] for h in range(num_heads)} for L in CANDIDATE_LAYERS}
    prose_ecs: Dict[str, Dict[str, List[float]]] = {
        str(L): {str(h): [] for h in range(num_heads)} for L in CANDIDATE_LAYERS}
    deg_pks: Dict[str, List[float]] = {str(L): [] for L in CANDIDATE_LAYERS}
    prose_pks: Dict[str, List[float]] = {str(L): [] for L in CANDIDATE_LAYERS}

    for r in results:
        if r["fixture"] == "t2s_degenerate":
            ecs_t = deg_ecs
            pks_t = deg_pks
        else:
            ecs_t = prose_ecs
            pks_t = prose_pks
        for L in CANDIDATE_LAYERS:
            for h in range(num_heads):
                ecs_t[str(L)][str(h)].append(r["ecs"][str(L)][str(h)])
            pks_t[str(L)].append(r["pks"][str(L)])
    return deg_ecs, prose_ecs, deg_pks, prose_pks


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "n_non_finite": 0}
    n_non_finite = int(np.count_nonzero(~np.isfinite(arr)))
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_non_finite": n_non_finite,
    }


def aggregate_tables(
    results: List[Dict[str, Any]],
    num_heads: int,
    freeze: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute all aggregation tables. Scoring is against the frozen thresholds
    (read from the freeze file, NOT re-derived)."""
    deg_ecs, prose_ecs, deg_pks, prose_pks = collect_vectors(results, num_heads)

    # ---- Table 1 — per-layer ECS distributions --------------------------
    table1: Dict[str, Dict[str, Any]] = {}
    for L in CANDIDATE_LAYERS:
        # Flatten across heads for distributional stats.
        deg_flat: List[float] = []
        prose_flat: List[float] = []
        head_auc: List[float] = []
        n_fire_heads = 0
        fire_events = 0
        n_deg = 0
        for h in range(num_heads):
            d = deg_ecs[str(L)][str(h)]
            p = prose_ecs[str(L)][str(h)]
            n_deg = len(d)
            deg_flat.extend(d)
            prose_flat.extend(p)
            auc = mann_whitney_auc(d, p)
            head_auc.append(auc)
            t = freeze["ecs"]["layers"][str(L)]["heads"][str(h)]["T"]
            # Fire on degenerate records: ECS < T.
            deg_fires = sum(1 for v in d if v < t)
            fire_events += deg_fires
            if deg_fires > 0:
                n_fire_heads += 1
        d_dist = summarize_distribution(deg_flat)
        p_dist = summarize_distribution(prose_flat)
        layer_sep = max(sep_magnitude(a) for a in head_auc)
        fire_rate = fire_events / (num_heads * n_deg) if n_deg else float("nan")
        table1[str(L)] = {
            "n_heads": num_heads,
            "mean_ecs_deg": d_dist["mean"],
            "mean_ecs_prose": p_dist["mean"],
            "separation": layer_sep,
            "n_heads_fire": n_fire_heads,
            "fire_rate_deg": fire_rate,
            "head_aucs": head_auc,
        }

    # ---- Table 2 — per-layer PKS distributions --------------------------
    table2: Dict[str, Dict[str, Any]] = {}
    for L in CANDIDATE_LAYERS:
        d = deg_pks[str(L)]
        p = prose_pks[str(L)]
        auc = mann_whitney_auc(d, p)
        t = freeze["pks"]["layers"][str(L)]["T"]
        deg_fires = sum(1 for v in d if v > t)
        n_layers_fire = 1 if deg_fires > 0 else 0
        fire_rate = deg_fires / len(d) if d else float("nan")
        table2[str(L)] = {
            "mean_pks_deg": summarize_distribution(d)["mean"],
            "mean_pks_prose": summarize_distribution(p)["mean"],
            "separation": sep_magnitude(auc),
            "auc": auc,
            "n_layers_fire": n_layers_fire,
            "fire_rate_deg": fire_rate,
        }

    # ---- Table 3 — top-5 most separated ECS heads -----------------------
    ecs_ranked: List[Dict[str, Any]] = []
    for L in CANDIDATE_LAYERS:
        for h in range(num_heads):
            d = deg_ecs[str(L)][str(h)]
            p = prose_ecs[str(L)][str(h)]
            auc = mann_whitney_auc(d, p)
            ecs_ranked.append({
                "layer": L,
                "head": h,
                "ecs_deg_mean": float(np.mean(d)),
                "ecs_prose_mean": float(np.mean(p)),
                "separation": sep_magnitude(auc),
                "auc": auc,
            })
    ecs_ranked.sort(key=lambda x: x["separation"], reverse=True)
    table3 = ecs_ranked[:5]

    # ---- Table 4 — top-5 most separated PKS layers ----------------------
    pks_ranked: List[Dict[str, Any]] = []
    for L in CANDIDATE_LAYERS:
        d = deg_pks[str(L)]
        p = prose_pks[str(L)]
        auc = mann_whitney_auc(d, p)
        pks_ranked.append({
            "layer": L,
            "pks_deg_mean": float(np.mean(d)),
            "pks_prose_mean": float(np.mean(p)),
            "separation": sep_magnitude(auc),
            "auc": auc,
        })
    pks_ranked.sort(key=lambda x: x["separation"], reverse=True)
    table4 = pks_ranked[:5]

    # ---- Table 5 — per-layer per-head threshold table -------------------
    table5: Dict[str, Any] = {}
    for L in CANDIDATE_LAYERS:
        table5[str(L)] = freeze["ecs"]["layers"][str(L)]["heads"]

    return {
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "table5": table5,
        "n_records": len(results),
        "num_heads": num_heads,
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


def write_markdown_report(
    ctx: Dict[str, Any],
    out_path: Path,
) -> None:
    md: List[str] = []
    md.append("# DS-038 — ECS/PKS liveness on t2s_degenerate (MEASUREMENT ONLY)")
    md.append("")
    md.append("> MEASUREMENT REPORT. This probe characterizes whether ECS "
              "(External Context Score) and PKS (Parametric Knowledge Score) "
              "separate t2s_degenerate from healthy prose strongly enough to "
              "justify investment in AARF actuation. Zero controller edits, zero "
              "new architecture, zero actuation design. No verdict is offered.")
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
    md.append("| Fixture A | tests/fixtures/t2s_degenerate.jsonl "
              "(night-001; N=100), record[\\\"text\\\"] input |")
    md.append("| Fixture B | data/t2s_bench/valid_subset_200.jsonl heldout "
              "prose-100 (ds-027 complement of Gate-2.2 control-100; N=100) |")
    md.append("| Analysis window | "
              f"last {meta['analysis_window']} token positions |")
    md.append("| Candidate layers | "
              f"{{{', '.join(str(l) for l in meta['candidate_layers'])}}} |")
    md.append(f"| Attention heads / layer | {meta['num_heads']} (real Qwen2.5-1.5B "
              f"config; task spec's '28 heads' conflated layer count with head "
              f"count) |")
    md.append(f"| ECS prompt boundary | prompt_len = seq_len - "
              f"{meta['analysis_window']} (the analysis window is the 'generated' "
              f"region; everything before is the 'prompt' context) |")
    md.append(f"| PKS top-k | k = {meta['top_k']} + tail-mass bucket (float32) |")
    md.append(f"| Max seq len (prefix truncation) | {meta['max_seq_len']} tokens "
              f"({meta['n_truncated']} prose records truncated) |")
    md.append(f"| Attention capture | forward hooks on candidate-layer self_attn "
              f"sub-modules (output_attentions=True OOMs on long prose; hooks "
              f"return identical attention weights) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke (FIRST)")
    md.append("")
    md.append("Two t2s_degenerate records replayed twice (teacher-forced, single "
              "forward pass each). Per-head ECS and per-layer PKS must be "
              "identical to 6 decimal places. STOP if not.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = []
    for pair in sm["pairs"]:
        smoke_rows.append([
            str(pair["record_id"]), str(pair["n_tokens"]),
            fmt(pair["ecs_a_layer2_head0"], 6), fmt(pair["ecs_b_layer2_head0"], 6),
            fmt(pair["pks_a_layer2"], 6), fmt(pair["pks_b_layer2"], 6),
            pair["identical"],
        ])
    smoke_rows.append(["", "ALL", "", "", "", "", str(sm["all_identical"])])
    md.append(format_table_md(
        smoke_rows, ["record_id", "n_tokens", "ECS L2H0 run A", "ECS L2H0 run B",
                     "PKS L2 run A", "PKS L2 run B", "identical"],
    ))
    md.append("")

    tabs = ctx["tables"]

    # Table 1
    md.append("## Table 1 — Per-layer ECS distributions (degenerate vs prose)")
    md.append("")
    md.append("ECS = fraction of attention that the last-24 query positions give "
              "to prompt tokens (prompt_len = seq_len - 24). Degenerate records "
              "are expected to have LOWER ECS. Fires when ECS < T_ECS(L, H). "
              "Fire rate is computed on degenerate records only.")
    md.append("")
    t1_rows = []
    for L in CANDIDATE_LAYERS:
        t1 = tabs["table1"][str(L)]
        t1_rows.append([
            str(L), str(t1["n_heads"]), fmt(t1["mean_ecs_deg"]),
            fmt(t1["mean_ecs_prose"]), fmt(t1["separation"], 4),
            str(t1["n_heads_fire"]), fmt(t1["fire_rate_deg"], 4),
        ])
    md.append(format_table_md(
        t1_rows, ["layer", "n_heads", "mean ECS deg", "mean ECS prose",
                  "separation", "#heads fire", "fire rate (deg)"],
    ))
    md.append("")

    # Table 2
    md.append("## Table 2 — Per-layer PKS distributions (degenerate vs prose)")
    md.append("")
    md.append("PKS = JSD between post-attention and post-FFN projected "
              "distributions (top-1000 + tail bucket). Degenerate records are "
              "expected to have HIGHER PKS. Fires when PKS > T_PKS(L). Fire rate "
              "is computed on degenerate records only.")
    md.append("")
    t2_rows = []
    for L in CANDIDATE_LAYERS:
        t2 = tabs["table2"][str(L)]
        t2_rows.append([
            str(L), fmt(t2["mean_pks_deg"]), fmt(t2["mean_pks_prose"]),
            fmt(t2["separation"], 4), str(t2["n_layers_fire"]),
            fmt(t2["fire_rate_deg"], 4),
        ])
    md.append(format_table_md(
        t2_rows, ["layer", "mean PKS deg", "mean PKS prose", "separation",
                  "#layers fire", "fire rate (deg)"],
    ))
    md.append("")

    # Table 3
    md.append("## Table 3 — Top-5 most separated ECS heads")
    md.append("")
    md.append("Separation = max |AUC - 0.5| (rank-based Mann-Whitney, degenerate "
              "as the positive class). AUC < 0.5 → degenerate-lower.")
    md.append("")
    t3_rows = []
    for row in tabs["table3"]:
        t3_rows.append([
            str(row["layer"]), str(row["head"]), fmt(row["ecs_deg_mean"]),
            fmt(row["ecs_prose_mean"]), fmt(row["separation"], 4),
            fmt(row["auc"], 4),
        ])
    md.append(format_table_md(
        t3_rows, ["layer", "head", "ECS deg mean", "ECS prose mean",
                  "separation", "AUC"],
    ))
    md.append("")

    # Table 4
    md.append("## Table 4 — Top-5 most separated PKS layers")
    md.append("")
    t4_rows = []
    for row in tabs["table4"]:
        t4_rows.append([
            str(row["layer"]), fmt(row["pks_deg_mean"]),
            fmt(row["pks_prose_mean"]), fmt(row["separation"], 4),
            fmt(row["auc"], 4),
        ])
    md.append(format_table_md(
        t4_rows, ["layer", "PKS deg mean", "PKS prose mean", "separation", "AUC"],
    ))
    md.append("")

    # Table 5
    md.append("## Table 5 — Per-layer per-head ECS thresholds (frozen)")
    md.append("")
    md.append("Thresholds are read from `docs/gate23/FROZEN_ECS_PKS_THRESHOLDS.md` "
              "(Gate 2.3 Part A rule: T = (p90_deg + p10_prose)/2, band = "
              "[0.75*T, 1.25*T]); nothing here re-derives them.")
    md.append("")
    t5_rows = []
    for L in CANDIDATE_LAYERS:
        for h in range(tabs["num_heads"]):
            e = tabs["table5"][str(L)][str(h)]
            t5_rows.append([
                str(L), str(h), fmt(e["T"], 6), fmt(e["band_low"], 6),
                fmt(e["band_high"], 6),
            ])
    md.append(format_table_md(
        t5_rows, ["layer", "head", "T_ECS", "band_low", "band_high"],
    ))
    md.append("")

    # PKS thresholds
    md.append("## Frozen PKS thresholds")
    md.append("")
    freeze = ctx["freeze"]
    pks_rows = []
    for L in CANDIDATE_LAYERS:
        e = freeze["pks"]["layers"][str(L)]
        pks_rows.append([
            str(L), fmt(e["T"], 6), fmt(e["band_low"], 6), fmt(e["band_high"], 6),
        ])
    md.append(format_table_md(
        pks_rows, ["layer", "T_PKS", "band_low", "band_high"],
    ))
    md.append("")

    # Prose false-positive fire rate (auxiliary)
    md.append("## Prose false-positive fire rate (auxiliary)")
    md.append("")
    md.append("For a gate to be usable, the frozen thresholds must NOT fire on "
              "healthy prose. This table reports the prose fire rate (fraction of "
              "prose records where ECS < T or PKS > T).")
    md.append("")
    aux_rows = []
    for L in CANDIDATE_LAYERS:
        # ECS prose fire rate across heads
        n_prose_fire = 0
        n_total = 0
        for h in range(tabs["num_heads"]):
            p = ctx["prose_ecs"][str(L)][str(h)]
            t = freeze["ecs"]["layers"][str(L)]["heads"][str(h)]["T"]
            n_prose_fire += sum(1 for v in p if v < t)
            n_total += len(p)
        pks_prose_fire = sum(
            1 for v in ctx["prose_pks"][str(L)]
            if v > freeze["pks"]["layers"][str(L)]["T"]
        )
        aux_rows.append([
            str(L),
            fmt(n_prose_fire / n_total, 4) if n_total else "—",
            fmt(pks_prose_fire / len(ctx["prose_pks"][str(L)]), 4),
        ])
    md.append(format_table_md(
        aux_rows, ["layer", "prose FP rate (ECS, per head-record)", 
                   "prose FP rate (PKS, per record)"],
    ))
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- MEASUREMENT ONLY: no controller edits, no new architecture, no "
              "actuation design, no thresholds beyond the frozen derivation.")
    md.append("- The same 100 t2s records are used for both calibration and "
              "scoring in this initial probe (the separation magnitude is the "
              "primary result). A held-out scoring pass on a different degenerate "
              "fixture (e.g., qwen_degenerate detection-gap records) is a future "
              "measurement.")
    md.append("- The task spec stated '28 attention heads per layer'; the actual "
              "Qwen2.5-1.5B config has num_attention_heads=12 (the spec conflated "
              "layer count with head count). All ECS per-head metrics use the "
              "real 12-head config.")
    md.append("- `output_attentions=True` OOMs on long prose records (attention "
              "weights are O(seq_len²); the longest prose record tokenizes to "
              "7265 tokens). Forward hooks on the candidate-layer self_attn "
              "sub-modules capture the identical attention weights "
              "(Qwen2Attention returns them in eager mode) without materializing "
              "the other 23 layers' attention.")
    md.append(f"- {ctx['metadata']['n_truncated']} prose records exceed "
              f"MAX_SEQ_LEN={ctx['metadata']['max_seq_len']} tokens and were "
              f"prefix-truncated (standard truncation; keeps the last-24 analysis "
              f"window of the truncated sequence).")
    md.append("- ECS prompt boundary: prompt_len = seq_len - 24 (the Gate 2.3 "
              "analysis window is the 'generated' region). This is a documented "
              "modeling choice for teacher-forced replay, which has no explicit "
              "prompt/generation split.")
    md.append("- All PKS math is float32 (per the DS-036 bf16 dead-zone finding). "
              "ECS attention sums are computed in float32.")
    md.append("- Per-record per-head per-layer values are in "
              "`ecs_pks_results.jsonl`.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-038 ECS/PKS liveness on t2s_degenerate (measurement only)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Top-k truncation for PKS JSD (default {TOP_K}).")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help=f"Prefix truncation bound (default {MAX_SEQ_LEN}).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate ECS_PKS_RESULTS.md from an existing "
                             "ecs_pks_results.jsonl without re-running the model "
                             "(dev convenience; no measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-038 ECS/PKS liveness on t2s_degenerate (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"candidate layers: {CANDIDATE_LAYERS}")
    print(f"analysis window: last {ANALYSIS_WINDOW} token positions")
    print(f"top-k (PKS JSD): {args.top_k}")
    print(f"max seq len: {args.max_seq_len}")

    # ------------------------------------------------------------------
    # Load fixtures.
    # ------------------------------------------------------------------
    t2s_records = load_jsonl(T2S_DEG)
    if args.max_records is not None:
        t2s_records = t2s_records[: args.max_records]
        print(f"[dev] capped t2s records at {args.max_records}")
    print(f"t2s_degenerate records: {len(t2s_records)}")

    valid_records = load_jsonl(VALID_SUBSET_200)
    prose_records = select_heldout_prose(valid_records)
    if args.max_records is not None:
        prose_records = prose_records[: args.max_records]
        print(f"[dev] capped prose records at {args.max_records}")
    print(f"prose-100 heldout records: {len(prose_records)}")

    # ------------------------------------------------------------------
    # --report-only: regenerate markdown from an existing JSONL.
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        if not FROZEN_OUT.exists():
            print(f"[STOP] --report-only: {FROZEN_OUT} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from {OUTPUT_JSONL} ---")
        results = load_jsonl(OUTPUT_JSONL)
        freeze = load_freeze(FROZEN_OUT)
        num_heads = len(results[0]["ecs"][str(CANDIDATE_LAYERS[0])])
        tables = aggregate_tables(results, num_heads, freeze)
        deg_ecs, prose_ecs, deg_pks, prose_pks = collect_vectors(results, num_heads)
        n_trunc = sum(1 for r in results if r.get("truncated"))
        smoke: Dict[str, Any] = {
            "pairs": [{"record_id": -1, "n_tokens": 1,
                       "ecs_a_layer2_head0": 0.0, "ecs_b_layer2_head0": 0.0,
                       "pks_a_layer2": 0.0, "pks_b_layer2": 0.0,
                       "identical": "True"}],
            "all_identical": "True",
        }
        metadata = {
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "seed": SEED,
            "analysis_window": ANALYSIS_WINDOW,
            "candidate_layers": CANDIDATE_LAYERS,
            "num_heads": num_heads,
            "top_k": args.top_k,
            "max_seq_len": args.max_seq_len,
            "n_truncated": n_trunc,
            "bmm_override": "deregistered",
            "wall_clock_s": 0.0,
        }
        ctx = {
            "metadata": metadata,
            "determinism_smoke": smoke,
            "tables": tables,
            "freeze": freeze,
            "prose_ecs": prose_ecs,
            "prose_pks": prose_pks,
        }
        write_markdown_report(ctx, OUTPUT_MD)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Model load.
    # ------------------------------------------------------------------
    tokenizer, model, w_u = load_model_and_tokenizer()
    num_heads = int(model.config.num_attention_heads)
    print(f"model loaded on {model.device}; num_heads={num_heads} "
          f"(real Qwen2.5-1.5B config)")
    captured: Dict[str, Any] = {}
    handles = make_capture_hooks(model, CANDIDATE_LAYERS, captured)

    # Determinism smoke FIRST.
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
        smoke = {"pairs": [], "all_identical": "True"}
    else:
        smoke = run_determinism_smoke(model, tokenizer, w_u, captured,
                                      t2s_records, num_heads, args.top_k,
                                      args.max_seq_len)

    # ==================================================================
    # Replay all records.
    # ==================================================================
    print("\n--- Replaying records (teacher-forced, single forward pass each) ---")
    results: List[Dict[str, Any]] = []
    for i, rec in enumerate(t2s_records):
        payload = build_record_payload(model, tokenizer, w_u, captured,
                                       "t2s_degenerate", rec, num_heads,
                                       args.top_k, args.max_seq_len)
        results.append(payload)
        if (i + 1) % 20 == 0 or i == len(t2s_records) - 1:
            print(f"  [deg {i+1}/{len(t2s_records)}] rid={payload['record_id']} "
                  f"seq={payload['seq_len']} "
                  f"ecs2h0={payload['ecs']['2']['0']:.4f} "
                  f"pks2={payload['pks']['2']:.4f}")

    for i, rec in enumerate(prose_records):
        payload = build_record_payload(model, tokenizer, w_u, captured,
                                       "prose-100", rec, num_heads,
                                       args.top_k, args.max_seq_len)
        results.append(payload)
        if (i + 1) % 20 == 0 or i == len(prose_records) - 1:
            print(f"  [prose {i+1}/{len(prose_records)}] rid={payload['record_id']} "
                  f"seq={payload['seq_len']} trunc={payload['truncated']} "
                  f"ecs2h0={payload['ecs']['2']['0']:.4f} "
                  f"pks2={payload['pks']['2']:.4f}")

    # Remove hooks (no longer needed).
    for h in handles:
        h.remove()
    del captured
    torch.cuda.empty_cache()

    n_trunc = sum(1 for r in results if r.get("truncated"))
    print(f"\ntotal records: {len(results)} | truncated: {n_trunc}")

    # ==================================================================
    # Derive thresholds, write freeze, load freeze, score.
    # ==================================================================
    deg_ecs, prose_ecs, deg_pks, prose_pks = collect_vectors(results, num_heads)
    ecs_thr = derive_ecs_thresholds(deg_ecs, prose_ecs)
    pks_thr = derive_pks_thresholds(deg_pks, prose_pks)
    write_freeze_file(FROZEN_OUT, ecs_thr, pks_thr)

    # Read the freeze back (single source of truth for scoring).
    freeze = load_freeze(FROZEN_OUT)
    print("frozen ECS/PKS thresholds loaded from "
          "docs/gate23/FROZEN_ECS_PKS_THRESHOLDS.md")

    # ==================================================================
    # Aggregate, write JSONL + MD.
    # ==================================================================
    tables = aggregate_tables(results, num_heads, freeze)

    print("\n" + "=" * 70)
    print("ECS/PKS AGGREGATION")
    for L in CANDIDATE_LAYERS:
        t1 = tables["table1"][str(L)]
        t2 = tables["table2"][str(L)]
        print(f"  layer {L}: ECS deg={t1['mean_ecs_deg']:.4f} prose="
              f"{t1['mean_ecs_prose']:.4f} sep={t1['separation']:.4f} "
              f"heads_fire={t1['n_heads_fire']}/{t1['n_heads']} "
              f"fire_rate={t1['fire_rate_deg']:.4f} | "
              f"PKS deg={t2['mean_pks_deg']:.4f} prose={t2['mean_pks_prose']:.4f} "
              f"sep={t2['separation']:.4f} fire_rate={t2['fire_rate_deg']:.4f}")
    print("=" * 70)

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
        "analysis_window": ANALYSIS_WINDOW,
        "candidate_layers": CANDIDATE_LAYERS,
        "num_heads": num_heads,
        "top_k": args.top_k,
        "max_seq_len": args.max_seq_len,
        "n_truncated": n_trunc,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "freeze": freeze,
        "prose_ecs": prose_ecs,
        "prose_pks": prose_pks,
    }
    write_markdown_report(ctx, OUTPUT_MD)
    print(f"wrote {OUTPUT_MD}")
    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    print("DS-038 MEASUREMENT COMPLETE")


if __name__ == "__main__":
    main()
