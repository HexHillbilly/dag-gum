#!/usr/bin/env python3
"""FP32 residual-kick liveness probe (MEASUREMENT ONLY).

DS-036 concluded the residual channel is DEAD under greedy decoding: a
0.50-L2 kick along the Jacobian's optimal direction v1 produced mean
KL ~= 10^-5 (9.4e-6). But DS-036's own caveat records this is a bf16
quantization artifact: at layer 26 the residual norm is O(250) and bf16's
absolute step is O(0.6), so a 0.5-L2 kick rounds to ~0 for 99.2% of
directions. Crucially, DS-036 computed v1 in fp32 but RESTORED bf16 before
Phase 2 (the kick + KL), so the "dead" verdict reflects bf16 quantization
of the kick, not the true effect of the optimal direction.

This probe re-runs Phase 2 in FULL fp32: same record, same v1 (fp32
finite-difference), same 0.50-L2 force, same KL formula — only the
kick/decoding dtype changes. It also re-runs Phase 2 in bf16 as an
in-run comparison to confirm the dead-zone reproduces. If fp32 KL >> 10^-5,
the "residual is dead" verdict is overturned to "residual is
precision-starved."

MEASUREMENT ONLY. No controller edits. Reuses DS-036's functions via
importlib so the methodology (KL formula, orthogonalization, probe
directions, power iteration) is identical.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Reuse DS-036's exact functions (byte-identical methodology).
_DS036 = Path(__file__).resolve().parent / "measure_greens_function.py"
_spec = importlib.util.spec_from_file_location("ds036", str(_DS036))
mgr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mgr)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
INTERVENTION_LAYER = 26
ETA = 0.50
EPSILON = 1e-3
K_PROBES = 128
NUM_POWER_ITER = 10
MAX_NEW_TOKENS = 10
LIVENESS_KL_ALIVE = 1e-2
LIVENESS_KL_DEAD = 1e-3

HELDOUT_DEG_V2 = Path("tests/fixtures/heldout_degenerate_v2.jsonl")
OUTPUT_JSONL = Path("docs/gate23/fp32_residual_results.jsonl")
OUTPUT_MD = Path("docs/gate23/FP32_RESIDUAL_RESULTS.md")


def classify(mean_kl: float) -> str:
    if mean_kl > LIVENESS_KL_ALIVE:
        return "ALIVE"
    if mean_kl < LIVENESS_KL_DEAD:
        return "DEAD"
    return "INCONCLUSIVE (between 1e-4 and 1e-2)"


@torch.no_grad()
def phase2(model, tokenizer, prompt, h_t, v1, eta, layer, max_new_tokens):
    """Dormant + kicked greedy generation, per-token KL. Returns a dict."""
    dormant = mgr.greedy_generate_dormant(model, tokenizer, prompt, max_new_tokens)
    kicked = mgr.greedy_generate_kicked(
        model, tokenizer, prompt, h_t, v1, eta, layer, max_new_tokens
    )
    n = min(len(kicked["logits"]), len(dormant["logits"]), max_new_tokens)
    per_token_kl = [
        mgr.kl_divergence(kicked["logits"][t], dormant["logits"][t]) for t in range(n)
    ]
    mean_kl = float(np.mean(per_token_kl)) if per_token_kl else float("nan")
    return {
        "per_token_kl": per_token_kl,
        "mean_kl": mean_kl,
        "first_token_kl": per_token_kl[0] if per_token_kl else float("nan"),
        "n_generated": kicked["n_generated"],
        "applied_l2_delta": kicked["state"]["l2_delta"],
        "cos_residual": kicked["state"]["cos_residual"],
        "kicked_text": tokenizer.decode(kicked["generated_ids"][0], skip_special_tokens=True),
        "dormant_text": tokenizer.decode(dormant["generated_ids"][0], skip_special_tokens=True),
        "verdict": classify(mean_kl),
    }


def main() -> None:
    t0 = time.time()
    mgr.seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("FP32 residual-kick liveness probe (MEASUREMENT ONLY)")
    print("=" * 70)
    print(f"model {MODEL_NAME}@{MODEL_REVISION} | device {device} | layer "
          f"{INTERVENTION_LAYER} | eta {ETA} | eps {EPSILON} | K {K_PROBES}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    # ---- Phase 1: estimate v1 in FULL fp32 (no dead zone) ----
    print("\n--- Load fp32 model + Phase 1 (v1 via finite differences, fp32) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, torch_dtype=torch.float32
    ).eval()
    if device == "cuda":
        model = model.to("cuda")
    print(f"model dtype: {model.dtype}")

    heldout = mgr.load_jsonl(HELDOUT_DEG_V2)
    rec = next((r for r in heldout if int(r["id"]) == 0), None)
    assert rec is not None, "no record id=0"
    prompt = rec["mutated_prompt"]

    r1 = mgr.run_phase1_probe(model, tokenizer, prompt, INTERVENTION_LAYER,
                              K_PROBES, SEED, EPSILON)
    ph1, v1 = mgr.finalize_phase1(
        r1["est"], r1["E"], r1["h_t"], r1["baseline_logits"],
        EPSILON, SEED, NUM_POWER_ITER, "fp32",
    )
    prompt_len = int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1])

    # ---- Phase 2a: fp32 kick + KL ----
    print("\n--- Phase 2a: bounded kick along v1 in FP32 ---")
    h_t_fp32 = mgr.capture_baseline(model, tokenizer, prompt, INTERVENTION_LAYER)["h_t"]
    ph2_fp32 = phase2(model, tokenizer, prompt, h_t_fp32, v1, ETA,
                      INTERVENTION_LAYER, MAX_NEW_TOKENS)
    print(f"  fp32 applied L2 delta = {ph2_fp32['applied_l2_delta']:.4f} "
          f"(target {ETA})")
    print(f"  fp32 per-token KL: " + ", ".join(f"{v:.4e}" for v in ph2_fp32["per_token_kl"]))
    print(f"  fp32 mean KL = {ph2_fp32['mean_kl']:.4e} -> {ph2_fp32['verdict']}")

    # ---- Phase 2b: bf16 kick + KL (in-run dead-zone reproduction) ----
    print("\n--- Phase 2b: same kick in production bf16 (comparison) ---")
    if device == "cuda":
        model = model.to(torch.bfloat16)
        torch.cuda.empty_cache()
    h_t_bf16 = mgr.capture_baseline(model, tokenizer, prompt, INTERVENTION_LAYER)["h_t"]
    ph2_bf16 = phase2(model, tokenizer, prompt, h_t_bf16, v1, ETA,
                      INTERVENTION_LAYER, MAX_NEW_TOKENS)
    print(f"  bf16 applied L2 delta = {ph2_bf16['applied_l2_delta']:.4f} "
          f"(target {ETA})")
    print(f"  bf16 per-token KL: " + ", ".join(f"{v:.4e}" for v in ph2_bf16["per_token_kl"]))
    print(f"  bf16 mean KL = {ph2_bf16['mean_kl']:.4e} -> {ph2_bf16['verdict']}")

    # ---- Write outputs ----
    metadata = {
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "device": device, "torch_version": torch.__version__, "seed": SEED,
        "record_id": 0, "prompt_len": prompt_len,
        "layer": INTERVENTION_LAYER, "eta": ETA, "epsilon": EPSILON,
        "k_probes": K_PROBES, "num_power_iter": NUM_POWER_ITER,
        "max_new_tokens": MAX_NEW_TOKENS,
        "vocab_dim": ph1["V"], "hidden_dim": ph1["d"],
        "wall_clock_s": time.time() - t0,
    }
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": metadata}) + "\n")
        f.write(json.dumps({"phase1": ph1}) + "\n")
        f.write(json.dumps({"phase2_fp32": ph2_fp32}) + "\n")
        f.write(json.dumps({"phase2_bf16": ph2_bf16}) + "\n")

    # ---- Markdown report ----
    lines: List[str] = []
    lines.append("# FP32 residual-kick liveness probe")
    lines.append("")
    lines.append("> MEASUREMENT REPORT. Single-record diagnostic. No controller edits,")
    lines.append("> no thresholds, no new architecture. Re-opens DS-036's \"residual")
    lines.append("> is dead\" verdict with a specific new hypothesis: the null result")
    lines.append("> was a bf16 quantization artifact, not a mathematical null space.")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append("| condition | mean KL (10 tokens) | token-1 KL | verdict |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **fp32** kick | **{ph2_fp32['mean_kl']:.4e}** | "
                 f"{ph2_fp32['first_token_kl']:.4e} | {ph2_fp32['verdict']} |")
    lines.append(f"| bf16 kick (DS-036 reproduction) | {ph2_bf16['mean_kl']:.4e} | "
                 f"{ph2_bf16['first_token_kl']:.4e} | {ph2_bf16['verdict']} |")
    lines.append("")
    lines.append(f"- DS-036 bf16 mean KL (reference): 9.3983e-06.")
    lines.append(f"- fp32 applied L2 delta = {ph2_fp32['applied_l2_delta']:.4f}; "
                 f"bf16 applied L2 delta = {ph2_bf16['applied_l2_delta']:.4f} "
                 f"(both target {ETA}).")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if ph2_fp32["mean_kl"] > LIVENESS_KL_ALIVE:
        lines.append("**The residual channel is ALIVE in fp32.** The DS-036 \"dead\" "
                     "verdict was a bf16 quantization artifact: the 0.5-L2 kick along "
                     "v1 is below bf16's resolution at layer 26 (residual norm O(250), "
                     "bf16 step O(0.6)) and rounds to zero. In fp32 the same kick "
                     "resolvably shifts the output. This re-opens RARI (DS-037) — the "
                     "retrieval-vs-optimal-direction comparison is only meaningful in "
                     "fp32.")
    elif ph2_fp32["mean_kl"] < LIVENESS_KL_DEAD:
        lines.append("**The residual channel is genuinely DEAD — not a precision "
                     "artifact.** Even in fp32, with the optimal direction v1, a "
                     "0.5-L2 kick produces KL ~= 10^-5. The DS-036 deprecation stands "
                     "on stronger grounds than before (now precision-independent).")
    else:
        lines.append("**INCONCLUSIVE.** fp32 KL is between 1e-4 and 1e-2 — neither "
                     "clearly alive nor clearly dead at the 0.5-L2 production force.")
    lines.append("")
    lines.append("## Method (identical to DS-036)")
    lines.append("")
    lines.append("- Record 0 of `heldout_degenerate_v2.jsonl`; layer 26, last prompt token.")
    lines.append("- v1 = dominant right-singular direction of the empirical Jacobian "
                 "(128 QR-orthonormal finite-difference probes, eps=1e-3, 10 power iters), "
                 "estimated in fp32.")
    lines.append("- Kick: h_t' = h_t + 0.5 * orth(v1, h_t), production double Gram-Schmidt.")
    lines.append("- KL = softmax(kicked) || softmax(dormant), eps=1e-12, clamp at 0.")
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {OUTPUT_JSONL}")
    print(f"wrote {OUTPUT_MD}")
    print(f"total wall clock: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
