#!/usr/bin/env python3
"""RFC-003 Phase 2b gate harness (Prose Immunity only).

Gate 2.2 (Degenerate-corpus shadow accuracy) is intentionally out of scope until
that corpus lands.

Gate 2.1b (Schema Suppression) is PARKED per human decision: the existing Track B
structural-code detector does not recognize bare JSON schema blocks, so the
measured suppression rate on T2S-Bench reference_frame is 0/200. The placeholder
below prints that evidence and exits cleanly rather than asserting a threshold.
No detector extension or re-scoping is performed.

Seed provenance: seed=42 is the canonical RFC-003 calibration seed, used for
controller startup (Gate 1.3), T2S-Bench subset certification (Amendment A1),
and these gates so that all Phase-2 observations are reproducible from a single
random-state contract.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from AVG.governor.controller import ActiveVarietyGovernor
from AVG.core.metrics import is_code_syntax_context

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
T2S_VALID = Path("data/t2s_bench/valid_subset_200.jsonl")
MAX_PROMPT_TOKENS = 512
MAX_NEW_TOKENS = 16
SEED = 42


def load_certified_samples():
    samples = []
    with open(T2S_VALID, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def make_governor(model, tokenizer):
    return ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        use_dual_resolution=True,
        use_code_filter=False,
    )


def tokenize_prompt(tokenizer, text: str):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )["input_ids"]


def gate_2_1a_prose_immunity(model, tokenizer, device, samples):
    """Prose Immunity: 0 fires and shadow-log engagement on certified Prose-200."""
    print("\n[Gate 2.1a] Prose Immunity over certified Prose-200")

    trusted = [
        tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]
    ]

    total_fires = 0
    total_delta = 0.0
    engagement_failures = []

    n = len(samples)
    for i, sample in enumerate(samples):
        governor = make_governor(model, tokenizer)
        governor.calibrate(trusted)

        prompt_text = sample.get("text", "")
        inputs = tokenize_prompt(tokenizer, prompt_text).to(device)

        with torch.no_grad():
            out = governor.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.3,
                top_p=0.9,
                intervene=True,
            )

        fires = len(out["decisions"])
        log = out.get("intervention_log", [])
        deltas = [r["residual_delta_norm"] for r in log]
        shadow_log = governor.get_shadow_log()

        total_fires += fires
        total_delta += sum(deltas)

        if not shadow_log:
            engagement_failures.append(sample.get("id"))

        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"  Progress: {i + 1}/{n}")

    print(f"  Samples evaluated: {n}")
    print(f"  Total fires: {total_fires}")
    print(f"  Total Δres: {total_delta:.4f} L2")
    print(f"  Shadow-log engagement failures: {len(engagement_failures)}")

    passed = True
    if total_fires != 0:
        print(f"❌ [Gate 2.1a FAIL] Expected 0 fires, got {total_fires}")
        passed = False
    if total_delta != 0.0:
        print(f"❌ [Gate 2.1a FAIL] Expected 0.00 L2 total Δres, got {total_delta:.4f}")
        passed = False
    if engagement_failures:
        print(f"❌ [Gate 2.1a FAIL] Shadow log empty for samples: {engagement_failures}")
        passed = False

    if passed:
        print("✅ [Gate 2.1a PASS] 0 fires, 0.00 L2 Δres, shadow log engaged for all samples.")
    return passed


def gate_2_1b_schema_suppression_parked(samples):
    """Parked gate: prints measured evidence, does not assert a threshold."""
    print("\n[Gate 2.1b] Schema Suppression — PARKED")
    print("  Reason: the existing Track B structural-code detector (core/metrics.py)")
    print("  matches C-style comments, Python/JS keywords, and markdown fences, but")
    print("  not bare JSON. T2S-Bench reference_frame blocks are JSON.")

    hits = sum(1 for s in samples if is_code_syntax_context(s.get("reference_frame", "")))
    total = len([s for s in samples if s.get("reference_frame")])
    print(f"  Measured Track B suppression on certified Prose-200 reference_frame:")
    print(f"    {hits}/{total} blocks flagged as code-syntax context")
    print(f"    suppression rate = {hits / total * 100:.2f}%")
    print("  This gate is parked pending a human decision on detector extension.")
    return True


def main():
    print("=" * 70)
    print("RFC-003 Phase 2b Gate Harness (2.1a active; 2.1b parked)")
    print("=" * 70)

    # Deterministic seed contract for Phase 2.
    set_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if not T2S_VALID.exists():
        print(f"❌ Missing certified fixture: {T2S_VALID}. Run scripts/vendor_t2s_bench.py first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    samples = load_certified_samples()

    ok = True
    ok &= gate_2_1a_prose_immunity(model, tokenizer, device, samples)
    ok &= gate_2_1b_schema_suppression_parked(samples)

    print("\n" + "=" * 70)
    if ok:
        print("✅ PHASE 2b GATES REPORT COMPLETE (2.1a PASS, 2.1b PARKED)")
        sys.exit(0)
    else:
        print("❌ PHASE 2b GATE 2.1a FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
