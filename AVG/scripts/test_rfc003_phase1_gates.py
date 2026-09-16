#!/usr/bin/env python3
"""RFC-003 Phase 1 Quality Gate harness (does not modify v2.4 gate scripts)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from AVG.governor.controller import ActiveVarietyGovernor

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TRUSTED_TEXT = "The quick brown fox jumps over the lazy dog."
PROMPT = "The history of science shows that fundamental discoveries often emerge from"
TOKENS = 128


def make_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    return model, tokenizer, device


def gate_1_3_sigma1_determinism(model, tokenizer, device):
    print("\n[Gate 1.3] σ₁⁰ determinism across 5 runs at seed=42")
    values = []
    for run in range(1, 6):
        set_seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        gov = ActiveVarietyGovernor(model, tokenizer=tokenizer, use_dual_resolution=True)
        trusted = [tokenizer(TRUSTED_TEXT, return_tensors="pt")["input_ids"]]
        gov.calibrate(trusted)
        sigma = gov.get_sigma1_0()
        values.append(sigma)
        print(f"  Run {run}: σ₁⁰ = {sigma:.6f}")

    if len(set(round(v, 6) for v in values)) != 1:
        print("❌ [Gate 1.3 FAIL] σ₁⁰ values differ across runs!")
        return False
    print("✅ [Gate 1.3 PASS] σ₁⁰ is deterministic.")
    return True


def gate_1_1_factual_safety(model, tokenizer, device):
    print("\n[Gate 1.1] Factual safety with use_dual_resolution=True")
    set_seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    gov = ActiveVarietyGovernor(model, tokenizer=tokenizer, use_dual_resolution=True)
    trusted = [tokenizer(TRUSTED_TEXT, return_tensors="pt")["input_ids"]]
    gov.calibrate(trusted)

    factual_prompts = [
        "The capital of France is",
        "The square root of 64 is",
        "Water freezes at a temperature of",
        "The first element on the periodic table is",
    ]

    total_fires = 0
    for prompt in factual_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = gov.generate(inputs, max_new_tokens=32, temperature=0.3, top_p=0.9, intervene=True)
        total_fires += len(out["decisions"])
        print(f"  '{prompt[:40]}...' -> fires={len(out['decisions'])}")

    if total_fires != 0:
        print(f"❌ [Gate 1.1 FAIL] Expected 0 fires, got {total_fires}!")
        return False
    print("✅ [Gate 1.1 PASS] 0 fires on factual prompts.")
    return True


def gate_1_2_latency(model, tokenizer, device):
    print("\n[Gate 1.2] Absolute governor cost with use_dual_resolution=True")
    gov = ActiveVarietyGovernor(model, tokenizer=tokenizer, use_dual_resolution=True)
    trusted = [tokenizer(TRUSTED_TEXT, return_tensors="pt")["input_ids"]]
    gov.calibrate(trusted)

    inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    def time_raw():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(inputs, max_new_tokens=TOKENS, min_new_tokens=TOKENS, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - start

    def time_dormant():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = gov.generate(inputs, max_new_tokens=TOKENS, temperature=0.7, top_p=0.9, intervene=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - start

    raw_times = [time_raw() for _ in range(3)]
    dormant_times = [time_dormant() for _ in range(3)]

    raw_ms = (min(raw_times) / TOKENS) * 1000.0
    dormant_ms = (min(dormant_times) / TOKENS) * 1000.0
    cost_ms = dormant_ms - raw_ms

    print(f"  RAW min:    {raw_ms:.2f} ms/token")
    print(f"  AVG min:    {dormant_ms:.2f} ms/token")
    print(f"  Governor cost: {cost_ms:.2f} ms/token (budget <= 6.0 ms/token)")

    if cost_ms > 6.0:
        print("❌ [Gate 1.2 FAIL] Governor cost exceeds 6.0 ms/token!")
        return False
    print("✅ [Gate 1.2 PASS] Governor cost within budget.")
    return True


def main():
    print("=" * 70)
    print("RFC-003 Phase 1 Quality Gate Harness")
    print("=" * 70)

    model, tokenizer, device = make_model_and_tokenizer()

    ok = True
    ok &= gate_1_3_sigma1_determinism(model, tokenizer, device)
    ok &= gate_1_1_factual_safety(model, tokenizer, device)
    ok &= gate_1_2_latency(model, tokenizer, device)

    print("\n" + "=" * 70)
    if ok:
        print("✅ ALL PHASE 1 GATES PASS")
        sys.exit(0)
    else:
        print("❌ PHASE 1 GATES FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
