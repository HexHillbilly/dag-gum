#!/usr/bin/env python3
import sys
import time
import argparse
from pathlib import Path

# Fix sys.path so 'AVG.core...' resolves cleanly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from AVG.governor.controller import ActiveVarietyGovernor


def _time_generate_raw(model, inputs, tokens: int) -> float:
    """Return elapsed seconds for one RAW baseline generation."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(inputs, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def _time_generate_dormant(governor, inputs, tokens: int) -> float:
    """Return elapsed seconds for one AVG dormant generation."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = governor.generate(inputs, max_new_tokens=tokens, temperature=0.7, top_p=0.9, intervene=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def benchmark_latency(model_name: str, tokens: int, max_governor_cost_ms: float):
    print("=" * 70)
    print("AVG Path 1 Dormant Latency Guardrail Benchmark")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    ).eval()

    prompt = "The history of science shows that fundamental discoveries often emerge from"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # 1. RAW Generation Baseline — 3 trials, minimum ms/token
    print("\n[RAW Baseline] Running 3 trials...")
    raw_times = []
    for i in range(1, 4):
        elapsed = _time_generate_raw(model, inputs, tokens)
        raw_times.append(elapsed)
        ms_per_tok = (elapsed / tokens) * 1000.0
        print(f"  Trial {i}: {ms_per_tok:.2f} ms/token ({tokens / elapsed:.2f} tok/s)")

    raw_time = min(raw_times)
    raw_ms_per_tok = (raw_time / tokens) * 1000.0
    print(f"  -> Minimum used: {raw_ms_per_tok:.2f} ms/token")

    # 2. AVG Governed (Dormant Regime) — 3 trials, minimum ms/token
    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer)
    trusted = [tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)

    print("\n[AVG Dormant] Running 3 trials...")
    dormant_times = []
    for i in range(1, 4):
        elapsed = _time_generate_dormant(governor, inputs, tokens)
        dormant_times.append(elapsed)
        ms_per_tok = (elapsed / tokens) * 1000.0
        print(f"  Trial {i}: {ms_per_tok:.2f} ms/token ({tokens / elapsed:.2f} tok/s)")

    dormant_time = min(dormant_times)
    dormant_ms_per_tok = (dormant_time / tokens) * 1000.0
    print(f"  -> Minimum used: {dormant_ms_per_tok:.2f} ms/token")

    # The guardrail is an absolute governor-cost budget, not a percentage ratio.
    # Dormant AVG overhead is CPU-side bookkeeping that is roughly constant in
    # ms/token, while the RAW baseline is GPU-load-dependent. When the GPU is
    # least loaded, the RAW floor drops and the ratio inverts, producing a red
    # gate even though the dormant leg is healthy (observed +36.87% failure on
    # 2026-08-04 with a stable 21.7 ms/token dormant leg). Recalibration of this
    # budget is a documented human decision tied to instrument changes — never a
    # response to a red gate.
    governor_cost_ms = dormant_ms_per_tok - raw_ms_per_tok
    overhead_pct = (governor_cost_ms / raw_ms_per_tok) * 100.0

    print(f"\nResults over {tokens} tokens (minimum-of-3 per leg):")
    print(f"  RAW Baseline:      {raw_ms_per_tok:.2f} ms/token ({tokens / raw_time:.2f} tok/s)")
    print(f"  AVG Dormant:       {dormant_ms_per_tok:.2f} ms/token ({tokens / dormant_time:.2f} tok/s)")
    print(f"  Governor cost:     {governor_cost_ms:.2f} ms/token (budget: <= {max_governor_cost_ms:.1f} ms/token)")
    print(f"  Overhead (info):   +{overhead_pct:.2f}%")

    print("\n" + "=" * 70)
    if governor_cost_ms > max_governor_cost_ms:
        print(f"❌ [CI FAIL] Governor cost breach! Measured: {governor_cost_ms:.2f} ms/token > Budget: {max_governor_cost_ms:.1f} ms/token")
        sys.exit(1)
    else:
        print(f"✅ [CI PASS] Governor cost within budget ({governor_cost_ms:.2f} ms/token <= {max_governor_cost_ms:.1f} ms/token).")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVG Dormant Latency Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--max-governor-cost-ms", type=float, default=7.0)
    args = parser.parse_args()

    benchmark_latency(args.model, args.tokens, args.max_governor_cost_ms)
