#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Fix sys.path so 'AVG.core...' resolves cleanly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from AVG.governor.controller import ActiveVarietyGovernor

SMOKE_PROBES = [
    {"mode": "Single-Token Spam", "prompt": "A A A A A A A A A A A A A A A A"},
    {"mode": "Number Loop", "prompt": "1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2"}
]


def _numeric_token_fraction(text: str) -> float:
    """Fraction of whitespace-separated tokens that contain at least one digit."""
    tokens = text.strip().split()
    if not tokens:
        return 0.0
    numeric = sum(1 for tok in tokens if any(ch.isdigit() for ch in tok))
    return numeric / float(len(tokens))


def _make_governor(model, tokenizer, device):
    """Return a freshly calibrated governor for a single probe."""
    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer)
    trusted = [tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)
    return governor


def run_degeneracy_smoke(model_name: str = "Qwen/Qwen2.5-1.5B"):
    print("=" * 70)
    print("AVG Path 1 Degeneracy Recovery Smoke Test")
    print("=" * 70)

    # Deterministic seed: induces a degenerate loop signature on Qwen/Qwen2.5-1.5B
    set_seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    ).eval()

    total_fires = 0

    for probe in SMOKE_PROBES:
        mode = probe["mode"]
        prompt = probe["prompt"]
        print(f"\n[Smoke] Mode: {mode}")

        # Isolate probes so Spam's residual/hook state cannot leak into Number Loop.
        governor = _make_governor(model, tokenizer, device)

        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            out = governor.generate(
                inputs,
                max_new_tokens=48,
                temperature=0.3,
                top_p=0.9,
                intervene=True
            )

        # B2 contract: engagement is measured via the "collapse" telemetry
        # (residual path detached per Amendment A4; "decisions" is always []).
        collapse = out.get("collapse", {})
        fired = collapse.get("bigram_fires", 0) + collapse.get("spectral_fires", 0)

        total_fires += fired

        print(
            f"  Collapse Fires: {fired} "
            f"(bigram={collapse.get('bigram_fires', 0)}, "
            f"spectral={collapse.get('spectral_fires', 0)})"
        )

        if mode == "Number Loop":
            gen_text = out.get("gen_only_text") or ""
            num_frac = _numeric_token_fraction(gen_text)
            immunity_steps = out.get("numeric_immunity_steps", 0)
            print(f"  Numeric-token fraction (observability): {num_frac:.3f}")
            print(f"  Numeric-immunity steps: {immunity_steps}")

            # B1 contract: immunity now comes from the EXPLICIT numeric rule, not
            # the retired digit-crush. Assert 0 fires AND that the predicate
            # actually engaged (non-vacuous). The old output-fraction floor (0.04)
            # is retired: it measured the digit-crush side effect and is
            # RNG-fragile under B1 (0.000 at seed=1 despite the predicate firing).
            if fired != 0:
                print(f"❌ [CI FAIL] Number Loop fired {fired} time(s); expected immunity (0 fires)!")
                sys.exit(1)
            if immunity_steps == 0:
                print("❌ [CI FAIL] Numeric-immunity predicate never engaged (0 steps) — vacuous pass!")
                sys.exit(1)

    print("\n" + "=" * 70)

    # Assertion: Require at least one collapse fire across severe loops
    if total_fires == 0:
        print("❌ [CI FAIL] Collapse detector failed to engage on Single-Token / Number loops!")
        sys.exit(1)

    print(f"✅ [CI PASS] Degeneracy Smoke Passed! Total Collapse Fires: {total_fires}.")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVG Degeneracy Smoke Test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()

    run_degeneracy_smoke(args.model)
