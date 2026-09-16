#!/usr/bin/env python3
"""
Comprehensive Evaluation Harness for Active Variety Governor (AVG).

Evaluates generalization across multiple degeneracy failure modes and measures false positives.
Tracks Distinct-2 and Distinct-3 strictly on newly generated tokens and prints raw continuations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.sae_loader import load_sae
from AVG.governor.controller import ActiveVarietyGovernor
from AVG.core.metrics import compute_coherent_token_ratio

def calculate_distinct_n(text: str, n: int = 2) -> float:
    """Computes Distinct-N ratio over token string."""
    tokens = text.split()
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def make_diverse_calibration_prompts(tokenizer, n_samples: int = 8) -> List[torch.Tensor]:
    """Generates multi-domain calibration tokens."""
    raw_texts = [
        "In physics, spacetime is any mathematical model which fuses the three dimensions of space.",
        "The economic policy of the government was focused on reducing inflation and lowering interest rates.",
        "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
        "The history of philosophy is the system of ideas that have been developed over centuries.",
        "Artificial intelligence systems rely on large neural network architectures trained on massive text corpora.",
        "The recipe calls for two cups of flour, one egg, and a tablespoon of butter blended smoothly.",
        "Geologists study the Earth's solid material, rocks, and the processes by which they change.",
        "Operating systems manage computer hardware and software resources while providing common services.",
    ]
    return [tokenizer(txt, return_tensors="pt")["input_ids"] for txt in raw_texts[:n_samples]]


def make_test_suite() -> List[Tuple[str, str]]:
    return [
        # --- Degeneracy / Collapse Failure Modes ---
        ("Word Loop", "repeating word word word word word word word word word word"),
        ("Single-Token Spam", "A A A A A A A A A A A A A A A A A A A A"),
        ("Number Loop", "1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2"),
        ("Phrase Repetition", "The fox jumped over The fox jumped over The fox jumped over"),

        # --- Clean Factual Control Group ---
        ("Factual - Science", "The chemical symbol for gold is"),
        ("Factual - History", "In the year 1969, humans first landed on the"),
        ("Factual - Math", "The square root of 144 is"),
        ("Factual - General", "Water boils at 100 degrees Celsius at standard"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AVG Hybrid Governor")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--sae-path", type=str, default="gpt2-small-res-jb")
    parser.add_argument("--every-k", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Loading {args.model} on {device}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    ).eval()

    sae_map = {}
    for layer in [7, 9]:
        sae = load_sae(args.sae_path, layer=layer, d_model=model.config.hidden_size, device=device)
        sae.to(device)
        sae_map[layer] = sae

    target_layers = sorted(list(sae_map.keys()))

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        layer_indices=target_layers,
        every_k=args.every_k,
        sae_map=sae_map,
        device=device,
        hallucination_z=-2.2,  # Strict gating
        use_causal_filter=True,
    )

    print("[evaluate] Calibrating governor baseline...")
    trusted = make_diverse_calibration_prompts(tokenizer)
    governor.calibrate(trusted)

    suite = make_test_suite()
    print(f"[evaluate] Running {len(suite)} benchmark prompts...\n")

    print(f"{'Category':<20} | {'Decisions':<9} | {'Fired':<5} | {'Δres':<7} | {'KL':<8} | {'Gen Dist-2':<10} | {'Gen Dist-3':<10}")
    print("-" * 85)

    for category, prompt in suite:
        out = governor.generate(prompt, max_new_tokens=24, temperature=0.8, intervene=True)
        summary = out.get("intervention_summary", {})
        gen_only = out.get("gen_only_text", "")

        # Calculate distinctness strictly on newly generated tokens
        dist_2 = calculate_distinct_n(gen_only, n=2)
        dist_3 = calculate_distinct_n(gen_only, n=3)

        delta = summary.get("mean_residual_delta")
        kl = summary.get("mean_kl_next_token")
        delta_str = f"{delta:.2f}" if delta is not None else "0.00"
        kl_str = f"{kl:.4f}" if kl is not None else "0.0000"

        print(
            f"{category:<20} | "
            f"{len(out['decisions']):<9} | "
            f"{summary.get('n_interventions', 0):<5} | "
            f"{delta_str:<7} | "
            f"{kl_str:<8} | "
            f"{dist_2:<10.2f} | "
            f"{dist_3:<10.2f}"
        )
        print(f"  └─ Continuation: {gen_only.strip()!r}\n")


if __name__ == "__main__":
    main()
