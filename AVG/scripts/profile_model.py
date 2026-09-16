#!/usr/bin/env python3
"""Baseline calibration script for the Active Variety Governor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.sae_loader import load_sae
from AVG.governor.controller import ActiveVarietyGovernor
from AVG.core.metrics import compute_coherent_token_ratio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate AVG baselines")
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--n-trusted", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--every-k", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default="avg_baseline.pt")
    p.add_argument("--sae-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_synthetic_trusted(tokenizer, n: int, max_length: int, seed: int) -> List[torch.Tensor]:
    torch.manual_seed(seed)
    templates = [
        "In physics, spacetime is any mathematical model which fuses space and time.",
        "The economic policy of the government was focused on reducing inflation.",
        "Photosynthesis is a process used by plants to convert light energy.",
        "The history of philosophy is the system of ideas developed over centuries.",
        "Artificial intelligence systems rely on neural network architectures.",
        "The recipe calls for two cups of flour, one egg, and butter.",
        "Geologists study the Earth's solid material and rocks.",
        "Operating systems manage computer hardware and software resources.",
    ]
    traces = []
    for i in range(n):
        txt = templates[i % len(templates)]
        enc = tokenizer(txt, return_tensors="pt", max_length=max_length, truncation=True)["input_ids"]
        traces.append(enc)
    return traces


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"[profile_model] Loading {args.model} on {device}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    sae_map = {}
    if args.sae_path:
        mid = getattr(model.config, "n_layer", 12) // 2
        d_model = getattr(model.config, "n_embd", 768)
        sae = load_sae(args.sae_path, layer=mid, d_model=d_model, device=device)
        sae_map[mid] = sae

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        every_k=args.every_k,
        sae_map=sae_map,
        device=device,
    )

    print(f"[profile_model] Building {args.n_trusted} trusted traces...")
    trusted = make_synthetic_trusted(tokenizer, args.n_trusted, args.max_length, args.seed)

    print("[profile_model] Running calibration...")
    baseline = governor.calibrate(trusted)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean_attenuation": baseline.mean_attenuation,
        "std_attenuation": baseline.std_attenuation,
        "mean_pr": baseline.mean_pr,
        "p10_attenuation": baseline.p10_attenuation,
        "p90_attenuation": baseline.p90_attenuation,
        "model": args.model,
        "every_k": args.every_k,
    }
    torch.save(payload, out_path)
    print(f"[profile_model] Saved calibration payload to {out_path}")


if __name__ == "__main__":
    main()
