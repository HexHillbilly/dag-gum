#!/usr/bin/env python3
"""Demonstration of generation with the Active Variety Governor enabled vs disabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.sae_loader import load_sae
from AVG.governor.controller import ActiveVarietyGovernor, BaselineStats
from AVG.core.metrics import compute_coherent_token_ratio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AVG-controlled generation demo")
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--every-k", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--baseline", type=str, default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_baseline(path: str, governor: ActiveVarietyGovernor) -> None:
    data = torch.load(path, map_location="cpu")
    b = BaselineStats(
        mean_attenuation=data.get("mean_attenuation", {}),
        std_attenuation=data.get("std_attenuation", {}),
        mean_pr=data.get("mean_pr", {}),
        p10_attenuation=data.get("p10_attenuation", {}),
        p90_attenuation=data.get("p90_attenuation", {}),
    )
    governor.baseline = b


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[generate] Loading {args.model} on {device}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        every_k=args.every_k,
        device=device,
    )

    if args.baseline and Path(args.baseline).exists():
        load_baseline(args.baseline, governor)
        print(f"[generate] Loaded baseline from {args.baseline}")
    else:
        print("[generate] Calibrating governor on dummy trace...")
        dummy_trace = tokenizer("Sample text for calibration", return_tensors="pt")["input_ids"]
        governor.calibrate([dummy_trace])

    def run(intervene: bool):
        return governor.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            intervene=intervene,
        )

    if args.compare:
        print("\n=== Generation WITHOUT AVG ===")
        out_off = run(intervene=False)
        print(out_off["text"])

        print("\n=== Generation WITH AVG ===")
        out_on = run(intervene=True)
        print(out_on["text"])
        if out_on["decisions"]:
            print("Interventions applied:")
            for d in out_on["decisions"]:
                print(f"  * {d.message} -> action={d.action} strength={d.strength:.2f}")
        else:
            print("(no interventions triggered)")
    else:
        out = run(intervene=True)
        print("\n=== AVG-controlled generation ===")
        print(out["text"])


if __name__ == "__main__":
    main()
