#!/usr/bin/env python3
"""
Side-by-side generation comparison: Ungoverned GPT-2 vs. Active Variety Governor
"""

import sys
from pathlib import Path

# Fix module resolution for running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from AVG.core.sae_loader import load_sae
from AVG.governor.controller import ActiveVarietyGovernor
from AVG.scripts.profile_model import make_synthetic_trusted
from AVG.core.metrics import compute_coherent_token_ratio

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "gpt2"

    print(f"[compare] Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    # Target layers in semantic commitment zone (~60% and ~75% depth)
    n_layers = getattr(model.config, "num_hidden_layers", 12)
    d_model = model.config.hidden_size
    mid_layers = sorted(set([max(0, int(n_layers * 0.60)), max(0, int(n_layers * 0.75))]))

    sae_map = {}
    for layer in mid_layers:
        sae = load_sae("gpt2-small-res-jb", layer=layer, d_model=d_model, device=device)
        sae.to(device)
        sae_map[layer] = sae

    target_layers = sorted(list(sae_map.keys()))

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        layer_indices=target_layers,
        every_k=2,
        sae_map=sae_map,
        device=device,
        hallucination_z=-1.5,
        spoof_z=2.0,
        use_causal_filter=True,
        history_len=16,
    )

    # Calibrate baseline stats (pass 8 positionally)
    print("[compare] Calibrating governor baseline...")
    trusted = make_synthetic_trusted(tokenizer, 8, max_length=64, seed=42)
    governor.calibrate(trusted)

    prompt = "repeating word word word word word word word word word word"
    max_new_tokens = 32

    print(f"\nPrompt: '{prompt}'\n" + "=" * 60)

    # 1. Ungoverned Run
    out_raw = governor.generate(
        prompt, max_new_tokens=max_new_tokens, intervene=False, do_sample=True, temperature=0.8
    )
    print(f"\n[RAW UNGOVERNED Output]:\n{out_raw['text']}\n")

    # 2. Governed Run
    out_avg = governor.generate(
        prompt, max_new_tokens=max_new_tokens, intervene=True, do_sample=True, temperature=0.8
    )
    print(f"[AVG GOVERNED Output]:\n{out_avg['text']}\n")

    summary = out_avg.get("intervention_summary", {})
    print("=" * 60)
    print(f"Interventions Fired : {summary.get('n_interventions', 0)}")
    print(f"Mean Residual Δ     : {summary.get('mean_residual_delta', 'n/a')}")
    print(f"Mean Next-Token KL  : {summary.get('mean_kl_next_token', 'n/a')}")


if __name__ == "__main__":
    main()
