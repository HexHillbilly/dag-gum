#!/usr/bin/env python3
"""Multi-Pass Evaluation Harness featuring Hard Zero-Context Stress Suite & VWR Metrics."""

import sys
import re
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.sae_loader import load_sae
from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast
from AVG.core.metrics import compute_coherent_token_ratio

PROMPTS = [
    ("Word Loop", "word word word word word word word word word word"),
    ("Single-Token Spam", "A A A A A A A A A A A A A A A A"),
    ("Number Loop", "1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2"),
    ("Alphabet Sequence Trap", "A B C D A B C D A B C D A B C D"),
    ("Symbol/Punctuation Spam", "# % # % # % # % # % # % # %"),
    ("Phrase Repetition", "The fox jumped over The fox jumped over The fox jumped over"),
    ("Factual - History", "In 1969, Apollo 11 landed on the"),
]

def compute_valid_word_ratio(text: str) -> float:
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    raw_tokens = text.strip().split()
    if not raw_tokens:
        return 0.0
    return len(words) / float(len(raw_tokens))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[multi-pass] Loading GPT-2...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    print("[multi-pass] Loading SAE...")
    sae = load_sae("gpt2-small-res-jb", layer=7, d_model=768, device=device)
    sae_map = {7: sae, 9: sae}

    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer, every_k=2, sae_map=sae_map, device=device)

    print("[multi-pass] Calibrating governor baseline...")
    trusted = [tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)

    N_PASSES = 5
    print(f"\n================ Running {N_PASSES}-Pass Hard Zero-Context Evaluation ================\n")

    for cat, prompt in PROMPTS:
        print(f"--- Category: {cat} ---")
        distinct_scores = []
        vwr_scores = []

        for p in range(N_PASSES):
            torch.manual_seed(42 + p)
            out = governor.generate(prompt, max_new_tokens=24, do_sample=True, temperature=0.8, intervene=True)
            gen_text = out["gen_only_text"]

            gen_ids = out["sequences"][:, out["prompt_len"]:]
            dist2, _ = compute_token_distinct_2_fast(gen_ids, window_len=24)
            vwr = compute_valid_word_ratio(gen_text)

            distinct_scores.append(dist2)
            vwr_scores.append(vwr)

            print(f"  Trial {p+1}: (Dist-2: {dist2:.2f} | VWR: {vwr:.2f}) -> {repr(gen_text[:65])}")

        mean_dist = sum(distinct_scores) / len(distinct_scores)
        mean_vwr = sum(vwr_scores) / len(vwr_scores)
        print(f"  ==> Mean Dist-2: {mean_dist:.2f} | Mean Valid-Word-Ratio (VWR): {mean_vwr:.2f}\n")

if __name__ == "__main__":
    main()
