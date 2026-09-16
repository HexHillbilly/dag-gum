#!/usr/bin/env python3
"""
Degeneracy Trigger Verification Probe.
Confirms that genuine repetition loops still satisfy the joint conjunction gate
(diversity < 0.30 AND CTR < 0.30) and receive calibrated residual resets in [0.35, 0.75] L2.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.sae_loader import BaseSAE, SAEOutput
from AVG.governor.controller import ActiveVarietyGovernor
from AVG.core.metrics import compute_coherent_token_ratio

DEGENERATE_PROMPTS = [
    ("Word Loop", "word word word word word word word word word word"),
    ("Single-Token Spam", "A A A A A A A A A A A A A A A A"),
    ("Number Loop", "1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2"),
    ("Symbol Spam", "# % # % # % # % # % # % # %"),
    ("Alphabet Trap", "A B C D A B C D A B C D A B C D"),
]


class UnembedDictionarySAE(BaseSAE):
    def __init__(self, unembed_weight: torch.Tensor, device: torch.device):
        super().__init__()
        self.W_dec = nn.Parameter(F.normalize(unembed_weight.detach().float().to(device), dim=-1))
        self.d_model = unembed_weight.shape[1]
        self.dict_size = unembed_weight.shape[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = F.normalize(x.float(), dim=-1)
        return torch.matmul(norm_x, self.W_dec.T)

    def decode(self, sparse_codes: torch.Tensor) -> torch.Tensor:
        return torch.matmul(sparse_codes.float(), self.W_dec)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        orig_dtype = x.dtype
        x_float = x.float()
        codes = self.encode(x_float)
        top_codes, _ = torch.topk(codes, k=32, dim=-1)
        thresh = top_codes[..., -1:]
        sparse_codes = torch.where(codes >= thresh, codes, torch.zeros_like(codes))
        recon = self.decode(sparse_codes).to(dtype=orig_dtype)
        l0 = (sparse_codes.abs() > 1e-6).float().sum(dim=-1).mean().item()
        recon_err = (x_float - recon.float()).norm(dim=-1).mean() / (x_float.norm(dim=-1).mean() + 1e-8)
        return SAEOutput(sparse_codes=sparse_codes, reconstructed=recon, l0=l0, recon_error=recon_err, reconstruction_error=recon_err)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"[degen-probe] Loading {model_id} on {device}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    ).eval()

    qwen_sae = UnembedDictionarySAE(model.lm_head.weight, device=model.lm_head.weight.device)
    sae_map = {16: qwen_sae, 20: qwen_sae}

    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer, every_k=2, sae_map=sae_map, device=device)
    trusted = [tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)

    print("\n================ Active Failure Mode Trigger Probe (64 Tokens) ================\n")

    for cat, prompt in DEGENERATE_PROMPTS:
        print(f"--- Mode: {cat} ---")
        for p in range(2):
            seed = 200 + p
            torch.manual_seed(seed)
            out = governor.generate(prompt, max_new_tokens=64, do_sample=True, temperature=0.8, intervene=True)
            gen_text = out["gen_only_text"]
            fired = len(out["decisions"])
            log = out["intervention_log"]

            mean_delta = sum(r["residual_delta_norm"] for r in log) / len(log) if log else 0.0
            ctr = compute_coherent_token_ratio(gen_text)

            print(f"  Pass {p+1} (Seed {seed}): Fired={fired:2d} | Mean Δres={mean_delta:.2f} L2 | CTR={ctr:.2f} | Text: {repr(gen_text[:75])}")
        print()


if __name__ == "__main__":
    main()
