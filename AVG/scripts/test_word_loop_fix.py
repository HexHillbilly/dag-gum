#!/usr/bin/env python3
"""
Targeted Verification Probe: Word Loop Hard-Cap & Trajectory Check.
Tests Word Loop across 3 seeds over 128 tokens to confirm Δres <= 0.75 L2 and coherent text exit.
"""

import sys
import dataclasses
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repo parent directory to sys.path so 'AVG.core...' imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.sae_loader import BaseSAE, SAEOutput
from AVG.governor.controller import ActiveVarietyGovernor
from AVG.core.metrics import compute_coherent_token_ratio

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
    print(f"[word-loop-probe] Loading {model_id} on {device}...")
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

    prompt = "word word word word word word word word word word"
    print("\n================ Targeted Word Loop Force & Text Verification ================\n")

    for p in range(3):
        seed = 100 + p
        torch.manual_seed(seed)
        out = governor.generate(prompt, max_new_tokens=128, do_sample=True, temperature=0.8, intervene=True)
        gen_text = out["gen_only_text"]
        log = out["intervention_log"]

        mean_delta = sum(r["residual_delta_norm"] for r in log) / len(log) if log else 0.0
        max_delta = max((r["residual_delta_norm"] for r in log), default=0.0)
        ctr = compute_coherent_token_ratio(gen_text)

        print(f"Pass {p+1} (Seed {seed}):")
        print(f"  Fired: {len(out['decisions'])} | Mean Δres: {mean_delta:.2f} L2 | Max Δres: {max_delta:.2f} L2 | CTR: {ctr:.2f}")
        print(f"  Text Preview: {repr(gen_text[:120])}...\n")

if __name__ == "__main__":
    main()
