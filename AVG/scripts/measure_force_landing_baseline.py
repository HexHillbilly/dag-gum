#!/usr/bin/env python3
"""
Baseline Measurement Probe: Force Magnitude (Δres) vs. Post-Exit Landing Domain.

Evaluates active intervention modes:
- Single-Token Spam
- Number Loop
- Symbol/Punctuation Spam

Measures per-pass Δres L2 force and classifies post-exit domain:
[ENGLISH_PROSE, CODE_SYNTAX, CJK_UNICODE, PATTERNED_TEMPLATE]
"""

import sys
import re
import dataclasses
import inspect
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.sae_loader import BaseSAE, SAEOutput
from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2
from AVG.core.metrics import compute_coherent_token_ratio

ACTIVE_PROMPTS = [
    ("Single-Token Spam", "A A A A A A A A A A A A A A A A"),
    ("Number Loop", "1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2"),
    ("Symbol/Punctuation Spam", "# % # % # % # % # % # % # %"),
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

        return SAEOutput(
            sparse_codes=sparse_codes,
            reconstructed=recon,
            l0=l0,
            recon_error=recon_err,
            reconstruction_error=recon_err
        )


def classify_landing_domain(text: str) -> str:
    """Classifies generated text into landing categories."""
    raw_tokens = text.strip().split()
    if not raw_tokens:
        return "EMPTY"

    cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fa5' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    code_keywords = sum(1 for tok in raw_tokens if tok in ("typedef", "struct", "var", "const", "import", "def", "return", "function", "->", "=="))
    english_words = len(re.findall(r'\b[a-zA-Z]{2,}\b', text))

    total = float(len(raw_tokens))

    if cjk_chars / max(1.0, float(len(text))) > 0.20:
        return "CJK_UNICODE"
    elif code_keywords >= 2 or (code_keywords > 0 and english_words / total < 0.40):
        return "CODE_SYNTAX"
    elif english_words / total >= 0.50:
        return "ENGLISH_PROSE"
    else:
        return "PATTERNED_TEMPLATE"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"[force-probe] Loading {model_id} on {device}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    ).eval()

    lm_head_weight = model.lm_head.weight
    qwen_sae = UnembedDictionarySAE(lm_head_weight, device=lm_head_weight.device)
    sae_map = {16: qwen_sae, 20: qwen_sae}

    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer, every_k=2, sae_map=sae_map, device=device)

    trusted = [tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)

    N_PASSES = 5
    MAX_TOKENS = 64

    print(f"\n================ Baseline Measurement: Force (Δres) vs. Landing Quality ================\n")

    for cat, prompt in ACTIVE_PROMPTS:
        print(f"--- Mode: {cat} ---")

        for p in range(N_PASSES):
            seed = 200 + p
            torch.manual_seed(seed)
            out = governor.generate(prompt, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.8, intervene=True)

            gen_text = out["gen_only_text"]
            interventions = len(out["decisions"])
            log = out["intervention_log"]

            mean_delta = sum(r["residual_delta_norm"] for r in log) / len(log) if log else 0.0
            ctr = compute_coherent_token_ratio(gen_text)
            landing = classify_landing_domain(gen_text)

            print(f"  Pass {p+1} (Seed {seed}): Δres={mean_delta:.2f} L2 | Fired={interventions:2d} | CTR={ctr:.2f} | Landing: [{landing}]")
            print(f"     Preview: {repr(gen_text[:75])}\n")


if __name__ == "__main__":
    main()
