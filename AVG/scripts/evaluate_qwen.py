#!/usr/bin/env python3
"""
Cross-Architecture Evaluation Harness for AVG on Qwen/Qwen2.5-1.5B.

Features:
- Language & Code-Aware Coherent Token Ratio (CTR): Evaluates English, CJK unicode,
  and code constructs without metric distortion.
- Vocabulary-Aligned Unembed Dictionary SAE: Binds Layer-16 & Layer-20 decoder
  dictionaries W_dec mapped to d_model = 1536.
- Full SAEOutput Field Compliance: Computes and populates L0 sparsity for dataclass compatibility.
- Multi-Pass Benchmark (5 Seeds) over 64-token extended rollout horizons.
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


class UnembedDictionarySAE(BaseSAE):
    """
    Constructs a 151,646-feature Sparse Dictionary for Qwen2.5 directly
    from its LM-head unembedding matrix (W_unembed), providing real
    vocabulary-aligned feature directions [vocab_size, d_model].
    """
    def __init__(self, unembed_weight: torch.Tensor, device: torch.device):
        super().__init__()
        # W_dec shape: [151646, 1536]
        self.W_dec = nn.Parameter(F.normalize(unembed_weight.detach().float().to(device), dim=-1))
        self.d_model = unembed_weight.shape[1]
        self.dict_size = unembed_weight.shape[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Cosine activation over vocabulary dictionary
        norm_x = F.normalize(x.float(), dim=-1)
        return torch.matmul(norm_x, self.W_dec.T)

    def decode(self, sparse_codes: torch.Tensor) -> torch.Tensor:
        return torch.matmul(sparse_codes, self.W_dec)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        orig_dtype = x.dtype
        x_float = x.float()
        codes = self.encode(x_float)

        # Soft top-k activation (k=32)
        top_codes, _ = torch.topk(codes, k=32, dim=-1)
        thresh = top_codes[..., -1:]
        sparse_codes = torch.where(codes >= thresh, codes, torch.zeros_like(codes))

        recon = self.decode(sparse_codes).to(dtype=orig_dtype)

        # Calculate L0 norm (average number of non-zero active features)
        l0 = (sparse_codes.abs() > 1e-6).float().sum(dim=-1).mean().item()

        # Relative reconstruction error
        recon_err = (x_float - recon.float()).norm(dim=-1).mean() / (x_float.norm(dim=-1).mean() + 1e-8)

        # Standard dataclass kwargs expected by SAEOutput
        kwargs = {
            "sparse_codes": sparse_codes,
            "reconstructed": recon,
            "l0": l0,
        }

        # Dynamically map reconstruction error field across variable naming schemes
        if dataclasses.is_dataclass(SAEOutput):
            field_names = {f.name for f in dataclasses.fields(SAEOutput)}
            for err_field in ["recon_error", "reconstruction_error", "error", "mse", "loss"]:
                if err_field in field_names:
                    kwargs[err_field] = recon_err
                    break
        else:
            sig = inspect.signature(SAEOutput.__init__)
            params = set(sig.parameters.keys())
            for err_field in ["recon_error", "reconstruction_error", "error", "mse", "loss"]:
                if err_field in params:
                    kwargs[err_field] = recon_err
                    break

        out = SAEOutput(**kwargs)
        setattr(out, "reconstruction_error", recon_err)
        setattr(out, "recon_error", recon_err)
        return out


def compute_coherent_token_ratio(text: str) -> float:
    """
    Language & Code-Aware Coherence Metric (CTR).
    Scores English words, CJK natural language, and code keywords/identifiers as valid.
    """
    raw_tokens = text.strip().split()
    if not raw_tokens:
        return 0.0

    valid_count = 0
    for tok in raw_tokens:
        clean = tok.strip("(),;:{}[]\"'<>`#$%^&*=-+/")
        # 1. English alphabetic word
        if clean in ("a", "I", "A") or (len(clean) >= 2 and all(c.isalpha() or c in ("'", "-") for c in clean)):
            valid_count += 1
        # 2. CJK / Chinese / Japanese / Korean characters
        elif any('\u4e00' <= c <= '\u9fa5' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in tok):
            valid_count += 1
        # 3. Code keywords, identifiers, and variables ($var, _idx, struct, import)
        elif re.match(r'^\$?[a-zA-Z_][a-zA-Z0-9_]*$', clean) and len(clean) >= 2:
            valid_count += 1
        # 4. Standard structural code tokens or operators (->, ==, <=, &&, ||, ::)
        elif tok in ("->", "==", "!=", "<=", ">=", "&&", "||", "::", "=>", "++", "--", "/*", "*/", "//"):
            valid_count += 1

    return min(1.0, valid_count / float(len(raw_tokens)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-1.5B"
    print(f"[qwen-eval] Loading {model_id} on {device}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    ).eval()

    # Bind LM-Head Vocabulary Dictionary SAE to Layers 16 and 20 (d_model = 1536)
    print("[qwen-eval] Binding Unembed-Aligned Vocabulary SAE Dictionaries on Layers 16 & 20...")
    lm_head_weight = model.lm_head.weight
    qwen_sae = UnembedDictionarySAE(lm_head_weight, device=lm_head_weight.device)
    sae_map = {16: qwen_sae, 20: qwen_sae}

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        every_k=2,
        sae_map=sae_map,
        device=device
    )

    print("[qwen-eval] Calibrating governor baseline...")
    trusted = [tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)

    N_PASSES = 5
    print(f"\n================ Running Qwen2.5-1.5B Language/Code-Aware Evaluation ================\n")

    for cat, prompt in PROMPTS:
        print(f"--- Category: {cat} ---")
        distinct_scores = []
        ctr_scores = []

        for p in range(N_PASSES):
            torch.manual_seed(42 + p)
            # Evaluate 64-token extended horizon
            out = governor.generate(prompt, max_new_tokens=64, do_sample=True, temperature=0.8, intervene=True)
            gen_text = out["gen_only_text"]

            gen_ids = out["sequences"][:, out["prompt_len"]:]
            dist2, _ = compute_token_distinct_2_fast(gen_ids, window_len=32)
            ctr = compute_coherent_token_ratio(gen_text)

            distinct_scores.append(dist2)
            ctr_scores.append(ctr)

            print(f"  Trial {p+1}: (Dist-2: {dist2:.2f} | CTR: {ctr:.2f}) -> {repr(gen_text[:70])}")

        mean_dist = sum(distinct_scores) / len(distinct_scores)
        mean_ctr = sum(ctr_scores) / len(ctr_scores)
        print(f"  ==> Mean Dist-2: {mean_dist:.2f} | Mean Coherent Token Ratio (CTR): {mean_ctr:.2f}\n")


if __name__ == "__main__":
    main()
