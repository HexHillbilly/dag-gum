#!/usr/bin/env python3
"""Measure high-diversity (emoji/symbol) degeneracy and the non-linguistic density detector.

3-way isolation on llama3-8b 4-bit (the observed failure model, temp 0.8,
top_p 0.85, 256-token horizon):

    dormant  — intervene=False (true no-actuation baseline)
    det_off  — intervene=True, non_linguistic_density_threshold=2.0 (detector disabled)
    det_on   — intervene=True, non_linguistic_density_threshold=0.45 (detector enabled)

Finding (2026-08-15): the emoji/symbol tail-spam is INDUCED by the governor's
suppression actuation (dormant baseline = 0.0 non-ASCII), not an intrinsic
model property. The reactive detector reduces it (full non-ASCII 0.25-0.55 ->
0.13-0.19) but does not eliminate it; the proactive fix (suppress non-ASCII
DURING suppression) is the root-cause follow-up.

Usage:
    python scripts/measure_high_diversity_degeneracy.py [--seeds 0 42 123] \
        [--max-new-tokens 256] [--out /tmp/hdd_results.jsonl]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from AVG.governor.controller import ActiveVarietyGovernor
from AVG.core.metrics import compute_non_linguistic_density

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
BAND_LOW = 7.8703

PERSONA = (
    "You are Daisy, a warm, curious, slightly naive conversationalist. "
    "A line in your own voice: 'i love making new sentences and learning more about the world'. "
    "Weave its spirit into your reply in your own Daisy voice.\n\n"
    "Request: hi daisy, how are you today?"
)


def run(gov, templated, seed, intervene, threshold, max_new_tokens):
    gov.non_linguistic_density_threshold = threshold
    torch.manual_seed(seed)
    res = gov.generate(templated, max_new_tokens=max_new_tokens, do_sample=True,
                       temperature=0.8, intervene=intervene)
    gen = res["sequences"][0].cpu()[res["prompt_len"]:]
    f_na = gov._non_ascii_mask[gen.to(gov._non_ascii_mask.device)].cpu().tolist()
    full_na = sum(f_na) / len(f_na) if f_na else 0.0
    max_na = max(sum(f_na[i:i + 24]) / 24 for i in range(len(f_na) - 23)) if len(f_na) >= 24 else 0.0
    non_ling = compute_non_linguistic_density(gen, gov.tokenizer)
    return {
        "seed": seed,
        "condition": "dormant" if not intervene else ("det_off" if threshold > 1.0 else "det_on"),
        "intervene": intervene,
        "symbol_fires": res["collapse"]["symbol_fires"],
        "full_nonascii": round(full_na, 3),
        "maxwin24_nonascii": round(max_na, 3),
        "non_ling_density": round(non_ling, 3),
        "gen_tokens": len(gen),
        "text": res["gen_only_text"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 123])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", type=str, default="/tmp/hdd_results.jsonl")
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    gov = ActiveVarietyGovernor(model, tok, band_low=BAND_LOW)
    templated = tok.apply_chat_template(
        [{"role": "user", "content": PERSONA}], tokenize=False, add_generation_prompt=True,
    )
    print(f"non_ascii_mask: {int(gov._non_ascii_mask.sum())} tokens flagged", flush=True)

    conds = [("dormant", False, 0.45), ("det_off", True, 2.0), ("det_on", True, 0.45)]
    records = []
    for seed in args.seeds:
        parts = [f"seed={seed}:"]
        for tag, intervene, thr in conds:
            r = run(gov, templated, seed, intervene, thr, args.max_new_tokens)
            records.append(r)
            parts.append(
                f"{tag}(fires={r['symbol_fires']},fullNA={r['full_nonascii']},"
                f"max24={r['maxwin24_nonascii']},nLing={r['non_ling_density']})"
            )
        print("  ".join(parts), flush=True)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[written {args.out}]", flush=True)


if __name__ == "__main__":
    main()
