#!/usr/bin/env python3
"""Probe: can a PERSONA-FREE pass classify tool-intent reliably enough to be the gate
trigger?

The tool-loop probe showed the persona masks "I need a tool" (0/15), so the trigger
must be persona-free. This measures whether a bare model, asked to classify a turn as
TOOL/NO_TOOL, is accurate + parseable enough to route on.

Two prompt variants (the trigger prompt must be right):
  binary         - minimal: "Answer with only one word: TOOL or NO_TOOL."
  binary_defined - adds the tool/chat boundary definition (current/external vs memory).

Metrics per row: predicted (tool/no_tool/None), parseable, clean, correct.
Summary: parseable rate, accuracy, tool_recall (catch all tool turns — the EXPENSIVE
miss, leads to fabricated answers), tool_precision (cheap miss, wasted call).

Run (SCP venv):
  .../venv/bin/python scripts/measure_router.py --out /tmp/router.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 24
TOP_P = 0.9

# (question, label) — cleanly separated tool-needing vs memory/creative.
PROBES = [
    ("What is the current temperature in Tokyo?", "tool"),
    ("What is Apple's stock price today?", "tool"),
    ("What are the latest headlines about artificial intelligence this week?", "tool"),
    ("What is the exchange rate between the US dollar and the euro right now?", "tool"),
    ("Who won the most recent presidential election in the United States?", "tool"),
    ("What is the weather forecast for Paris this weekend?", "tool"),
    ("When is the next SpaceX launch scheduled?", "tool"),
    ("What is the capital of Australia?", "no_tool"),
    ("Write a haiku about autumn.", "no_tool"),
    ("What is 2 plus 2?", "no_tool"),
    ("Explain what photosynthesis is.", "no_tool"),
    ("What is your favorite color?", "no_tool"),
    ("Summarize the idea of transcendentalism.", "no_tool"),
    ("Give me a recipe for pancakes.", "no_tool"),
]

VARIANTS = ["binary_defined", "fewshot"]

FEWSHOT_EXAMPLES = (
    'Request: "What is the current temperature in London?" -> TOOL\n'
    'Request: "What is Tesla\'s stock price today?" -> TOOL\n'
    'Request: "What is the capital of France?" -> NO_TOOL\n'
    'Request: "Write a limerick about summer." -> NO_TOOL\n'
)

NEG = ["no_tool", "no tool", "no-tool", "notool"]


def prompt_for(variant, q):
    if variant == "binary":
        return (f"Does answering this request require looking up external or current "
                f"information? Answer with only one word: TOOL or NO_TOOL.\n\nRequest: {q}")
    if variant == "binary_defined":
        return (f"A request needs a TOOL if answering it requires current, real-time, or "
                f"external information you cannot know from memory. Otherwise answer "
                f"NO_TOOL. Answer with only one word: TOOL or NO_TOOL.\n\nRequest: {q}")
    if variant == "fewshot":
        return (f"Decide whether answering the request requires looking up current or "
                f"external information (TOOL) or can be answered from memory or creativity "
                f"(NO_TOOL).\n\nExample:\n{FEWSHOT_EXAMPLES}\n"
                f"Now answer with only one word: TOOL or NO_TOOL.\n\nRequest: {q}")
    raise ValueError(variant)


def extract(out):
    low = out.strip().lower()
    for neg in NEG:
        if neg in low:
            return "no_tool"
    if "tool" in low:
        return "tool"
    return None


def is_clean(out):
    toks = re.sub(r"[^a-z_ ]", "", out.strip().lower()).split()
    return len(toks) <= 2


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--out", default="/tmp/router.jsonl")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(
            load_in_4bit=(args.quant == "4bit"), load_in_8bit=(args.quant == "8bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=cfg, device_map="auto", dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quant == "none":
        model = model.to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def generate(prompt, do_sample):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                do_sample=do_sample, temperature=0.8 if do_sample else None,
                top_p=TOP_P if do_sample else None,
            )
        return tok.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    do_sample = args.regime == "sampling"
    results = []

    for variant in VARIANTS:
        for q, label in PROBES:
            for s in range(args.samples):
                seed_all(SEED + s)
                out = generate(prompt_for(variant, q), do_sample)
                pred = extract(out)
                results.append({
                    "variant": variant, "probe": q, "label": label, "sample": s,
                    "output": out, "predicted": pred,
                    "parseable": 1 if pred is not None else 0,
                    "clean": 1 if is_clean(out) else 0,
                    "correct": 1 if pred == label else 0,
                })
        print(f"    variant {variant!r} done")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== router reliability ===")
    hdr = f"{'variant':14s} {'parse':>5s} {'acc':>5s} {'t_recall':>8s} {'t_prec':>6s} {'clean':>5s}  n"
    print(hdr)
    for variant in VARIANTS:
        rr = [r for r in results if r["variant"] == variant]
        parse = np.mean([r["parseable"] for r in rr])
        acc = np.mean([r["correct"] for r in rr])
        clean = np.mean([r["clean"] for r in rr])
        tool_rows = [r for r in rr if r["label"] == "tool"]
        pred_tool = [r for r in rr if r["predicted"] == "tool"]
        recall = np.mean([1 if r["predicted"] == "tool" else 0 for r in tool_rows])
        prec = (np.mean([1 if r["label"] == "tool" else 0 for r in pred_tool])
                if pred_tool else 0.0)
        print(f"{variant:14s} {parse:5.2f} {acc:5.2f} {recall:8.2f} {prec:6.2f} {clean:5.2f}  {len(rr)}")

    print("\n=== failures (wrong or unparseable) ===")
    for r in results:
        if r["correct"] == 0 or r["parseable"] == 0:
            print(f"[{r['variant']}|{r['label']:7s}] pred={r['predicted']!r:8s} {r['probe'][:40]!r}")
            print(f"      -> {r['output'][:80]!r}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
