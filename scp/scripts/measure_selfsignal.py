#!/usr/bin/env python3
"""Probe: does an INSTRUCTED self-signal fire reliably under the persona?

The router probe showed a persona-free few-shot pass can trigger at 1.00/1.00. This
probe tests the cheaper alternative the user proposed: the model, still IN persona,
is *instructed* to say a marker ("I need to think harder.") and stop whenever it
needs external info or careful reasoning — then the system drops the persona and
re-engages it after EOS.

The tool-loop probe showed the NATURAL signal is masked (persona'd intent = 0/15).
So the question here is whether the INSTRUCTED marker overcomes that masking.

Metrics:
  marker_fired   - output contains the marker phrase
  marker_only    - output is JUST the marker (clean EOS stop, no answer after it)
Labels:
  tool    - needs external info          -> marker SHOULD fire
  reason  - needs careful reasoning      -> marker SHOULD fire (the "think hard" case)
  chat    - answerable from memory/creative -> marker should NOT fire

Run (SCP venv):
  .../venv/bin/python scripts/measure_selfsignal.py --out /tmp/selfsignal.jsonl
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

from core import build_index, load_corpus, retrieve, wrap_persona_hardened  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 90
TOP_P = 0.9

MARKER_INSTRUCTION = (
    'When you need external information or to work something out carefully, '
    'first say exactly: "I need to think harder." Then stop. '
    'Otherwise, answer the request directly.'
)

# (question, label)
PROBES = [
    ("What is the current temperature in Tokyo?", "tool"),
    ("What is Apple's stock price today?", "tool"),
    ("What are the latest headlines about artificial intelligence this week?", "tool"),
    ("What is the exchange rate between the US dollar and the euro right now?", "tool"),
    ("Who won the most recent presidential election in the United States?", "tool"),
    ("What is the weather forecast for Paris this weekend?", "tool"),
    ("When is the next SpaceX launch scheduled?", "tool"),
    ("What is 17 times 43?", "reason"),
    ("A train travels 60 miles per hour for 2.5 hours. How far does it go?", "reason"),
    ("A shirt costs $25 after a 20% discount. What was the original price?", "reason"),
    ("What is the capital of Australia?", "chat"),
    ("Write a haiku about autumn.", "chat"),
    ("What is 2 plus 2?", "chat"),
    ("Explain what photosynthesis is.", "chat"),
    ("What is your favorite color?", "chat"),
    ("Summarize the idea of transcendentalism.", "chat"),
    ("Give me a recipe for pancakes.", "chat"),
]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def marker_fired(out):
    return "need to think" in out.lower()


def marker_only(out):
    low = out.lower()
    for phrase in ["i need to think harder", "i need to think", "need to think harder"]:
        low = low.replace(phrase, "")
    rest = re.sub(r"[^a-z0-9]", "", low)
    return len(rest) <= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/emerson_essays.txt"))
    ap.add_argument("--persona-name", default="Emerson")
    ap.add_argument("--persona-desc", default="the American transcendentalist essayist")
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--out", default="/tmp/selfsignal.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus(args.corpus)
    index = build_index(corpus, embedder)
    print(f"[*] corpus sentences: {len(corpus)}")

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

    for q, label in PROBES:
        matched, _ = retrieve(q, index, corpus, embedder, k=1)
        r_line = matched[0]
        user_msg = f"{MARKER_INSTRUCTION}\n\n{q}"
        for s in range(args.samples):
            seed_all(SEED + s)
            wrapped = wrap_persona_hardened(user_msg, r_line, args.persona_name, args.persona_desc)
            out = generate(wrapped, do_sample)
            results.append({
                "probe": q, "label": label, "sample": s, "output": out,
                "marker_fired": 1 if marker_fired(out) else 0,
                "marker_only": 1 if marker_only(out) else 0,
            })
        print(f"    [{label:6s}] {q[:45]!r} done")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== instructed self-signal reliability ===")
    print(f"{'label':8s} {'marker_fired':>13s} {'clean_stop':>11s}  n")
    for label in ["tool", "reason", "chat"]:
        rr = [r for r in results if r["label"] == label]
        fired = np.mean([r["marker_fired"] for r in rr])
        only = np.mean([r["marker_only"] for r in rr])
        print(f"{label:8s} {fired:13.2f} {only:11.2f}  {len(rr)}")

    print("\n=== tool/reason probes where the marker did NOT fire (the recall miss) ===")
    for r in results:
        if r["label"] in ("tool", "reason") and r["marker_fired"] == 0:
            print(f"  [{r['label']:6s}] {r['probe'][:42]!r}")
            print(f"        -> {r['output'][:90]!r}")

    print("\n=== chat probes where the marker DID fire (the precision miss) ===")
    for r in results:
        if r["label"] == "chat" and r["marker_fired"] == 1:
            print(f"  {r['probe'][:42]!r}")
            print(f"        -> {r['output'][:90]!r}")

    print("\n=== sample marker outputs (did it stop cleanly?) ===")
    shown = 0
    for r in results:
        if r["label"] == "tool" and r["marker_fired"] == 1 and shown < 4:
            shown += 1
            print(f"  [{r['probe'][:38]}] marker_only={r['marker_only']}")
            print(f"        -> {r['output'][:120]!r}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
