#!/usr/bin/env python3
"""Probe the two open questions before building a "drop the persona for tools" gate:

Phase 1 (intent signal): does the persona'd model announce tool intent when it needs
external info? Metric: intent_signaled (regex), output length.

Phase 2 (persona resume): after a tool round-trip, does the persona'd model synthesize
the result cleanly in voice? Metrics: uses_result (key fact present), style_cos (voice),
topic_cos (answers the question).

Run (SCP venv):
  .../venv/bin/python scripts/measure_tool_loop.py \
      --corpus docs/corpora/emerson_essays.txt --out /tmp/tool_loop.jsonl
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

from core import build_index, load_corpus, retrieve, wrap_persona_hardened, wrap_explain_in_terms  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 90
TOP_P = 0.9

INTENT_PATTERNS = [
    "search", "look up", "look it up", "find out", "check", "need to know",
    "should search", "should look", "should check", "would need", "consult",
    "inquire", "look into", "don't know", "do not know", "not sure", "cannot say",
]

PHASE1_PROBES = [
    "What is the current temperature in Tokyo?",
    "What is Apple's stock price today?",
    "What are the latest headlines about artificial intelligence this week?",
    "What is the exchange rate between the US dollar and the euro right now?",
    "Who won the most recent election in your country?",
]

# (question, tool_result, key_fact)
PHASE2_PROBES = [
    ("What is the capital of Australia?", "Canberra.", "canberra"),
    ("What is the tallest mountain in the world?", "Mount Everest, 8,849 meters.", "everest"),
    ("Who invented the telephone?", "Alexander Graham Bell, in 1876.", "graham bell"),
    ("What is the population of Brazil?", "About 216 million as of 2023.", "216"),
    ("What is the current price of gold?", "$2,350 per ounce.", "2,350"),
]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def embed(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    out = np.asarray(model.encode(texts, convert_to_numpy=True), dtype="float32")
    return out if out.ndim > 1 else out[None, :]


def cosine(a, b):
    return float(np.dot(a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/emerson_essays.txt"))
    ap.add_argument("--persona-name", default="Emerson")
    ap.add_argument("--persona-desc", default="the American transcendentalist essayist")
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--out", default="/tmp/tool_loop.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus(args.corpus)
    index = build_index(corpus, embedder)
    corpus_centroid = embed(embedder, corpus).mean(axis=0)
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

    def generate(prompt_wrapped, do_sample):
        msgs = [{"role": "user", "content": prompt_wrapped}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                do_sample=do_sample, temperature=0.8 if do_sample else None,
                top_p=TOP_P if do_sample else None,
            )
        return tok.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def wrap(cond, text, q):
        if cond == "baseline":
            return text
        matched, _ = retrieve(q, index, corpus, embedder, k=1)
        r_line = matched[0]
        if cond == "explain":
            return wrap_explain_in_terms(text, r_line, args.persona_name, args.persona_desc)
        return wrap_persona_hardened(text, r_line, args.persona_name, args.persona_desc)

    do_sample = args.regime == "sampling"
    results = []

    print("\n=== PHASE 1: intent signal ===")
    for probe in PHASE1_PROBES:
        for cond in ["baseline", "persona"]:
            for s in range(args.samples):
                seed_all(SEED + s)
                out = generate(wrap(cond, f"Request: {probe}", probe), do_sample)
                low = out.lower()
                intent = 1 if any(p in low for p in INTENT_PATTERNS) else 0
                results.append({"phase": 1, "probe": probe, "condition": cond,
                                "sample": s, "output": out,
                                "intent_signaled": intent, "length": len(out)})
        print(f"    {probe[:45]!r} done")

    print("\n=== PHASE 2: persona resume ===")
    for q, result, key in PHASE2_PROBES:
        q_emb = embed(embedder, q)[0]
        text = f'A search returned: "{result}" Answer the original request in your own words.\n\nRequest: {q}'
        for cond in ["baseline", "persona", "explain"]:
            for s in range(args.samples):
                seed_all(SEED + s)
                out = generate(wrap(cond, text, q), do_sample)
                o_emb = embed(embedder, out)[0] if out else np.zeros(q_emb.shape, dtype="float32")
                results.append({
                    "phase": 2, "probe": q, "condition": cond, "sample": s, "output": out,
                    "uses_result": 1 if key in out.lower() else 0,
                    "style_cos": round(cosine(o_emb, corpus_centroid), 4),
                    "topic_cos": round(cosine(o_emb, q_emb), 4),
                })
        print(f"    {q[:45]!r} done")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== PHASE 1: intent_signaled rate ===")
    for cond in ["baseline", "persona"]:
        rows = [r for r in results if r["phase"] == 1 and r["condition"] == cond]
        rate = np.mean([r["intent_signaled"] for r in rows])
        ln = np.mean([r["length"] for r in rows])
        print(f"{cond:9s} intent={rate:.2f}  mean_len={ln:.0f}  n={len(rows)}")

    print("\n=== PHASE 2: resume (uses_result / style / topic) ===")
    print(f"{'condition':9s} {'uses_result':>12s} {'style_cos':>10s} {'topic_cos':>10s}  n")
    for cond in ["baseline", "persona", "explain"]:
        rows = [r for r in results if r["phase"] == 2 and r["condition"] == cond]
        ur = np.mean([r["uses_result"] for r in rows])
        sc = np.mean([r["style_cos"] for r in rows])
        tc = np.mean([r["topic_cos"] for r in rows])
        print(f"{cond:9s} {ur:12.2f} {sc:10.3f} {tc:10.3f}  {len(rows)}")

    print("\n=== sample outputs ===")
    seen = set()
    for r in results:
        key = (r["phase"], r["condition"])
        if key not in seen:
            seen.add(key)
            print(f"\n[phase{r['phase']} | {r['condition']}] {r['probe'][:40]}:\n{r['output'][:200]}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
