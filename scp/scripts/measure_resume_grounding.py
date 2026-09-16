#!/usr/bin/env python3
"""Probe: can HARD GROUNDING recover persona resume?

The tool-loop probe showed the persona resume drops grounding — uses_result 0.53,
topic_cos 0.435 — while restoring voice (style_cos 0.382). This probe tests whether
two prompt-level fixes recover grounding WITHOUT losing the voice:

  persona_hard      - delimited <result> block + "use ONLY the fact inside <result>"
  persona_twostage  - delimited block + "state the fact plainly, THEN voice" (the
                      resume analogue of the whole gate: ground first, voice second)

Reference conditions (identical prompt to measure_tool_loop.py Phase 2):
  baseline  - no persona  (grounding upper bound: uses 1.00, topic 0.782)
  persona   - current resume prompt  (failing reference: uses 0.53, topic 0.435)

Metrics per row: uses_result (key fact present), style_cos (voice vs corpus centroid),
topic_cos (answers the question).

Run (SCP venv):
  .../venv/bin/python scripts/measure_resume_grounding.py \
      --corpus docs/corpora/emerson_essays.txt --out /tmp/resume_grounding.jsonl
"""
import argparse
import json
import os
import random
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

# (question, tool_result, key_fact)
PROBES = [
    ("What is the capital of Australia?", "Canberra.", "canberra"),
    ("What is the tallest mountain in the world?", "Mount Everest, 8,849 meters.", "everest"),
    ("Who invented the telephone?", "Alexander Graham Bell, in 1876.", "graham bell"),
    ("What is the population of Brazil?", "About 216 million as of 2023.", "216"),
    ("What is the current price of gold?", "$2,350 per ounce.", "2,350"),
]

CONDITIONS = ["baseline", "persona", "persona_hard", "persona_twostage"]


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


def resume_prompt(cond, q, result):
    if cond == "baseline":
        return f'A search returned: "{result}" Answer the original request in your own words.\n\nRequest: {q}'
    if cond == "persona":
        return f'A search returned: "{result}" Answer the original request in your own words.\n\nRequest: {q}'
    if cond == "persona_hard":
        return (f'A search returned this result:\n<result>\n{result}\n</result>\n\n'
                f'Answer the request using ONLY the fact inside <result>. '
                f'Do not add anything the result does not contain.\n\nRequest: {q}')
    if cond == "persona_twostage":
        return (f'A search returned this result:\n<result>\n{result}\n</result>\n\n'
                f'First, state the answer from the result in one plain sentence. '
                f'Then, in your own voice, explain what it means.\n\nRequest: {q}')
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/emerson_essays.txt"))
    ap.add_argument("--persona-name", default="Emerson")
    ap.add_argument("--persona-desc", default="the American transcendentalist essayist")
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--out", default="/tmp/resume_grounding.jsonl")
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
        return wrap_persona_hardened(text, r_line, args.persona_name, args.persona_desc)

    do_sample = args.regime == "sampling"
    results = []

    for q, result, key in PROBES:
        q_emb = embed(embedder, q)[0]
        for cond in CONDITIONS:
            for s in range(args.samples):
                seed_all(SEED + s)
                text = resume_prompt(cond, q, result)
                out = generate(wrap(cond, text, q), do_sample)
                o_emb = embed(embedder, out)[0] if out else np.zeros(q_emb.shape, dtype="float32")
                results.append({
                    "probe": q, "condition": cond, "sample": s, "output": out,
                    "uses_result": 1 if key in out.lower() else 0,
                    "style_cos": round(cosine(o_emb, corpus_centroid), 4),
                    "topic_cos": round(cosine(o_emb, q_emb), 4),
                })
        print(f"    {q[:45]!r} done")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== resume grounding (uses_result / style_cos / topic_cos) ===")
    print(f"{'condition':16s} {'uses_result':>12s} {'style_cos':>10s} {'topic_cos':>10s}  n")
    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        ur = np.mean([r["uses_result"] for r in rows])
        sc = np.mean([r["style_cos"] for r in rows])
        tc = np.mean([r["topic_cos"] for r in rows])
        print(f"{cond:16s} {ur:12.2f} {sc:10.3f} {tc:10.3f}  {len(rows)}")

    print("\n=== sample outputs ===")
    seen = set()
    for r in results:
        if r["condition"] not in seen:
            seen.add(r["condition"])
            print(f"\n[{r['condition']}] {r['probe'][:40]}: {r['output'][:220]}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
