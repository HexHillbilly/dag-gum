#!/usr/bin/env python3
"""Reframed research question: does persona injection transfer the author's
LEXICON (voice/terms) while keeping the SUBJECT anchored to the question?

The "topic bleed" mechanism, read as a feature: feed a single author's writing
(the stand-in "user's lexicon"), retrieve a line, inject it as persona, and ask
a cross-subject question. The useful outcome is "explain the question IN THEIR
TERMS" — high style_cos (their voice) AND high topic_cos (still your question).

Joint metric:
  topic_cos  = cosine(output, question)   high = still answering the question
  style_cos  = cosine(output, corpus centroid)  high = using the author's voice

  good transfer (explain in their terms): HIGH topic_cos AND HIGH style_cos
  subject swap   (the old "contamination"): LOW topic_cos, HIGH style_cos
  no transfer    (baseline):                HIGH topic_cos, LOW style_cos

Run (SCP venv):
  python scripts/measure_explain_in_terms.py \
      --corpus docs/corpora/emerson_essays.txt --out /tmp/explain_in_terms.jsonl
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

from core import build_index, load_corpus, retrieve, wrap_persona_hardened, wrap_explain_in_terms  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 90
TOP_P = 0.9

# Cross-subject probes — technical/scientific, topically distant from Emerson's
# moral philosophy. Retrieval will return an Emerson line (philosophy), and the
# question is whether the output explains the probe in Emerson's terms or drifts
# into philosophy.
PROBES = [
    "Explain how a jet engine works.",
    "How does a computer's CPU process instructions?",
    "How do vaccines train the immune system?",
    "What causes the seasons to change?",
    "How does a combustion engine convert fuel into motion?",
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
    ap.add_argument("--probes", nargs="*", default=PROBES)
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default="/tmp/explain_in_terms.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] corpus: {args.corpus}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus(args.corpus)
    print(f"[*] corpus sentences: {len(corpus)}")
    index = build_index(corpus, embedder)
    corpus_centroid = embed(embedder, corpus).mean(axis=0)

    print(f"[*] model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
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

    do_sample = args.regime == "sampling"
    results = []
    for probe in args.probes:
        q_emb = embed(embedder, probe)[0]
        matched, _ = retrieve(probe, index, corpus, embedder, k=1)
        r_line = matched[0]
        persona_prompt = wrap_persona_hardened(probe, r_line, args.persona_name, args.persona_desc)
        explain_prompt = wrap_explain_in_terms(probe, r_line, args.persona_name, args.persona_desc)
        print(f"\n[*] probe: {probe!r}")
        print(f"    retrieved: {r_line[:90]!r}")

        for cond in ["baseline", "persona", "explain"]:
            if cond == "baseline":
                prompt_wrapped = probe
            elif cond == "explain":
                prompt_wrapped = explain_prompt
            else:
                prompt_wrapped = persona_prompt
            for i in range(args.samples):
                seed_all(SEED + i)
                out = generate(prompt_wrapped, do_sample)
                o_emb = embed(embedder, out)[0] if out else np.zeros(q_emb.shape, dtype="float32")
                results.append({
                    "probe": probe, "condition": cond, "regime": args.regime,
                    "sample": i, "output": out, "retrieved_line": r_line,
                    "topic_cos": round(cosine(q_emb, o_emb), 4),
                    "style_cos": round(cosine(o_emb, corpus_centroid), 4),
                })

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== SUMMARY (joint topic_cos / style_cos) ===")
    for cond in ["baseline", "persona", "explain"]:
        rows = [r for r in results if r["condition"] == cond]
        tc = np.mean([r["topic_cos"] for r in rows])
        sc = np.mean([r["style_cos"] for r in rows])
        print(f"{cond:9s} topic_cos={tc:.3f}  style_cos={sc:.3f}  n={len(rows)}")

    print("\n=== per-probe (topic_cos / style_cos) ===")
    for probe in args.probes:
        for cond in ["baseline", "persona", "explain"]:
            rows = [r for r in results if r["probe"] == probe and r["condition"] == cond]
            tc = np.mean([r["topic_cos"] for r in rows])
            sc = np.mean([r["style_cos"] for r in rows])
            print(f"{probe[:38]:38s} {cond:9s} topic={tc:.3f} style={sc:.3f}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
