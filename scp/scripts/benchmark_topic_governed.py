#!/usr/bin/env python3
"""Track 2: does AVG governance suppress topic-bleed under persona injection?

Phase 3b finding: with a NARROW corpus, persona injection bleeds the retrieved
line's topic into the answer (topic_cos drops). This tests whether the AVG
governed backend (SCP persona pre-gen + AVG mid-gen logit penalties) changes it.

Conditions (all through the chat template):
  baseline          raw prompt, plain HF generate
  persona           wrap_persona + top-1 retrieved line, plain HF generate
  persona_governed  same persona prompt, generation via ActiveVarietyGovernor
                    (top_p matched to the plain path to isolate the logit-penalty
                    effect from the sampling regime)

Expected (measure, don't assume): AVG's penalties target loops/degeneracy, not
coherent topic drift, so governance likely does NOT restore topic fidelity.

Run:
  python scripts/benchmark_topic_governed.py \
      --corpus docs/corpora/huckleberry_finn.txt
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # AVG package parent

from core import build_index, load_corpus, retrieve, wrap_persona, wrap_persona_topic  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
BAND_LOW = 6.95  # Phi-3-mini derived valve
SEED = 42
MAX_NEW_TOKENS = 80
TOP_P = 0.9  # match the plain benchmark sampling regime

PROBES = [
    "Explain how a jet engine works.",
    "How do I fix a broken network connection?",
    "What's the meaning of life?",
    "Tell me something about yourself.",
    "What's your favorite food?",
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


def rep_rate(text):
    import re
    toks = re.findall(r"[a-z0-9']+", text.lower())
    if len(toks) < 3:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/huckleberry_finn.txt"))
    ap.add_argument("--persona-name", default="Daisy")
    ap.add_argument("--persona-desc", default="a warm, curious, slightly naive conversationalist")
    ap.add_argument("--probes", nargs="*", default=PROBES)
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="greedy")
    ap.add_argument("--out", default="/tmp/track2_results.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from AVG.governor.controller import ActiveVarietyGovernor

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

    gov = ActiveVarietyGovernor(model, tok, band_low=BAND_LOW, top_p=TOP_P)
    print("[*] AVG governor armed (band_low=%.2f, top_p=%.2f)" % (BAND_LOW, TOP_P))

    def retrieve_fn(prompt):
        return retrieve(prompt, index, corpus, embedder, k=3)

    def plain_generate(prompt_wrapped, do_sample):
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

    def governed_generate(prompt_wrapped, do_sample):
        msgs = [{"role": "user", "content": prompt_wrapped}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        res = gov.generate(text, max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample,
                           temperature=0.8, intervene=True)
        return (res.get("gen_only_text") or "").strip()

    results = []
    for probe in args.probes:
        p_emb = embed(embedder, probe)[0]
        matched, dists = retrieve_fn(probe)
        persona_prompt = wrap_persona(probe, matched[0], args.persona_name, args.persona_desc)
        persona_topic_prompt = wrap_persona_topic(probe, matched[0], args.persona_name, args.persona_desc)

        for cond in ["baseline", "persona", "persona_topic", "persona_governed"]:
            if cond == "baseline":
                prompt_wrapped = probe
            elif cond == "persona_topic":
                prompt_wrapped = persona_topic_prompt
            else:
                prompt_wrapped = persona_prompt
            do_sample = args.regime == "sampling"
            n = 3 if do_sample else 1
            samples = []
            for i in range(n):
                seed_all(SEED + i)
                gen = (governed_generate if cond == "persona_governed" else plain_generate)(
                    prompt_wrapped, do_sample)
                samples.append(gen)
            for i, gen in enumerate(samples):
                o_emb = embed(embedder, gen)[0]
                results.append({
                    "probe": probe, "condition": cond, "regime": args.regime,
                    "sample": i, "output": gen,
                    "style_cos": cosine(o_emb, corpus_centroid),
                    "topic_cos": cosine(p_emb, o_emb),
                    "rep_rate": rep_rate(gen),
                    "out_len_chars": len(gen),
                })

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== SUMMARY (mean topic_cos / style_cos / rep_rate) ===")
    for cond in ["baseline", "persona", "persona_topic", "persona_governed"]:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            continue
        print(f"{cond:16s} topic_cos={np.mean([r['topic_cos'] for r in rows]):.3f} "
              f"style_cos={np.mean([r['style_cos'] for r in rows]):.3f} "
              f"rep_rate={np.mean([r['rep_rate'] for r in rows]):.3f} "
              f"n={len(rows)}")
    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
