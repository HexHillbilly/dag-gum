#!/usr/bin/env python3
"""Phase 2 benchmark for semantic-context-proxy: prompt-injection vs logit-bias.

Adds a `logit_bias` condition (decode-time corpus style-prior, see
style_prior.py) and sweeps its valve λ, head-to-head against the Phase-1
prompt-injection conditions. Adds `corpus_ppl` (perplexity of the output under
the corpus n-gram model) as the fingerprint metric — lower = closer to target.

Conditions:
  baseline          raw prompt, nothing
  markov            v1.0: retrieve -> Markov "misfire" -> chaotic-override wrap
  rag               mode B: retrieved lines as delimited grounding
  persona           mode A: coherent Daisy voice + top-1 line
  logit_bias_lX     raw prompt + decode-time λ·log P_corpus bias (the "skew")

Regimes: greedy (deterministic) + sampling (temp 0.8 / top_p 0.9, 3 samples).
Model: microsoft/Phi-3-mini-4k-instruct (see Phase-1 results for why Qwen base
was rejected). Every condition is passed through the chat template — the regime
the real proxy runs in.

Run: python scripts/benchmark.py
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

from core import (  # noqa: E402
    build_index,
    load_corpus,
    markov_synthesize,
    retrieve,
    wrap_misfire,
    wrap_persona,
    wrap_rag,
)
from style_prior import (  # noqa: E402
    CorpusNGramBias,
    build_ngram_model,
    corpus_ngram_ppl,
)

CORPUS_PATH = os.path.join(ROOT, "docs/corpora/emerson_essays.txt")
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 80
SAMPLING_KWARGS = dict(do_sample=True, temperature=0.8, top_p=0.9)
N_SAMPLES = 3
LAMBDAS = [0.02, 0.05, 0.1, 0.2]  # the "skew" valve sweep
NGRAM_N = 2

PROBES = [
    "How are you feeling today?",
    "What do you like to do for fun?",
    "Tell me something about yourself.",
    "What's the meaning of life?",
    "Do you have any friends?",
    "How do I fix a broken network connection?",
    "Explain how a jet engine works.",
    "What's your favorite food?",
    "Are you afraid of anything?",
    "Write me a short poem about the moon.",
]

BASE_CONDITIONS = ["baseline", "markov", "rag", "persona"]
CONDITIONS = BASE_CONDITIONS + [f"logit_bias_l{l}" for l in LAMBDAS]


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def embed(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    out = model.encode(texts, convert_to_numpy=True)
    out = np.asarray(out, dtype="float32")
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


def jaccard_distance(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 1.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def build_condition(condition, user_msg, retrieve_fn, ngram_model, persona_name, persona_desc):
    """Return (prompt, diag, logits_processors) for a condition."""
    if condition == "baseline":
        return user_msg, None, []
    matched, dists = retrieve_fn(user_msg)
    if condition == "markov":
        misfire, fallback = markov_synthesize(matched)
        return wrap_misfire(user_msg, misfire), {
            "retr_lines": matched,
            "retr_l2": dists,
            "misfire": misfire,
            "markov_fallback": int(fallback),
            "misfire_novelty": min(jaccard_distance(misfire, m) for m in matched),
        }, []
    if condition == "rag":
        return wrap_rag(user_msg, matched), {"retr_lines": matched, "retr_l2": dists}, []
    if condition == "persona":
        return wrap_persona(user_msg, matched[0], persona_name, persona_desc), {
            "retr_lines": matched,
            "retr_l2": dists,
        }, []
    if condition.startswith("logit_bias_l"):
        lam = float(condition.split("l")[-1])
        proc = CorpusNGramBias(ngram_model, n=NGRAM_N, lam=lam)
        return user_msg, {"lambda": lam}, [proc]
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS_PATH, help="sentence-per-line corpus path")
    ap.add_argument("--persona-name", default="Daisy")
    ap.add_argument(
        "--persona-desc",
        default="a warm, curious, slightly naive conversationalist",
    )
    ap.add_argument("--out", default=None, help="results jsonl path")
    args = ap.parse_args()
    corpus_path = args.corpus

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] Loading embedder + corpus: {corpus_path}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus(corpus_path)
    index = build_index(corpus, embedder)
    corpus_centroid = embed(embedder, corpus).mean(axis=0)

    print(f"[*] Loading LLM: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ngram_model = build_ngram_model(tok, corpus, n=NGRAM_N)
    print(f"[*] Corpus n-gram model: {len(ngram_model)} contexts")

    def retrieve_fn(prompt):
        return retrieve(prompt, index, corpus, embedder, k=3)

    def generate(prompt_wrapped, processors):
        msgs = [{"role": "user", "content": prompt_wrapped}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        for p in processors:
            p.prompt_len = prompt_len

        out = {}
        for regime, kwargs, n in [
            ("greedy", dict(do_sample=False), 1),
            ("sampling", SAMPLING_KWARGS, N_SAMPLES),
        ]:
            samples = []
            for i in range(n):
                seed_all(SEED + i)
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tok.pad_token_id,
                        logits_processor=processors if processors else None,
                        **kwargs,
                    )
                samples.append(
                    tok.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                )
            out[regime] = samples
        return out

    results = []
    for probe in PROBES:
        p_emb = embed(embedder, probe)[0]
        for cond in CONDITIONS:
            prompt_wrapped, diag, processors = build_condition(
                cond, probe, retrieve_fn, ngram_model, args.persona_name, args.persona_desc
            )
            samples_by_regime = generate(prompt_wrapped, processors)
            for regime, samples in samples_by_regime.items():
                for i, gen in enumerate(samples):
                    o_emb = embed(embedder, gen)[0]
                    cppl = corpus_ngram_ppl(tok, ngram_model, gen, n=NGRAM_N)
                    row = {
                        "probe": probe,
                        "condition": cond,
                        "regime": regime,
                        "sample": i,
                        "prompt": prompt_wrapped,
                        "output": gen,
                        "style_cos": cosine(o_emb, corpus_centroid),
                        "topic_cos": cosine(p_emb, o_emb),
                        "rep_rate": rep_rate(gen),
                        "corpus_ppl": cppl,
                        "out_len_chars": len(gen),
                    }
                    if diag and i == 0:
                        row.update(diag)
                    results.append(row)

    out_path = args.out or os.path.join(
        ROOT, "scripts", f"results_phase2_{Path(corpus_path).stem}.jsonl"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[+] Wrote {len(results)} rows -> {out_path}")

    # --- Summary ---
    print("\n=== SUMMARY (mean over probes; sampling = mean of %d samples) ===" % N_SAMPLES)
    hdr = f"{'condition':16s} {'regime':8s} {'style_cos':>10s} {'topic_cos':>10s} {'rep_rate':>10s} {'corpus_ppl':>12s}"
    print(hdr)
    for regime in ["greedy", "sampling"]:
        for c in CONDITIONS:
            rows = [r for r in results if r["condition"] == c and r["regime"] == regime]
            if not rows:
                continue
            cppls = [r["corpus_ppl"] for r in rows if r["corpus_ppl"] is not None]
            cppl_s = f"{np.mean(cppls):12.1f}" if cppls else f"{'n/a':>12s}"
            print(f"{c:16s} {regime:8s} "
                  f"{np.mean([r['style_cos'] for r in rows]):10.3f} "
                  f"{np.mean([r['topic_cos'] for r in rows]):10.3f} "
                  f"{np.mean([r['rep_rate'] for r in rows]):10.3f} "
                  f"{cppl_s}")

    mk = [r for r in results if r["condition"] == "markov" and "markov_fallback" in r]
    if mk:
        print(f"\nMarkov fallback rate: {np.mean([r['markov_fallback'] for r in mk]):.2f}")
        print(f"Markov misfire novelty: {np.mean([r['misfire_novelty'] for r in mk]):.3f}")
    print(f"Retrieval top-1 L2 (mean): "
          f"{np.mean([r['retr_l2'][0] for r in results if 'retr_l2' in r]):.3f}")


if __name__ == "__main__":
    main()
