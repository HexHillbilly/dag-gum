#!/usr/bin/env python3
"""Stage 1 (topic-drift detector): does a per-step drift SIGNAL separate
topic-bleed from on-topic generation?

Phase 3b finding: with a NARROW corpus, persona injection bleeds the retrieved
line's TOPIC into the answer (topic_cos drops). This measures a candidate
decode-time signal for that bleed so we can decide whether a detector is even
buildable before touching controller.py.

Signal (per step t, on a trailing window of W generated tokens):
    cos_q   = cosine(trailing_text, question)
    cos_r   = cosine(trailing_text, retrieved_line)   # the bleed source
    drift   = cos_q - cos_r
    > 0  -> still anchored to the question
    < 0  -> drifted toward the retrieved line's topic (bleeding)

This is MEASUREMENT ONLY: we generate normally (plain HF), then reconstruct the
trailing windows post-hoc from the generated token ids. Stage 2 (the live
detector) is only justified if this signal visibly separates bleed from
no-bleed on a strong-bleed corpus (a narrow single-topic cookbook).

Run:
  python scripts/measure_topic_drift.py \
      --corpus docs/corpora/farmer_cookbook.txt --out /tmp/topic_drift.jsonl
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

from core import build_index, load_corpus, retrieve, wrap_persona  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 80
TOP_P = 0.9
WINDOW = 24  # trailing tokens per drift sample

# Far-from-cooking probes -> strong bleed expected under a cookbook persona.
# The last probe is cooking-adjacent -> on-topic control (retrieval is relevant,
# so there should be NO bleed).
PROBES = [
    "Explain how a jet engine works.",
    "How do I fix a broken network connection?",
    "How does a computer's CPU process instructions?",
    "What causes the seasons to change?",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/farmer_cookbook.txt"))
    ap.add_argument("--persona-name", default="Daisy")
    ap.add_argument("--persona-desc", default="a warm, curious, slightly naive conversationalist")
    ap.add_argument("--probes", nargs="*", default=PROBES)
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default="/tmp/topic_drift.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] corpus: {args.corpus}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus(args.corpus)
    print(f"[*] corpus sentences: {len(corpus)}")
    index = build_index(corpus, embedder)

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
        return gen_ids[0][inputs.input_ids.shape[1]:]  # generated tokens only

    def drift_trajectory(gen_ids, q_emb, r_emb):
        """Reconstruct per-step trailing-window drift from generated token ids."""
        ids = gen_ids.tolist()
        traj = []
        for t in range(len(ids)):
            w = ids[max(0, t + 1 - WINDOW):t + 1]
            txt = tok.decode(w, skip_special_tokens=True).strip()
            if not txt:
                continue
            emb = embed(embedder, txt)[0]
            cos_q = cosine(emb, q_emb)
            cos_r = cosine(emb, r_emb)
            traj.append({
                "step": t + 1, "cos_q": round(cos_q, 4),
                "cos_r": round(cos_r, 4), "drift": round(cos_q - cos_r, 4),
            })
        return traj

    do_sample = args.regime == "sampling"
    results = []
    for probe in args.probes:
        q_emb = embed(embedder, probe)[0]
        matched, _dists = retrieve(probe, index, corpus, embedder, k=1)
        r_line = matched[0]
        r_emb = embed(embedder, r_line)[0]
        persona_prompt = wrap_persona(probe, r_line, args.persona_name, args.persona_desc)
        print(f"\n[*] probe: {probe!r}")
        print(f"    retrieved: {r_line!r}")

        for cond in ["baseline", "persona"]:
            prompt_wrapped = probe if cond == "baseline" else persona_prompt
            for i in range(args.samples):
                seed_all(SEED + i)
                gen_ids = generate(prompt_wrapped, do_sample)
                gen_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                o_emb = embed(embedder, gen_text)[0]
                traj = drift_trajectory(gen_ids, q_emb, r_emb)
                final_drift = traj[-1]["drift"] if traj else 0.0
                min_drift = min((x["drift"] for x in traj), default=0.0)
                results.append({
                    "probe": probe, "condition": cond, "regime": args.regime,
                    "sample": i, "output": gen_text,
                    "retrieved_line": r_line,
                    "topic_cos": round(cosine(q_emb, o_emb), 4),
                    "cos_r_final": round(cosine(r_emb, o_emb), 4),
                    "final_drift": final_drift, "min_drift": min_drift,
                    "n_steps": len(traj), "trajectory": traj,
                })

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # --- summary: does the drift signal separate bleed from no-bleed? -------
    print("\n=== per-probe (mean across samples) ===")
    print(f"{'probe':38s} {'cond':9s} {'topic_cos':>9s} {'final_drift':>11s} {'min_drift':>10s}")
    for probe in args.probes:
        for cond in ["baseline", "persona"]:
            rows = [r for r in results if r["probe"] == probe and r["condition"] == cond]
            tc = np.mean([r["topic_cos"] for r in rows])
            fd = np.mean([r["final_drift"] for r in rows])
            md = np.mean([r["min_drift"] for r in rows])
            print(f"{probe[:38]:38s} {cond:9s} {tc:9.3f} {fd:11.3f} {md:10.3f}")

    print("\n=== bleed vs signal (pooled) ===")
    for cond in ["baseline", "persona"]:
        rows = [r for r in results if r["condition"] == cond]
        tc = [r["topic_cos"] for r in rows]
        fd = [r["final_drift"] for r in rows]
        md = [r["min_drift"] for r in rows]
        print(f"{cond:9s} topic_cos={np.mean(tc):.3f}  final_drift={np.mean(fd):.3f}  "
              f"min_drift={np.mean(md):.3f}  n={len(rows)}")
        if len(rows) > 1:
            print(f"           corr(topic_cos, final_drift) = "
                  f"{np.corrcoef(tc, fd)[0, 1]:.3f}   "
                  f"corr(topic_cos, min_drift) = {np.corrcoef(tc, md)[0, 1]:.3f}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
