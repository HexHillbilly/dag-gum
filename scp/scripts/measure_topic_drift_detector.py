#!/usr/bin/env python3
"""Stage 2: does the AVG topic-drift detector (both actuations) reduce topic-bleed?

Compares the governor WITH drift references (det_on) vs WITHOUT (det_off), on
the SCP persona-injection setup (narrow retrieval -> topic bleed).

Conditions (both through ActiveVarietyGovernor, intervene=True):
  det_off  persona prompt, NO drift refs (Track-2 baseline: governor doesn't fix
           coherent topic drift)
  det_on   same prompt + drift_embedder + question/retrieved embeddings +
           bleed token ids (temperature cooling + content-token suppression)

Metric: topic_cos = cosine(whole output emb, question emb). Higher = less bleed.
Also records drift_fires (governor telemetry) to confirm the detector actually
fired on the bleeding outputs.

Run (SCP venv: sentence-transformers + faiss + transformers):
  python scripts/measure_topic_drift_detector.py \
      --corpus docs/corpora/huckleberry_finn.txt --out /tmp/drift_detector.jsonl
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

from core import build_index, load_corpus, retrieve, wrap_persona  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
BAND_LOW = 6.95  # Phi-3-mini derived valve (Track 2)
SEED = 42
MAX_NEW_TOKENS = 80
TOP_P = 0.9

PROBES = [
    "Explain how a jet engine works.",
    "How do I fix a broken network connection?",
    "How does a computer's CPU process instructions?",
    "What causes the seasons to change?",
    "What's your favorite food?",
]

STOPWORDS = set(
    "a an the and or but if then else of in on at to for with from by as is are "
    "was were be been being it its this that these those he she they we you i my "
    "your our their his her him them me us not no yes do does did done have has "
    "had will would can could should shall may might must about into over under "
    "again further once here there when where why how all any both each few more "
    "most other some such only own same so than too very just because what which "
    "who whom whose".split()
)


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


def extract_bleed_token_ids(line, tok):
    """Distinctive content-word token ids from the retrieved line.

    Keeps alphabetic tokens >= 3 chars that are not stopwords — the words that
    carry the line's topic. Suppressing these on drift blocks the bleed source's
    vocabulary. (Subword over-suppression is a known, acceptable first-cut cost.)
    """
    ids = tok.encode(line, add_special_tokens=False)
    out = []
    for tid in ids:
        s = tok.decode([tid]).strip().lower()
        if s.isalpha() and len(s) >= 3 and s not in STOPWORDS:
            out.append(tid)
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/huckleberry_finn.txt"))
    ap.add_argument("--persona-name", default="Daisy")
    ap.add_argument("--persona-desc", default="a warm, curious, slightly naive conversationalist")
    ap.add_argument("--probes", nargs="*", default=PROBES)
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default="/tmp/drift_detector.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from AVG.governor.controller import ActiveVarietyGovernor

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

    gov = ActiveVarietyGovernor(model, tok, band_low=BAND_LOW, top_p=TOP_P)
    print(f"[*] governor armed (band_low={BAND_LOW}, top_p={TOP_P}, "
          f"drift_threshold={gov.drift_threshold}, drift_temp={gov.drift_temp})")

    def run(probe, cond):
        q_emb = embed(embedder, probe)[0]
        matched, _ = retrieve(probe, index, corpus, embedder, k=1)
        r_line = matched[0]
        r_emb = embed(embedder, r_line)[0]
        bleed_ids = extract_bleed_token_ids(r_line, tok)
        persona_prompt = wrap_persona(probe, r_line, args.persona_name, args.persona_desc)

        if cond == "det_on":
            kwargs = dict(
                drift_embedder=embedder,
                topic_anchor_emb=q_emb,
                topic_bleed_emb=r_emb,
                topic_bleed_token_ids=bleed_ids,
            )
        else:
            kwargs = {}

        do_sample = args.regime == "sampling"
        rows = []
        for i in range(args.samples):
            seed_all(SEED + i)
            res = gov.generate(
                persona_prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample,
                temperature=0.8, intervene=True, **kwargs,
            )
            out = (res.get("gen_only_text") or "").strip()
            o_emb = embed(embedder, out)[0] if out else np.zeros(q_emb.shape, dtype="float32")
            rows.append({
                "probe": probe, "condition": cond, "regime": args.regime,
                "sample": i, "output": out, "retrieved_line": r_line,
                "n_bleed_tokens": len(bleed_ids),
                "topic_cos": round(cosine(q_emb, o_emb), 4) if out else 0.0,
                "drift_fires": res["collapse"].get("drift_fires", 0),
            })
        return rows

    results = []
    for probe in args.probes:
        print(f"\n[*] probe: {probe!r}")
        for cond in ["det_off", "det_on"]:
            rows = run(probe, cond)
            results.extend(rows)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== SUMMARY (mean topic_cos; drift_fires) ===")
    for cond in ["det_off", "det_on"]:
        rows = [r for r in results if r["condition"] == cond]
        tc = np.mean([r["topic_cos"] for r in rows])
        df = np.mean([r["drift_fires"] for r in rows])
        print(f"{cond:9s} topic_cos={tc:.3f}  drift_fires={df:.2f}  n={len(rows)}")

    print("\n=== per-probe (topic_cos det_off -> det_on) ===")
    for probe in args.probes:
        off = [r["topic_cos"] for r in results if r["probe"] == probe and r["condition"] == "det_off"]
        on = [r["topic_cos"] for r in results if r["probe"] == probe and r["condition"] == "det_on"]
        print(f"{probe[:40]:40s} {np.mean(off):.3f} -> {np.mean(on):.3f}  ({np.mean(on)-np.mean(off):+.3f})")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
