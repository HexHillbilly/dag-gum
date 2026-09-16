#!/usr/bin/env python3
"""Phase 4: statistical AI-text detection score (GLTR-style).

Measures two signals that statistical AI-text detectors (GLTR / DetectGPT /
Panagram-class) key on:

  * perplexity  — mean token surprise under a reference LLM. LLM output tends
                  to be LOW perplexity (the model predicts its own text well);
                  human text is HIGHER perplexity.
  * burstiness  — spread of per-sentence perplexity. Human text is "burstier"
                  (varies sentence-to-sentence); LLM output is more uniform.

Scores three groups and reports the deltas:
  baseline  : the model's default assistant output
  persona   : style-injected output (from a benchmark results jsonl)
  human     : a held-out human-author corpus (reference)

The claim under test: persona injection moves the output's statistical profile
TOWARD the human reference, away from the default LLM fingerprint.

Usage:
  python scripts/score_detector.py <results.jsonl> <human_reference.txt>
"""
import json
import math
import re
import statistics
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 10]


@torch.no_grad()
def text_ppl(model, tok, text):
    ids = tok.encode(text, return_tensors="pt").to(model.device)
    if ids.shape[1] < 2:
        return None
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


def metrics(model, tok, text):
    m = text_ppl(model, tok, text)
    if m is None:
        return None
    ppls = [text_ppl(model, tok, s) for s in sentences(text)]
    ppls = [p for p in ppls if p is not None]
    burst = statistics.pstdev(ppls) if len(ppls) >= 2 else 0.0
    return {"mean_ppl": m, "burstiness": burst, "n_sents": len(ppls)}


def main():
    results_path, human_path = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    rows = [json.loads(l) for l in open(results_path)]
    baseline = [r["output"] for r in rows if r["condition"] == "baseline"
                and r["regime"] == "sampling" and r["sample"] == 0]
    persona = [r["output"] for r in rows if r["condition"] == "persona"
               and r["regime"] == "sampling" and r["sample"] == 0]
    # Chunk the one-sentence-per-line human reference into ~3-sentence groups so
    # within-text burstiness is comparable to the multi-sentence outputs.
    human_lines = [l.strip() for l in open(human_path, encoding="utf-8") if l.strip()]
    human = [" ".join(human_lines[i:i + 3]) for i in range(0, min(len(human_lines), 60), 3)]

    groups = {"baseline": baseline, "persona": persona, "human": human}
    agg = {}
    for name, texts in groups.items():
        ms = [metrics(model, tok, t) for t in texts]
        ms = [m for m in ms if m is not None]
        if not ms:
            print(f"{name}: no scored texts")
            continue
        agg[name] = {
            "mean_ppl": statistics.mean(m["mean_ppl"] for m in ms),
            "burstiness": statistics.mean(m["burstiness"] for m in ms),
            "n": len(ms),
        }
        print(f"{name:10s} n={len(ms):3d}  mean_ppl={agg[name]['mean_ppl']:10.1f}  "
              f"burstiness={agg[name]['burstiness']:9.2f}")

    if set(["baseline", "persona", "human"]) <= set(agg):
        b, p, h = agg["baseline"], agg["persona"], agg["human"]
        gap = h["mean_ppl"] - b["mean_ppl"]
        shift = p["mean_ppl"] - b["mean_ppl"]
        print("\n=== deltas (perplexity, higher = more human-like) ===")
        print(f"baseline -> human gap : {gap:+.1f}")
        print(f"persona shift        : {shift:+.1f} "
              f"({shift / gap * 100:.0f}% of the gap toward human)" if gap else "gap=0")
        print(f"burstiness: baseline={b['burstiness']:.2f} persona={p['burstiness']:.2f} "
              f"human={h['burstiness']:.2f}")


if __name__ == "__main__":
    main()
