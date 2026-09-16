#!/usr/bin/env python3
"""Phase B (SSP): trained statistical AI-text detector.

Upgrades the Phase-4 GLTR-style proxy (mean perplexity + burstiness) to a
*trained* classifier: logistic regression over [log-ppl, burstiness, rep_rate]
features, fit to label AI text (the benchmark baseline outputs) vs human text
(a held-out public-domain author — Austen — NOT the target style, so the
detector learns "general human vs AI", not "target-style vs AI").

The claim under test (S3): style transfer moves the output's detector score
TOWARD the human reference, away from the baseline AI fingerprint. This is the
detection *mirror* of the style product — reported, never the goal.

Usage:
  python scripts/detector.py <results.jsonl> <human_corpus.txt>
"""
import json
import math
import random
import re
import statistics
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 10]


@torch.no_grad()
def text_ppl(model, tok, text):
    ids = tok.encode(text, return_tensors="pt").to(model.device)
    if ids.shape[1] < 2:
        return None
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


def burstiness(model, tok, text):
    ppls = [text_ppl(model, tok, s) for s in sentences(text)]
    ppls = [p for p in ppls if p is not None]
    return statistics.pstdev(ppls) if len(ppls) >= 2 else 0.0


def rep_rate(text):
    toks = re.findall(r"[a-z0-9']+", text.lower())
    if len(toks) < 3:
        return 0.0
    bigrams = list(zip(toks, toks[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


def featurize(model, tok, text):
    p = text_ppl(model, tok, text)
    if p is None:
        return None
    return [math.log(p), burstiness(model, tok, text), rep_rate(text)]


class LogisticReg:
    """Tiny, dependency-free logistic regression (gradient descent on z-scored
    features) — transparent enough to read, unlike a sklearn black box."""

    def __init__(self, lr=0.05, iters=3000):
        self.lr, self.iters = lr, iters

    def fit(self, X, y):
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sd
        Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        w = np.zeros(Xs.shape[1])
        for _ in range(self.iters):
            z = Xs @ w
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xs.T @ (p - y) / len(y)
            w -= self.lr * grad
        self.w = w
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype="float64")
        Xs = (X - self.mu) / self.sd
        Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        z = Xs @ self.w
        return 1.0 / (1.0 + np.exp(-z))


def main():
    results_path, human_path = sys.argv[1], sys.argv[2]
    target_path = sys.argv[3] if len(sys.argv) > 3 else None

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    rows = [json.loads(l) for l in open(results_path, encoding="utf-8") if l.strip()]

    def group(cond):
        return [r["output"] for r in rows if r["condition"] == cond]

    ai_texts = group("baseline")
    styled_conds = sorted({r["condition"] for r in rows if r["condition"] != "baseline"})

    human_lines = [l.strip() for l in open(human_path, encoding="utf-8") if l.strip()]
    rng = random.Random(SEED)
    rng.shuffle(human_lines)
    # Chunk one-sentence-per-line human text into ~3-sentence groups so
    # per-text burstiness is comparable to the multi-sentence outputs.
    chunks = [" ".join(human_lines[i:i + 3]) for i in range(0, len(human_lines), 3)]
    human_train, human_test = chunks[:40], chunks[40:160]

    target_chunks = None
    if target_path:
        tl = [l.strip() for l in open(target_path, encoding="utf-8") if l.strip()]
        target_chunks = [" ".join(tl[i:i + 3]) for i in range(0, len(tl), 3)]

    def feats(texts):
        out = []
        for t in texts:
            f = featurize(model, tok, t)
            if f is not None:
                out.append(f)
        return out

    F = {
        "baseline(AI)": feats(ai_texts),
        **{c: feats(group(c)) for c in styled_conds},
        "human-train": feats(human_train),
        "human-test": feats(human_test),
    }
    if target_chunks is not None:
        F["human-target"] = feats(target_chunks)

    # Variance-aware transform: replace raw burstiness with its distance from the
    # human reference, so both too-low (AI-uniform) and too-high (overshoot)
    # burstiness signal AI — not the naive "more bursty = more human".
    hb = [f[1] for f in F["human-train"]]
    if not hb:
        hb = [f[1] for f in F["human-test"]]
    ref_burst = float(np.mean(hb)) if hb else 0.0

    def vad(fs):
        return [[f[0], abs(f[1] - ref_burst), f[2]] for f in fs]

    F_vad = {name: vad(fs) for name, fs in F.items()}

    X_train = F_vad["baseline(AI)"] + F_vad["human-train"]
    y_train = [1.0] * len(F_vad["baseline(AI)"]) + [0.0] * len(F_vad["human-train"])
    clf = LogisticReg().fit(X_train, y_train)

    tr = clf.predict_proba(X_train)
    train_acc = float(np.mean((tr > 0.5).astype(float) == np.array(y_train)))

    print("=== variance-aware detector (logreg: log-ppl, |burstiness-ref|, rep_rate) ===")
    print(f"ref_burstiness={ref_burst:.2f}")
    print(f"features={clf.w.tolist()}  (weights: log-ppl, burst-dist, rep_rate, bias)")
    print(f"train: AI={len(F_vad['baseline(AI)'])} human={len(F_vad['human-train'])} "
          f"train_acc={train_acc:.2f}")

    print("\n=== P(AI) by group (1.0 = AI-typical, 0.0 = human-typical) ===")
    for name, fs in F_vad.items():
        if not fs:
            print(f"{name:14s} no scored texts")
            continue
        p = clf.predict_proba(fs)
        print(f"{name:14s} n={len(p):3d}  mean={p.mean():.3f}  "
              f"min={p.min():.3f}  max={p.max():.3f}")

    b = clf.predict_proba(F_vad["baseline(AI)"]).mean()
    h = clf.predict_proba(F_vad["human-test"]).mean()
    print("\n=== detection delta ===")
    print(f"baseline {b:.3f}  human {h:.3f}")
    if b > 0:
        for c in styled_conds:
            fs = F_vad.get(c)
            if not fs:
                continue
            p = clf.predict_proba(fs).mean()
            print(f"{c:14s} {p:.3f}  shift vs baseline: {p - b:+.3f}  ({100 * (p - b) / b:+.0f}%)")
    print("\n=== feature profile (mean per group) ===")
    print(f"{'group':14s} {'log-ppl':>8s} {'burstiness':>10s} {'rep_rate':>8s}")
    for name, fs in F.items():
        if not fs:
            continue
        lp = np.mean([f[0] for f in fs])
        bu = np.mean([f[1] for f in fs])
        rr = np.mean([f[2] for f in fs])
        print(f"{name:14s} {lp:8.2f} {bu:10.2f} {rr:8.3f}")


if __name__ == "__main__":
    main()
