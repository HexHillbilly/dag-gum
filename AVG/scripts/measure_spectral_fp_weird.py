#!/usr/bin/env python3
"""Spectral detector false-positive rate on weird-but-benign inputs (MEASUREMENT ONLY).

The existing DS-034c-hazard FP gate was VACUOUS: 445/450 structured inputs made the model emit
<eos> at step 0 (complete blocks), so the detector had no generated tokens to evaluate. This
probe fixes that by TRUNCATING each input to an open prefix (so the model actually continues),
then measuring whether the spectral / bigram detector fires on the benign continuation.

Classification (per record):
  - dormant_degenerate = dormant (no-governor) continuation loops (tail d2 < 0.5).
  - FP = predicate fired AND NOT dormant_degenerate  (fired on text that was NOT degenerating).
  - TP = predicate fired AND dormant_degenerate      (fired on text that WAS degenerating).

MEASUREMENT ONLY. No controller edits. Greedy decoding (deterministic, matches DS-034c-hazard).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast

SEED = 42
MODEL = "Qwen/Qwen2.5-1.5B"
MAX_NEW = 64
LOOP_D2 = 0.5

CORPORA: List[Tuple[str, str]] = [
    ("code_torture", "tests/fixtures/code_torture.jsonl"),
    ("schema_corpus_v2", "tests/fixtures/schema_corpus_v2.jsonl"),
    ("schema_markdown_fenced", "tests/fixtures/schema_markdown_fenced.jsonl"),
    ("schema_urls_strings", "tests/fixtures/schema_urls_strings.jsonl"),
    ("prose_code_switch", "tests/fixtures/prose_code_switch.jsonl"),
]


def seed_all(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tail_d2(ids: torch.Tensor) -> float:
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)


def truncate_prefix(text: str, ratio: float) -> str:
    """Cut `text` to the first `ratio` fraction at a whitespace boundary (open prefix)."""
    cut = int(len(text) * ratio)
    if cut <= 0 or cut >= len(text):
        return text
    while cut < len(text) and not text[cut].isspace():
        cut += 1
    return text[:cut].rstrip()


@torch.no_grad()
def run_record(governor, tokenizer, prompt: str, device) -> Dict[str, Any]:
    seed_all(SEED)
    inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    o_d = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=False, intervene=False)
    dorm = o_d["sequences"][:, o_d["prompt_len"]:]

    seed_all(SEED)
    o_a = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=False, intervene=True)
    act = o_a["sequences"][:, o_a["prompt_len"]:]
    col = o_a["collapse"]

    dd = tail_d2(dorm)
    return {
        "dormant_d2": dd,
        "active_d2": tail_d2(act),
        "dormant_degenerate": bool(dd < LOOP_D2),
        "bigram_fires": int(col["bigram_fires"]),
        "spectral_fires": int(col["spectral_fires"]),
        "n_gen": int(dorm.shape[-1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5, help="prefix truncation ratio")
    ap.add_argument("--n-per", type=int, default=0, help="cap records per corpus (0 = all)")
    ap.add_argument("--out", default="docs/gate23/spectral_fp_weird.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer)

    rows: List[Dict[str, Any]] = []
    for corpus, path in CORPORA:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        if args.n_per:
            recs = recs[: args.n_per]
        print(f"=== {corpus}: {len(recs)} records ===", flush=True)
        for i, rec in enumerate(recs):
            prompt = truncate_prefix(rec["text"], args.ratio)
            r = run_record(governor, tokenizer, prompt, device)
            r["corpus"] = corpus
            r["id"] = int(rec.get("id", i))
            rows.append(r)
        # per-corpus summary
        fired = [r for r in rows if r["corpus"] == corpus and (r["bigram_fires"] or r["spectral_fires"])]
        fps = [r for r in fired if not r["dormant_degenerate"]]
        print(f"  {corpus}: fired {len(fired)}/{len(recs)}, spectral-fired "
              f"{sum(1 for r in rows if r['corpus']==corpus and r['spectral_fires']>0)}, "
              f"FP {len(fps)} (fired & not-degenerate)", flush=True)

    n = len(rows)
    fired = [r for r in rows if (r["bigram_fires"] or r["spectral_fires"])]
    spec_fired = [r for r in rows if r["spectral_fires"] > 0]
    bg_fired = [r for r in rows if r["bigram_fires"] > 0]
    fp = [r for r in fired if not r["dormant_degenerate"]]
    tp = [r for r in fired if r["dormant_degenerate"]]
    degen = [r for r in rows if r["dormant_degenerate"]]

    print(f"\n=== SUMMARY (N={n}, ratio={args.ratio}) ===", flush=True)
    print(f"dormant-degenerate records: {len(degen)}/{n} ({100*len(degen)/n:.1f}%)", flush=True)
    print(f"spectral-fired: {len(spec_fired)}/{n} | bigram-fired: {len(bg_fired)}/{n}", flush=True)
    print(f"FALSE POSITIVES (fired & NOT degenerate): {len(fp)}/{n} "
          f"({100*len(fp)/n:.2f}%)", flush=True)
    print(f"  spectral FP: {sum(1 for r in fp if r['spectral_fires']>0)} | "
          f"bigram FP: {sum(1 for r in fp if r['bigram_fires']>0)}", flush=True)
    print(f"TRUE POSITIVES (fired & degenerate): {len(tp)}/{n}", flush=True)
    for r in fp:
        print(f"  FP: {r['corpus']} id={r['id']} dormant_d2={r['dormant_d2']:.3f} "
              f"spectral={r['spectral_fires']} bigram={r['bigram_fires']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[wrote] {out}", flush=True)


if __name__ == "__main__":
    main()
