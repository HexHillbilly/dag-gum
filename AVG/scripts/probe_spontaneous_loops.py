#!/usr/bin/env python3
"""Spontaneous loop formation probe (MEASUREMENT ONLY).

Question (reframed from the PR-floor root-cause): on NON-looping prompts — natural prose,
code, schema — does a model SPONTANEOUSLY form a short loop (<=8 tokens), or only long-range
phrase repetition? What sets the period of whatever repetition forms?

For each prompt, generate greedily for a long horizon and measure:
  (1) min trailing-24 distinct-token count over the generation (short-loop detector: a short
      loop of k tokens has distinct ~k; <9 means a short loop formed);
  (2) the strongest period p of the trailing 256 tokens via self-agreement at lag p
      (long-loop / phrase-repetition detector: agreement(p) = fraction of positions i where
      seq[i]==seq[i-p]).

Report the loop spectrum: do short periods form spontaneously, or is spontaneous repetition
always long-period (phrase-level)?

MEASUREMENT ONLY: no controller, no threshold, no actuation. Writes a jsonl.
"""
from __future__ import annotations

import argparse
import json

import torch


def load_fixtures(path, field, n):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recs.append(rec[field] if field else (rec.get("text") or rec.get("prompt")))
            if len(recs) >= n:
                break
    return recs


def min_trailing_distinct(gen_ids, window=24):
    """Min distinct-token count over any trailing `window` slice of the generation."""
    best = None
    for i in range(window, len(gen_ids) + 1):
        d = len(set(gen_ids[i - window : i]))
        if best is None or d < best:
            best = d
    return best


def strongest_period(gen_ids, max_p=64, tail=256, min_agreement=0.30):
    """Find the strongest period p in the trailing `tail` tokens by self-agreement.

    Returns (period, agreement) of the best lag, or (None, None) if no lag clears
    min_agreement. Period 0 = no periodic structure detected.
    """
    seq = gen_ids[-tail:] if len(gen_ids) > tail else gen_ids
    L = len(seq)
    if L < 8:
        return None, None
    best_p, best_a = None, 0.0
    for p in range(2, min(max_p, L // 2) + 1):
        hits = sum(1 for i in range(p, L) if seq[i] == seq[i - p])
        a = hits / (L - p)
        if a > best_a:
            best_p, best_a = p, a
    if best_a >= min_agreement:
        return best_p, round(best_a, 4)
    return None, round(best_a, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--revision", default="8faed761d45a263340a0528343f099c05c9a4323")
    ap.add_argument("--n-per", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default="docs/gate23/spontaneous_loops.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpora = [
        ("prose", "data/t2s_bench/valid_subset_200.jsonl", "text"),
        ("code", "tests/fixtures/code_torture.jsonl", "text"),
        ("schema", "tests/fixtures/schema_corpus_v2.jsonl", "text"),
    ]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    out_recs = []
    for corpus, path, field in corpora:
        texts = load_fixtures(path, field, args.n_per)
        print(f"=== {corpus}: {len(texts)} prompts ===", flush=True)
        for ti, text in enumerate(texts):
            inp = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inp, use_cache=True)
            past = out.past_key_values
            logits = out.logits
            gen_ids = []
            with torch.no_grad():
                for _ in range(args.max_new):
                    nxt = torch.argmax(logits[:, -1, :], dim=-1).item()
                    gen_ids.append(nxt)
                    nxt_inp = torch.tensor([[nxt]], device=model.device)
                    out = model(input_ids=nxt_inp, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits
            mt = min_trailing_distinct(gen_ids)
            period, agreement = strongest_period(gen_ids)
            rec = {
                "corpus": corpus,
                "id": ti,
                "n_gen": len(gen_ids),
                "min_trailing24_distinct": mt,
                "period": period,
                "agreement": agreement,
                "tail_tokens": [tok.decode(i) for i in gen_ids[-24:]],
            }
            out_recs.append(rec)
            print(
                f"  [{ti}] min_distinct={mt} period={period} agreement={agreement} "
                f"tail={rec['tail_tokens'][:10]}",
                flush=True,
            )

    with open(args.out, "w") as f:
        for r in out_recs:
            f.write(json.dumps(r) + "\n")

    # aggregate
    import statistics
    short = [r for r in out_recs if r["min_trailing24_distinct"] < 9]
    looped = [r for r in out_recs if r["period"] is not None]
    print(f"\n=== SUMMARY (n={len(out_recs)}) ===", flush=True)
    print(f"short-loop rate (min distinct<9): {len(short)}/{len(out_recs)} ({len(short)/len(out_recs):.1%})", flush=True)
    print(f"periodic (agreement>=0.30): {len(looped)}/{len(out_recs)} ({len(looped)/len(out_recs):.1%})", flush=True)
    if looped:
        ps = [r["period"] for r in looped]
        print(f"spontaneous loop periods: mean {statistics.mean(ps):.1f} median {statistics.median(ps):.0f} "
              f"range [{min(ps)}, {max(ps)}]", flush=True)
        from collections import Counter
        print("period histogram:", dict(sorted(Counter(ps).items())), flush=True)
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
