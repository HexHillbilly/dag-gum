#!/usr/bin/env python3
"""Probe the 1.5B EOS-death anomaly at the logit level.

The release sweep showed guard_off EOS-death of 8% (0.5B) / 47% (1.5B) / 25% (3B),
with first_<eos> at positions 0-6 on 1.5B. This probe answers WHY by looking at the
first-token logits (the cold-start step where suppression -5.0 + kickstart -1e4 fire
on the trailing-24 PROMPT tokens of a degenerate word-loop).

For each model and each discovered looper, it records the <eos> RANK and softmax
PROBABILITY at three stages:
  raw        - no penalties
  +supp      - suppression -5.0 on repeated prompt-tail tokens
  +kick      - + kickstart -1e4 on those same tokens
"""
import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import compute_token_distinct_2_fast

SUPP = 5.0
KICK = 1e4


def unique_loopers(jsonl):
    rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    seen, out = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r["text"])
    return out


def rank_of(logits, tid):
    order = torch.argsort(logits[0], descending=True)
    return int((order == tid).nonzero()[0]) + 1


def prob_of(logits, tid):
    p = torch.softmax(logits, dim=-1)[0]
    return float(p[tid])


def main():
    models = [
        ("0.5B", "Qwen/Qwen2.5-0.5B", "<repo-root>/AVG/docs/gate23/qwen2.5-0.5b_eos_release_sweep.jsonl"),
        ("1.5B", "Qwen/Qwen2.5-1.5B", "<repo-root>/AVG/docs/gate23/qwen2.5-1.5b_eos_release_sweep.jsonl"),
        ("3B", "Qwen/Qwen2.5-3B", "<repo-root>/AVG/docs/gate23/qwen2.5-3b_eos_release_sweep.jsonl"),
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, mname, jpath in models:
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForCausalLM.from_pretrained(
            mname, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
        eos = tokenizer.eos_token_id
        loopers = unique_loopers(jpath)

        raw_r, supp_r, kick_r = [], [], []
        raw_p, supp_p, kick_p = [], [], []
        n_raw, n_supp, n_kick = 0, 0, 0

        for t in loopers:
            ids = tokenizer(t, return_tensors="pt")["input_ids"].to(device)
            plen = ids.shape[-1]
            with torch.no_grad():
                out = model(ids, use_cache=True)
            logits = out.logits[:, -1, :].clone()
            _, active = compute_token_distinct_2_fast(ids, prompt_len=plen, window_len=24)

            rr = rank_of(logits, eos)
            raw_r.append(rr); raw_p.append(prob_of(logits, eos))
            if rr == 1:
                n_raw += 1

            ls = logits.clone()
            for tid in active:
                ls[:, tid] -= SUPP
            sr = rank_of(ls, eos)
            supp_r.append(sr); supp_p.append(prob_of(ls, eos))
            if sr == 1:
                n_supp += 1

            lk = ls.clone()
            for tid in active:
                lk[:, tid] -= KICK
            kr = rank_of(lk, eos)
            kick_r.append(kr); kick_p.append(prob_of(lk, eos))
            if kr == 1:
                n_kick += 1

        print(f"\n=== {name} (eos_id={eos}, {len(loopers)} loopers) ===")
        print(f"  <eos> rank   raw mean={statistics.mean(raw_r):.1f} med={statistics.median(raw_r):.0f} "
              f"| +supp mean={statistics.mean(supp_r):.1f} med={statistics.median(supp_r):.0f} "
              f"| +kick mean={statistics.mean(kick_r):.1f} med={statistics.median(kick_r):.0f}")
        print(f"  <eos> P      raw mean={statistics.mean(raw_p):.4f} med={statistics.median(raw_p):.4f} "
              f"| +supp mean={statistics.mean(supp_p):.4f} med={statistics.median(supp_p):.4f} "
              f"| +kick mean={statistics.mean(kick_p):.4f} med={statistics.median(kick_p):.4f}")
        print(f"  <eos> argmax raw {n_raw}/{len(loopers)} | +supp {n_supp}/{len(loopers)} | +kick {n_kick}/{len(loopers)}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
