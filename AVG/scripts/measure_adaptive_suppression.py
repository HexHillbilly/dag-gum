#!/usr/bin/env python3
"""Adaptive-suppression measurement — does an EOS-aware guard eliminate EOS-death
while holding rescue, at the fixed penalty strength that otherwise forces full rescue?

Baseline (fixed) already measured by sweep_suppression.py. This harness adds three
adaptive variants, each still using a FIXED penalty strength but with a guard:

  baseline          : next_logits[tid] -= strength; temp 0.70 during suppression (current).
  eos-exclusion     : while suppression is active, mask <eos> (set -1e9) so the model
                      cannot terminate mid-rescue; it must pick a non-EOS token.
  eos-cap           : cap the penalty so the loop token stays >= MARGIN above <eos>
                      (penalty = min(strength, loop_logit - eos_logit - MARGIN));
                      never push the loop token below EOS.
  eos-exclusion-temp: eos-exclusion + raise temp to 1.0 during suppression (more diversity).

Measures rescue + EOS-death at strengths {5.0, 7.0} per variant, sampling regime.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import compute_token_distinct_2_fast, sample_top_p

SEED = 42
MAX_NEW = 128
LOOP_D2 = 0.5
COOLDOWN = 8
TEMPERATURE = 0.80
TOP_P = 0.85
MARGIN = 1.0
STRENGTHS = [5.0, 7.0]
VARIANTS = ["baseline", "eos-exclusion", "eos-cap", "eos-exclusion-temp"]


def seed_all(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def d2(ids):
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)


def tokens_before_loop(gen_ids, eos_id):
    toks = gen_ids[0]
    n = toks.numel()
    first_eos = n
    ep = (toks == eos_id).nonzero()
    if len(ep) > 0:
        first_eos = int(ep[0].item())
    loop_onset = n
    for step in range(24, n):
        window = toks[step - 23:step + 1]
        t1 = window[:-1]; t2 = window[1:]
        bg = torch.stack([t1, t2], dim=1)
        dd = float(torch.unique(bg, dim=0).shape[0]) / (len(window) - 1)
        if dd < LOOP_D2:
            loop_onset = step
            break
    return min(first_eos, loop_onset)


def raw_generate(model, tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    plen = ids.shape[-1]
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    for _ in range(MAX_NEW):
        logits = logits / max(TEMPERATURE, 1e-5)
        logits = sample_top_p(logits, top_p=TOP_P)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=-1)
        if int(nxt.item()) == eos:
            break
        with torch.no_grad():
            out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    return ids[:, plen:]


def suppressed_generate(model, tokenizer, prompt, strength, variant):
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    plen = ids.shape[-1]
    eos = tokenizer.eos_token_id
    suppress = {}
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    for _ in range(MAX_NEW):
        dist2, active_loop_ids = compute_token_distinct_2_fast(
            ids, prompt_len=plen, window_len=24
        )
        for tid in active_loop_ids:
            suppress[tid] = COOLDOWN
        for tid, steps_left in list(suppress.items()):
            if steps_left > 0:
                if variant == "eos-cap":
                    loop_logit = logits[0, tid].item()
                    eos_logit = logits[0, eos].item()
                    max_pen = loop_logit - eos_logit - MARGIN
                    pen = min(strength, max_pen) if max_pen > 0 else 0.0
                    logits[:, tid] -= pen
                else:
                    logits[:, tid] -= strength
                suppress[tid] -= 1
            else:
                del suppress[tid]
        if suppress:
            if variant in ("eos-exclusion", "eos-exclusion-temp"):
                logits[:, eos] = -1e9
            logits = sample_top_p(logits, top_p=TOP_P)
            cur_temp = 1.0 if variant == "eos-exclusion-temp" else 0.70
        else:
            cur_temp = TEMPERATURE
        logits = logits / max(cur_temp, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=-1)
        if int(nxt.item()) == eos:
            break
        with torch.no_grad():
            out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    return ids[:, plen:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--fixtures", default="tests/fixtures/t2s_degenerate.jsonl,tests/fixtures/qwen_degenerate.jsonl")
    ap.add_argument("--keys", default="text,prompt")
    ap.add_argument("--n-fixture", type=int, default=40)
    ap.add_argument("--out", default="docs/gate23/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    eos = tokenizer.eos_token_id

    loopers = []
    for fpath, key in zip(args.fixtures.split(","), args.keys.split(",")):
        recs = [json.loads(l) for l in open(fpath) if l.strip()][:args.n_fixture]
        for rec in recs:
            seed_all(SEED)
            gen = raw_generate(model, tokenizer, rec[key])
            if d2(gen) < LOOP_D2:
                loopers.append((fpath.split("/")[-1].replace(".jsonl", ""), rec, key))
    print(f"sampling loopers: {len(loopers)}", flush=True)

    dorm = {}
    for i, (fix, rec, key) in enumerate(loopers):
        seed_all(SEED)
        g = raw_generate(model, tokenizer, rec[key])
        dorm[i] = (d2(g), tokens_before_loop(g, eos))

    out_base = Path(args.out) / f"{args.model.split('/')[-1].lower()}_adaptive_suppression.jsonl"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\n{'variant':20s} {'strength':>8s} | rescue | EOS-death | mean_dD2 | act_tb_mean", flush=True)
    for variant in VARIANTS:
        for s in STRENGTHS:
            n = len(loopers)
            resc = 0; eosd = 0; dds = []; tbas = []
            for i, (fix, rec, key) in enumerate(loopers):
                seed_all(SEED)
                g = suppressed_generate(model, tokenizer, rec[key], strength=s, variant=variant)
                ad = d2(g); tba = tokens_before_loop(g, eos)
                dd, tbd = dorm[i]
                delta = ad - dd
                rescued = delta > 0.01
                eos_death = ad > 0.5 and tba < tbd
                if rescued: resc += 1
                if eos_death: eosd += 1
                dds.append(delta); tbas.append(tba)
                rows.append({"model": args.model, "variant": variant, "strength": s,
                             "fixture": fix, "orig_id": rec.get("id"), "text": rec[key],
                             "dorm_d2": round(dd, 6), "act_d2": round(ad, 6),
                             "tb_dorm": int(tbd), "tb_act": int(tba),
                             "rescued": bool(rescued), "eos_death": bool(eos_death)})
            print(f"{variant:20s} {s:8.1f} | {resc}/{n}  | {eosd:3d}/{n}   | "
                  f"{np.mean(dds):+.3f}   | {np.mean(tbas):.1f}", flush=True)

    with open(out_base, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_base}", flush=True)


if __name__ == "__main__":
    main()
