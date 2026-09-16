#!/usr/bin/env python3
"""Suppression-valve sweep — measure rescue + EOS-death vs suppression strength.

Isolates the SUPPRESSION valve only (replicates controller.py's suppression path
faithfully: repeated-token penalty with cooldown, temp drop to 0.70 and top_p during
suppression). Kickstart + spectral detection are EXCLUDED so the curve is pure suppression.

Dormant baseline + discovery use a PURE raw sampler (temp 0.8, top_p 0.85 always) — the same
convention as measure_cross_model_rescue.py. For each looper (raw sampling d2 < 0.5) and each
strength in the sweep, records rescue + EOS-death; outputs a per-strength table + jsonl.
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


def raw_generate(model, tokenizer, prompt, do_sample):
    """Pure raw sampler (temp 0.8, top_p 0.85 always) — no governor, no suppression."""
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    plen = ids.shape[-1]
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    for _ in range(MAX_NEW):
        if do_sample:
            logits = logits / max(TEMPERATURE, 1e-5)
            logits = sample_top_p(logits, top_p=TOP_P)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=-1)
        if int(nxt.item()) == eos:
            break
        with torch.no_grad():
            out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    return ids[:, plen:]


def suppressed_generate(model, tokenizer, prompt, strength, do_sample):
    """Replicate controller suppression path at a given penalty strength (>0)."""
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
                logits[:, tid] -= strength
                suppress[tid] -= 1
            else:
                del suppress[tid]
        if suppress:
            logits = sample_top_p(logits, top_p=TOP_P)
            cur_temp = 0.70
        else:
            cur_temp = TEMPERATURE
        logits = logits / max(cur_temp, 1e-5)
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
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
    ap.add_argument("--strengths", default="1.0,2.0,3.0,4.0,5.0,7.0,10.0,15.0")
    ap.add_argument("--out", default="docs/gate23/")
    args = ap.parse_args()
    strengths = [float(x) for x in args.strengths.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    eos = tokenizer.eos_token_id

    # discovery: sampling loopers (raw sampling d2 < 0.5)
    loopers = []
    for fpath, key in zip(args.fixtures.split(","), args.keys.split(",")):
        recs = [json.loads(l) for l in open(fpath) if l.strip()][:args.n_fixture]
        for rec in recs:
            seed_all(SEED)
            gen = raw_generate(model, tokenizer, rec[key], do_sample=True)
            if d2(gen) < LOOP_D2:
                loopers.append((fpath.split("/")[-1].replace(".jsonl", ""), rec, key))
    print(f"sampling loopers: {len(loopers)}", flush=True)

    out_base = Path(args.out) / f"{args.model.split('/')[-1].lower()}_suppression_sweep.jsonl"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    dorm = {}
    for i, (fix, rec, key) in enumerate(loopers):
        seed_all(SEED)
        g = raw_generate(model, tokenizer, rec[key], do_sample=True)
        dorm[i] = (d2(g), tokens_before_loop(g, eos))

    print(f"strength | rescue | EOS-death | mean_dD2 | act_tb_mean", flush=True)
    for s in strengths:
        n = len(loopers)
        resc = 0; eosd = 0; dds = []; tbas = []
        for i, (fix, rec, key) in enumerate(loopers):
            seed_all(SEED)
            g = suppressed_generate(model, tokenizer, rec[key], strength=s, do_sample=True)
            ad = d2(g); tba = tokens_before_loop(g, eos)
            dd, tbd = dorm[i]
            delta = ad - dd
            rescued = delta > 0.01
            if rescued:
                resc += 1
            eos_death = ad > 0.5 and tba < tbd
            if eos_death:
                eosd += 1
            dds.append(delta); tbas.append(tba)
            rows.append({"model": args.model, "strength": s, "fixture": fix,
                         "orig_id": rec.get("id"), "text": rec[key],
                         "dorm_d2": round(dd, 6), "act_d2": round(ad, 6),
                         "tb_dorm": int(tbd), "tb_act": int(tba),
                         "rescued": bool(rescued), "eos_death": bool(eos_death)})
        print(f"{s:8.1f} | {resc}/{n}  | {eosd:3d}/{n}   | {np.mean(dds):+.3f}   | {np.mean(tbas):.1f}", flush=True)

    with open(out_base, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_base}", flush=True)


if __name__ == "__main__":
    main()
