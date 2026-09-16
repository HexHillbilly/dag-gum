#!/usr/bin/env python3
"""Verify the EOS-guard on the REAL governor (not the replicated sweep suppression).

Confirms that ActiveVarietyGovernor with eos_guard=True eliminates EOS-death while
holding rescue, at the same strengths the isolated sweep predicted. Uses the governor's
own generate(intervene=False) as the dormant baseline (matches raw, no actuation).

Configs: guard_off@5.0 (current default), guard_on@5.0, guard_on@7.0.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast

SEED = 42
MAX_NEW = 128
LOOP_D2 = 0.5


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--fixtures", default="tests/fixtures/t2s_degenerate.jsonl,tests/fixtures/qwen_degenerate.jsonl")
    ap.add_argument("--keys", default="text,prompt")
    ap.add_argument("--n-fixture", type=int, default=20)
    ap.add_argument("--out", default="docs/gate23/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    eos = tokenizer.eos_token_id

    # dormant governor (intervene=False = true raw baseline)
    g_dormant = ActiveVarietyGovernor(model, tokenizer=tokenizer)

    # discover loopers via dormant (raw) sampling
    loopers = []
    for fpath, key in zip(args.fixtures.split(","), args.keys.split(",")):
        recs = [json.loads(l) for l in open(fpath) if l.strip()][:args.n_fixture]
        for rec in recs:
            seed_all(SEED)
            inp = tokenizer(rec[key], return_tensors="pt")["input_ids"].to(device)
            o = g_dormant.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.8, intervene=False)
            gen = o["sequences"][:, o["prompt_len"]:]
            if d2(gen) < LOOP_D2:
                loopers.append((fpath.split("/")[-1].replace(".jsonl", ""), rec, key))
    print(f"sampling loopers: {len(loopers)}", flush=True)

    # dormant baselines
    dorm = {}
    for i, (fix, rec, key) in enumerate(loopers):
        seed_all(SEED)
        inp = tokenizer(rec[key], return_tensors="pt")["input_ids"].to(device)
        o = g_dormant.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                               temperature=0.8, intervene=False)
        gen = o["sequences"][:, o["prompt_len"]:]
        dorm[i] = (d2(gen), tokens_before_loop(gen, eos))

    configs = [
        ("guard_off_s5", dict(eos_guard=False, suppression_strength=5.0)),
        ("guard_on_s5", dict(eos_guard=True, suppression_strength=5.0)),
        ("guard_on_s7", dict(eos_guard=True, suppression_strength=7.0)),
    ]

    out_base = Path(args.out) / f"{args.model.split('/')[-1].lower()}_eos_guard_verify.jsonl"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\n{'config':14s} | rescue | EOS-death | mean_dD2 | act_tb_mean", flush=True)
    for name, cfg in configs:
        g = ActiveVarietyGovernor(model, tokenizer=tokenizer, **cfg)
        n = len(loopers)
        resc = 0; eosd = 0; dds = []; tbas = []
        for i, (fix, rec, key) in enumerate(loopers):
            seed_all(SEED)
            inp = tokenizer(rec[key], return_tensors="pt")["input_ids"].to(device)
            o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                           temperature=0.8, intervene=True)
            act = o["sequences"][:, o["prompt_len"]:]
            ad = d2(act); tba = tokens_before_loop(act, eos)
            dd, tbd = dorm[i]
            delta = ad - dd
            rescued = delta > 0.01
            eos_death = ad > 0.5 and tba < tbd
            if rescued: resc += 1
            if eos_death: eosd += 1
            dds.append(delta); tbas.append(tba)
            rows.append({"model": args.model, "config": name, "fixture": fix,
                         "orig_id": rec.get("id"), "text": rec[key],
                         "dorm_d2": round(dd, 6), "act_d2": round(ad, 6),
                         "tb_dorm": int(tbd), "tb_act": int(tba),
                         "rescued": bool(rescued), "eos_death": bool(eos_death)})
        print(f"{name:14s} | {resc}/{n}  | {eosd:3d}/{n}   | {np.mean(dds):+.3f}   | {np.mean(tbas):.1f}", flush=True)

    with open(out_base, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_base}", flush=True)


if __name__ == "__main__":
    main()
