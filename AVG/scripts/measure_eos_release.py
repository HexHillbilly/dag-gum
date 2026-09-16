#!/usr/bin/env python3
"""Bounded EOS-release sweep on the REAL governor.

Measures the effect of `eos_release_after` (unmask <eos> after N consecutive
loop-free steps) on the degenerate-prompt run-length tradeoff. With the guard
on and no release rule, degenerate prompts run to max_new (healthy prose, no
<eos>) — this sweep finds the N that lets natural termination return WITHOUT
reintroducing EOS-death or losing rescue.

Conventions match scripts/verify_eos_guard.py exactly:
  rescue   = active d2 > dormant d2 + 0.01
  eos_death = active d2 > 0.5 AND active tokens_before_loop < dormant tb
  tokens_before_loop = min(first_<eos> position, loop-onset position)

New metrics for the release effect:
  first_eos  = index of first <eos> in the generated sequence (max_new = never)
  no_eos     = True if the sequence contains no <eos> at all

Configs: guard_off (reference), release_off (current default = no release rule),
release_N for N in {1,2,3,4,6,8,12}.
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
RELEASE_NS = [1, 2, 3, 4, 6, 8, 12]


def seed_all(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def d2(ids):
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)


def first_eos_idx(gen_ids, eos_id):
    toks = gen_ids[0]
    ep = (toks == eos_id).nonzero()
    return int(ep[0].item()) if len(ep) > 0 else toks.numel()


def tokens_before_loop(gen_ids, eos_id):
    toks = gen_ids[0]
    n = toks.numel()
    first_eos = first_eos_idx(gen_ids, eos_id)
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
    ap.add_argument("--release-ns", default=",".join(map(str, RELEASE_NS)))
    ap.add_argument("--out", default="docs/gate23/")
    args = ap.parse_args()
    release_ns = [int(x) for x in args.release_ns.split(",") if x.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    eos = tokenizer.eos_token_id

    g_dormant = ActiveVarietyGovernor(model, tokenizer=tokenizer)

    # discover sampling loopers via dormant (raw) generation
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

    configs = [("guard_off", dict(eos_guard=False, suppression_strength=5.0)),
               ("release_off", dict(eos_guard=True, suppression_strength=5.0, eos_release_after=0))]
    for n in release_ns:
        configs.append((f"release_{n}", dict(eos_guard=True, suppression_strength=5.0, eos_release_after=n)))

    out_base = Path(args.out) / f"{args.model.split('/')[-1].lower()}_eos_release_sweep.jsonl"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\n{'config':12s} | rescue  | EOS-death | mean_dD2 | act_tb  | 1st_eos | no_eos", flush=True)
    for name, cfg in configs:
        g = ActiveVarietyGovernor(model, tokenizer=tokenizer, **cfg)
        n = len(loopers)
        resc = 0; eosd = 0; dds = []; tbas = []; fe = []; noe = 0
        for i, (fix, rec, key) in enumerate(loopers):
            seed_all(SEED)
            inp = tokenizer(rec[key], return_tensors="pt")["input_ids"].to(device)
            o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                           temperature=0.8, intervene=True)
            act = o["sequences"][:, o["prompt_len"]:]
            ad = d2(act); tba = tokens_before_loop(act, eos); fei = first_eos_idx(act, eos)
            dd, tbd = dorm[i]
            delta = ad - dd
            rescued = delta > 0.01
            eos_death = ad > 0.5 and tba < tbd
            if rescued: resc += 1
            if eos_death: eosd += 1
            if fei >= MAX_NEW: noe += 1
            dds.append(delta); tbas.append(tba); fe.append(fei)
            rows.append({"model": args.model, "config": name, "fixture": fix,
                         "orig_id": rec.get("id"), "text": rec[key],
                         "dorm_d2": round(dd, 6), "act_d2": round(ad, 6),
                         "tb_dorm": int(tbd), "tb_act": int(tba),
                         "first_eos": int(fei),
                         "rescued": bool(rescued), "eos_death": bool(eos_death)})
        print(f"{name:12s} | {resc}/{n}   | {eosd:3d}/{n}   | {np.mean(dds):+.3f}   | "
              f"{np.mean(tbas):6.1f} | {np.mean(fe):6.1f} | {noe}/{n}", flush=True)

    with open(out_base, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_base}", flush=True)


if __name__ == "__main__":
    main()
