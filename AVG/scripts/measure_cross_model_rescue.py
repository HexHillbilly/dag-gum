#!/usr/bin/env python3
"""Cross-model rescue measurement (reusable) — Qwen2.5-0.5B scale test.

Measures, for a given model + fixture set (greedy AND sampling):
  1. rescue rate  = active d2 > dormant d2 (delta > 0.01)
  2. tokens-before-degeneration (loop onset OR EOS) for raw vs active  <-- NEW
  3. per-record fire counts (mechanism attribution)
  4. spectral-PR collapse floor (degenerate) vs healthy-prose baseline  <-- invariant check

Loop onset = first step >= 24 where trailing-24 distinct-2 < 0.5, else EOS, else max_new.
"""
import sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast
from AVG.core.metrics import participation_ratio

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
    """Step index of loop onset (trailing-24 d2 < LOOP_D2) or EOS, whichever first."""
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

def raw_generate(model, tokenizer, prompt, do_sample, temp=0.8, top_p=0.85, max_new=MAX_NEW):
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    plen = ids.shape[-1]
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    for _ in range(max_new):
        if do_sample:
            logits = logits / max(temp, 1e-5)
            sl, si = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
            rm = cum > top_p; rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = 0
            idx_rm = rm.scatter(1, si, rm)
            logits = logits.masked_fill(idx_rm, -1e4)
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
    ap.add_argument("--keys", default="text,prompt")  # text field per fixture
    ap.add_argument("--prose", default="data/t2s_bench/valid_subset_200.jsonl")
    ap.add_argument("--n-fixture", type=int, default=40)
    ap.add_argument("--out", default="docs/gate23/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    g = ActiveVarietyGovernor(model, tokenizer=tokenizer)
    eos = tokenizer.eos_token_id

    # --- discovery: which fixture records loop under raw greedy ---
    loopers = []
    screened = 0
    for fpath, key in zip(args.fixtures.split(","), args.keys.split(",")):
        recs = [json.loads(l) for l in open(fpath) if l.strip()][:args.n_fixture]
        for rec in recs:
            screened += 1
            seed_all(SEED)
            gen = raw_generate(model, tokenizer, rec[key], do_sample=False)
            if d2(gen) < LOOP_D2:
                loopers.append((fpath.split("/")[-1].replace(".jsonl", ""), rec, key))
    print(f"loopers: {len(loopers)}/{screened}", flush=True)

    # --- formal: rescue + tokens-before-loop ---
    out_base = Path(args.out) / f"{args.model.split('/')[-1].lower()}_rescue_results.jsonl"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    res = open(out_base, "w")
    summ = {}
    print(f"{'id':>3s} {'g_dorm':>6s} {'g_act':>6s} {'g_resc':>6s} {'g_tb_d':>6s} {'g_tb_a':>6s} "
          f"{'s_dorm':>6s} {'s_act':>6s} {'s_resc':>6s} {'s_tb_d':>6s} {'s_tb_a':>6s}", flush=True)
    for rid, (fix, rec, key) in enumerate(loopers):
        text = rec[key]
        row = {"id": rid, "fixture": fix, "orig_id": rec.get("id"), "text": text}
        for regime, do_sample in (("greedy", False), ("sampling", True)):
            seed_all(SEED)
            dorm = raw_generate(model, tokenizer, text, do_sample=do_sample)
            dd = d2(dorm); tbd = tokens_before_loop(dorm, eos)
            seed_all(SEED)
            inp = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=do_sample,
                           temperature=0.8, top_p=0.85, intervene=True)
            act = o["sequences"][:, o["prompt_len"]:]
            ad = d2(act); tba = tokens_before_loop(act, eos)
            col = o["collapse"]
            delta = ad - dd; rescued = delta > 0.01
            s = summ.setdefault(regime, {"n": 0, "rescue": 0, "tb_dorm": [], "tb_act": []})
            s["n"] += 1
            if rescued: s["rescue"] += 1
            s["tb_dorm"].append(tbd); s["tb_act"].append(tba)
            row[f"{regime}_dorm_d2"] = round(dd, 6)
            row[f"{regime}_act_d2"] = round(ad, 6)
            row[f"{regime}_rescue"] = bool(rescued)
            row[f"{regime}_tb_dorm"] = int(tbd)
            row[f"{regime}_tb_act"] = int(tba)
            row["bigram_fires"] = col["bigram_fires"]
            row["spectral_fires"] = col["spectral_fires"]
        res.write(json.dumps(row) + "\n"); res.flush()
        print(f"{rid:3d} {row['greedy_dorm_d2']:6.3f} {row['greedy_act_d2']:6.3f} {str(row['greedy_rescue']):>6s} "
              f"{row['greedy_tb_dorm']:6d} {row['greedy_tb_act']:6d} "
              f"{row['sampling_dorm_d2']:6.3f} {row['sampling_act_d2']:6.3f} {str(row['sampling_rescue']):>6s} "
              f"{row['sampling_tb_dorm']:6d} {row['sampling_tb_act']:6d}", flush=True)
    res.close()

    print("\n=== SUMMARY ===", flush=True)
    for regime in ("greedy", "sampling"):
        s = summ[regime]
        td = np.array(s["tb_dorm"]); ta = np.array(s["tb_act"])
        print(f"{regime}: rescue {s['rescue']}/{s['n']} | "
              f"tokens-before-degen RAW mean={td.mean():.1f} med={np.median(td):.0f} | "
              f"ACTIVE mean={ta.mean():.1f} med={np.median(ta):.0f}", flush=True)

    # --- spectral PR collapse floor (degenerate) vs prose baseline ---
    print("\n=== spectral PR: degenerate vs prose ===", flush=True)
    deg_min = []
    pr_buffer = []
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:
            pr_buffer.append(hs[:, -1:, :].detach().to(torch.float32))
        return out
    handle = g._get_layer_blocks()[2].register_forward_hook(hook)
    for fix, rec, key in loopers[:12]:
        seed_all(SEED)
        enc = tokenizer(rec[key], return_tensors="pt"); ids = enc["input_ids"].to(device)
        pr_buffer.clear()
        with torch.no_grad(): out = model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
        prs = []
        for _ in range(MAX_NEW):
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            if int(nxt.item()) == eos: break
            if len(pr_buffer) >= 2:
                buf = pr_buffer[-24:]
                if len(buf) >= 4:
                    prs.append(float(participation_ratio(torch.cat(buf, dim=1)).item()))
            with torch.no_grad(): out = model(nxt, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :].clone()
        if prs: deg_min.append(min(prs))
    prose_recs = [json.loads(l) for l in open(args.prose) if l.strip()]
    step = max(1, len(prose_recs) // 12)
    prose_mean = []
    for rec in prose_recs[::step][:12]:
        seed_all(SEED)
        enc = tokenizer(rec["text"][:600], return_tensors="pt"); ids = enc["input_ids"].to(device)
        pr_buffer.clear()
        with torch.no_grad(): out = model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
        prs = []
        for _ in range(MAX_NEW):
            logits2 = logits / 0.8
            probs = torch.softmax(logits2, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if int(nxt.item()) == eos: break
            if len(pr_buffer) >= 2:
                buf = pr_buffer[-24:]
                if len(buf) >= 4:
                    prs.append(float(participation_ratio(torch.cat(buf, dim=1)).item()))
            with torch.no_grad(): out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].clone()
        if prs: prose_mean.append(sum(prs) / len(prs))
    handle.remove()
    if deg_min and prose_mean:
        d = np.array(deg_min); p = np.array(prose_mean)
        p90 = float(np.percentile(d, 90)); p10 = float(np.percentile(p, 10))
        T = (p90 + p10) / 2
        print(f"  degenerate minPR p90={p90:.2f} (range {d.min():.2f}-{d.max():.2f})", flush=True)
        print(f"  prose meanPR p10={p10:.2f} (range {p.min():.2f}-{p.max():.2f})", flush=True)
        print(f"  -> T={T:.4f} band_low={0.75*T:.4f}  [Qwen1.5B band_low=8.22, GPT2 ~7.04]", flush=True)

if __name__ == "__main__":
    main()
