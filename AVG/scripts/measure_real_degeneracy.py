#!/usr/bin/env python3
"""Real-generation degeneracy validation.

Every rescue so far was measured on synthetic word-loops. This probe asks the
portfolio question: does the governor help NATURAL degeneracy — the gradual
repetition/rambling a small model drifts into on LONG continuations of real prose?

Method: for real prose prompts (data/t2s_bench/valid_subset_200.jsonl), generate a
LONG continuation (default 256 tokens, sampling) both RAW (intervene=False) and
GOVERNED (intervene=True). Measure:
  - tail distinct-2 (trailing 24 tokens) — is the model currently looping?
  - full distinct-2 (whole 256-token continuation)
  - tokens-before-degeneration (first step where trailing-24 d2 < 0.5)
  - collapse fires (bigram + spectral) — is the governor actually engaging?

Reports the natural degeneration rate and the rescue rate on the degenerate subset.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast

SEED = 42
LOOP_D2 = 0.5


def seed_all(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distinct_n(ids, n):
    """distinct-n over the full generated sequence (0 if too short)."""
    toks = ids[0]
    m = toks.numel()
    if m < n:
        return 0.0
    grams = [tuple(toks[i:i + n].tolist()) for i in range(m - n + 1)]
    return len(set(grams)) / float(len(grams))


def tail_d2(ids):
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)


def tokens_before_loop(gen_ids):
    toks = gen_ids[0]
    n = toks.numel()
    for step in range(24, n):
        window = toks[step - 23:step + 1]
        t1 = window[:-1]; t2 = window[1:]
        bg = torch.stack([t1, t2], dim=1)
        dd = float(torch.unique(bg, dim=0).shape[0]) / (len(window) - 1)
        if dd < LOOP_D2:
            return step
    return n


def perplexity(model, tokenizer, prompt_text, cont_ids, device):
    """Teacher-forced perplexity of the continuation given the prompt.

    Answers "is the governed text as natural (likely) as the raw text?" — if the
    suppression increases diversity at the cost of naturalness, governed ppl >> raw ppl.
    """
    enc = tokenizer(prompt_text, return_tensors="pt")
    p_ids = enc["input_ids"].to(device)
    cont = cont_ids.to(device).long()
    full = torch.cat([p_ids, cont], dim=-1)
    P = p_ids.shape[-1]
    with torch.no_grad():
        out = model(full)
    logits = out.logits[:, P - 1:-1, :]  # logits[t] predicts token[t+1]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), cont.reshape(-1))
    return float(torch.exp(nll).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prose", default="data/t2s_bench/valid_subset_200.jsonl")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--band-low", type=float, default=None,
                    help="Override the governor's band_low (per-model valve).")
    ap.add_argument("--greedy", action="store_true",
                    help="Greedy decoding (default: sampling).")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature (ignored with --greedy).")
    ap.add_argument("--top-p", type=float, default=None,
                    help="Optional nucleus top-p (sampling only).")
    ap.add_argument("--out", default="docs/gate23/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    governor = ActiveVarietyGovernor(
        model, tokenizer=tokenizer,
        **({"band_low": args.band_low} if args.band_low is not None else {}),
    )

    prose = [json.loads(l) for l in open(args.prose) if l.strip()][:args.n_prompts]

    rows = []
    regime_name = "greedy" if args.greedy else f"sampling (temp {args.temperature})"
    print(f"=== real-generation validation: {args.model}, {len(prose)} prompts, "
          f"max_new={args.max_new}, {regime_name} ===", flush=True)
    gen = dict(max_new_tokens=args.max_new, do_sample=not args.greedy,
               temperature=args.temperature)
    if args.top_p is not None:
        gen["top_p"] = args.top_p
    for i, rec in enumerate(prose):
        text = rec["text"]
        # RAW (dormant)
        seed_all(SEED)
        inp = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        o = governor.generate(inp, intervene=False, **gen)
        raw = o["sequences"][:, o["prompt_len"]:]
        # GOVERNED (active)
        seed_all(SEED)
        o = governor.generate(inp, intervene=True, **gen)
        act = o["sequences"][:, o["prompt_len"]:]

        rt = tail_d2(raw); at = tail_d2(act)
        rf = distinct_n(raw, 2); af = distinct_n(act, 2)
        r3 = distinct_n(raw, 3); a3 = distinct_n(act, 3)
        tbd_r = tokens_before_loop(raw); tbd_a = tokens_before_loop(act)
        col = o["collapse"]
        bigram_fires = int(col["bigram_fires"])
        spectral_fires = int(col["spectral_fires"])
        fires = bigram_fires + spectral_fires
        degenerate = rt < LOOP_D2
        rescued = at > rt + 0.01
        rp = perplexity(model, tokenizer, text, raw, device)
        ap = perplexity(model, tokenizer, text, act, device)

        rows.append({"id": i, "text": text, "raw_tail_d2": round(rt, 4),
                     "act_tail_d2": round(at, 4), "raw_full_d2": round(rf, 4),
                     "act_full_d2": round(af, 4), "raw_d3": round(r3, 4),
                     "act_d3": round(a3, 4), "raw_tb": int(tbd_r), "act_tb": int(tbd_a),
                     "raw_ppl": round(rp, 2), "act_ppl": round(ap, 2),
                     "degenerate": bool(degenerate), "rescued": bool(rescued),
                     "fires": int(fires), "bigram_fires": bigram_fires,
                     "spectral_fires": spectral_fires})
        print(f"  [{i:02d}] raw_tail={rt:.3f} act_tail={at:.3f} "
              f"raw_full={rf:.3f} act_full={af:.3f} raw_tb={tbd_r} act_tb={tbd_a} "
              f"ppl {rp:.1f}->{ap:.1f} fires={fires} "
              f"{'DEGEN' if degenerate else '     '} "
              f"{'RESCUE' if rescued else ''}", flush=True)

    n = len(rows)
    degen = [r for r in rows if r["degenerate"]]
    rescued = [r for r in degen if r["rescued"]]
    all_rt = [r["raw_tail_d2"] for r in rows]; all_at = [r["act_tail_d2"] for r in rows]
    all_rf = [r["raw_full_d2"] for r in rows]; all_af = [r["act_full_d2"] for r in rows]
    all_tbr = [r["raw_tb"] for r in rows]; all_tba = [r["act_tb"] for r in rows]
    all_fires = [r["fires"] for r in rows]

    print("\n=== SUMMARY ===", flush=True)
    print(f"prompts: {n}", flush=True)
    print(f"natural degeneration rate (raw tail d2 < 0.5): {len(degen)}/{n} "
          f"({100*len(degen)/max(n,1):.0f}%)", flush=True)
    print(f"rescue on degenerate subset: {len(rescued)}/{len(degen)} "
          f"({100*len(rescued)/max(len(degen),1):.0f}%)", flush=True)
    print(f"tail d2:   raw mean={np.mean(all_rt):.4f}  governed mean={np.mean(all_at):.4f}  "
          f"(Δ {np.mean(all_at)-np.mean(all_rt):+.4f})", flush=True)
    print(f"full d2:   raw mean={np.mean(all_rf):.4f}  governed mean={np.mean(all_af):.4f}  "
          f"(Δ {np.mean(all_af)-np.mean(all_rf):+.4f})", flush=True)
    print(f"distinct-3: raw mean={np.mean([r['raw_d3'] for r in rows]):.4f}  "
          f"governed mean={np.mean([r['act_d3'] for r in rows]):.4f}", flush=True)
    print(f"tokens-before-degen: raw mean={np.mean(all_tbr):.1f}  governed mean={np.mean(all_tba):.1f}", flush=True)
    all_rp = [r["raw_ppl"] for r in rows]; all_ap = [r["act_ppl"] for r in rows]
    print(f"perplexity: raw mean={np.mean(all_rp):.2f}  governed mean={np.mean(all_ap):.2f}  "
          f"(Δ {np.mean(all_ap)-np.mean(all_rp):+.2f})", flush=True)
    all_bf = [r["bigram_fires"] for r in rows]
    all_sf = [r["spectral_fires"] for r in rows]
    print(f"collapse fires: bigram mean={np.mean(all_bf):.2f} (engaged "
          f"{sum(1 for f in all_bf if f>0)}/{n}), spectral mean={np.mean(all_sf):.2f} "
          f"(engaged {sum(1 for f in all_sf if f>0)}/{n}), total engaged "
          f"{sum(1 for f in all_fires if f>0)}/{n}", flush=True)

    stem = args.model.split("/")[-1].lower()
    regime = "_greedy" if args.greedy else ""
    out = Path(args.out) / f"{stem}_real_degeneracy_{args.max_new}{regime}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[wrote] {out}", flush=True)


if __name__ == "__main__":
    main()
