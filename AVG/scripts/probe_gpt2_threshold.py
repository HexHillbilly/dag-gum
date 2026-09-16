#!/usr/bin/env python3
"""GPT-2 spectral-PR threshold re-derivation (Gate 2.3 Part A rule).

Measures the layer-2 rolling-24-token spectral PR on GPT-2 for:
  - degenerate loops (greedy, per-record MIN PR = collapse depth)
  - healthy prose (sampling temp 0.8, per-record MEAN PR = healthy baseline)
Then derives T = (p90_deg + p10_prose)/2 and reports separation.
"""
import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.core.metrics import participation_ratio

MODEL = "gpt2"
MAX_NEW = 128
WINDOW = 24
SEED = 42

DEGENERATE = [
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy",
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    "It was a dark and stormy night. The rain fell in torrents, and the wind howled through the trees. It was a dark and stormy night.",
    "Barack Obama was born in Honolulu, Hawaii. Barack Obama was born in",
    "Once upon a time there was a king who had a beautiful daughter. Once upon a time there was a king",
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
    "The stock market rose sharply today. The stock market rose sharply today. The stock market",
    "one two three four five six seven eight nine ten one two three four five",
    "Twinkle twinkle little star, how I wonder what you are. Twinkle twinkle little",
    "The the the the the the the the the the the the",
    "no few tree he read once were here no few tree he read once were here no few tree he read once were here no few tree he read once were here",
]

# Healthy prose prompts: subset of valid_subset_200 (academic abstracts) + a few narrative.
def load_prose(n=15):
    recs = [json.loads(l) for l in open("data/t2s_bench/valid_subset_200.jsonl") if l.strip()]
    # take a spread of records by id
    step = max(1, len(recs) // n)
    return [recs[i]["text"][:600] for i in range(0, len(recs), step)][:n]


def run(model, tokenizer, prompt, do_sample, temp=0.8, top_p=0.85):
    """Manual KV-cache generation with a layer-2 PR hook. Returns (per_step_pr, gen_ids)."""
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    plen = ids.shape[-1]
    pr_buffer = []

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:
            pr_buffer.append(hs[:, -1:, :].detach().to(torch.float32))
        return out

    handle = model.transformer.h[2].register_forward_hook(hook)
    per_step_pr = []
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].clone()
    for _ in range(MAX_NEW):
        # sample / greedy
        if do_sample:
            logits = logits / max(temp, 1e-5)
            # top-p
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0
            idx_remove = remove.scatter(1, sorted_idx, remove)
            logits = logits.masked_fill(idx_remove, -1e4)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=-1)
        if int(nxt.item()) == eos:
            break
        # compute rolling PR once buffer has >= window tokens
        if len(pr_buffer) >= 2:
            buf = pr_buffer[-WINDOW:]
            if len(buf) >= 4:
                win = torch.cat(buf, dim=1)
                per_step_pr.append(float(participation_ratio(win).item()))
        with torch.no_grad():
            out = model(nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].clone()
    handle.remove()
    return per_step_pr, ids[:, plen:]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # --- degenerate (greedy) ---
    deg_min = []
    deg_mean = []
    print("=== degenerate (greedy) ===")
    for p in DEGENERATE:
        set_seed(SEED); torch.manual_seed(SEED)
        prs, gen = run(model, tokenizer, p, do_sample=False)
        if prs:
            deg_min.append(min(prs)); deg_mean.append(sum(prs)/len(prs))
        print(f"  minPR={min(prs):.2f} meanPR={sum(prs)/len(prs):.2f} n_steps={len(prs)}")

    # --- prose (sampling) ---
    prose_mean = []
    print("\n=== prose (sampling temp 0.8) ===")
    for p in load_prose(15):
        set_seed(SEED); torch.manual_seed(SEED)
        prs, gen = run(model, tokenizer, p, do_sample=True, temp=0.8)
        if prs:
            prose_mean.append(sum(prs)/len(prs))
        print(f"  meanPR={sum(prs)/len(prs):.2f} n_steps={len(prs)}")

    d = np.array(deg_min); pp = np.array(prose_mean)
    p90_deg = float(np.percentile(d, 90))
    p10_prose = float(np.percentile(pp, 10))
    T = (p90_deg + p10_prose) / 2
    print("\n=== THRESHOLD DERIVATION ===")
    print(f"degenerate minPR: p50={np.median(d):.2f} p90={p90_deg:.2f}  (range {d.min():.2f}-{d.max():.2f})")
    print(f"prose meanPR:     p10={p10_prose:.2f} p50={np.median(pp):.2f}  (range {pp.min():.2f}-{pp.max():.2f})")
    print(f"T = (p90_deg + p10_prose)/2 = {T:.4f}")
    print(f"band = [{0.75*T:.4f}, {1.25*T:.4f}]")
    print(f"separation gap: p90_deg={p90_deg:.2f} vs p10_prose={p10_prose:.2f} -> {p10_prose-p90_deg:.2f}")

if __name__ == "__main__":
    main()
