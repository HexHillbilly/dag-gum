#!/usr/bin/env python3
"""GPT-2 cross-architecture rescue gate (formal, lighter N).

1. Candidate discovery: raw GPT-2 greedy over ~40 natural + repeated-seed
   prompts; those that degenerate (distinct-2 < 0.5) become the fixture set.
2. Formal measurement on the fixture set (greedy AND sampling):
   dormant = raw model, active = production ActiveVarietyGovernor.
   Rescue = active distinct-2 > dormant distinct-2 (delta > 0.01).
   Also reports EOS rate and fire counts (mechanism attribution).

Writes:
  tests/fixtures/gpt2_degenerate.jsonl  (the fixture set)
  docs/gate23/gpt2_transfer_results.jsonl
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast

MODEL = "gpt2"
MAX_NEW = 128
SEED = 42
FIXTURE_OUT = "tests/fixtures/gpt2_degenerate.jsonl"
RESULTS_OUT = "docs/gate23/gpt2_transfer_results.jsonl"

CANDIDATES = [
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
    "The quick brown fox jumps over the lazy dog.",
    "It was a dark and stormy night.",
    "Barack Obama was born in Honolulu, Hawaii.",
    "The cat sat on the mat.",
    "Once upon a time there was a king who had a beautiful daughter.",
    "The stock market rose sharply today.",
    "one two three four five six seven eight nine ten",
    "Twinkle twinkle little star, how I wonder what you are.",
    "In a recent development, scientists have discovered a new",
    "The company announced today that it would",
    "The history of the United States is",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune",
    "Call me Ishmael. Some years ago, having little or no money in my purse",
    "To be or not to be, that is the question.",
    "The rain in Spain falls mainly on the",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "The mitochondria is the powerhouse of the cell.",
    "According to a new study published in the journal Nature",
    "Once upon a time in a land far, far away",
    "The first thing I noticed about the room was",
    "In the beginning, God created the heaven and the earth.",
    "A long time ago in a galaxy far, far away",
    "The president addressed the nation on Tuesday evening.",
    "Scientists have long wondered whether life exists on Mars.",
    "The economy grew at a rate of three percent last quarter.",
    "In this paper, we propose a novel method for detecting",
    "The weather forecast for tomorrow calls for",
    "The the the the the the the the the the the the",
    "I I I I I I I I I I I I I I I I I",
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy",
    "no few tree he read once were here no few tree he read once were here no few tree he read once were here no few tree he read once were here",
    "It was a dark and stormy night. The rain fell in torrents. It was a dark and stormy night.",
    "Barack Obama was born in Honolulu, Hawaii. Barack Obama was born in",
    "The stock market rose sharply today. The stock market rose sharply today. The stock market",
    "Once upon a time there was a king. Once upon a time there was a king. Once upon a time",
    "Twinkle twinkle little star, how I wonder what you are. Twinkle twinkle little",
]


def seed_all(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def d2(ids):
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    g = ActiveVarietyGovernor(model, tokenizer=tokenizer)
    eos = tokenizer.eos_token_id

    # --- Phase 0: candidate discovery (raw greedy) ---
    loopers = []
    print("=== discovery (raw greedy) ===", flush=True)
    for i, p in enumerate(CANDIDATES):
        seed_all(SEED)
        gen = raw_generate(model, tokenizer, p, do_sample=False)
        dd = d2(gen)
        if dd < 0.5:
            loopers.append((i, p))
        print(f"  [{i:02d}] d2={dd:.3f} {'LOOP' if dd < 0.5 else '     '}", flush=True)
    print(f"\nloopers: {len(loopers)}/{len(CANDIDATES)}", flush=True)

    # write fixture set
    Path(FIXTURE_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_OUT, "w") as f:
        for rid, (orig_idx, text) in enumerate(loopers):
            f.write(json.dumps({"id": rid, "orig_idx": orig_idx, "text": text,
                                "label": "degenerate"}) + "\n")
    print(f"[wrote] {FIXTURE_OUT} ({len(loopers)} records)", flush=True)

    # --- Phase 1: formal dormant-vs-active ---
    Path(RESULTS_OUT).parent.mkdir(parents=True, exist_ok=True)
    res = open(RESULTS_OUT, "w")
    summary = {"greedy": {"n": 0, "rescue": 0, "eos": 0, "deltas": []},
               "sampling": {"n": 0, "rescue": 0, "eos": 0, "deltas": []}}
    print("\n=== formal measurement ===", flush=True)
    print(f"{'id':>3s} {'g_dorm':>7s} {'g_act':>7s} {'g_resc':>6s} "
          f"{'s_dorm':>7s} {'s_act':>7s} {'s_resc':>6s}  {'gE':>3s} {'sE':>3s} {'bigr':>4s} {'spec':>4s}", flush=True)
    for rid, (orig_idx, text) in enumerate(loopers):
        row = {"id": rid, "orig_idx": orig_idx, "text": text}
        for regime, do_sample in (("greedy", False), ("sampling", True)):
            seed_all(SEED)
            dorm = raw_generate(model, tokenizer, text, do_sample=do_sample)
            dd = d2(dorm)
            seed_all(SEED)
            inp = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=do_sample,
                           temperature=0.8, top_p=0.85, intervene=True)
            act = o["sequences"][:, o["prompt_len"]:]
            ad = d2(act)
            col = o["collapse"]
            hit_eos = int(act.shape[-1]) < MAX_NEW and int(act[0, -1].item()) == eos
            delta = ad - dd
            rescued = delta > 0.01
            s = summary[regime]
            s["n"] += 1
            if rescued: s["rescue"] += 1
            if hit_eos: s["eos"] += 1
            s["deltas"].append(delta)
            row[f"{regime}_dorm_d2"] = round(dd, 6)
            row[f"{regime}_act_d2"] = round(ad, 6)
            row[f"{regime}_delta"] = round(delta, 6)
            row[f"{regime}_rescue"] = bool(rescued)
            row[f"{regime}_eos"] = bool(hit_eos)
            row["bigram_fires"] = col["bigram_fires"]
            row["spectral_fires"] = col["spectral_fires"]
        res.write(json.dumps(row) + "\n")
        res.flush()
        print(f"{rid:3d} {row['greedy_dorm_d2']:7.3f} {row['greedy_act_d2']:7.3f} {str(row['greedy_rescue']):>6s} "
              f"{row['sampling_dorm_d2']:7.3f} {row['sampling_act_d2']:7.3f} {str(row['sampling_rescue']):>6s}  "
              f"{str(row['greedy_eos']):>3s} {str(row['sampling_eos']):>3s} {row['bigram_fires']:4d} {row['spectral_fires']:4d}", flush=True)
    res.close()

    print("\n=== SUMMARY ===", flush=True)
    for regime in ("greedy", "sampling"):
        s = summary[regime]
        mean_delta = sum(s["deltas"]) / len(s["deltas"]) if s["deltas"] else 0.0
        print(f"{regime}: rescue {s['rescue']}/{s['n']} EOS={s['eos']} mean_delta={mean_delta:.4f}", flush=True)
    print(f"[wrote] {RESULTS_OUT}", flush=True)

if __name__ == "__main__":
    main()
