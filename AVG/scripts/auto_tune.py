#!/usr/bin/env python3
"""Auto-tune the governor for a NEW model — zero hand-tuning.

Pipeline (fully automatic; no per-model thresholds supplied by hand):
  0. DISCOVERY   — generate model-agnostic degenerate candidates (natural prose,
                   repeated-seed, and programmatic word-loops) plus any --fixtures;
                   collect the prompts that degenerate (distinct-2 < 0.5) under
                   GREEDY and under SAMPLING (dormant, intervene=False).
  1. BAND_LOW    — derive the spectral-PR collapse threshold from the model's own
                   degenerate floor (p90 of min-PR over loopers) and healthy-prose
                   baseline (p10 of mean-PR over prose):
                     T = (p90_degenerate + p10_prose) / 2 ;  band_low = 0.75 * T
                   (matches the frozen 1.5B derivation: 0.75*(3.13+18.78)/2 = 8.216).
  2. GATE        — rescue (greedy + sampling) and EOS-death (sampling), dormant
                   (intervene=False) vs active (intervene=True) with the auto-derived
                   band_low and the shared defaults (suppression 5.0 + eos_guard).
  3. VALVE CARD  — JSON + MD: band_low, rescue, EOS-death, tokens-before-degen.

The only thing a caller must supply is a model id and (optionally) a prose corpus.
"""
import sys, json, argparse
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

# Model-agnostic natural + repeated-seed candidates.
CANDIDATES = [
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains.",
    "The quick brown fox jumps over the lazy dog.",
    "It was a dark and stormy night.",
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
    "A A A A A A A A A A A A A A A A",
    "1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2",
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy",
    "no few tree he read once were here no few tree he read once were here no few tree he read once were here",
    "It was a dark and stormy night. The rain fell in torrents. It was a dark and stormy night.",
    "Once upon a time there was a king. Once upon a time there was a king. Once upon a time",
    "The stock market rose sharply today. The stock market rose sharply today. The stock market",
]

# Programmatic word-loop patterns — model-agnostic degenerate fixtures (~200 tokens each),
# the same "synthetic word-loop" shape as the Qwen t2s/qwen fixtures.
WORD_LOOP_PATTERNS = [
    ["the", "cat", "the", "dog"],
    ["foo", "bar"],
    ["one", "two", "three"],
    ["1", "2", "1", "2"],
    ["alpha", "beta", "gamma"],
    ["red", "green", "blue"],
    ["a", "b", "c", "d", "e"],
    ["up", "down", "left", "right"],
    ["this", "that", "these", "those"],
    ["king", "queen", "prince", "princess"],
]


def make_word_loops():
    out = []
    for pat in WORD_LOOP_PATTERNS:
        seq = (pat * 60)[:200]
        out.append(" ".join(seq))
    return out


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
    fe = first_eos_idx(gen_ids, eos_id)
    loop_onset = n
    for step in range(24, n):
        window = toks[step - 23:step + 1]
        t1 = window[:-1]; t2 = window[1:]
        bg = torch.stack([t1, t2], dim=1)
        dd = float(torch.unique(bg, dim=0).shape[0]) / (len(window) - 1)
        if dd < LOOP_D2:
            loop_onset = step
            break
    return min(fe, loop_onset)


def derive_band_low(model, tokenizer, device, governor, loopers, prose_recs, eos):
    """Degenerate min-PR (p90) vs prose mean-PR (p10) -> T -> band_low = 0.75*T."""
    pr_buffer = []

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1:
            pr_buffer.append(hs[:, -1:, :].detach().to(torch.float32))
        return out

    handle = governor._get_layer_blocks()[2].register_forward_hook(hook)

    deg_min = []
    for text in loopers[:12]:
        seed_all(SEED)
        enc = tokenizer(text, return_tensors="pt"); ids = enc["input_ids"].to(device)
        pr_buffer.clear()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
        prs = []
        for _ in range(MAX_NEW):
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            if int(nxt.item()) == eos:
                break
            if len(pr_buffer) >= 2:
                buf = pr_buffer[-24:]
                if len(buf) >= 4:
                    prs.append(float(participation_ratio(torch.cat(buf, dim=1)).item()))
            with torch.no_grad():
                out = model(nxt, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :].clone()
        if prs:
            deg_min.append(min(prs))

    step = max(1, len(prose_recs) // 12)
    prose_mean = []
    for rec in prose_recs[::step][:12]:
        seed_all(SEED)
        enc = tokenizer(rec["text"][:600], return_tensors="pt"); ids = enc["input_ids"].to(device)
        pr_buffer.clear()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
        prs = []
        for _ in range(MAX_NEW):
            logits2 = logits / 0.8
            probs = torch.softmax(logits2, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            if int(nxt.item()) == eos:
                break
            if len(pr_buffer) >= 2:
                buf = pr_buffer[-24:]
                if len(buf) >= 4:
                    prs.append(float(participation_ratio(torch.cat(buf, dim=1)).item()))
            with torch.no_grad():
                out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].clone()
        if prs:
            prose_mean.append(sum(prs) / len(prs))
    handle.remove()

    if not deg_min or not prose_mean:
        print("!! band_low derivation failed (no PR samples)", flush=True)
        return None, None, None
    d = np.array(deg_min); p = np.array(prose_mean)
    p90 = float(np.percentile(d, 90)); p10 = float(np.percentile(p, 10))
    T = (p90 + p10) / 2
    band_low = 0.75 * T
    print(f"  band_low: degenerate p90={p90:.2f}  prose p10={p10:.2f}  "
          f"T={T:.4f}  -> band_low={band_low:.4f}", flush=True)
    return band_low, p90, p10


def run_gate(governor, tokenizer, device, loopers, eos, do_sample):
    """Return (n, rescue, eos_death, mean_delta, mean_tb_active)."""
    dorm = {}
    for i, text in enumerate(loopers):
        seed_all(SEED)
        inp = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        o = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=do_sample,
                              temperature=0.8, intervene=False)
        gen = o["sequences"][:, o["prompt_len"]:]
        dorm[i] = (d2(gen), tokens_before_loop(gen, eos))

    resc = 0; eosd = 0; dds = []; tbas = []
    for i, text in enumerate(loopers):
        seed_all(SEED)
        inp = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        o = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=do_sample,
                              temperature=0.8, intervene=True)
        act = o["sequences"][:, o["prompt_len"]:]
        ad = d2(act); tba = tokens_before_loop(act, eos)
        dd, tbd = dorm[i]
        delta = ad - dd
        if delta > 0.01:
            resc += 1
        if ad > 0.5 and tba < tbd:
            eosd += 1
        dds.append(delta); tbas.append(tba)
    n = len(loopers)
    return n, resc, eosd, float(np.mean(dds)) if dds else 0.0, float(np.mean(tbas)) if tbas else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--prose", default="data/t2s_bench/valid_subset_200.jsonl")
    ap.add_argument("--fixtures", default="")  # optional extra fixture files (comma-sep)
    ap.add_argument("--keys", default="")
    ap.add_argument("--out", default="docs/gate23/")
    ap.add_argument("--load_in_8bit", action="store_true",
                    help="load the model with bitsandbytes 8-bit (for models too big for fp16)")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="load the model with bitsandbytes 4-bit (smaller VRAM footprint)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.load_in_8bit or args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = (
            BitsAndBytesConfig(load_in_8bit=True)
            if args.load_in_8bit
            else BitsAndBytesConfig(load_in_4bit=True)
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_config, device_map="auto"
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    eos = tokenizer.eos_token_id

    governor = ActiveVarietyGovernor(model, tokenizer=tokenizer)
    prose_recs = [json.loads(l) for l in open(args.prose) if l.strip()]

    # ---- Phase 0: discovery (greedy + sampling, dormant) ----
    candidates = list(CANDIDATES) + make_word_loops()
    if args.fixtures:
        for fpath, key in zip(args.fixtures.split(","), args.keys.split(",")):
            candidates += [rec[key] for rec in [json.loads(l) for l in open(fpath) if l.strip()]]

    greedy_loopers, sampling_loopers = [], []
    print(f"=== discovery over {len(candidates)} candidates ===", flush=True)
    for p in candidates:
        seed_all(SEED)
        inp = tokenizer(p, return_tensors="pt")["input_ids"].to(device)
        o = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=False, intervene=False)
        if d2(o["sequences"][:, o["prompt_len"]:]) < LOOP_D2:
            greedy_loopers.append(p)
        seed_all(SEED)
        o = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                              temperature=0.8, intervene=False)
        if d2(o["sequences"][:, o["prompt_len"]:]) < LOOP_D2:
            sampling_loopers.append(p)
    print(f"greedy loopers: {len(greedy_loopers)} | sampling loopers: {len(sampling_loopers)}", flush=True)
    if not greedy_loopers and not sampling_loopers:
        print("!! no loopers discovered — cannot auto-tune", flush=True)
        sys.exit(2)

    # ---- Phase 1: band_low derivation ----
    print("=== band_low derivation ===", flush=True)
    union = list(dict.fromkeys(greedy_loopers + sampling_loopers))
    band_low, p90, p10 = derive_band_low(model, tokenizer, device, governor, union, prose_recs, eos)
    if band_low is None:
        sys.exit(2)
    governor.band_low = band_low

    # ---- Phase 2: gate ----
    card = {
        "model": args.model, "band_low": round(band_low, 4),
        "degenerate_p90": round(p90, 4), "prose_p10": round(p10, 4),
        "n_greedy_loopers": len(greedy_loopers), "n_sampling_loopers": len(sampling_loopers),
        "shared_defaults": {"suppression_strength": 5.0, "eos_guard": True},
    }
    if greedy_loopers:
        print("=== gate (greedy) ===", flush=True)
        n, resc, eosd, mean_dd, mean_tba = run_gate(governor, tokenizer, device, greedy_loopers, eos, do_sample=False)
        card["greedy"] = {"n": n, "rescue": resc, "eos_death": eosd,
                          "mean_delta_d2": round(mean_dd, 4), "mean_tb_active": round(mean_tba, 2)}
        print(f"greedy rescue {resc}/{n} ({100*resc/max(n,1):.0f}%) | EOS-death {eosd}/{n}", flush=True)
    if sampling_loopers:
        print("=== gate (sampling) ===", flush=True)
        n, resc, eosd, mean_dd, mean_tba = run_gate(governor, tokenizer, device, sampling_loopers, eos, do_sample=True)
        card["sampling"] = {"n": n, "rescue": resc, "eos_death": eosd,
                            "mean_delta_d2": round(mean_dd, 4), "mean_tb_active": round(mean_tba, 2)}
        print(f"sampling rescue {resc}/{n} ({100*resc/max(n,1):.0f}%) | EOS-death {eosd}/{n} ({100*eosd/max(n,1):.0f}%)", flush=True)

    # ---- Phase 3: valve card ----
    stem = args.model.split("/")[-1].lower()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}_autotune_card.json"
    with open(json_path, "w") as f:
        json.dump(card, f, indent=2)
    md_path = out_dir / f"{stem}_autotune_card.md"
    lines = [f"# Auto-tune valve card — {args.model}", "",
             f"- **band_low** = `{card['band_low']}` (degenerate p90 = {card['degenerate_p90']}, prose p10 = {card['prose_p10']})",
             f"- loopers: {card['n_greedy_loopers']} greedy / {card['n_sampling_loopers']} sampling"]
    for regime in ("greedy", "sampling"):
        if regime in card:
            g = card[regime]
            lines.append(f"- **{regime}**: rescue {g['rescue']}/{g['n']} ({100*g['rescue']/max(g['n'],1):.0f}%), "
                         f"EOS-death {g['eos_death']}/{g['n']} ({100*g['eos_death']/max(g['n'],1):.0f}%), "
                         f"mean ΔD2 {g['mean_delta_d2']:+}, mean tb-active {g['mean_tb_active']}")
    lines.append("- shared defaults: suppression_strength 5.0, eos_guard True")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {json_path}\n[wrote] {md_path}", flush=True)
    print(json.dumps(card, indent=2), flush=True)


if __name__ == "__main__":
    main()
