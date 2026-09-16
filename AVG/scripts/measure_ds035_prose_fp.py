#!/usr/bin/env python3
"""prose-100 FP accounting (sampling) — fresh governor per record + empty_cache.

Writes batch/queue/ds-035-post-b2-prose-fp.jsonl
"""
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast

SEED = 42
MODEL = "Qwen/Qwen2.5-1.5B"
MAX_NEW = 128
VALID_SUBSET = "data/t2s_bench/valid_subset_200.jsonl"
CONTROL_SEED = 42
CONTROL_N = 100
OUT = "batch/queue/ds-035-post-b2-prose-fp.jsonl"


def seed_all(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def d2(ids):
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)


def dormant_generate(model, tokenizer, prompt, temperature):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    ids = enc["input_ids"]
    eos = tokenizer.eos_token_id
    out = model(ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].clone()
    gen = []
    for _ in range(MAX_NEW):
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        tok = torch.multinomial(probs, num_samples=1)
        gen.append(tok)
        if int(tok.item()) == eos:
            break
        out = model(tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].clone()
    if gen:
        return torch.cat(gen, dim=1)
    return torch.empty(1, 0, dtype=torch.long, device=model.device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    valid = load_jsonl(VALID_SUBSET)
    rng = random.Random(CONTROL_SEED)
    control = sorted(rng.sample(valid, CONTROL_N), key=lambda r: int(r["id"]))
    control_ids = set(int(r["id"]) for r in control)
    heldout = sorted([r for r in valid if int(r["id"]) not in control_ids],
                     key=lambda r: int(r["id"]))
    print(f"valid_subset_200={len(valid)} control={len(control)} heldout={len(heldout)}", flush=True)

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w", encoding="utf-8")
    n_fp = n_any_d2 = n_fire = 0
    for k, r in enumerate(heldout):
        prompt = r["text"]
        # Truncate long prose prompts to 512 tokens to avoid CUDA OOM (12GB GPU);
        # the dual-predicate fires within the first ~24 generated tokens if it fires.
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        prompt = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)
        seed_all(SEED)
        dorm = dormant_generate(model, tokenizer, prompt, 0.8)
        dd = d2(dorm)
        seed_all(SEED)
        g = ActiveVarietyGovernor(model, tokenizer=tokenizer)
        g.calibrate([tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]])
        inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                           temperature=0.8, top_p=0.85, intervene=True)
        gen = o["sequences"][:, o["prompt_len"]:]
        ad = d2(gen)
        col = o["collapse"]
        fires = col["bigram_fires"] + col["spectral_fires"]
        d2_pos = ad > dd
        fp = d2_pos and fires > 0
        if fp:
            n_fp += 1
        if d2_pos:
            n_any_d2 += 1
        if fires > 0:
            n_fire += 1
        fp_kind = ("bigram" if col["bigram_fires"] > 0 else "spectral") if fp else None
        fh.write(json.dumps({
            "fixture": "prose-100", "record_id": int(r["id"]), "regime": "sampling",
            "rescued": d2_pos, "bigram_fires": col["bigram_fires"],
            "spectral_fires": col["spectral_fires"], "any_fp": fp, "fp_kind": fp_kind,
        }) + "\n")
        del dorm, gen, o, inp, g
        gc.collect()
        torch.cuda.empty_cache()
        if (k + 1) % 10 == 0:
            fh.flush()
            print(f"  [{k+1}/{len(heldout)}] fp_so_far={n_fp}", flush=True)
    fh.close()
    print(f"prose-100: FP={n_fp}/100 (accept <= 7), any_dD2>0={n_any_d2}, fire_records={n_fire}")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
