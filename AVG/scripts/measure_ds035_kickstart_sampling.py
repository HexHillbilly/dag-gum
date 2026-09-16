#!/usr/bin/env python3
"""DS-035 sampling rescue + EOS under kickstart-under-sampling (experiment).

Production ActiveVarietyGovernor with kickstart re-enabled under sampling.
Measures rescue (act_d2 > dorm_d2, notop dormant convention matching the
79/100 baseline) and EOS rate for t2s + qwen fixtures, SEED=42.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast

SEED = 42
MODEL = "Qwen/Qwen2.5-1.5B"
MAX_NEW = 128
FIXTURES = {
    "qwen_degenerate": ("tests/fixtures/qwen_degenerate.jsonl", "prompt"),
    "t2s_degenerate": ("tests/fixtures/t2s_degenerate.jsonl", "text"),
}
OUT = "batch/queue/ds-035-kickstart-sampling-experiment.jsonl"


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
    eos = tokenizer.eos_token_id

    def make_governor():
        g = ActiveVarietyGovernor(model, tokenizer=tokenizer)
        g.calibrate([tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]])
        return g

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for name, (path, key) in FIXTURES.items():
            records = load_jsonl(path)
            n_rescue = n_eos = 0
            deltas = []
            for i, rec in enumerate(records):
                prompt = rec[key]
                seed_all(SEED)
                dorm = dormant_generate(model, tokenizer, prompt, 0.8)
                dd = d2(dorm)
                seed_all(SEED)
                g = make_governor()
                inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                with torch.no_grad():
                    o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.8, top_p=0.85, intervene=True)
                gen = o["sequences"][:, o["prompt_len"]:]
                ad = d2(gen)
                delta = ad - dd
                n_gen = int(gen.shape[-1])
                hit_eos = n_gen < MAX_NEW and int(gen[0, -1].item()) == eos
                rescued = delta > 0
                if rescued:
                    n_rescue += 1
                if hit_eos:
                    n_eos += 1
                deltas.append(delta)
                col = o["collapse"]
                fh.write(json.dumps({
                    "fixture": name, "record_id": i, "rescued": rescued,
                    "eos": hit_eos, "n_gen": n_gen, "dorm_d2": round(dd, 6),
                    "act_d2": round(ad, 6), "delta": round(delta, 6),
                    "bigram_fires": col["bigram_fires"], "spectral_fires": col["spectral_fires"],
                    "numeric_immunity_steps": o["numeric_immunity_steps"],
                }) + "\n")
                fh.flush()
            rate = n_rescue / len(records)
            eos_rate = n_eos / len(records)
            mean_delta = sum(deltas) / len(deltas)
            summary[name] = {"n": len(records), "rescued": n_rescue, "rate": rate,
                             "eos": n_eos, "eos_rate": eos_rate, "mean_delta": round(mean_delta, 4)}
            print(f"{name}: {n_rescue}/{len(records)} rescued (rate={rate:.3f}) "
                  f"EOS={n_eos} ({eos_rate:.3f}) mean_delta={mean_delta:.4f}", flush=True)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
