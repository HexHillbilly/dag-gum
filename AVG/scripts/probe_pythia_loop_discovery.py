#!/usr/bin/env python3
"""Pythia (GPT-NeoX) loop discovery — find model-native degenerate-loop prompts.

Candidate pool = GPT-2 native fixtures (same scale, likely to transfer) + a few
extra natural openers. Raw greedy only; a prompt is a "looper" if trailing-24
distinct-2 < 0.5 over its generated continuation.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import compute_token_distinct_2_fast

MODEL = "EleutherAI/pythia-160m"
SEED = 42
MAX_NEW = 128

def d2(ids, window=24):
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=window)
    return float(d)

def raw_greedy(model, tokenizer, prompt):
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    plen = ids.shape[-1]
    eos = tokenizer.eos_token_id
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    for _ in range(MAX_NEW):
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=-1)
        if int(nxt.item()) == eos:
            break
        with torch.no_grad():
            out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].clone()
    return ids[:, plen:]

def main():
    set_seed(SEED)
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16
    ).to("cuda").eval()

    # candidate pool: GPT-2 native fixtures + extras
    pool = [json.loads(l)["text"] for l in open("tests/fixtures/gpt2_degenerate.jsonl") if l.strip()]
    extras = [
        "The weather today is",
        "Once upon a time,",
        "The meaning of life is",
        "In the beginning,",
        "It is a truth universally acknowledged,",
        "Call me Ishmael.",
        "I think therefore I",
        "To be or not to be,",
        "The mitochondria is the powerhouse of",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        # repetition-heavy seeds (small models reliably degenerate)
        "the the the the the the the the the the",
        "and and and and and and and and and and",
        "a a a a a a a a a a a a a",
        "hello hello hello hello hello hello hello",
        "yes yes yes yes yes yes yes yes yes",
        "no no no no no no no no no no",
        "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
        "Once upon a time. Once upon a time. Once upon a time. Once upon a time.",
        "It was a dark and stormy night. It was a dark and stormy night. It was a dark and stormy night.",
        "I am. I am. I am. I am. I am. I am. I am.",
        "This is a test. This is a test. This is a test. This is a test.",
        "To be or not to be. To be or not to be. To be or not to be.",
        "The quick brown fox. The quick brown fox. The quick brown fox.",
        "He said. He said. He said. He said. He said.",
        "very very very very very very very very",
        "the end. the end. the end. the end. the end.",
        "one. one. one. one. one. one. one. one.",
        "I I I I I I I I I I I I I I I I I I I I",
    ]
    pool = pool + extras

    loopers = []
    for i, p in enumerate(pool):
        set_seed(SEED); torch.manual_seed(SEED)
        gen = raw_greedy(model, tok, p)
        dd = d2(gen)
        flag = "LOOP" if dd < 0.5 else "ok"
        print(f"{i:2d} {flag:5s} d2={dd:.3f}  {p[:60]!r}", flush=True)
        if dd < 0.5:
            loopers.append({"id": len(loopers), "orig_idx": i, "text": p, "label": "degenerate"})

    with open("tests/fixtures/pythia_degenerate.jsonl", "w") as f:
        for r in loopers:
            f.write(json.dumps(r) + "\n")
    print(f"\nloopers: {len(loopers)}/{len(pool)} -> tests/fixtures/pythia_degenerate.jsonl", flush=True)

if __name__ == "__main__":
    main()
