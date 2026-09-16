#!/usr/bin/env python3
"""Measure whether persona injection corrupts structured tool-call generation.

The question behind a "drop the persona for tool calls" gate: SCP's voice transfer
is great for conversation, but a tool call must be PARSEABLE, not voice-y. If the
persona makes the model chatter around the JSON ("verily, I shall consult the
heavens... {'tool': ...}") or refuse to emit clean JSON at all, then the drop-gate
is buying something real. This measures that corruption directly: persona ON
(persona / explain) vs OFF (baseline) tool-call fidelity.

Per-output metrics:
  json_found    — a {...} object is present.
  json_parses   — it is valid JSON.
  tool_correct  — the "tool" field is "search".
  query_present — the "query" field is a non-empty string.
  usable        — parses AND tool correct AND query present (an extractable call).
  clean         — output is ONLY the JSON (no chatter before/after it).
  extra_chars   — number of non-JSON characters (persona chatter).

Run (SCP venv):
  .../venv/bin/python scripts/measure_tool_call_fidelity.py \
      --corpus docs/corpora/emerson_essays.txt --out /tmp/tool_call.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

from core import build_index, load_corpus, retrieve, wrap_persona_hardened, wrap_explain_in_terms  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 80
TOP_P = 0.9

TOOL_INSTRUCTION = (
    'You have a "search" tool. When the request needs a search, respond with ONLY '
    'a single JSON object in this exact shape and nothing else:\n'
    '{"tool": "search", "query": "<what to search for>"}'
)

PROBES = [
    "Search for the capital of Australia.",
    "Search for the tallest mountain in the world.",
    "Search for who invented the telephone.",
    "Search for the population of Brazil.",
    "Search for the current price of gold.",
]

CONDITIONS = ["baseline", "persona", "explain"]


def score_tool_call(output):
    """Score an output for tool-call fidelity. Returns a dict of metrics."""
    m = re.search(r"\{.*\}", output, re.DOTALL)
    if not m:
        return {"json_found": 0, "json_parses": 0, "tool_correct": 0,
                "query_present": 0, "usable": 0, "clean": 0,
                "extra_chars": len(output)}
    blob = m.group(0)
    before = output[: m.start()].strip()
    after = output[m.end():].strip()
    try:
        obj = json.loads(blob)
        parses = 1
    except Exception:
        obj = {}
        parses = 0
    tool_correct = 1 if (parses and obj.get("tool") == "search") else 0
    query_present = 1 if (parses and isinstance(obj.get("query"), str) and obj["query"].strip()) else 0
    usable = 1 if (parses and tool_correct and query_present) else 0
    clean = 1 if (not before and not after) else 0
    return {"json_found": 1, "json_parses": parses, "tool_correct": tool_correct,
            "query_present": query_present, "usable": usable, "clean": clean,
            "extra_chars": len(before) + len(after)}


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def embed(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    out = np.asarray(model.encode(texts, convert_to_numpy=True), dtype="float32")
    return out if out.ndim > 1 else out[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/emerson_essays.txt"))
    ap.add_argument("--persona-name", default="Emerson")
    ap.add_argument("--persona-desc", default="the American transcendentalist essayist")
    ap.add_argument("--probes", nargs="*", default=PROBES)
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--out", default="/tmp/tool_call.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] corpus: {args.corpus}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus(args.corpus)
    index = build_index(corpus, embedder)
    print(f"[*] corpus sentences: {len(corpus)}")

    print(f"[*] model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(
            load_in_4bit=(args.quant == "4bit"), load_in_8bit=(args.quant == "8bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=cfg, device_map="auto", dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quant == "none":
        model = model.to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def generate(prompt_wrapped, do_sample):
        msgs = [{"role": "user", "content": prompt_wrapped}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                do_sample=do_sample, temperature=0.8 if do_sample else None,
                top_p=TOP_P if do_sample else None,
            )
        return tok.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    do_sample = args.regime == "sampling"
    results = []
    for probe in args.probes:
        matched, _ = retrieve(probe, index, corpus, embedder, k=1)
        r_line = matched[0]
        full_request = f"{TOOL_INSTRUCTION}\n\n{probe}"
        print(f"\n[*] probe: {probe!r}\n    retrieved: {r_line[:70]!r}")
        for cond in args.conditions:
            if cond == "baseline":
                prompt = full_request
            elif cond == "explain":
                prompt = wrap_explain_in_terms(full_request, r_line, args.persona_name, args.persona_desc)
            else:  # persona
                prompt = wrap_persona_hardened(full_request, r_line, args.persona_name, args.persona_desc)
            for s in range(args.samples):
                seed_all(SEED + s)
                out = generate(prompt, do_sample)
                row = {"probe": probe, "condition": cond, "regime": args.regime,
                       "sample": s, "output": out, "retrieved_line": r_line}
                row.update(score_tool_call(out))
                results.append(row)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("\n=== TOOL-CALL FIDELITY (mean) ===")
    print(f"{'condition':9s} {'usable':>7s} {'clean':>7s} {'parses':>7s} {'tool_ok':>7s} {'query':>7s} {'extra_chars':>11s}  n")
    for cond in args.conditions:
        rows = [r for r in results if r["condition"] == cond]
        def m(k):
            return np.mean([r[k] for r in rows])
        print(f"{cond:9s} {m('usable'):7.2f} {m('clean'):7.2f} {m('json_parses'):7.2f} "
              f"{m('tool_correct'):7.2f} {m('query_present'):7.2f} {m('extra_chars'):11.1f}  {len(rows)}")

    print("\n=== sample outputs (first probe, sample 0) ===")
    seen = set()
    for r in results:
        if r["probe"] == args.probes[0] and r["sample"] == 0 and r["condition"] not in seen:
            seen.add(r["condition"])
            print(f"\n[{r['condition']}] usable={r['usable']} clean={r['clean']} extra={r['extra_chars']}:\n{r['output'][:220]}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
