#!/usr/bin/env python3
"""Multi-turn cadence benchmark: does persona conditioning hold BOTH voice and
thread fidelity over a long conversation, or does it collapse "all or nothing"?

The single-turn harness (measure_explain_in_terms.py) scores voice + topic on
isolated generations. The live production proxy (llama3-8b + Daisy persona) showed
something those probes can't measure: cadence held at length, complex concepts held
across turns, and cross-turn reference while staying in voice. The failure mode the
owner calls "all or nothing" — you get the voice OR the content, never both — did
not appear, even though it normally shows up "even on a giant model."

This benchmark reproduces that observation under controlled conditions. A multi-turn
thread (progressive follow-ups, each referencing the prior answer) is generated under
four conditions:

  baseline    — plain conversation, no persona.
  persona     — v2 "weave its spirit" context injection (retrieved line).
  explain     — v4 "explain in your own terms" (production default).
  hard_prompt — a hard identity directive with NO retrieval. This is the
                "constraint" approach (command the voice) and is the expected
                "all or nothing" negative control.
  hard_prompt_retrieval — the same hard directive PLUS a retrieved line, to
                isolate "retrieval" from "framing" as the active variable.

Per-turn metrics:
  style_cos  = cosine(output_t, corpus centroid)           — voice (their terms)
  topic_cos  = cosine(output_t, question_t)                — answers this turn
  thread_cos = cosine(output_t, mean(all Q so far + all A so far)) — holds the thread
  ref_cos    = cosine(output_t, answer_{t-1})              — builds on prior turn

Research question: over turns, does hard_prompt trade voice for thread (or vice
versa) while persona/explain hold both? Read as the style_cos / thread_cos
trajectory from turn 1 to turn N.

Run (SCP venv):
  python scripts/measure_multiturn.py \
      --corpus docs/corpora/emerson_essays.txt --out /tmp/multiturn.jsonl
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

from core import build_index, load_corpus, retrieve, wrap_persona_hardened, wrap_explain_in_terms  # noqa: E402

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SEED = 42
MAX_NEW_TOKENS = 90
TOP_P = 0.9

CONDITIONS = ["baseline", "persona", "explain", "hard_prompt", "hard_prompt_retrieval"]

# Multi-turn threads: each later question embeds a referent to the prior answer's
# expected content, so answering correctly REQUIRES holding the thread. Two threads
# (mechanics, biology) average out single-topic noise.
THREADS = [
    [
        "Explain how a jet engine works.",
        "You mentioned the compressor — why must the air be compressed before combustion?",
        "Given that compression, what keeps the turbine blades from melting?",
        "How does a high bypass ratio change that cooling problem on modern airliners?",
        "If high bypass works so well, why do fighter jets use low-bypass engines instead?",
        "How would that fighter-jet tradeoff change for a hypersonic aircraft?",
    ],
    [
        "Explain how the immune system defends the body against infection.",
        "You mentioned antibodies — how do they recognize a specific pathogen?",
        "Given that recognition, why does a second infection by the same pathogen clear faster?",
        "How do vaccines exploit that faster response without causing the disease?",
        "If vaccines work that way, why do some viruses like influenza need a new vaccine every year?",
        "What would that annual-update problem mean for a universal flu vaccine?",
    ],
]


def wrap_hard_prompt(user_msg: str, name: str, desc: str) -> str:
    """Constraint-based persona control — the "all or nothing" negative control.

    Unlike wrap_persona_hardened / wrap_explain_in_terms, which CONDITION the model
    by injecting a retrieved line of the persona's own words, this COMMANDS the
    voice with a hard identity directive and no example. This is the "tell it to be
    X" approach, expected to reproduce the observed all-or-nothing failure: the
    model either locks voice and drops the thread, or keeps the thread and drops
    the voice.
    """
    return (
        f"You are {name}, {desc}. You MUST write every response in {name}'s exact "
        f"voice, vocabulary, and sentence rhythm. Never break character. Never fall "
        f"back into a neutral, default assistant voice, even when explaining "
        f"technical subjects.\n\n"
        f"Question: {user_msg}"
    )


def wrap_hard_prompt_retrieval(user_msg: str, persona_line: str, name: str, desc: str) -> str:
    """Hard directive PLUS a retrieved line — isolates retrieval from framing.
    Same constraint language as wrap_hard_prompt, but it also injects the persona's
    own words as context. The three-way comparison is:
      persona (retrieval + soft "weave its spirit")   = condition, soft framing
      explain (retrieval + "explain in your terms")   = condition, anchored framing
      hard_prompt_retrieval (retrieval + hard MUST)   = condition, constraint framing
      hard_prompt (no retrieval + hard MUST)          = constraint, no context
    retrieval effect = hard_prompt vs hard_prompt_retrieval.
    framing effect   = persona/explain vs hard_prompt_retrieval (all have retrieval).
    """
    return (
        f"You are {name}, {desc}. "
        f"A line in your own voice: '{persona_line}'. "
        f"You MUST write every response in {name}'s exact voice, vocabulary, and "
        f"sentence rhythm. Never break character. Never fall back into a neutral, "
        f"default assistant voice, even when explaining technical subjects.\n\n"
        f"Question: {user_msg}"
    )


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


def cosine(a, b):
    return float(np.dot(a / (np.linalg.norm(a) + 1e-12), b / (np.linalg.norm(b) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs/corpora/emerson_essays.txt"))
    ap.add_argument("--persona-name", default="Emerson")
    ap.add_argument("--persona-desc", default="the American transcendentalist essayist")
    ap.add_argument("--regime", choices=["greedy", "sampling"], default="sampling")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    ap.add_argument("--out", default="/tmp/multiturn.jsonl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[*] corpus: {args.corpus}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = load_corpus(args.corpus)
    print(f"[*] corpus sentences: {len(corpus)}")
    index = build_index(corpus, embedder)
    corpus_centroid = embed(embedder, corpus).mean(axis=0)

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

    def generate(msgs, do_sample):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id,
                do_sample=do_sample, temperature=0.8 if do_sample else None,
                top_p=TOP_P if do_sample else None,
            )
        return tok.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def run_thread(questions, condition, name, desc, do_sample):
        """Generate the thread turn by turn; history carries clean Q+A."""
        history = []          # list of {"role", "content"} for prior turns
        answers = []          # prior assistant answers (strings), for ref_cos
        rows = []
        for t, q in enumerate(questions):
            r_line = ""
            if condition in ("persona", "explain", "hard_prompt_retrieval"):
                matched, _ = retrieve(q, index, corpus, embedder, k=1)
                r_line = matched[0]
            if condition == "baseline":
                user_content = q
            elif condition == "persona":
                user_content = wrap_persona_hardened(q, r_line, name, desc)
            elif condition == "explain":
                user_content = wrap_explain_in_terms(q, r_line, name, desc)
            elif condition == "hard_prompt_retrieval":
                user_content = wrap_hard_prompt_retrieval(q, r_line, name, desc)
            else:  # hard_prompt
                user_content = wrap_hard_prompt(q, name, desc)

            msgs = history + [{"role": "user", "content": user_content}]
            out = generate(msgs, do_sample)

            q_emb = embed(embedder, q)[0]
            o_emb = embed(embedder, out)[0] if out else np.zeros(q_emb.shape, dtype="float32")
            thread_parts = [embed(embedder, qq)[0] for qq in questions[: t + 1]]
            thread_parts += [embed(embedder, aa)[0] for aa in answers]
            thread_emb = np.mean(thread_parts, axis=0)
            ref_emb = embed(embedder, answers[-1])[0] if answers else None

            rows.append({
                "thread": questions[0], "turn": t, "condition": condition,
                "regime": args.regime, "output": out, "retrieved_line": r_line,
                "style_cos": round(cosine(o_emb, corpus_centroid), 4),
                "topic_cos": round(cosine(o_emb, q_emb), 4),
                "thread_cos": round(cosine(o_emb, thread_emb), 4),
                "ref_cos": round(cosine(o_emb, ref_emb), 4) if ref_emb is not None else None,
            })

            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": out})
            answers.append(out)
        return rows

    do_sample = args.regime == "sampling"
    results = []
    for ti, thread in enumerate(THREADS):
        print(f"\n[*] thread {ti}: {thread[0]!r}")
        for cond in args.conditions:
            for s in range(args.samples):
                seed_all(SEED + s)
                rows = run_thread(thread, cond, args.persona_name, args.persona_desc, do_sample)
                for r in rows:
                    r["sample"] = s
                results.extend(rows)
            print(f"    {cond:11s} done ({len(thread)} turns x {args.samples} samples)")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # ---- summary ----
    print("\n=== OVERALL (mean over all turns) ===")
    print(f"{'condition':12s} {'style':>7s} {'topic':>7s} {'thread':>7s} {'ref':>7s}  n")
    for cond in args.conditions:
        rows = [r for r in results if r["condition"] == cond]
        sc = np.mean([r["style_cos"] for r in rows])
        tc = np.mean([r["topic_cos"] for r in rows])
        th = np.mean([r["thread_cos"] for r in rows])
        rc = np.mean([r["ref_cos"] for r in rows if r["ref_cos"] is not None])
        print(f"{cond:12s} {sc:7.3f} {tc:7.3f} {th:7.3f} {rc:7.3f}  {len(rows)}")

    print("\n=== TRAJECTORY (style_cos / thread_cos per turn) ===")
    nturns = max(len(t) for t in THREADS)
    for cond in args.conditions:
        style_by_turn, thread_by_turn = [], []
        for t in range(nturns):
            rows = [r for r in results if r["condition"] == cond and r["turn"] == t]
            style_by_turn.append(np.mean([r["style_cos"] for r in rows]))
            thread_by_turn.append(np.mean([r["thread_cos"] for r in rows]))
        s = " ".join(f"{v:.2f}" for v in style_by_turn)
        th = " ".join(f"{v:.2f}" for v in thread_by_turn)
        print(f"{cond:12s} style : {s}")
        print(f"{'':12s} thread: {th}")

    print(f"\n[+] {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
