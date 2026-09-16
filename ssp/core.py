"""Pure, importable logic extracted from proxy.py (v1.0).

Mirrors the exact behavior of the original proxy without its side effects
(the "Booting up" print, the process exit when the brain is missing, and the
uvicorn server). This module is the single source of truth the benchmark
measures against.

The original proxy.py does:
  1. embed the prompt (all-MiniLM-L6-v2)
  2. FAISS L2 top-3 from the corpus
  3. markovify(state_size=1) blend of the 3 lines  -> the "misfire"
  4. wrap: "You MUST adopt this exact underlying thought and chaotic tone ..."
  5. forward to the local LLM

`wrap_misfire` is v1.0 behavior (mode: chaotic persona override). `wrap_rag`
and `wrap_persona` are the two target modes from the research program (mode B
= grounding, mode A = coherent Daisy voice).
"""
import json
from typing import List, Tuple

import faiss
import markovify
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_corpus(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_index(lines: List[str], model: SentenceTransformer) -> faiss.IndexFlatL2:
    embs = model.encode(lines, convert_to_numpy=True).astype("float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index


def retrieve(
    prompt: str,
    index: faiss.IndexFlatL2,
    text_map: List[str],
    model: SentenceTransformer,
    k: int = 3,
) -> Tuple[List[str], List[float]]:
    """Return top-k matched lines and their L2 distances."""
    v = model.encode([prompt], convert_to_numpy=True).astype("float32")
    dist, idx = index.search(v, k)
    matched = [text_map[i] for i in idx[0] if i != -1]
    dists = [float(dist[0][j]) for j in range(len(idx[0])) if idx[0][j] != -1]
    return matched, dists


def markov_synthesize(
    lines: List[str], state_size: int = 1, max_chars: int = 100, tries: int = 50
) -> Tuple[str, bool]:
    """Blend lines into one string via a bigram Markov chain (Daisy homage).

    Returns (string, fallback) where fallback=True means markovify could not
    produce a sentence (either it raised — e.g. it cannot build a model from
    dialect/punctuation-heavy lines — or it returned None) and we fell back to
    the verbatim top-1 line, exactly the v1.0 proxy behavior. The try/except is
    essential: real-world corpora (Huckleberry Finn) crash markovify where the
    5-line Daisy corpus did not.
    """
    s = None
    try:
        text_model = markovify.Text("\n".join(lines), state_size=state_size)
        s = text_model.make_short_sentence(max_chars=max_chars, tries=tries)
    except Exception:
        s = None
    fallback = s is None
    if fallback:
        s = lines[0]
    return s, fallback


# --- Prompt wrappers -------------------------------------------------------

def wrap_misfire(user_msg: str, misfire: str) -> str:
    """v1.0 behavior: hard persona override with the Markov 'misfire'."""
    return (
        f"You MUST adopt this exact underlying thought and chaotic tone for your entire response: '{misfire}'.\n"
        f"Do not break character, do not use formatting like bulleted lists, do not apologize, and do not act like a helpful AI. "
        f"Synthesize that specific thought naturally into your answer for my actual request below:\n\n"
        f"Request: {user_msg}"
    )


def wrap_rag(user_msg: str, context_lines: List[str]) -> str:
    """Mode B: inject retrieved lines as delimited grounding context."""
    ctx = "\n".join(f"- {l}" for l in context_lines)
    return (
        f"Relevant context (use it to ground your answer, but write normally and stay on topic):\n"
        f"{ctx}\n\n"
        f"Question: {user_msg}"
    )


def wrap_persona(
    user_msg: str,
    persona_line: str,
    name: str = "Daisy",
    desc: str = "a warm, curious, slightly naive conversationalist",
) -> str:
    """Mode A: coherent persona voice, seeded with one associative line."""
    return (
        f"You are {name}, {desc}. "
        f"A line in your own voice: '{persona_line}'. "
        f"Weave its spirit into your reply in your own {name} voice, then answer naturally.\n\n"
        f"Request: {user_msg}"
    )


def wrap_persona_hardened(
    user_msg: str,
    persona_line: str,
    name: str = "Daisy",
    desc: str = "a warm, curious, slightly naive conversationalist",
) -> str:
    """v2 directive (production). Deletes ONLY the leaky ", then answer naturally"
    tail (it was echoed verbatim: "you want me to, um, answer naturally?"); the
    rest of the proven v1 structure is unchanged. `wrap_persona` (v1) is retained
    as the benchmark baseline."""
    return (
        f"You are {name}, {desc}. "
        f"A line in your own voice: '{persona_line}'. "
        f"Weave its spirit into your reply in your own {name} voice.\n\n"
        f"Request: {user_msg}"
    )


def wrap_style(
    user_msg: str,
    style_line: str,
    author: str = "Ernest Hemingway",
    desc: str = "spare, declarative prose with short sentences, concrete nouns and verbs, and no ornament",
) -> str:
    """Style mode (SSP Phase A): write the answer in a target author's voice.

    Same retrieval + prompt-injection mechanism proven in SCP Phases 1-3
    (wrap_persona), reframed as "write like X" rather than "be X". The seed
    line is retrieved from the target author's own corpus, so the injection is
    both in-voice and semantically near the user's request.
    """
    return (
        f"Write in the style of {author}: {desc}. "
        f"A line in that style: '{style_line}'. "
        f"Weave its spirit into your reply in that voice.\n\n"
        f"Request: {user_msg}"
    )
