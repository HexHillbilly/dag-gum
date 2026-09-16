#!/usr/bin/env python3
"""Build a sentence-per-line style corpus from a Project Gutenberg plain-text
book. Reproducible: run this to regenerate the corpus in docs/corpora/.

Usage:
    python scripts/build_book_corpus.py <raw_gutenberg.txt> <output.txt> [--start-after STR]

Steps:
  1. strip the Gutenberg START/END boilerplate
  2. drop the table-of-contents front matter (before --start-after, default
     "NOTICE." for Huck Finn)
  3. drop chapter-heading lines (CHAPTER I. / II. / ...)
  4. unwrap hard-wrapped paragraphs (blank-line separated blocks joined)
  5. sentence-split on terminal punctuation
  6. filter fragments shorter than MIN_CHARS (removes headings / illustration
     captions / abbreviation splits)
"""
import re
import sys

MIN_CHARS = 15
# Matches "CHAPTER I." (Roman, e.g. Huck Finn) and "chapter 1" (Arabic,
# e.g. Hemingway's In Our Time). Case-insensitive, optional trailing period.
CHAPTER_RE = re.compile(r"^chapter\s+([IVXL]+|\d+)\.?$", re.IGNORECASE)


def strip_illustrations(lines):
    """Drop Gutenberg `[Illustration: ...]` caption blocks (common in illustrated
    editions, e.g. Austen's Pride and Prejudice). A block runs from a line
    containing "[Illustration" to the next line containing "]". Captions are
    quotes re-printed from the novel body, so dropping them loses nothing unique.
    """
    out = []
    in_ill = False
    for l in lines:
        if in_ill:
            if "]" in l:
                in_ill = False
            continue
        if "[Illustration" in l:
            if "]" in l:
                continue  # single-line caption
            in_ill = True
            continue
        out.append(l)
    return out


def build(raw_path: str, out_path: str, start_after: str = "NOTICE.") -> int:
    raw = open(raw_path, encoding="utf-8", errors="replace").read()

    start = raw.index("*** START OF")
    end = raw.index("*** END OF")
    lines = raw[start:end].split("\n")

    # Drop front matter (title + table of contents) up to and including the
    # start_after heading line.
    for i, l in enumerate(lines):
        if l.strip() == start_after:
            lines = lines[i + 1:]
            break

    lines = strip_illustrations(lines)

    # Join into paragraphs (unwrapping hard line wraps), dropping headings.
    paragraphs, cur = [], []
    for l in lines:
        s = l.strip()
        if CHAPTER_RE.match(s):
            continue
        if s:
            cur.append(s)
        else:
            if cur:
                paragraphs.append(" ".join(cur))
                cur = []
    if cur:
        paragraphs.append(" ".join(cur))

    # Sentence split + filter short fragments.
    sentences = []
    for p in paragraphs:
        for sent in re.split(r"(?<=[.!?])\s+", p):
            s = sent.strip()
            if len(s) >= MIN_CHARS:
                sentences.append(s)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    return len(sentences)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build a sentence-per-line style corpus from a Project Gutenberg text."
    )
    ap.add_argument("raw_path", help="path to raw Gutenberg plain text")
    ap.add_argument("out_path", help="output sentence-per-line corpus path")
    ap.add_argument(
        "--start-after",
        default="NOTICE.",
        help="drop front matter up to (and including) this heading line",
    )
    args = ap.parse_args()
    n = build(args.raw_path, args.out_path, args.start_after)
    print(f"[+] Wrote {n} sentences to {args.out_path}")
