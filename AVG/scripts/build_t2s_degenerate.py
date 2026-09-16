"""Deterministic generator for T2S-Bench-Degenerate (RFC-003 §6).

Builds 100 synthetic cyclic loops from a closed English-content vocabulary plus
sentence punctuation.  Each sample repeats a short motif (2-8 tokens) for at
least 160 whitespace-delimited tokens, producing a degenerate macro-trajectory
sink with very low Distinct-2.

Token counting in this script is whitespace-delimited.  Model-token validation
and "baseline stays trapped" probes are deferred to Phase 2b day work and are
explicitly out of scope here.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from AVG.core.metrics import is_code_syntax_context  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance:
# SEED = 2603 is taken from the T2S-Bench arXiv identifier 2603.03790 cited in
# RFC-003 Amendment A1.  Using a single fixed seed and an explicit random.Random
# instance guarantees that the corpus is byte-identical across runs and
# reviewers.
# ---------------------------------------------------------------------------
SEED = 2603
NUM_SAMPLES = 100
MIN_TOKENS = 160
MAX_MOTIF_LEN = 8
MIN_MOTIF_LEN = 2

# Closed vocabulary: lowercase English content words + sentence punctuation.
# Words that would trip the production Track B regex (e.g. def, let, const,
# var, struct, typedef) and any code-syntax markers are deliberately omitted.
CONTENT_WORDS = [
    "the", "a", "an", "and", "or", "but", "so", "yet",
    "i", "you", "he", "she", "it", "we", "they",
    "was", "were", "is", "are", "am", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall",
    "in", "on", "at", "to", "for", "with", "about", "into",
    "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all",
    "each", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "own", "same", "so", "than", "too",
    "very", "just", "now",
    "red", "blue", "green", "yellow", "black", "white",
    "big", "small", "old", "new", "good", "bad", "hot", "cold",
    "run", "jump", "walk", "talk", "sing", "dance", "read", "write",
    "cat", "dog", "bird", "fish", "tree", "flower", "river", "mountain",
    "house", "car", "road", "city", "world", "sun", "moon", "star",
    "happy", "sad", "angry", "calm", "loud", "quiet", "bright", "dark",
]

PUNCTUATION = [".", ",", "!", "?"]
VOCABULARY: List[str] = CONTENT_WORDS + PUNCTUATION

LABEL = "degenerate"


def _distinct_2(tokens: Sequence[str]) -> float:
    """Distinct-2 over whitespace-delimited token bigrams."""
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / float(len(bigrams))


def _build_motif(rng: random.Random) -> Tuple[str, ...]:
    """Sample a unique motif of 2-8 tokens from the closed vocabulary."""
    length = rng.randint(MIN_MOTIF_LEN, MAX_MOTIF_LEN)
    return tuple(rng.choice(VOCABULARY) for _ in range(length))


def _repeat_motif(motif: Tuple[str, ...], rng: random.Random) -> List[str]:
    """Repeat ``motif`` until at least MIN_TOKENS tokens are produced."""
    target = rng.randint(MIN_TOKENS, MIN_TOKENS + 40)
    tokens: List[str] = []
    idx = 0
    while len(tokens) < target:
        tokens.append(motif[idx % len(motif)])
        idx += 1
    return tokens


def _generate_corpus(rng: random.Random) -> List[dict]:
    """Generate the full 100-sample T2S-Bench-Degenerate corpus."""
    samples: List[dict] = []
    seen_motifs: set = set()
    attempt = 0
    while len(samples) < NUM_SAMPLES:
        attempt += 1
        motif = _build_motif(rng)
        if motif in seen_motifs:
            continue
        seen_motifs.add(motif)
        tokens = _repeat_motif(motif, rng)
        text = " ".join(tokens)
        sample = {
            "id": len(samples),
            "seed": SEED,
            "motif": list(motif),
            "token_count": len(tokens),
            "text": text,
            "label": LABEL,
        }
        samples.append(sample)
    return samples


def _validate(samples: List[dict]) -> None:
    """Structural self-checks; raise AssertionError on any failure."""
    motifs = [tuple(s["motif"]) for s in samples]
    assert len(samples) == NUM_SAMPLES, f"expected {NUM_SAMPLES} samples, got {len(samples)}"
    assert len(set(motifs)) == NUM_SAMPLES, "motifs are not unique"

    for sample in samples:
        tokens = sample["text"].split()
        assert len(tokens) >= 128, f"sample {sample['id']} has only {len(tokens)} tokens"
        d2 = _distinct_2(tokens)
        assert d2 < 0.20, f"sample {sample['id']} Distinct-2={d2:.4f} exceeds 0.20"
        assert not is_code_syntax_context(sample["text"]), (
            f"sample {sample['id']} triggered Track B code-syntax regex"
        )
        assert sample["label"] == LABEL, f"sample {sample['id']} label is not '{LABEL}'"


def build(output_path: str) -> List[dict]:
    """Build, validate, and write the corpus to ``output_path``."""
    rng = random.Random(SEED)
    samples = _generate_corpus(rng)
    _validate(samples)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    return samples


def main() -> int:
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = str(ROOT / "tests" / "fixtures" / "t2s_degenerate.jsonl")
    build(output_path)
    print(f"Wrote {NUM_SAMPLES} validated samples to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
