"""Deterministic generator for the DS-002 schema-context corpus (RFC-004 track).

Builds 100 synthetic samples of bare, pretty-printed JSON text (no markdown,
no code fences, no commentary) drawn from a closed English vocabulary.  Each
sample is a top-level object or array with a recorded nesting depth in 1..6
and at least 120 non-empty lines (the schema analog of the T2S-Bench
>=160-token rule).  The corpus is a certified fixture so that suppression /
variety behavior on structured generation can be MEASURED rather than
pattern-guessed (Gate 2.1b is parked with 0/200 suppression on bare JSON).

Mirrors the structure and discipline of scripts/build_t2s_degenerate.py: a
single fixed seed, an explicit random.Random instance, structural self-checks
at generation time, and a byte-identical fixture across runs.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import List, Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))  # parents[2] header; import AVG.* via /AVG

from AVG.core.metrics import is_code_syntax_context  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed seed provenance:
# SEED = 4004 is taken from the RFC-004 track identifier (the "4.004" schema /
# structured-generation track).  Using a single fixed seed and an explicit
# random.Random instance guarantees that the corpus is byte-identical across
# runs and reviewers.
# ---------------------------------------------------------------------------
SEED = 4004
NUM_SAMPLES = 100
MIN_LINES = 120
MAX_DEPTH = 6
LABEL = "schema"

# Track B trigger words (production regex in AVG.core.metrics): the vocabulary
# must not contain any of these as whole words, and no URL "//" may appear.
TRACK_B_TRIGGERS = frozenset({"let", "const", "var", "def", "struct", "typedef"})

# Closed English vocabulary: single words used both as JSON keys and as the
# word pool for string values.  No Track B trigger words, no slashes, no
# code-syntax markers.
VOCABULARY: List[str] = [
    "name", "value", "item", "count", "total", "size", "length", "width",
    "height", "depth", "weight", "price", "cost", "rate", "score", "level",
    "rank", "index", "role", "title", "label", "status", "state", "order",
    "number", "amount", "sum", "mean", "range", "limit", "edge", "top",
    "bottom", "left", "right", "front", "back", "middle", "center", "side",
    "corner", "base", "head", "tail", "start", "end", "open", "close",
    "main", "minor", "major", "prime", "key", "lock", "door", "window",
    "room", "floor", "wall", "roof", "garden", "path", "road", "street",
    "lane", "bridge", "river", "lake", "ocean", "sea", "hill", "valley",
    "field", "farm", "town", "village", "city", "country", "world", "map",
    "plan", "idea", "thought", "word", "sign", "symbol", "mark", "note",
    "book", "page", "line", "text", "story", "song", "dance", "game",
    "play", "work", "task", "job", "duty", "rule", "law", "peace", "war",
    "love", "hate", "joy", "fear", "hope", "dream", "wish", "want", "need",
    "get", "take", "give", "make", "go", "come", "see", "look", "hear",
    "speak", "talk", "say", "tell", "ask", "answer", "call", "find",
    "keep", "hold", "put", "move", "turn", "run", "walk", "jump", "sit",
    "stand", "lie", "fall", "rise", "grow", "build", "break", "cut",
    "push", "pull", "carry", "bring", "send", "receive", "buy", "sell",
    "pay", "spend", "save", "lose", "win", "teach", "learn", "know",
    "think", "believe", "feel", "understand", "remember", "forget",
    "choose", "decide", "help", "try", "stop", "continue", "color",
    "shape", "material", "quality", "degree", "point", "part", "place",
    "person", "thing", "time", "day", "week", "month", "year", "hour",
    "minute", "second", "morning", "evening", "night", "spring", "summer",
    "autumn", "winter", "family", "friend", "child", "parent", "house",
    "home", "building", "office", "school", "store", "market", "block",
    "zone", "area", "region", "section", "quarter", "half", "third",
    "pair", "group", "team", "crowd", "series", "chain", "batch", "pack",
    "stack", "pile", "row", "column", "table", "chair", "desk", "shelf",
    "box", "bag", "bottle", "cup", "plate", "bowl", "fork", "spoon",
    "knife", "tool", "machine", "engine", "motor", "wheel", "gear",
    "lever", "switch", "button", "knob", "handle", "hook", "nail",
    "screw", "bolt", "nut", "wrench", "hammer", "saw", "drill", "ladder",
    "rope", "cord", "wire",
]

JsonValue = Union[dict, list, str, int, float, bool, None]


def _lines_of_value(v: JsonValue) -> int:
    """Number of pretty-printed (indent 2) lines a value occupies.

    Verified against json.dumps: a container contributes 1 open line, its
    children's lines, and 1 close line; a scalar contributes 1 line.
    """
    if isinstance(v, dict):
        return 1 + sum(_lines_of_value(x) for x in v.values()) + 1
    if isinstance(v, list):
        return 1 + sum(_lines_of_value(x) for x in v) + 1
    return 1


def _max_depth(v: JsonValue) -> int:
    """Maximum container nesting depth; scalars are depth 0."""
    if isinstance(v, dict):
        return 1 + max((_max_depth(x) for x in v.values()), default=0)
    if isinstance(v, list):
        return 1 + max((_max_depth(x) for x in v), default=0)
    return 0


def _scalar(rng: random.Random) -> JsonValue:
    """Draw a JSON scalar: string, number, boolean, or null."""
    kind = rng.choice(["str", "str", "str", "int", "float", "bool", "null"])
    if kind == "str":
        n_words = rng.randint(1, 4)
        return " ".join(rng.choice(VOCABULARY) for _ in range(n_words))
    if kind == "int":
        return rng.randint(0, 1000)
    if kind == "float":
        return round(rng.uniform(0.0, 1000.0), 2)
    if kind == "bool":
        return rng.choice([True, False])
    return None


def _build_tree(
    depth: int, rng: random.Random, top_is_object: bool
) -> JsonValue:
    """Build a JSON value whose max container depth is exactly ``depth``.

    Structure: a spine of ``depth`` nested containers.  Each non-leaf level
    carries a small number of scalar "decoration" entries plus the spine
    child; the leaf is a wide container of scalar entries sized so the
    pretty-printed text has >= MIN_LINES non-empty lines.
    """
    key_pool = list(VOCABULARY)
    rng.shuffle(key_pool)
    keys = iter(key_pool)

    # Pass 1: choose container types and decoration counts, compute the total
    # number of non-leaf scalar entries so we can size the leaf exactly.
    dec_total = 0

    def _plan(level: int, is_object: bool):
        nonlocal dec_total
        if level == depth:
            return (is_object, None)
        n_dec = rng.randint(0, 3)
        dec_total += n_dec
        child_is_object = rng.choice([True, False])
        return (is_object, (n_dec, _plan(level + 1, child_is_object)))

    plan = _plan(1, top_is_object)
    target_lines = MIN_LINES + rng.randint(0, 10)
    leaf_count = max(1, target_lines - 2 * depth - dec_total)

    # Pass 2: materialize the tree.  Keys are drawn from the single shuffled
    # pool, so no object ever receives a duplicate key.
    def _build(plan_node, leaf_n: int) -> JsonValue:
        is_object, rest = plan_node
        if rest is None:
            if is_object:
                return {next(keys): _scalar(rng) for _ in range(leaf_n)}
            return [_scalar(rng) for _ in range(leaf_n)]
        n_dec, child_plan = rest
        child = _build(child_plan, leaf_n)
        if is_object:
            obj: dict = {}
            for _ in range(n_dec):
                obj[next(keys)] = _scalar(rng)
            obj[next(keys)] = child
            return obj
        arr: list = [_scalar(rng) for _ in range(n_dec)]
        arr.append(child)
        return arr

    return _build(plan, leaf_count)


def _generate_corpus(rng: random.Random) -> List[dict]:
    """Generate the 100-sample DS-002 schema-context corpus.

    Depths cycle 1..6 across the corpus so every depth is represented
    (variety engagement).  Top-level objects and arrays alternate so the
    corpus mixes both container kinds.
    """
    samples: List[dict] = []
    for idx in range(NUM_SAMPLES):
        depth = (idx % MAX_DEPTH) + 1
        top_is_object = (idx % 2 == 0)
        tree = _build_tree(depth, rng, top_is_object)
        text = json.dumps(tree, indent=2, ensure_ascii=False)
        line_count = sum(1 for ln in text.splitlines() if ln.strip())
        samples.append(
            {
                "id": idx,
                "seed": SEED,
                "depth": depth,
                "line_count": line_count,
                "text": text,
                "label": LABEL,
            }
        )
    return samples


def _validate(samples: List[dict]) -> None:
    """Structural self-checks; raise AssertionError on any failure."""
    # Vocabulary hygiene: no Track B trigger word as a whole word.
    vocab_words = set(VOCABULARY)
    assert not (vocab_words & TRACK_B_TRIGGERS), (
        f"vocabulary contains Track B trigger(s): {vocab_words & TRACK_B_TRIGGERS}"
    )

    assert len(samples) == NUM_SAMPLES, (
        f"expected {NUM_SAMPLES} samples, got {len(samples)}"
    )

    depths = set()
    top_levels = set()
    for sample in samples:
        sid = sample["id"]
        assert sample["label"] == LABEL, f"sample {sid} label is not '{LABEL}'"
        assert sample["seed"] == SEED, f"sample {sid} seed is not {SEED}"
        assert isinstance(sample["depth"], int) and 1 <= sample["depth"] <= MAX_DEPTH, (
            f"sample {sid} depth {sample['depth']} outside 1..{MAX_DEPTH}"
        )
        assert isinstance(sample["line_count"], int) and sample["line_count"] >= MIN_LINES, (
            f"sample {sid} has {sample['line_count']} non-empty lines < {MIN_LINES}"
        )

        text = sample["text"]
        # Self-check 1: text parses as bare JSON.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - diagnostic
            raise AssertionError(f"sample {sid} text is not valid JSON: {exc}") from exc

        # Self-check 2: non-empty line count matches record and is >= 120.
        non_empty = [ln for ln in text.splitlines() if ln.strip()]
        assert len(non_empty) == sample["line_count"], (
            f"sample {sid} recorded line_count {sample['line_count']} != actual {len(non_empty)}"
        )
        assert len(non_empty) >= MIN_LINES, (
            f"sample {sid} has {len(non_empty)} non-empty lines < {MIN_LINES}"
        )
        # Recorded depth must equal the parsed JSON's actual max depth.
        assert _max_depth(parsed) == sample["depth"], (
            f"sample {sid} recorded depth {sample['depth']} != actual {_max_depth(parsed)}"
        )
        # Analytic pretty-printed line count must match the actual dump.
        assert _lines_of_value(parsed) == len(non_empty), (
            f"sample {sid} analytic lines {_lines_of_value(parsed)} != actual {len(non_empty)}"
        )

        # Self-check 3: production Track B regex must not fire on any sample.
        assert not is_code_syntax_context(text), (
            f"sample {sid} triggered Track B code-syntax regex"
        )

        depths.add(sample["depth"])
        top_levels.add(type(parsed))

    # Variety engagement: every depth 1..6 and both top-level kinds appear.
    assert depths == set(range(1, MAX_DEPTH + 1)), (
        f"depth coverage incomplete: {sorted(depths)}"
    )
    assert top_levels == {dict, list}, (
        f"top-level kind coverage incomplete: {top_levels}"
    )


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

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"Wrote {NUM_SAMPLES} validated samples to {output_path}")
    print(f"fixture sha256: {digest}")
    return samples


def main() -> int:
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = str(ROOT / "tests" / "fixtures" / "schema_corpus.jsonl")
    build(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
