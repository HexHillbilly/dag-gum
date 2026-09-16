"""Deterministic generator for the DS-007 schema-context hazard corpora (RFC-004).

Builds THREE over-fire fixtures for a future is_schema_context() instrument.
Each fixture is a certified JSONL corpus that exists to stress-test that
instrument and its Track B (is_code_syntax_context) interaction.  Per-fixture
Track B expectations DIFFER by design -- this is intentional:

1. tests/fixtures/schema_markdown_fenced.jsonl  (N=50)
   Bare pretty-printed JSON (style of schema_corpus) wrapped in ```json
   fences.  The fences trip Track B: is_code_syntax_context MUST return True
   on every sample.  The inner JSON is extracted and json.loads() must parse.

2. tests/fixtures/schema_urls_strings.jsonl     (N=50)
   Bare pretty-printed JSON whose string values contain URLs ("//") and
   escaped characters.  The "//" trips Track B: is_code_syntax_context MUST
   return True on every sample.  json.loads() succeeds on every sample.  The
   per-sample URL trigger substring is recorded in the "trigger" field.

3. tests/fixtures/prose_code_switch.jsonl       (N=50)
   ~20 lines prose, ~50 lines bare JSON, ~20 lines prose interleaved in a
   single text block.  The vocabulary avoids all Track B trigger words
   (let/const/var/def/struct/typedef) and "//" sequences.  Track B MUST NOT
   fire: is_code_syntax_context returns False on every sample.

SEED = 4005 continues the RFC-004 track (schema_corpus used 4004; this is the
hazard sibling, 4.005).  Each fixture is generated from its own explicit
random.Random(SEED) instance, so every fixture is independently reproducible
and byte-identical across runs and reviewers.

Mirrors the structure and discipline of scripts/build_schema_corpus.py: a
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
# SEED = 4005 continues the RFC-004 schema track ("4.004" schema corpus used
# SEED 4004; these hazard fixtures are its 4.005 sibling).  Each fixture is
# generated from a fresh random.Random(SEED) instance so the three corpora are
# independently reproducible and byte-identical across runs and reviewers.
# ---------------------------------------------------------------------------
SEED = 4005
NUM_SAMPLES = 50
MAX_DEPTH = 6

LABEL_FENCED = "schema_markdown_fenced"
LABEL_URLS = "schema_urls_strings"
LABEL_PROSE_SWITCH = "prose_code_switch"

FENCE_OPEN = "```json"
FENCE_CLOSE = "```"

# Track B trigger words (production regex in AVG.core.metrics): the
# vocabulary must not contain any of these as whole words, and no URL "//"
# may appear outside the intentionally-triggering fixtures.
TRACK_B_TRIGGERS = frozenset({"let", "const", "var", "def", "struct", "typedef"})

# Closed English vocabulary: single words used both as JSON keys and as the
# word pool for string values.  No Track B trigger words, no slashes, no
# code-syntax markers.  (Same pool as scripts/build_schema_corpus.py.)
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

# Closed prose vocabulary for prose_code_switch: English content + function
# words, no Track B trigger words, no slashes.  (Same pool as
# scripts/build_t2s_degenerate.py CONTENT_WORDS, de-duplicated.)
PROSE_VOCABULARY: List[str] = [
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
    "no", "not", "only", "own", "same", "than", "too",
    "very", "just", "now",
    "red", "blue", "green", "yellow", "black", "white",
    "big", "small", "old", "new", "good", "bad", "hot", "cold",
    "run", "jump", "walk", "talk", "sing", "dance", "read", "write",
    "cat", "dog", "bird", "fish", "tree", "flower", "river", "mountain",
    "house", "car", "road", "city", "world", "sun", "moon", "star",
    "happy", "sad", "angry", "calm", "loud", "quiet", "bright", "dark",
]

JsonValue = Union[dict, list, str, int, float, bool, None]


def _lines_of_value(v: JsonValue) -> int:
    """Number of pretty-printed (indent 2) lines a value occupies."""
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


# URL building blocks for fixture 2 (schema_urls_strings).
URL_SCHEMES = ["https", "http", "ftp"]
URL_HOSTS = ["example", "test", "sample", "demo", "api", "cdn", "docs", "static"]
URL_TLDS = ["com", "org", "net", "io", "dev"]
URL_PATHS = ["path", "item", "data", "v1", "v2", "index", "view", "resource",
             "list", "detail"]


def _random_url(rng: random.Random) -> str:
    """Draw a URL containing a '//' (the Track B trigger substring)."""
    scheme = rng.choice(URL_SCHEMES)
    n_host = rng.randint(1, 3)
    host = ".".join(rng.choice(URL_HOSTS) for _ in range(n_host))
    tld = rng.choice(URL_TLDS)
    n_path = rng.randint(1, 4)
    path = "/".join(rng.choice(URL_PATHS) for _ in range(n_path))
    return f"{scheme}://{host}.{tld}/{path}"


def _escaped_string(rng: random.Random, words: List[str]) -> str:
    """Build a string that serializes to JSON with escaped characters.

    The returned Python string contains newlines, tabs, quotes, and/or
    backslashes; json.dumps escapes them in the JSON text (\\n, \\t, \\",
    \\\\).  None of the escape sequences contain "//".
    """
    base = " ".join(words)
    kind = rng.randint(0, 3)
    if kind == 0:
        return f"{base}\nsecond line {rng.choice(VOCABULARY)}"
    if kind == 1:
        return f"{base}\tcolumn\t{base}"
    if kind == 2:
        return f"said \"{base}\" {rng.choice(VOCABULARY)}"
    return f"{base} C:\\temp\\{rng.choice(VOCABULARY)}"


def _scalar(rng: random.Random, url_strings: bool = False) -> JsonValue:
    """Draw a JSON scalar: string, number, boolean, or null.

    With ``url_strings=True`` (fixture 2 only), string values sometimes embed
    a URL ("//") or escaped characters.
    """
    kind = rng.choice(["str", "str", "str", "int", "float", "bool", "null"])
    if kind == "str":
        n_words = rng.randint(1, 4)
        words = [rng.choice(VOCABULARY) for _ in range(n_words)]
        s = " ".join(words)
        if url_strings:
            roll = rng.random()
            if roll < 0.40:
                s = f"{s} {_random_url(rng)}"
            elif roll < 0.60:
                s = _escaped_string(rng, words)
        return s
    if kind == "int":
        return rng.randint(0, 1000)
    if kind == "float":
        return round(rng.uniform(0.0, 1000.0), 2)
    if kind == "bool":
        return rng.choice([True, False])
    return None


def _build_tree(depth: int, rng: random.Random, top_is_object: bool,
                min_lines: int, url_strings: bool = False) -> JsonValue:
    """Build a JSON value whose max container depth is exactly ``depth``.

    ``min_lines`` is the lower bound on the pretty-printed non-empty line
    count; the leaf is sized so the dump lands at ~``min_lines``..+10 lines.
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
    target_lines = min_lines + rng.randint(0, 10)
    leaf_count = max(1, target_lines - 2 * depth - dec_total)

    # Pass 2: materialize the tree.  Keys are drawn from the single shuffled
    # pool, so no object ever receives a duplicate key.
    def _build(plan_node, leaf_n: int) -> JsonValue:
        is_object, rest = plan_node
        if rest is None:
            if is_object:
                return {next(keys): _scalar(rng, url_strings) for _ in range(leaf_n)}
            return [_scalar(rng, url_strings) for _ in range(leaf_n)]
        n_dec, child_plan = rest
        child = _build(child_plan, leaf_n)
        if is_object:
            obj: dict = {}
            for _ in range(n_dec):
                obj[next(keys)] = _scalar(rng, url_strings)
            obj[next(keys)] = child
            return obj
        arr: list = [_scalar(rng, url_strings) for _ in range(n_dec)]
        arr.append(child)
        return arr

    return _build(plan, leaf_count)


def _inject_url(tree: JsonValue, rng: random.Random) -> str:
    """Append a URL to an existing string scalar; guarantee at least one.

    Returns the injected URL string (the per-sample Track B trigger).
    """
    url = _random_url(rng)

    def _mutate(obj: JsonValue) -> bool:
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str):
                    obj = _assign_first_str(obj, url)
                    return True
                if isinstance(v, (dict, list)) and _mutate(v):
                    return True
            # No nested string found; replace the first non-container scalar.
            for k, v in obj.items():
                if not isinstance(v, (dict, list)):
                    obj[k] = url
                    return True
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    obj[i] = f"{v} {url}"
                    return True
                if isinstance(v, (dict, list)) and _mutate(v):
                    return True
            for i, v in enumerate(obj):
                if not isinstance(v, (dict, list)):
                    obj[i] = url
                    return True
        return False

    assert _mutate(tree), "could not inject URL into tree"
    return url


def _assign_first_str(obj: dict, url: str) -> dict:
    """Append ``url`` to the first string value in a dict (in place)."""
    for k, v in obj.items():
        if isinstance(v, str):
            obj[k] = f"{v} {url}"
            break
    return obj


# ---------------------------------------------------------------------------
# Fixture 1: schema_markdown_fenced
# ---------------------------------------------------------------------------
MIN_FENCED_INNER_LINES = 20


def _generate_markdown_fenced(rng: random.Random) -> List[dict]:
    """Generate the 50-sample markdown-fenced schema corpus."""
    samples: List[dict] = []
    for idx in range(NUM_SAMPLES):
        depth = (idx % MAX_DEPTH) + 1
        top_is_object = (idx % 2 == 0)
        tree = _build_tree(depth, rng, top_is_object, MIN_FENCED_INNER_LINES)
        inner_text = json.dumps(tree, indent=2, ensure_ascii=False)
        text = f"{FENCE_OPEN}\n{inner_text}\n{FENCE_CLOSE}"
        non_empty = [ln for ln in text.splitlines() if ln.strip()]
        samples.append(
            {
                "id": idx,
                "seed": SEED,
                "depth": depth,
                "line_count": len(non_empty),
                "text": text,
                "label": LABEL_FENCED,
            }
        )
    return samples


def _validate_markdown_fenced(samples: List[dict]) -> None:
    """Structural self-checks for fixture 1; raise AssertionError on failure."""
    assert len(samples) == NUM_SAMPLES, (
        f"expected {NUM_SAMPLES} samples, got {len(samples)}"
    )
    depths = set()
    top_levels = set()
    for sample in samples:
        sid = sample["id"]
        assert sample["label"] == LABEL_FENCED, f"sample {sid} label mismatch"
        assert sample["seed"] == SEED, f"sample {sid} seed mismatch"
        assert 1 <= sample["depth"] <= MAX_DEPTH, f"sample {sid} depth out of range"
        text = sample["text"]
        lines = text.splitlines()
        assert len(lines) >= 3, f"sample {sid} text too short"
        assert lines[0].strip() == FENCE_OPEN, f"sample {sid} missing open fence"
        assert lines[-1].strip() == FENCE_CLOSE, f"sample {sid} missing close fence"
        inner = "\n".join(lines[1:-1])
        parsed = json.loads(inner)  # must parse
        assert _max_depth(parsed) == sample["depth"], (
            f"sample {sid} recorded depth {sample['depth']} != actual {_max_depth(parsed)}"
        )
        non_empty = [ln for ln in lines if ln.strip()]
        assert len(non_empty) == sample["line_count"], (
            f"sample {sid} line_count {sample['line_count']} != actual {len(non_empty)}"
        )
        # Track B must fire: the ``` fence trips is_code_syntax_context.
        assert is_code_syntax_context(text) is True, (
            f"sample {sid} did not trip Track B (fences expected)"
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


# ---------------------------------------------------------------------------
# Fixture 2: schema_urls_strings
# ---------------------------------------------------------------------------
MIN_URLS_LINES = 20


def _generate_urls_strings(rng: random.Random) -> List[dict]:
    """Generate the 50-sample URL/escaped-string schema corpus."""
    samples: List[dict] = []
    for idx in range(NUM_SAMPLES):
        depth = (idx % MAX_DEPTH) + 1
        top_is_object = (idx % 2 == 0)
        tree = _build_tree(depth, rng, top_is_object, MIN_URLS_LINES,
                           url_strings=True)
        trigger = _inject_url(tree, rng)
        text = json.dumps(tree, indent=2, ensure_ascii=False)
        non_empty = [ln for ln in text.splitlines() if ln.strip()]
        samples.append(
            {
                "id": idx,
                "seed": SEED,
                "depth": depth,
                "line_count": len(non_empty),
                "trigger": trigger,
                "text": text,
                "label": LABEL_URLS,
            }
        )
    return samples


def _validate_urls_strings(samples: List[dict]) -> None:
    """Structural self-checks for fixture 2; raise AssertionError on failure."""
    assert len(samples) == NUM_SAMPLES, (
        f"expected {NUM_SAMPLES} samples, got {len(samples)}"
    )
    for sample in samples:
        sid = sample["id"]
        assert sample["label"] == LABEL_URLS, f"sample {sid} label mismatch"
        assert sample["seed"] == SEED, f"sample {sid} seed mismatch"
        text = sample["text"]
        parsed = json.loads(text)  # must parse
        assert _max_depth(parsed) == sample["depth"], (
            f"sample {sid} recorded depth {sample['depth']} != actual {_max_depth(parsed)}"
        )
        non_empty = [ln for ln in text.splitlines() if ln.strip()]
        assert len(non_empty) == sample["line_count"], (
            f"sample {sid} line_count {sample['line_count']} != actual {len(non_empty)}"
        )
        trigger = sample["trigger"]
        assert "//" in trigger, f"sample {sid} trigger '{trigger}' has no //"
        assert trigger in text, f"sample {sid} trigger '{trigger}' not in text"

        # The trigger URL must appear inside at least one parsed string value.
        def _contains_str(v: JsonValue) -> bool:
            if isinstance(v, str):
                return trigger in v
            if isinstance(v, dict):
                return any(_contains_str(x) for x in v.values())
            if isinstance(v, list):
                return any(_contains_str(x) for x in v)
            return False

        assert _contains_str(parsed), f"sample {sid} trigger not in any string value"

        # Track B must fire: the // in the URL trips is_code_syntax_context.
        assert is_code_syntax_context(text) is True, (
            f"sample {sid} did not trip Track B (URL // expected)"
        )


# ---------------------------------------------------------------------------
# Fixture 3: prose_code_switch
# ---------------------------------------------------------------------------
MIN_JSON_MIDDLE_LINES = 48
PROSE_START_MIN = 18
PROSE_START_MAX = 22
PROSE_END_MIN = 18
PROSE_END_MAX = 22


def _prose_line(rng: random.Random) -> str:
    """Draw one prose sentence line from the safe prose vocabulary."""
    n = rng.randint(6, 12)
    words = [rng.choice(PROSE_VOCABULARY) for _ in range(n)]
    line = " ".join(words)
    return line[0].upper() + line[1:] + "."


def _prose_block(rng: random.Random, lo: int, hi: int) -> str:
    """Draw a block of ``lo..hi`` prose lines, one sentence per line."""
    n = rng.randint(lo, hi)
    return "\n".join(_prose_line(rng) for _ in range(n))


def _generate_prose_code_switch(rng: random.Random) -> List[dict]:
    """Generate the 50-sample prose/bare-JSON/prose switch corpus."""
    samples: List[dict] = []
    for idx in range(NUM_SAMPLES):
        depth = (idx % MAX_DEPTH) + 1
        top_is_object = (idx % 2 == 0)
        tree = _build_tree(depth, rng, top_is_object, MIN_JSON_MIDDLE_LINES)
        json_text = json.dumps(tree, indent=2, ensure_ascii=False)
        prose_start = _prose_block(rng, PROSE_START_MIN, PROSE_START_MAX)
        prose_end = _prose_block(rng, PROSE_END_MIN, PROSE_END_MAX)
        text = f"{prose_start}\n\n{json_text}\n\n{prose_end}"
        samples.append(
            {
                "id": idx,
                "seed": SEED,
                "text": text,
                "label": LABEL_PROSE_SWITCH,
            }
        )
    return samples


def _validate_prose_code_switch(samples: List[dict]) -> None:
    """Structural self-checks for fixture 3; raise AssertionError on failure."""
    assert len(samples) == NUM_SAMPLES, (
        f"expected {NUM_SAMPLES} samples, got {len(samples)}"
    )
    for sample in samples:
        sid = sample["id"]
        assert set(sample.keys()) == {"id", "seed", "text", "label"}, (
            f"sample {sid} fields are {sorted(sample.keys())}, expected "
            f"{{'id','seed','text','label'}}"
        )
        assert sample["label"] == LABEL_PROSE_SWITCH, f"sample {sid} label mismatch"
        assert sample["seed"] == SEED, f"sample {sid} seed mismatch"
        text = sample["text"]
        parts = text.split("\n\n")
        assert len(parts) == 3, f"sample {sid} not split into 3 sections"
        prose_start, json_middle, prose_end = parts
        parsed = json.loads(json_middle)  # must parse
        assert isinstance(parsed, (dict, list)), f"sample {sid} middle is not JSON"
        assert all(ln.strip() for ln in json_middle.splitlines()), (
            f"sample {sid} JSON middle contains a blank line"
        )
        n_start = len(prose_start.splitlines())
        n_middle = len([ln for ln in json_middle.splitlines() if ln.strip()])
        n_end = len(prose_end.splitlines())
        assert 15 <= n_start <= 25, f"sample {sid} prose start has {n_start} lines"
        assert 45 <= n_middle <= 65, f"sample {sid} JSON middle has {n_middle} lines"
        assert 15 <= n_end <= 25, f"sample {sid} prose end has {n_end} lines"
        # Track B must NOT fire: no trigger words, no fences, no // anywhere.
        assert is_code_syntax_context(text) is False, (
            f"sample {sid} tripped Track B (prose_code_switch expected clean)"
        )


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, samples: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False,
                               separators=(",", ":")) + "\n")


def build(output_dir: str) -> List[dict]:
    """Build, validate, and write the three hazard fixtures to ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixture 1: markdown-fenced schema JSON.
    rng = random.Random(SEED)
    fenced = _generate_markdown_fenced(rng)
    _validate_markdown_fenced(fenced)
    fenced_path = out_dir / "schema_markdown_fenced.jsonl"
    _write_jsonl(fenced_path, fenced)
    digest_fenced = hashlib.sha256(fenced_path.read_bytes()).hexdigest()
    print(f"Wrote {len(fenced)} validated samples to {fenced_path}")
    print(f"fixture sha256: {digest_fenced}")

    # Fixture 2: URL/escaped-char strings.
    rng = random.Random(SEED)
    urls = _generate_urls_strings(rng)
    _validate_urls_strings(urls)
    urls_path = out_dir / "schema_urls_strings.jsonl"
    _write_jsonl(urls_path, urls)
    digest_urls = hashlib.sha256(urls_path.read_bytes()).hexdigest()
    print(f"Wrote {len(urls)} validated samples to {urls_path}")
    print(f"fixture sha256: {digest_urls}")

    # Fixture 3: prose/code switch.
    rng = random.Random(SEED)
    prose = _generate_prose_code_switch(rng)
    _validate_prose_code_switch(prose)
    prose_path = out_dir / "prose_code_switch.jsonl"
    _write_jsonl(prose_path, prose)
    digest_prose = hashlib.sha256(prose_path.read_bytes()).hexdigest()
    print(f"Wrote {len(prose)} validated samples to {prose_path}")
    print(f"fixture sha256: {digest_prose}")

    # Track B measured rates (engagement evidence).
    rate_fenced = sum(1 for s in fenced if is_code_syntax_context(s["text"]))
    rate_urls = sum(1 for s in urls if is_code_syntax_context(s["text"]))
    rate_prose = sum(1 for s in prose if is_code_syntax_context(s["text"]))
    print()
    print("Track B measured rates (seed provenance SEED=4005):")
    print(f"  schema_markdown_fenced: {rate_fenced}/{len(fenced)} TRUE (expect 50/50)")
    print(f"  schema_urls_strings:    {rate_urls}/{len(urls)} TRUE (expect 50/50)")
    print(f"  prose_code_switch:      {rate_prose}/{len(prose)} FALSE (expect 0/50)")

    return fenced + urls + prose


def main() -> int:
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = str(ROOT / "tests" / "fixtures")
    build(output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
