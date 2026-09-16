"""
Fixture-based unit tests for scripts/vendor_t2s_bench.py.

Scope: PURE logic only. No network access, no Hugging Face dataset download,
and no model loading. The vendoring entry point ``vendor_valid_subset()``
deliberately triggers ``datasets.load_dataset`` and is NOT exercised here.
The script's module-level imports are offline-safe (hashlib/json/random plus
``AVG.core.metrics``), so the three pure helpers are loaded in-process:

  * ``_is_prose_fit``      -- certification filter via is_code_syntax_context()
  * ``_certified_subset``  -- seeded draw, tripper exclusion counting, refill
  * ``_compute_sha256``    -- sha256 pinning used for valid.jsonl / subset

There is no ``--help``/dry-run subprocess path in the script, and the module
imports cleanly without a download, so the pure functions are tested directly.

Fixtures are small synthetic JSONL corpora written to pytest's tmp_path. The
pool mixes three categories:

  * clean prose           -- kept by the prose-fitness filter
  * fenced-code trippers  -- ``` fences / ``def`` / ``//`` style markers
  * URL trippers          -- ``https://...`` trips the C-family ``//`` pattern
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

# Standard root-as-package header. parents[2] resolves to '/' in the container;
# `import AVG.*` resolves via the /AVG mount (correct, not a bug).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.metrics import is_code_syntax_context  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vendor_t2s_bench.py"

CLEAN_COUNT = 200
FENCED_COUNT = 10
URL_COUNT = 10
POOL_SIZE = CLEAN_COUNT + FENCED_COUNT + URL_COUNT

# Clean prose: no code delimiters (//, /*, # %%, def, let, const, var,
# struct, typedef, ```) and no URL double-slash.
_CLEAN_TEXTS = [
    f"Sample prose passage number {i} about quick brown foxes and lazy dogs."
    for i in range(CLEAN_COUNT)
]

# Fenced-code trippers: markdown code fences trip the ``` pattern.
_FENCED_TRIPPERS = [
    f"```python\nx = {i}\nprint(x)\n```"
    for i in range(FENCED_COUNT)
]

# URL trippers: http(s):// contains '//' which matches the C-family comment
# delimiter pattern, so URLs are treated as code-syntax context by the filter.
_URL_TRIPPERS = [
    f"See https://example.com/resource/{i} for more details."
    for i in range(URL_COUNT)
]


def build_pool_rows() -> list[dict]:
    """Build the synthetic MR-500 proxy pool: clean prose + two tripper kinds."""
    rows: list[dict] = []
    for text in _CLEAN_TEXTS:
        rows.append({"text": text, "source": "clean_prose"})
    for text in _FENCED_TRIPPERS:
        rows.append({"text": text, "source": "fenced_code_tripper"})
    for text in _URL_TRIPPERS:
        rows.append({"text": text, "source": "url_tripper"})
    return rows


def tripper_indices(rows: list[dict]) -> list[int]:
    """Indices whose text trips is_code_syntax_context()."""
    return [i for i, row in enumerate(rows) if is_code_syntax_context(row["text"])]


def write_pool_fixture(tmp_path: Path, rows: list[dict]) -> Path:
    """Write the synthetic pool as a JSONL fixture (script-style serialization)."""
    path = tmp_path / "mr500_proxy.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_pool_fixture(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_subset_fixture(tmp_path: Path, vendor_mod, rows: list[dict]) -> Path:
    """Run certification and write the subset exactly as the script documents."""
    final_ids, _ = vendor_mod._certified_subset(
        rows, random.Random(vendor_mod.SUBSET_SEED)
    )
    path = tmp_path / "valid_subset_200.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for idx in final_ids:
            record = {"id": idx, **rows[idx]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


@pytest.fixture(scope="module")
def vendor_mod():
    """Load scripts/vendor_t2s_bench.py in-process (offline-safe import)."""
    spec = importlib.util.spec_from_file_location(
        "vendor_t2s_bench_under_test", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_import_is_download_free(vendor_mod) -> None:
    """Pure helpers are importable without triggering datasets/download."""
    assert callable(vendor_mod._is_prose_fit)
    assert callable(vendor_mod._certified_subset)
    assert callable(vendor_mod._compute_sha256)
    assert vendor_mod.SUBSET_SIZE == 200
    assert vendor_mod.SUBSET_SEED == 42


def test_is_prose_fit_classifies_all_fixture_categories(vendor_mod) -> None:
    """Filter keeps clean prose and rejects fenced-code and URL trippers."""
    rows = build_pool_rows()
    assert len(rows) == POOL_SIZE
    for i, row in enumerate(rows):
        assert vendor_mod._is_prose_fit(row) is (i < CLEAN_COUNT)


def test_certified_subset_excludes_exactly_the_trippers(vendor_mod, tmp_path) -> None:
    """Certified subset is 0/200 trippers; exactly the trippers are excluded."""
    rows = read_pool_fixture(write_pool_fixture(tmp_path, build_pool_rows()))
    trippers = set(tripper_indices(rows))
    assert len(trippers) == FENCED_COUNT + URL_COUNT

    final_ids, excluded_count = vendor_mod._certified_subset(
        rows, random.Random(vendor_mod.SUBSET_SEED)
    )

    assert len(final_ids) == vendor_mod.SUBSET_SIZE
    assert len(set(final_ids)) == vendor_mod.SUBSET_SIZE
    # Closing assertion from the script: 0/200 trippers survive.
    assert all(vendor_mod._is_prose_fit(rows[i]) for i in final_ids)
    # Exactly the trippers are excluded; every clean row is kept.
    assert trippers.isdisjoint(final_ids)
    assert sorted(final_ids) == sorted(set(range(len(rows))) - trippers)

    # Exclusion counting: excluded_count equals the number of trippers in the
    # initial seeded draw (refill trippers are not double-counted).
    rng = random.Random(vendor_mod.SUBSET_SEED)
    all_ids = set(range(len(rows)))
    initial = rng.sample(list(all_ids), vendor_mod.SUBSET_SIZE)
    expected_excluded = sum(1 for i in initial if i in trippers)
    assert excluded_count == expected_excluded
    assert excluded_count >= 1  # the refill path was actually exercised


def test_certified_subset_refill_is_deterministic(vendor_mod, tmp_path) -> None:
    """Two seeded runs produce identical certified selections."""
    rows = read_pool_fixture(write_pool_fixture(tmp_path, build_pool_rows()))

    ids_a, excl_a = vendor_mod._certified_subset(
        rows, random.Random(vendor_mod.SUBSET_SEED)
    )
    ids_b, excl_b = vendor_mod._certified_subset(
        rows, random.Random(vendor_mod.SUBSET_SEED)
    )

    assert ids_a == ids_b
    assert excl_a == excl_b


def test_certified_subset_raises_when_pool_exhausted(vendor_mod) -> None:
    """A pool with no clean replacement left must raise, not underfill."""
    rows = [
        {"text": f"clean prose sample number {i} without markers."}
        for i in range(199)
    ]
    rows.append({"text": "```python\nx = 1\n```"})  # one tripper, pool == SUBSET_SIZE
    with pytest.raises(RuntimeError, match="Exhausted MR-500 pool"):
        vendor_mod._certified_subset(rows, random.Random(vendor_mod.SUBSET_SEED))


def test_output_records_match_documented_schema(vendor_mod, tmp_path) -> None:
    """Subset records are {id: <pool index>, **row} per the documented schema."""
    rows = read_pool_fixture(write_pool_fixture(tmp_path, build_pool_rows()))
    subset_path = write_subset_fixture(tmp_path, vendor_mod, rows)

    records = [
        json.loads(line)
        for line in subset_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == vendor_mod.SUBSET_SIZE

    # The script's documented record shape: {"id": idx, **rows[idx]}.
    for rec in records:
        assert isinstance(rec["id"], int)
        assert set(rec.keys()) == {"id", "text", "source"}
        assert rec["text"] and isinstance(rec["text"], str)
        assert rec["source"] in {"clean_prose", "fenced_code_tripper", "url_tripper"}
        # "id" is the pool index, and the referenced row must be prose-fit.
        assert vendor_mod._is_prose_fit(rows[rec["id"]])


def test_sha256_of_output_matches_recomputed_hash(vendor_mod, tmp_path) -> None:
    """_compute_sha256 pins the exact bytes of the written JSONL files."""
    rows = read_pool_fixture(write_pool_fixture(tmp_path, build_pool_rows()))
    subset_path = write_subset_fixture(tmp_path, vendor_mod, rows)

    assert vendor_mod._compute_sha256(subset_path) == hashlib.sha256(
        subset_path.read_bytes()
    ).hexdigest()

    # Also pin the full-pool valid.jsonl analog used by the vendor script.
    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    assert vendor_mod._compute_sha256(valid_path) == hashlib.sha256(
        valid_path.read_bytes()
    ).hexdigest()
