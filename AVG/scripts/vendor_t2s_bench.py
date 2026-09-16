#!/usr/bin/env python3
"""Vendor script for the T2S-Bench validation subset used by RFC-003 Phase 2.

Source material:
- Paper: arXiv:2603.03790, "T2S-Bench & Structure-of-Thought: Benchmarking and
  Prompting Comprehensive Text-to-Structure Reasoning"
- Project page: https://t2s-bench.github.io/T2S-Bench-Page/
- Hugging Face organization: https://huggingface.co/T2SBench

At the time of vendoring, the public T2S-Bench release is split across three
Hugging Face datasets (T2S-Train-1.2k, T2S-Bench-MR, T2S-Bench-E2E). This
script vendors the evaluation-split proxy T2SBench/T2S-Bench-MR (500 samples)
as the Phase-2a validation corpus. The full 1,800-sample validation release
remains a Phase-4 / release milestone per RFC-003 §5.

Seed provenance: seed=42 is chosen as the canonical RFC-003 calibration seed
(see Gate 1.3) so that subset selection is reproducible and tied to the same
random-state contract as the controller calibration.

Refill method (RFC-003 Amendment A1 / Gate 2.1a):
After the initial seeded 200-sample draw, every selected sample is run through
AVG's is_code_syntax_context() prose-fitness filter. Any tripper is discarded,
and a replacement is drawn from the remaining MR-500 pool using the same
seeded RNG (random.Random(42).choice on a sorted candidate list). The process
repeats until the sample is prose-clean. The final certified subset is
asserted to contain 0/200 trippers.
"""

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import List, Set

# Fix sys.path so 'AVG.*' resolves cleanly when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.metrics import is_code_syntax_context

DATASET_NAME = "T2SBench/T2S-Bench-MR"
SPLIT_NAME = "train"  # public evaluation split is exposed under the train split
OUTPUT_DIR = Path("data/t2s_bench")
VALID_FILE = OUTPUT_DIR / "valid.jsonl"
SUBSET_FILE = OUTPUT_DIR / "valid_subset_200.jsonl"
SOURCE_FILE = OUTPUT_DIR / "valid.source"
HASH_FILE = OUTPUT_DIR / "valid.jsonl.sha256"
SUBSET_HASH_FILE = OUTPUT_DIR / "valid_subset_200.jsonl.sha256"
CERT_FILE = OUTPUT_DIR / "valid_subset_200.cert.txt"
SUBSET_SEED = 42
SUBSET_SIZE = 200


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_prose_fit(row: dict) -> bool:
    """Return True iff the row's text field passes the prose-fitness filter."""
    text = row.get("text", "")
    return not is_code_syntax_context(text)


def _certified_subset(rows: List[dict], rng: random.Random) -> tuple[List[int], int]:
    """Return (final_ids, excluded_count) after prose-fitness certification.

    Initial draw: seeded sample of SUBSET_SIZE from the full pool.
    Refill: for every tripper, draw replacements from the remaining pool using
    the same RNG until a prose-fit sample is found.
    """
    pool_size = len(rows)
    all_ids = set(range(pool_size))
    selected = rng.sample(list(all_ids), SUBSET_SIZE)
    remaining: Set[int] = all_ids - set(selected)

    final_ids: List[int] = []
    excluded_count = 0

    for idx in selected:
        if _is_prose_fit(rows[idx]):
            final_ids.append(idx)
            continue

        excluded_count += 1
        while remaining:
            replacement = rng.choice(sorted(remaining))
            remaining.remove(replacement)
            if _is_prose_fit(rows[replacement]):
                final_ids.append(replacement)
                break
        else:
            raise RuntimeError(
                "Exhausted MR-500 pool while refilling prose-certified subset"
            )

    assert len(final_ids) == SUBSET_SIZE
    return final_ids, excluded_count


def vendor_valid_subset() -> List[int]:
    """Download, save, hash, and select the certified 200-sample validation subset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required to vendor T2S-Bench. "
            "Install it in the active environment or provide a pre-downloaded JSONL."
        ) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME, split=SPLIT_NAME)

    # Serialize rows, dropping the image column which is not usable in a JSONL.
    rows = []
    for item in ds:
        row = {k: v for k, v in item.items() if k != "fig"}
        rows.append(row)

    with open(VALID_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sha256 = _compute_sha256(VALID_FILE)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        f.write(f"{sha256}  {VALID_FILE.name}\n")

    # RFC-003 Amendment A1 / Gate 2.1a: prose-fitness certification.
    rng = random.Random(SUBSET_SEED)
    final_ids, excluded_count = _certified_subset(rows, rng)

    # Closing assertion: 0/200 trippers.
    tripper_count = sum(1 for idx in final_ids if not _is_prose_fit(rows[idx]))
    assert tripper_count == 0, f"Certification failed: {tripper_count}/200 trippers"

    with open(SUBSET_FILE, "w", encoding="utf-8") as f:
        for idx in final_ids:
            record = {"id": idx, **rows[idx]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    subset_sha256 = _compute_sha256(SUBSET_FILE)
    with open(SUBSET_HASH_FILE, "w", encoding="utf-8") as f:
        f.write(f"{subset_sha256}  {SUBSET_FILE.name}\n")

    with open(SOURCE_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"paper: arXiv:2603.03790\n"
            f"project_page: https://t2s-bench.github.io/T2S-Bench-Page/\n"
            f"dataset: {DATASET_NAME}\n"
            f"split: {SPLIT_NAME}\n"
            f"full_release_milestone: 1,800 samples (RFC-003 §5)\n"
            f"vendored_rows: {len(rows)}\n"
            f"sha256_full: {sha256}\n"
            f"sha256_certified_subset: {subset_sha256}\n"
            f"certification_excluded: {excluded_count}\n"
            f"certification_assertion: 0/{SUBSET_SIZE} trippers\n"
        )

    with open(CERT_FILE, "w", encoding="utf-8") as f:
        f.write("RFC-003 Amendment A1 / Gate 2.1a Prose-Fitness Certification\n")
        f.write("=" * 70 + "\n")
        f.write(f"pool_size: {len(rows)}\n")
        f.write(f"initial_selected: {SUBSET_SIZE}\n")
        f.write(f"excluded_count: {excluded_count}\n")
        f.write(
            "refill_method: For each selected sample that tripped "
            "is_code_syntax_context(), draw a replacement from the remaining "
            "MR-500 pool using the same seeded RNG (random.Random(42).choice on a "
            "sorted candidate list) until a prose-clean sample is found.\n"
        )
        f.write(f"final_ids: {final_ids}\n")
        f.write(f"tripper_assertion: 0/{SUBSET_SIZE} trippers\n")

    return final_ids


def load_subset_ids() -> List[int]:
    """Load the vendored 200-sample subset IDs without re-downloading."""
    ids = []
    with open(SUBSET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
    return ids


def verify_determinism() -> bool:
    """Run subset selection twice and confirm identical certified sample IDs."""
    ids_a = vendor_valid_subset()
    ids_b = vendor_valid_subset()
    return ids_a == ids_b


if __name__ == "__main__":
    print("Vendoring T2S-Bench validation subset with prose-fitness certification...")
    subset_ids = vendor_valid_subset()
    excluded = int(CERT_FILE.read_text().split("excluded_count: ")[1].split("\n")[0])
    print(f"Vendored {len(subset_ids)} certified sample IDs from {DATASET_NAME}")
    print(f"Pool size: 500")
    print(f"Excluded by prose-fitness filter: {excluded}")
    print(f"Closing assertion: 0/200 trippers")
    print(f"Saved: {VALID_FILE}")
    print(f"Certified subset: {SUBSET_FILE}")
    print(f"Certification: {CERT_FILE}")
    print(f"Source metadata: {SOURCE_FILE}")
    print(f"Full SHA256: {HASH_FILE.read_text().strip()}")
    print(f"Certified subset SHA256: {SUBSET_HASH_FILE.read_text().strip()}")

    print("\nVerifying seeded determinism (two certified runs)...")
    if verify_determinism():
        print("✅ Determinism verified: both runs produced identical certified sample IDs.")
    else:
        print("❌ Determinism failure: certified sample IDs differ between runs.")
        sys.exit(1)
