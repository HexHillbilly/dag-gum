"""
DS-003: metrics edge-case CHARACTERIZATION tests.

These tests encode the CURRENT observed behavior of is_code_syntax_context
and compute_token_distinct_2_fast. They are characterization tests, not
specifications: they freeze today's behavior so that deliberate changes
become visible. If a characterization is wrong, the production code must
change first and this file must be updated deliberately — never silently
"fixed" to match a hoped-for value.

Ground truth (verified by human tier — not re-derived here):
- is_code_syntax_context lives in core/metrics.py (Track B regex).
- The canonical Distinct-2 metric is compute_token_distinct_2_fast in
  governor/controller.py (tensor-based; signature
  (input_ids: torch.Tensor, prompt_len: int, window_len: int)).
- core/metrics.py has NO distinct-2 function.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Root-as-package import convention (parents[2] resolves to `/` in this
# container; `import AVG.*` resolves via the /AVG mount).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.metrics import is_code_syntax_context
from AVG.governor.controller import compute_token_distinct_2_fast

# Deterministic runs: no stochastic ops are exercised by these tests, but the
# seed pins the run for provenance reporting.
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Fixture/data paths (repo root = parents[1] of this test file)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_CORPUS_PATH = REPO_ROOT / "tests" / "fixtures" / "schema_corpus.jsonl"
T2S_BENCH_PATH = REPO_ROOT / "data" / "t2s_bench" / "valid_subset_200.jsonl"


def _load_first_schema_sample() -> str:
    """Return the `text` field of the first schema_corpus.jsonl sample."""
    with open(SCHEMA_CORPUS_PATH, encoding="utf-8") as fh:
        return json.loads(fh.readline())["text"]


def _load_first_t2s_samples(n: int = 3) -> list[str]:
    """Return the `text` fields of the first n t2s_bench samples."""
    texts: list[str] = []
    with open(T2S_BENCH_PATH, encoding="utf-8") as fh:
        for _ in range(n):
            texts.append(json.loads(fh.readline())["text"])
    return texts


# ---------------------------------------------------------------------------
# is_code_syntax_context — TRUE cases (Track B structural code indicators)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "// comment",                 # C/C++/Java/JS single-line comment
        "/* block */",                # multi-line comment open
        "```python\nx = 1",           # fenced code block
        "def f():",                   # Python function definition
        "let x = 1",                  # JS variable declaration
        "const y",                    # JS constant declaration
        "var z",                      # JS/Go variable declaration
        "struct S",                   # C/C++/Rust struct keyword
        "typedef int",                # C/C++ typedef keyword
        "# %%",                       # Jupyter / VSCode cell break
    ],
)
def test_is_code_syntax_context_true_cases(text: str) -> None:
    assert is_code_syntax_context(text) is True


# ---------------------------------------------------------------------------
# is_code_syntax_context — FALSE cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "letter",                     # contains "let" without word boundary
        "constantinople",             # contains "const" without word boundary
        "defined",                    # contains "def" without word boundary
        "The quick brown fox jumps over the lazy dog.",  # plain prose
    ],
)
def test_is_code_syntax_context_false_cases(text: str) -> None:
    assert is_code_syntax_context(text) is False


def test_is_code_syntax_context_false_schema_corpus_sample() -> None:
    """One generated schema-corpus sample must NOT be classified as code."""
    sample_text = _load_first_schema_sample()
    assert sample_text, "schema_corpus sample 0 text is empty"
    assert is_code_syntax_context(sample_text) is False


@pytest.mark.skipif(
    not T2S_BENCH_PATH.is_file(),
    reason="data/t2s_bench/valid_subset_200.jsonl not readable",
)
@pytest.mark.parametrize("idx", [0, 1, 2])
def test_is_code_syntax_context_false_t2s_bench_samples(idx: int) -> None:
    """First 3 t2s_bench samples must NOT be classified as code."""
    texts = _load_first_t2s_samples(3)
    assert texts[idx], f"t2s_bench sample {idx} text is empty"
    assert is_code_syntax_context(texts[idx]) is False


# ---------------------------------------------------------------------------
# compute_token_distinct_2_fast — characterization
# ---------------------------------------------------------------------------

def test_distinct2_all_identical_window() -> None:
    """
    All tokens identical in the window.

    Hand-computed: window [7,7,7,7,7,7] has n=6, so n-1=5 bigrams, all equal
    to (7,7). n_unique=1 => dist_2 = 1/5 = 0.2. The repeated-token list is
    [7] (count 6 >= 2).
    """
    input_ids = torch.tensor([[7, 7, 7, 7, 7, 7]])
    dist_2, repeated = compute_token_distinct_2_fast(
        input_ids, prompt_len=0, window_len=24
    )
    assert dist_2 == pytest.approx(0.2)
    assert repeated == [7]


def test_distinct2_all_unique_window() -> None:
    """
    All tokens unique in the window.

    Window [1,2,3,4,5,6] has n=6, so n-1=5 bigrams, all distinct =>
    dist_2 = 5/5 = 1.0. No token repeats => repeated list is empty.
    """
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    dist_2, repeated = compute_token_distinct_2_fast(
        input_ids, prompt_len=0, window_len=24
    )
    assert dist_2 == pytest.approx(1.0)
    assert repeated == []


def test_distinct2_hand_computed_mixed_case() -> None:
    """
    Hand-computed mixed case.

    Window [10,20,10,20,30,40], n=6, n-1=5 bigrams:
        (10,20), (20,10), (10,20), (20,30), (30,40)
    Unique bigrams: (10,20), (20,10), (20,30), (30,40) => 4
    dist_2 = 4/5 = 0.8.
    Repeated tokens (count >= 2): 10 (x2), 20 (x2). torch.unique returns
    sorted values => repeated list is [10, 20].
    """
    input_ids = torch.tensor([[10, 20, 10, 20, 30, 40]])
    dist_2, repeated = compute_token_distinct_2_fast(
        input_ids, prompt_len=0, window_len=24
    )
    assert dist_2 == pytest.approx(0.8)
    assert repeated == [10, 20]


def test_distinct2_prompt_only_empty_window() -> None:
    """
    Prompt-only (empty generated window).

    prompt_len == total sequence length, so the generated-token window is
    empty. Current behavior: the function falls back to the trailing prompt
    window and computes diversity over the prompt itself (no "no data"
    sentinel). For prompt [5,6,7,8,9,10] all bigrams are distinct =>
    dist_2 = 1.0 and no repeated tokens.
    """
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
    dist_2, repeated = compute_token_distinct_2_fast(
        input_ids, prompt_len=6, window_len=24
    )
    assert dist_2 == pytest.approx(1.0)
    assert repeated == []
