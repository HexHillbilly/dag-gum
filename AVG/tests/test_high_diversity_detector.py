"""
Unit tests for `compute_non_linguistic_density` — the char-based high-diversity
predicate that catches both non-ASCII emoji floods AND dense ASCII symbol loops
(----, ====, !?!?, markdown divider runs) that a token-class non-ASCII check is
blind to.

Model-free: uses a one-token-per-character mock tokenizer so the suite loads no
weights and no tokenizer files. The function under test only needs `decode(...)`;
the real-tokenizer path is exercised by the llama3-8b measure harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.metrics import compute_non_linguistic_density


class _CharTokenizer:
    """Minimal tokenizer: one token per character, decode reverses the mapping."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)


@pytest.fixture()
def tok():
    return _CharTokenizer()


def _density(text: str, tok, window_size: int = 1000) -> float:
    # Large default window so density is measured over the whole test string,
    # isolating the char-density math from window truncation (tested separately
    # in test_window_size_truncates).
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return compute_non_linguistic_density(ids, tok, window_size=window_size)


def test_emoji_run_is_high_density(tok):
    assert _density("🔥🔥🔥✨✨✨", tok) >= 0.80


def test_ascii_symbol_loops_are_high_density(tok):
    assert _density("-----=====", tok) >= 0.80
    assert _density("!?!?!?!?!?", tok) >= 0.80
    assert _density("----------", tok) >= 0.80  # markdown divider run


def test_clean_prose_is_low_density(tok):
    assert _density("The quick brown fox jumps over the lazy dog.", tok) <= 0.15


def test_code_snippet_is_low_density(tok):
    code = (
        "# compute the sum of squares\n"
        "def sum_of_squares(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total += value * value\n"
        "    return total"
    )
    assert _density(code, tok) <= 0.15


def test_empty_input_returns_zero(tok):
    assert _density("", tok) == 0.0


def test_pure_whitespace_returns_zero(tok):
    assert _density("   \n\t  ", tok) == 0.0


def test_single_punctuation_is_finite(tok):
    # A lone punctuation char is 100% non-linguistic -> 1.0, but must not
    # crash or divide by zero. (In practice the 24-token window never holds a
    # single char, and the controller's persistence-2 check absorbs it.)
    assert _density(".", tok) == 1.0


def test_batch_dimension_is_handled(tok):
    ids = torch.tensor(tok.encode("🔥🔥🔥"), dtype=torch.long).unsqueeze(0)  # (1, N)
    assert compute_non_linguistic_density(ids, tok) >= 0.80


def test_window_size_truncates(tok):
    # A long clean prefix must not dilute a symbol-spam tail: only the trailing
    # window is measured.
    text = "the quick brown fox " * 5 + "====="
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    assert compute_non_linguistic_density(ids, tok, window_size=5) >= 0.80
