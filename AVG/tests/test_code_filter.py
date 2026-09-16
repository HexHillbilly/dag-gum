"""
Unit tests for the Track B / Path 2 structural code syntax detector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.metrics import is_code_syntax_context


@pytest.mark.parametrize(
    "text,expected",
    [
        # Code syntax markers: comments, fences, keywords
        ("// comment", True),
        ("/* multi-line comment", True),
        ("# %% cell break", True),
        ("def foo():", True),
        ("let x = 1", True),
        ("const y = 2", True),
        ("var z = 3", True),
        ("struct Foo {", True),
        ("typedef int myint;", True),
        ("```python\nx = 1", True),
        # Plain prose and natural-language repetition traps
        ("The quick brown fox jumps over the lazy dog.", False),
        ("word word word word word", False),
        ("1 2 1 2 1 2", False),
        ("# % # % # %", False),
        ("A B C D A B C D", False),
        # Empty / whitespace input
        ("", False),
        ("   ", False),
        # English words that contain keyword substrings must not trigger
        ("variety", False),
        ("letter", False),
        ("constant", False),
        ("variables", False),
        ("definition", False),
        ("structural", False),
    ],
)
def test_is_code_syntax_context(text: str, expected: bool) -> None:
    assert is_code_syntax_context(text) is expected


def test_is_code_syntax_context_whitespace_stability() -> None:
    """Leading/trailing whitespace should not change the result."""
    assert is_code_syntax_context("  // comment  ") is True
    assert is_code_syntax_context("  word word word  ") is False


def test_is_code_syntax_context_mixed_content() -> None:
    """A code marker anywhere in the window flips the context to True."""
    assert is_code_syntax_context("Here is some prose.\n```python\nx = 1") is True
    assert is_code_syntax_context("Here is some prose with no code markers.") is False
