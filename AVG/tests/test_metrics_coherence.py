"""
DS-008: compute_coherent_token_ratio CHARACTERIZATION tests.

These tests encode the CURRENT observed behavior of the text-based coherence
metric in core/metrics.py. They are characterization tests, not
specifications: they freeze today's behavior so that deliberate changes
become visible. If a characterization is wrong, the production code must
change first and this file must be updated deliberately -- never silently
"fixed" to match a hoped-for value.

Function under test:
- compute_coherent_token_ratio(text: str) -> float  (core/metrics.py ~line 43)

Ground truth (verified by human tier -- not re-derived here):
- core/metrics.py contains exactly one str->float coherence helper in this
  family: compute_coherent_token_ratio. (is_code_syntax_context is a
  str->bool helper already covered by test_metrics_edge_cases.py / DS-003.)
- There is NO distinct-2 function in core/metrics.py.

Surprises encoded (flagged for human review):
- CTR does NOT penalize word-level repetition: "the the the ..." and every
  sample in tests/fixtures/t2s_degenerate.jsonl score 1.0.
- The punctuation strip set does NOT include '.', so a sentence-final token
  such as "dog." is counted as invalid, pulling prose CTR to 8/9.
- Pure numeric tokens score 0.0 (no valid category matches).
- A single alphabetic token scores 1.0, but a single alphabetic CHARACTER
  scores 0.15 (single-char-token majority branch).
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

from AVG.core.metrics import compute_coherent_token_ratio

# Deterministic runs: no stochastic ops are exercised by these tests, but the
# seed pins the run for provenance reporting.
torch.manual_seed(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
T2S_DEGENERATE_PATH = REPO_ROOT / "tests" / "fixtures" / "t2s_degenerate.jsonl"


def _load_first_t2s_degenerate_text() -> str:
    """Return the `text` field of the first t2s_degenerate.jsonl sample."""
    with open(T2S_DEGENERATE_PATH, encoding="utf-8") as fh:
        return json.loads(fh.readline())["text"]


# ---------------------------------------------------------------------------
# Empty / whitespace input
# ---------------------------------------------------------------------------

def test_ctr_empty_string() -> None:
    """Empty input: raw_tokens is empty -> returns 0.0."""
    assert compute_coherent_token_ratio("") == 0.0


def test_ctr_whitespace_only() -> None:
    """Whitespace-only input strips to no tokens -> returns 0.0."""
    assert compute_coherent_token_ratio("   \t  ") == 0.0


# ---------------------------------------------------------------------------
# Single token
# ---------------------------------------------------------------------------

def test_ctr_single_token() -> None:
    """Single alphabetic word: 1 valid token / 1 -> 1.0."""
    assert compute_coherent_token_ratio("hello") == 1.0


def test_ctr_single_character() -> None:
    """
    Single alphabetic character: 1/1 single-char majority (> 0.5) -> 0.15.

    Current behavior: the single-char-token majority branch fires before the
    valid-token ratio is computed.
    """
    assert compute_coherent_token_ratio("a") == 0.15


# ---------------------------------------------------------------------------
# Normal prose
# ---------------------------------------------------------------------------

def test_ctr_normal_prose_without_terminal_period() -> None:
    """
    Plain prose without a terminal period: all 9 tokens are valid words
    -> 9/9 = 1.0.
    """
    text = "The quick brown fox jumps over the lazy dog"
    assert compute_coherent_token_ratio(text) == pytest.approx(1.0)


def test_ctr_terminal_period_is_not_stripped() -> None:
    """
    SURPRISE characterization: the punctuation strip set in
    compute_coherent_token_ratio is
        "(),;:{}[]\"'<>`$%^&*=-+/"
    which does NOT include '.'. A sentence-final token "dog." therefore fails
    every valid-token branch and is counted invalid.

    "The quick brown fox jumps over the lazy dog." -> 8 valid / 9 tokens
    = 0.8888888888888888 (8/9).
    """
    text = "The quick brown fox jumps over the lazy dog."
    assert compute_coherent_token_ratio(text) == pytest.approx(8.0 / 9.0)
    # The lone token "dog." alone also scores 0.0 for the same reason.
    assert compute_coherent_token_ratio("dog.") == 0.0


# ---------------------------------------------------------------------------
# Degenerate repetition
# ---------------------------------------------------------------------------

def test_ctr_degenerate_word_repetition() -> None:
    """
    SURPRISE characterization: CTR does NOT penalize word-level repetition.

    "the the the the the the the the the the" -> every token is a valid
    word, so the ratio is 10/10 = 1.0, identical to normal prose.
    """
    text = "the the the the the the the the the the"
    assert compute_coherent_token_ratio(text) == 1.0


def test_ctr_t2s_degenerate_sample() -> None:
    """
    SURPRISE characterization: the first t2s_degenerate.jsonl sample (id=0,
    label="degenerate", token_count=199, repeated 8-token motif) also scores
    1.0. CTR is a token-validity ratio, not a repetition/diversity metric.
    """
    text = _load_first_t2s_degenerate_text()
    assert text, "t2s_degenerate sample 0 text is empty"
    assert compute_coherent_token_ratio(text) == 1.0


# ---------------------------------------------------------------------------
# Additional edge characterizations
# ---------------------------------------------------------------------------

def test_ctr_single_char_majority_branch() -> None:
    """10 single-char alphabetic tokens: 10/10 > 0.5 -> 0.15 (majority branch)."""
    text = "a a a a a a a a a a"
    assert compute_coherent_token_ratio(text) == 0.15


def test_ctr_numeric_tokens_score_zero() -> None:
    """
    SURPRISE characterization: pure numeric tokens match no valid-token
    branch -> 0/3 = 0.0.
    """
    text = "123 456 789"
    assert compute_coherent_token_ratio(text) == 0.0


def test_ctr_code_operators_are_valid() -> None:
    """C/JS-style operators are in the explicit valid-operator list -> 7/7 = 1.0."""
    text = "-> == != <= >= && ||"
    assert compute_coherent_token_ratio(text) == 1.0


def test_ctr_apostrophe_words_are_valid() -> None:
    """
    Words containing apostrophes are valid (apostrophe is in the allowed
    character set). "it's a fine day today": 4 valid words + 1 single char
    ('a') out of 5 tokens; single-char ratio 1/5 = 0.2 <= 0.5, so
    4/5 = 0.8.
    """
    text = "it's a fine day today"
    assert compute_coherent_token_ratio(text) == pytest.approx(0.8)
