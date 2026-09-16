"""
Wiring test: confirm generate() computes and passes is_code_context to the
collapse detector (B2: _evaluate_collapse, which replaced diagnose() in the
generate() hot path).
Uses distilgpt2 for speed; tiny-gpt2 has only 2 layers and is rejected by the
hook-discovery heuristic, while distilgpt2 has 6 layers and is accepted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.governor.controller import ActiveVarietyGovernor


@pytest.fixture(scope="module")
def small_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()
    return model, tokenizer


def _make_patched_governor(model, tokenizer, use_code_filter: bool):
    torch.manual_seed(0)

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        use_code_filter=use_code_filter,
        device=torch.device("cpu"),
    )

    # Calibrate baseline so the collapse detector is active.
    trusted = [
        tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
        )["input_ids"]
    ]
    governor.calibrate(trusted)

    # B2: patch _evaluate_collapse (which replaced diagnose() in generate()'s
    # hot path) to capture the is_code_context values it receives.
    seen_code_contexts: list[bool] = []
    original_evaluate = governor._evaluate_collapse

    def patched_evaluate(
        token_diversity: float = 1.0,
        trailing_ctr: float = 1.0,
        is_code_context: bool = False,
    ):
        seen_code_contexts.append(is_code_context)
        return original_evaluate(
            token_diversity, trailing_ctr, is_code_context
        )

    governor._evaluate_collapse = patched_evaluate
    return governor, seen_code_contexts


def test_generate_passes_code_context_to_collapse(small_gpt2) -> None:
    model, tokenizer = small_gpt2
    governor, seen_code_contexts = _make_patched_governor(
        model, tokenizer, use_code_filter=True
    )

    with torch.no_grad():
        out = governor.generate(
            "// // // // // // // // // //",
            max_new_tokens=16,
            temperature=0.8,
            intervene=True,
        )

    assert isinstance(out, dict)
    assert "collapse" in out
    assert seen_code_contexts, "_evaluate_collapse() was never called"
    assert any(seen_code_contexts), (
        f"Expected at least one True is_code_context, got {seen_code_contexts}"
    )


def test_generate_legacy_path_no_code_context_work(small_gpt2) -> None:
    model, tokenizer = small_gpt2
    governor, seen_code_contexts = _make_patched_governor(
        model, tokenizer, use_code_filter=False
    )

    with torch.no_grad():
        out = governor.generate(
            "// // // // // // // // // //",
            max_new_tokens=16,
            temperature=0.8,
            intervene=True,
        )

    assert isinstance(out, dict)
    assert "collapse" in out
    assert seen_code_contexts, "_evaluate_collapse() was never called"
    assert all(value is False for value in seen_code_contexts), (
        f"Legacy path should never see is_code_context=True, got {seen_code_contexts}"
    )
