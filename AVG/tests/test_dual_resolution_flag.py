"""
RFC-003 §4.3 flag-off byte-identical proof.

Seed provenance: seed=0 is used because it is the same deterministic seed
employed by the existing controller wiring tests (test_controller_generation_code_filter.py),
giving a stable, comparable baseline for distilgpt2 CPU generation.
"""

from __future__ import annotations

import subprocess
import sys
import types
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


def _load_head_controller_module():
    """Load the pre-Phase-1 controller from git HEAD as a separate module."""
    repo_root = Path(__file__).resolve().parents[1]  # AVG package root
    # Resolve the controller path relative to the actual git root so this runs
    # identically standalone (governor/controller.py) or nested in the dag-gum
    # monorepo (AVG/governor/controller.py).
    git_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    rel = (repo_root / "governor" / "controller.py").relative_to(git_root).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=str(git_root),
        capture_output=True,
        text=True,
        check=True,
    )
    module = types.ModuleType("AVG_governor_controller_head")
    module.__file__ = f"HEAD:{rel}"
    sys.modules[module.__name__] = module
    exec(result.stdout, module.__dict__)
    return module


def _make_governor(model, tokenizer, use_dual_resolution: bool):
    torch.manual_seed(0)

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        use_dual_resolution=use_dual_resolution,
    )

    trusted = [
        tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
        )["input_ids"]
    ]
    governor.calibrate(trusted)
    return governor


def test_flag_off_byte_identical_to_head(small_gpt2) -> None:
    """use_dual_resolution=False must match the git-HEAD controller token sequence."""
    model, tokenizer = small_gpt2
    prompt = "The history of science shows that"
    max_new_tokens = 12

    head_module = _load_head_controller_module()
    head_governor = head_module.ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    head_trusted = [
        tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
        )["input_ids"]
    ]
    torch.manual_seed(0)
    head_governor.calibrate(head_trusted)

    modified_governor = _make_governor(model, tokenizer, use_dual_resolution=False)

    with torch.no_grad():
        head_out = head_governor.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            intervene=True,
        )
        modified_out = modified_governor.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            intervene=True,
        )

    head_ids = head_out["sequences"]
    modified_ids = modified_out["sequences"]

    assert modified_ids.shape == head_ids.shape, (
        f"Shape mismatch: modified {modified_ids.shape} vs HEAD {head_ids.shape}"
    )
    assert torch.equal(modified_ids, head_ids), (
        "Flag-off modified governor diverged from git-HEAD governor output"
    )

    # Engagement evidence: the dual-resolution state must remain untouched.
    assert modified_governor._fast_token_buffer == [], "Fast-path buffer was touched with flag off"
    assert modified_governor._fast_path_distinct == 1.0, "Fast-path distinct was modified with flag off"
    assert modified_governor.get_sigma1_0() is None, "σ₁⁰ was calibrated with flag off"
    assert modified_governor._h_128_buffer is None, "H_128 buffer was touched with flag off"
    assert modified_governor.get_shadow_log() == [], "Shadow log was written with flag off"


def test_flag_on_engages_dual_resolution_state(small_gpt2) -> None:
    """use_dual_resolution=True must touch the fast-path buffer and calibrate σ₁⁰."""
    model, tokenizer = small_gpt2
    prompt = "The history of science shows that"
    max_new_tokens = 12

    governor = _make_governor(model, tokenizer, use_dual_resolution=True)

    # Calibration with the flag on must compute σ₁⁰.
    assert governor.get_sigma1_0() is not None, "σ₁⁰ was not calibrated with flag on"

    with torch.no_grad():
        out = governor.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            intervene=True,
        )

    assert isinstance(out, dict)
    assert "sequences" in out

    # Engagement evidence: dual-resolution state must have been populated.
    assert len(governor._fast_token_buffer) > 0, "Fast-path buffer was never populated with flag on"
    assert governor._fast_path_distinct != 1.0 or len(governor._fast_token_buffer) >= 4, (
        "Fast-path distinct metric was not computed after enough tokens"
    )
    assert governor._h_128_buffer is not None, "H_128 buffer was never populated with flag on"
    assert len(governor.get_shadow_log()) > 0, "Shadow PR log was never written with flag on"
