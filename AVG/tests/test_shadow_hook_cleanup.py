"""
DS-005: shadow-hook exception safety (try/finally) + regression tests.

Proves that _remove_shadow_hooks() always runs when governed generation
throws mid-run (try/finally guarantee), that the governor remains usable
afterward, and that the happy path is byte-identical to the committed
git-HEAD controller.

Seed provenance: seed=0 is used because it is the same deterministic seed
employed by the existing controller wiring tests
(test_dual_resolution_flag.py, test_controller_generation_code_filter.py),
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
    """Load the committed git-HEAD controller as a separate module.

    Reuses the HEAD-loading pattern from tests/test_dual_resolution_flag.py
    so the working-tree controller can be compared byte-for-byte against the
    committed controller.
    """
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


def _force_generation_exception(governor, monkeypatch):
    """Monkeypatch a shadow-mode step so governed generation throws mid-run."""
    def boom(self):
        raise RuntimeError("DS-005 forced mid-generation failure")

    monkeypatch.setattr(
        governor,
        "_update_h_128_buffer",
        types.MethodType(boom, governor),
    )


def test_shadow_hooks_removed_when_generation_throws(small_gpt2, monkeypatch) -> None:
    """try/finally guarantee: _remove_shadow_hooks() runs even on exception."""
    model, tokenizer = small_gpt2
    governor = _make_governor(model, tokenizer, use_dual_resolution=True)

    assert governor._shadow_hook_handles == [], "pre-condition: no hooks yet"

    _force_generation_exception(governor, monkeypatch)

    with pytest.raises(RuntimeError, match="DS-005 forced mid-generation failure"):
        with torch.no_grad():
            governor.generate(
                "The history of science shows that",
                max_new_tokens=8,
                do_sample=False,
                intervene=True,
            )

    # The exception propagated, but the finally block must have cleaned up.
    assert governor._shadow_hook_handles == [], (
        "Shadow hooks leaked onto the model after generation threw"
    )
    assert governor._shadow_acts == {}, (
        "Shadow activation state not cleared after generation threw"
    )


def test_normal_generation_still_works_after_exception(small_gpt2, monkeypatch) -> None:
    """After a forced exception + cleanup, a seeded generation still works."""
    model, tokenizer = small_gpt2
    governor = _make_governor(model, tokenizer, use_dual_resolution=True)

    _force_generation_exception(governor, monkeypatch)
    with pytest.raises(RuntimeError, match="DS-005 forced mid-generation failure"):
        with torch.no_grad():
            governor.generate(
                "The history of science shows that",
                max_new_tokens=4,
                do_sample=False,
                intervene=True,
            )

    # Undo the monkeypatch so the normal generation path is restored.
    monkeypatch.undo()

    prompt = "The history of science shows that"
    torch.manual_seed(0)
    with torch.no_grad():
        out = governor.generate(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            intervene=True,
        )

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
    assert out["sequences"].shape == (1, prompt_len + 8), (
        f"Expected {(1, prompt_len + 8)}, got {out['sequences'].shape}"
    )
    assert governor._shadow_hook_handles == [], (
        "Shadow hooks leaked after the normal post-exception generation"
    )
    assert governor._shadow_acts == {}, (
        "Shadow activation state not cleared after the normal generation"
    )


def test_happy_path_byte_identical_to_head(small_gpt2) -> None:
    """Happy path (dual-resolution ON) is unchanged vs committed git HEAD."""
    model, tokenizer = small_gpt2
    prompt = "The history of science shows that"
    max_new_tokens = 12

    head_module = _load_head_controller_module()
    head_governor = head_module.ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        use_dual_resolution=True,
    )
    head_trusted = [
        tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
        )["input_ids"]
    ]
    torch.manual_seed(0)
    head_governor.calibrate(head_trusted)

    modified_governor = _make_governor(model, tokenizer, use_dual_resolution=True)

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
        "Modified governor diverged from git-HEAD governor output"
    )

    # Engagement evidence: the shadow logger must have run on the happy path.
    assert len(modified_governor.get_shadow_log()) > 0, (
        "Shadow PR log was never written with dual-resolution on"
    )
