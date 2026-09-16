"""
Unit tests for Track B / Path 2 code-context awareness in diagnose().
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.governor.controller import ActiveVarietyGovernor
from AVG.governor.profiler import LayerVarietyStats, VarietyProfile


class DummyModel(nn.Module):
    def __init__(self, n_layers: int = 12):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_governor(use_code_filter: bool = False) -> ActiveVarietyGovernor:
    model = DummyModel(n_layers=12)
    governor = ActiveVarietyGovernor(
        model,
        device=torch.device("cpu"),
        use_code_filter=use_code_filter,
    )
    # Manually calibrate baseline for the layers diagnose() will consider.
    governor.baseline.mean_attenuation = {6: 1.0, 8: 1.0, 10: 1.0}
    governor.baseline.std_attenuation = {6: 0.1, 8: 0.1, 10: 0.1}
    return governor


def _make_profile() -> VarietyProfile:
    profile = VarietyProfile()
    profile.monitored_layers = [6, 8, 10]
    for layer in profile.monitored_layers:
        profile.stats[layer] = LayerVarietyStats(
            layer_idx=layer,
            participation_ratio=10.0,
            spectrum_entropy=0.5,
        )
    return profile


def test_diagnose_code_context_suppresses_decisions() -> None:
    """Code context forces dormancy even when metrics look like a loop."""
    governor = _make_governor(use_code_filter=True)
    profile = _make_profile()

    decisions = governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=True,
    )

    assert decisions == []
    assert governor._bigram_ctr == 0
    assert governor._consecutive_interventions == {}


def test_diagnose_code_context_resets_existing_persistence() -> None:
    """Transitioning into code context must purge stale collapse state."""
    governor = _make_governor(use_code_filter=True)
    profile = _make_profile()

    # Build persistence without code context.
    governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=False,
    )
    governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=False,
    )
    assert governor._bigram_ctr >= 2

    # A code-context call must reset everything and return no decisions.
    decisions = governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=True,
    )

    assert decisions == []
    assert governor._bigram_ctr == 0
    assert governor._consecutive_interventions == {}


def test_diagnose_natural_text_loop_fires_with_or_predicate() -> None:
    """Natural-text loop satisfies the Path 2 OR predicate and fires."""
    governor = _make_governor(use_code_filter=True)
    profile = _make_profile()

    # First call builds persistence to 1; not enough to fire.
    decisions = governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=False,
    )
    assert decisions == []
    assert governor._bigram_ctr == 1

    # Second call reaches persistence >= 2 and fires on the deepest layer.
    decisions = governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=False,
    )
    assert len(decisions) == 1
    assert decisions[0].layer_idx == 10
    assert "code_context=False" in decisions[0].message


def test_diagnose_path1_and_predicate_unchanged() -> None:
    """With use_code_filter=False, the original AND predicate is preserved."""
    governor = _make_governor(use_code_filter=False)
    profile = _make_profile()

    # Both metrics below threshold -> fires after persistence reaches 2.
    decisions = governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=False,
    )
    assert decisions == []
    assert governor._bigram_ctr == 1

    decisions = governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.10,
        is_code_context=False,
    )
    assert len(decisions) == 1
    assert decisions[0].layer_idx == 10
    assert "code_context" not in decisions[0].message

    # Only one metric below threshold -> AND false; persistence resets.
    governor._bigram_ctr = 0
    governor._consecutive_interventions = {}
    decisions = governor.diagnose(
        profile,
        token_diversity=0.10,
        trailing_ctr=0.50,
        is_code_context=False,
    )
    assert decisions == []
    assert governor._bigram_ctr == 0
