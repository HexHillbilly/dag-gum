"""
DS-008: tensor-family spectral CHARACTERIZATION tests.

These tests encode the CURRENT observed behavior of the tensor/SVD metrics in
core/metrics.py. They are characterization tests, not specifications: they
freeze today's behavior so that deliberate changes become visible. If a
characterization is wrong, the production code must change first and this
file must be updated deliberately -- never silently "fixed" to match a
hoped-for value.

Functions under test (core/metrics.py):
- participation_ratio(activations, eps=1e-8)                [~line 81]
- singular_value_spectrum_entropy(..., normalize=True)      [~line 105]
- effective_rank(activations, threshold=0.01)               [~line 138]
- feature_sparsity_ratio(sparse_codes, eps=1e-8)            [~line 163]
- residual_reconstruction_error(..., relative=True)         [~line 169]
- coherence_index(original, reconstructed, sparse_codes)    [~line 182]
- variety_attenuation_proxy(...)                            [~line 199]
- local_participation_ratio(activations, window=8)          [~line 219]

Surprises encoded (flagged for human review):
- participation_ratio(zero (8,16)) ~= 6.4e-7 (not 0.0): singular values are
  clamped to eps=1e-8, so the ratio collapses toward 0 but is not exactly 0.
- effective_rank(zero (8,16)) == 9.0 == min(n,d)+1: all-zero singular values
  produce cum=0 below every threshold, so every singular value is counted.
- singular_value_spectrum_entropy(zero (8,16)) == inf: normalized entropy
  divides by log(rank+eps) with rank clamped to 1 -> log(1+1e-8) rounds to
  log(1.0)=0.0 in float32 -> division by zero.
- singular_value_spectrum_entropy((8,1)) == nan: after mean-centering, a
  single column has one nonzero singular value; p=1.0 -> entropy ~0 and the
  same log(1.0)=0.0 denominator -> 0/0 = nan.
- local_participation_ratio returns a per-position scalar expanded across the
  batch dim (both batch rows are identical), not a per-(batch,pos) value.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Root-as-package import convention (parents[2] resolves to `/` in this
# container; `import AVG.*` resolves via the /AVG mount).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.metrics import (
    coherence_index,
    effective_rank,
    feature_sparsity_ratio,
    local_participation_ratio,
    participation_ratio,
    residual_reconstruction_error,
    singular_value_spectrum_entropy,
    variety_attenuation_proxy,
)

# Deterministic runs: every tensor below is built from torch.manual_seed(0)
# at the point of construction. See spectral_pair fixture.
torch.manual_seed(0)


@pytest.fixture
def spectral_pair():
    """
    near-rank-1 (8,16) and full-rank noise (8,16), both from seed 0.

    near_rank1 = outer(v, w) + 1e-3 * randn(8,16), with v=randn(8),
    w=randn(16) drawn immediately after the full_rank randn(8,16).

    Recorded values (seed 0):
      near_rank1: pr=1.00455904006958 ent=0.009605571627616882 er=1.0
      full_rank : pr=6.4035820960998535 ent=0.9123597741127014 er=7.0
    """
    torch.manual_seed(0)
    full_rank = torch.randn(8, 16)
    v = torch.randn(8)
    w = torch.randn(16)
    near_rank1 = torch.outer(v, w) + 1e-3 * torch.randn(8, 16)
    return near_rank1, full_rank


# ---------------------------------------------------------------------------
# participation_ratio
# ---------------------------------------------------------------------------

def test_pr_separates_near_rank1_from_full_rank(spectral_pair) -> None:
    """
    participation_ratio separates the two: near-rank-1 ~= 1.005, full-rank
    ~= 6.404 (6.37x separation on this 8x16 seed-0 pair).
    """
    near_rank1, full_rank = spectral_pair
    pr_nr1 = participation_ratio(near_rank1).item()
    pr_fr = participation_ratio(full_rank).item()

    # Recorded values: 1.00455904006958 vs 6.4035820960998535
    assert pr_nr1 == pytest.approx(1.004559, abs=1e-5)
    assert pr_fr == pytest.approx(6.403582, abs=1e-5)
    # Characterization: the near-rank-1 PR is at least 5x below full-rank PR.
    assert pr_fr / pr_nr1 > 5.0


def test_pr_zero_tensor_eps_collapse() -> None:
    """
    SURPRISE: zero (8,16) -> ~6.4e-7, not 0.0. All 8 singular values are
    clamped to eps=1e-8; pr = (8e-8)^2 / (8e-16 + 1e-8) ~= 6.4e-7.
    """
    pr = participation_ratio(torch.zeros(8, 16)).item()
    assert 0.0 < pr < 1e-5


def test_pr_single_row_returns_dimensionality() -> None:
    """
    Single-row (1,16): n<2 branch returns float(d) = 16.0.
    """
    x = torch.randn(1, 16)
    pr = participation_ratio(x).item()
    assert pr == pytest.approx(16.0)


def test_pr_1d_raises_value_error() -> None:
    with pytest.raises(ValueError):
        participation_ratio(torch.randn(10))


# ---------------------------------------------------------------------------
# singular_value_spectrum_entropy
# ---------------------------------------------------------------------------

def test_entropy_separates_near_rank1_from_full_rank(spectral_pair) -> None:
    """
    Normalized spectrum entropy separates the pair: near-rank-1 ~= 0.0096,
    full-rank ~= 0.9124.
    """
    near_rank1, full_rank = spectral_pair
    ent_nr1 = singular_value_spectrum_entropy(near_rank1).item()
    ent_fr = singular_value_spectrum_entropy(full_rank).item()

    assert ent_nr1 == pytest.approx(0.009606, abs=1e-5)
    assert ent_fr == pytest.approx(0.912360, abs=1e-5)
    assert ent_nr1 < ent_fr


def test_entropy_zero_tensor_is_inf() -> None:
    """
    SURPRISE: zero (8,16) -> inf. All singular values clamp to eps; rank
    (s > eps*10) is 0, clamped to 1; log(1 + 1e-8) rounds to log(1.0) = 0.0
    in float32 -> normalized entropy divides by zero.
    """
    ent = singular_value_spectrum_entropy(torch.zeros(8, 16)).item()
    assert math.isinf(ent)


def test_entropy_single_row_is_zero() -> None:
    """
    Single-row (1,16): n<2 branch returns 0.0 (normalized).
    """
    ent = singular_value_spectrum_entropy(torch.randn(1, 16)).item()
    assert ent == pytest.approx(0.0)


def test_entropy_single_column_is_nan() -> None:
    """
    SURPRISE: (8,1) -> nan. After mean-centering, the single column has one
    nonzero singular value; p=1.0 -> entropy ~= 0 and log(1+1e-8) -> 0.0 in
    float32 -> 0/0 = nan.
    """
    ent = singular_value_spectrum_entropy(torch.randn(8, 1)).item()
    assert math.isnan(ent)


# ---------------------------------------------------------------------------
# effective_rank
# ---------------------------------------------------------------------------

def test_effective_rank_separates_near_rank1_from_full_rank(spectral_pair) -> None:
    """
    effective_rank separates the pair: near-rank-1 -> 1.0, full-rank -> 7.0
    (a difference of 6.0 on this 8x16 seed-0 pair).
    """
    near_rank1, full_rank = spectral_pair
    er_nr1 = effective_rank(near_rank1).item()
    er_fr = effective_rank(full_rank).item()

    assert er_nr1 == pytest.approx(1.0)
    assert er_fr == pytest.approx(7.0)
    assert er_fr - er_nr1 == pytest.approx(6.0)


def test_effective_rank_zero_tensor_is_min_dim_plus_one() -> None:
    """
    SURPRISE: zero (8,16) -> 9.0. All-zero singular values give cum=0 below
    every threshold, so all min(n,d)=8 singular values count, plus the final
    +1 -> 9.0.
    """
    er = effective_rank(torch.zeros(8, 16)).item()
    assert er == pytest.approx(9.0)


def test_effective_rank_single_row_returns_dimensionality() -> None:
    """
    Single-row (1,16): n<2 branch returns float(d) = 16.0.
    """
    er = effective_rank(torch.randn(1, 16)).item()
    assert er == pytest.approx(16.0)


def test_effective_rank_1d_raises_value_error() -> None:
    with pytest.raises(ValueError):
        effective_rank(torch.randn(10))


# ---------------------------------------------------------------------------
# feature_sparsity_ratio
# ---------------------------------------------------------------------------

def test_feature_sparsity_ratio_dense_and_empty() -> None:
    """5 of 100 columns active -> 0.05; empty tensor -> 0.0."""
    codes = torch.zeros(4, 100)
    codes[:, :5] = 1.0
    assert feature_sparsity_ratio(codes).item() == pytest.approx(0.05, abs=1e-6)

    assert feature_sparsity_ratio(torch.zeros(4, 10)).item() == 0.0
    assert feature_sparsity_ratio(torch.ones(3, 5)).item() == 1.0
    assert feature_sparsity_ratio(torch.empty(0)).item() == 0.0


def test_feature_sparsity_ratio_eps_boundary() -> None:
    """
    Active is (abs > eps): a value exactly == eps (1e-8) is NOT active;
    a negative value with |value| > eps IS active.
    """
    x = torch.tensor([[1e-8, 0.0, -0.5]])
    assert feature_sparsity_ratio(x).item() == pytest.approx(1.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# residual_reconstruction_error
# ---------------------------------------------------------------------------

def test_reconstruction_error_identity_and_zero() -> None:
    """
    Identity reconstruction -> 0.0 (relative and absolute). Zero
    reconstruction, relative -> 1.0 (||x|| / ||x|| = 1).
    """
    torch.manual_seed(0)
    x = torch.randn(8, 16)

    assert residual_reconstruction_error(x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert residual_reconstruction_error(x, x, relative=False).item() == pytest.approx(0.0, abs=1e-6)
    assert residual_reconstruction_error(x, torch.zeros_like(x)).item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# coherence_index
# ---------------------------------------------------------------------------

def test_coherence_index_perfect_and_zero() -> None:
    """
    Perfect reconstruction -> 1.0; zero reconstruction -> ~0.0.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 16)

    assert coherence_index(x, x).item() == pytest.approx(1.0, abs=1e-6)
    assert coherence_index(x, torch.zeros_like(x)).item() == pytest.approx(0.0, abs=1e-3)


def test_coherence_index_with_sparse_codes() -> None:
    """
    With sparse codes: perfect reconstruction (explained=1.0) is multiplied
    by (1 - sparsity). Codes with 4/32 active -> sparsity 0.125 ->
    1.0 * 0.875 = 0.875.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    codes = torch.zeros(8, 32)
    codes[:, :4] = 1.0

    assert feature_sparsity_ratio(codes).item() == pytest.approx(0.125, abs=1e-6)
    assert coherence_index(x, x, codes).item() == pytest.approx(0.875, abs=1e-6)


# ---------------------------------------------------------------------------
# variety_attenuation_proxy
# ---------------------------------------------------------------------------

def test_attenuation_proxy_monotonic_inverse_pr() -> None:
    """
    Lower continuous PR (more variety risk) yields HIGHER attenuation.
    Recorded: low PR 30 -> 0.687826; high PR 400 -> 0.473333.
    """
    low_pr = variety_attenuation_proxy(
        torch.tensor(30.0), torch.tensor(0.5), torch.tensor(0.3)
    ).item()
    high_pr = variety_attenuation_proxy(
        torch.tensor(400.0), torch.tensor(0.5), torch.tensor(0.3)
    ).item()

    assert low_pr == pytest.approx(0.687826, abs=1e-5)
    assert high_pr == pytest.approx(0.473333, abs=1e-5)
    assert low_pr > high_pr


def test_attenuation_proxy_range() -> None:
    """
    Output is clamped to [0, 1]. All-max attenuation -> 0.4; all-min
    attenuation -> 0.6 (alpha=0.4 weights PR, so full PR-attenuation can
    never drive the score to 0).
    """
    all_max = variety_attenuation_proxy(
        torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0)
    ).item()
    all_min = variety_attenuation_proxy(
        torch.tensor(1e9), torch.tensor(0.0), torch.tensor(0.0)
    ).item()

    assert all_max == pytest.approx(0.4, abs=1e-5)
    assert all_min == pytest.approx(0.6, abs=1e-5)
    assert 0.0 <= all_max <= 1.0
    assert 0.0 <= all_min <= 1.0


# ---------------------------------------------------------------------------
# local_participation_ratio
# ---------------------------------------------------------------------------

def test_local_participation_ratio_shape_and_values() -> None:
    """
    (2, 10, 16) -> (2, 10). SURPRISE: the per-position PR is computed over a
    chunk pooled across the batch, then expanded -- both batch rows are
    identical (recorded batch0 == batch1).
    """
    torch.manual_seed(0)
    acts = torch.randn(2, 10, 16)
    lpr = local_participation_ratio(acts)

    assert lpr.shape == (2, 10)
    assert torch.equal(lpr[0], lpr[1])  # batch-expanded scalar, not per-batch

    # Recorded first-row values (seed 0):
    # [5.774345, 7.203494, 8.698192, 9.786089, 10.354174,
    #  10.642627, 10.818333, 9.575674, 8.377236, 7.331086]
    assert lpr[0, 0].item() == pytest.approx(5.774345, abs=1e-5)
    assert lpr[0, -1].item() == pytest.approx(7.331086, abs=1e-5)
    assert torch.all(lpr >= 0.0)
    assert torch.all(lpr <= 16.0)


def test_local_participation_ratio_short_sequence() -> None:
    """
    (2, 1, 16): seq length < 2 -> fills with float(d) = 16.0.
    """
    lpr = local_participation_ratio(torch.randn(2, 1, 16))
    assert lpr.shape == (2, 1)
    assert lpr[0, 0].item() == pytest.approx(16.0)


def test_local_participation_ratio_requires_3d() -> None:
    with pytest.raises(ValueError):
        local_participation_ratio(torch.randn(8, 16))
