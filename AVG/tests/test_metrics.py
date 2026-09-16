"""
Unit tests for dimensionality / variety metrics and the DummySAE.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.core.metrics import (
    coherence_index,
    effective_rank,
    feature_sparsity_ratio,
    participation_ratio,
    residual_reconstruction_error,
    singular_value_spectrum_entropy,
    variety_attenuation_proxy,
)
from AVG.core.sae_loader import DummySAE, load_sae

@pytest.fixture
def isotropic_activations():
    torch.manual_seed(0)
    return torch.randn(32, 64)

@pytest.fixture
def low_rank_activations():
    torch.manual_seed(1)
    basis = torch.randn(64, 4)
    coeffs = torch.randn(32, 4)
    return coeffs @ basis.T

def test_pr_isotropic_near_full(isotropic_activations):
    pr = participation_ratio(isotropic_activations)
    assert pr.item() > 20.0
    assert pr.item() <= 64.0 + 1e-3

def test_pr_low_rank(low_rank_activations):
    pr = participation_ratio(low_rank_activations)
    assert 3.0 < pr.item() < 8.0

def test_pr_single_sample():
    x = torch.randn(1, 128)
    pr = participation_ratio(x)
    assert math.isclose(pr.item(), 128.0, rel_tol=1e-5)

def test_pr_shape_error():
    with pytest.raises(ValueError):
        participation_ratio(torch.randn(10))

def test_entropy_range(isotropic_activations):
    ent = singular_value_spectrum_entropy(isotropic_activations, normalize=True)
    assert 0.0 <= ent.item() <= 1.0 + 1e-5

def test_entropy_low_rank_lower(isotropic_activations, low_rank_activations):
    e_iso = singular_value_spectrum_entropy(isotropic_activations, normalize=True)
    e_lr = singular_value_spectrum_entropy(low_rank_activations, normalize=True)
    assert e_lr.item() < e_iso.item()

def test_effective_rank_low(low_rank_activations):
    r = effective_rank(low_rank_activations, threshold=0.05)
    assert 3 <= r.item() <= 6

def test_feature_sparsity_ratio():
    codes = torch.zeros(4, 100)
    codes[:, :5] = 1.0
    s = feature_sparsity_ratio(codes)
    assert math.isclose(s.item(), 0.05, abs_tol=1e-6)

def test_reconstruction_error_zero():
    x = torch.randn(8, 32)
    err = residual_reconstruction_error(x, x, relative=True)
    assert err.item() < 1e-6

def test_coherence_perfect():
    x = torch.randn(8, 32)
    c = coherence_index(x, x)
    assert c.item() > 0.99

def test_attenuation_proxy_monotonic():
    high_pr = variety_attenuation_proxy(
        continuous_pr=torch.tensor(400.0),
        sparse_sparsity=torch.tensor(0.5),
        reconstruction_err=torch.tensor(0.3),
    )
    low_pr = variety_attenuation_proxy(
        continuous_pr=torch.tensor(30.0),
        sparse_sparsity=torch.tensor(0.5),
        reconstruction_err=torch.tensor(0.3),
    )
    assert low_pr.item() > high_pr.item()

def test_dummy_sae_shapes():
    d_model, n_features = 64, 128
    sae = DummySAE(d_model=d_model, dict_size=n_features)
    x = torch.randn(2, 10, d_model)
    out = sae(x)
    assert out.sparse_codes.shape == (2, 10, n_features)
    assert out.reconstructed.shape == (2, 10, d_model)
    assert out.l0 > 0

def test_dummy_sae_sparsity():
    sae = DummySAE(d_model=32, dict_size=512)
    x = torch.randn(16, 32)
    out = sae(x)
    assert out.l0 == pytest.approx(32.0, abs=1e-6)

def test_load_sae_dummy_sentinel():
    sae = load_sae("dummy", d_model=32)
    assert isinstance(sae, DummySAE)

def test_end_to_end_metrics_with_sae():
    torch.manual_seed(7)
    d = 64
    x = torch.randn(20, d)
    sae = DummySAE(d_model=d, dict_size=128)
    out = sae(x)

    pr = participation_ratio(x)
    spars = feature_sparsity_ratio(out.sparse_codes)
    err = residual_reconstruction_error(x, out.reconstructed)
    coh = coherence_index(x, out.reconstructed, out.sparse_codes)
    atten = variety_attenuation_proxy(pr, spars, err)

    assert pr.ndim == 0
    assert 0.0 <= spars.item() <= 1.0
    assert err.item() >= 0.0
    assert 0.0 <= coh.item() <= 1.0
    assert 0.0 <= atten.item() <= 1.0
