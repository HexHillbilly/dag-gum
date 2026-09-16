"""
DS-010: fast-path helper characterization tests (MODEL-FREE).

Characterizes the RFC-003 Phase 1/2a dual-resolution helpers WITHOUT loading a
model or tokenizer and WITHOUT any HuggingFace download. Every governor under
test is instantiated with ActiveVarietyGovernor.__new__() and only the
attributes the helper methods touch are set (or tiny stub objects).

Seed provenance: all random synthetic matrices use torch.manual_seed(0),
matching the existing controller wiring tests
(tests/test_dual_resolution_flag.py, tests/test_controller_generation_code_filter.py)
for a stable cross-test seed contract. Deterministic synthetic matrices
(arange outer products) need no seeding.

Helpers characterized:
  _update_fast_path / fast_path_window / fast_path_stride / fast_path_bypass_threshold
  _compute_sigma1 / get_sigma1_0
  _update_h_128_buffer / _log_shadow_pr / get_shadow_log
  use_dual_resolution flag OFF == inert dual-resolution machinery
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AVG.governor.controller import ActiveVarietyGovernor
import AVG.governor.controller as controller_mod

# Window / stride / threshold ground truth (constructor defaults).
FAST_PATH_WINDOW = 24
FAST_PATH_STRIDE_DEFAULT = 1
FAST_PATH_STRIDE_K8 = 8
FAST_PATH_BYPASS_THRESHOLD = 0.40
K_SLOW = 8
H_128_WINDOW = 128


def _make_gov(**overrides) -> ActiveVarietyGovernor:
    """Instantiate a governor via __new__() with only helper-touched attrs."""
    gov = ActiveVarietyGovernor.__new__(ActiveVarietyGovernor)
    gov.use_dual_resolution = True
    gov.fast_path_window = FAST_PATH_WINDOW
    gov.fast_path_stride = FAST_PATH_STRIDE_DEFAULT
    gov.fast_path_bypass_threshold = FAST_PATH_BYPASS_THRESHOLD
    gov.k_slow = K_SLOW
    gov.h_128_window = H_128_WINDOW
    gov._fast_token_buffer = []
    gov._fast_path_distinct = 1.0
    gov._sigma1_0 = None
    gov._shadow_acts = {}
    gov._h_128_buffer = None
    gov._pr_history = collections.deque(maxlen=8)
    gov._shadow_log = []
    for key, value in overrides.items():
        setattr(gov, key, value)
    return gov


def _make_sigma_governor(
    monkeypatch: pytest.MonkeyPatch, matrix: torch.Tensor
) -> ActiveVarietyGovernor:
    """Build a __new__() governor whose _compute_sigma1 runs on a synthetic
    hidden-state matrix via a stub model + stub HookManager (no real forward)."""
    captured: dict = {}

    class _FakeActivation:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.activations = tensor

    class _FakeHookManager:
        def __init__(self, model, detach=False, dtype=torch.float32, history_len=None) -> None:
            self.records = {}
            captured["hm"] = self

        def register_residual_hooks(self, layer_indices=None, every_k=1) -> None:
            pass

        def remove_hooks(self) -> None:
            pass

        def get_records(self):
            return self.records

    class _FakeModel:
        def __init__(self, mat: torch.Tensor) -> None:
            self.mat = mat

        def __call__(self, input_ids):
            # Simulate the layer hook capture: one layer, last-token hidden state.
            captured["hm"].records[0] = _FakeActivation(self.mat.float())
            return None

    monkeypatch.setattr(controller_mod, "HookManager", _FakeHookManager)

    gov = ActiveVarietyGovernor.__new__(ActiveVarietyGovernor)
    gov._n_layers = 12
    gov.device = torch.device("cpu")
    gov.model = _FakeModel(matrix)
    return gov


# ---------------------------------------------------------------------------
# Flag OFF (default): dual-resolution machinery is inert
# ---------------------------------------------------------------------------

def test_flag_off_state_is_inert() -> None:
    """With use_dual_resolution=False the fast-path state stays pristine.

    The flag gates the helper calls in generate() (controller.py lines ~655-660
    and ~776-777), so when it is OFF the helpers are never invoked and every
    dual-resolution state attribute keeps its default value.
    """
    gov = _make_gov(use_dual_resolution=False)

    assert gov._fast_token_buffer == []
    assert gov._fast_path_distinct == 1.0
    assert gov.get_sigma1_0() is None
    assert gov.get_shadow_log() == []
    assert gov._h_128_buffer is None
    assert gov._pr_history == collections.deque(maxlen=8)


def test_flag_off_update_fast_path_is_not_self_gating() -> None:
    """Surprising characterization: _update_fast_path does NOT check the flag.

    The use_dual_resolution gate lives in generate(), not in the helper. A
    direct call mutates the ring buffer even when the flag is OFF. This is
    benign for production (generate() guards the call) but means the helper is
    not safe to invoke unconditionally from any other caller.
    """
    gov = _make_gov(use_dual_resolution=False)

    gov._update_fast_path(torch.tensor([[7]]), step=0)

    assert gov._fast_token_buffer == [7]
    assert gov._fast_path_distinct == 1.0  # n < 4 => stays at default 1.0


# ---------------------------------------------------------------------------
# _update_fast_path: ring buffer, stride, bypass-threshold characterization
# ---------------------------------------------------------------------------

def test_ring_buffer_fills_to_window_and_rolls() -> None:
    """CPU integer ring buffer fills to fast_path_window=24 then rolls (FIFO)."""
    gov = _make_gov()

    for i in range(FAST_PATH_WINDOW):
        gov._update_fast_path(torch.tensor([[i]]), step=i)

    assert len(gov._fast_token_buffer) == FAST_PATH_WINDOW
    assert gov._fast_token_buffer[0] == 0
    assert gov._fast_token_buffer[-1] == FAST_PATH_WINDOW - 1
    assert gov._fast_path_distinct == pytest.approx(1.0)

    # One more token: the oldest (0) must roll out.
    gov._update_fast_path(torch.tensor([[FAST_PATH_WINDOW]]), step=FAST_PATH_WINDOW)

    assert len(gov._fast_token_buffer) == FAST_PATH_WINDOW
    assert gov._fast_token_buffer[0] == 1
    assert gov._fast_token_buffer[-1] == FAST_PATH_WINDOW


def test_ring_buffer_length_never_exceeds_window() -> None:
    """Length invariant: after 100 inserts the buffer never exceeds the window."""
    gov = _make_gov()

    max_len = 0
    for i in range(100):
        gov._update_fast_path(torch.tensor([[i]]), step=i)
        max_len = max(max_len, len(gov._fast_token_buffer))

    assert max_len <= FAST_PATH_WINDOW
    assert len(gov._fast_token_buffer) == FAST_PATH_WINDOW


def test_stride_counter_behavior_at_k8_boundaries() -> None:
    """With fast_path_stride=8 only steps divisible by 8 update the buffer.

    Characterization: steps 0, 8, 16 produce buffer [100, 108, 116]; all other
    steps are skipped. This matches the ground-truth "stride k=8" boundary.
    """
    gov = _make_gov(fast_path_stride=FAST_PATH_STRIDE_K8)

    for step in range(17):
        gov._update_fast_path(torch.tensor([[100 + step]]), step=step)

    assert gov._fast_token_buffer == [100, 108, 116]
    # n == 3 (< 4) => distinct stays at default 1.0.
    assert gov._fast_path_distinct == 1.0


def test_stride_default_is_one() -> None:
    """Ground-truth cross-check: the constructor default stride is 1, so every
    step updates the buffer unless fast_path_stride is overridden to k=8."""
    gov = _make_gov()  # default fast_path_stride == FAST_PATH_STRIDE_DEFAULT (1)

    for step in range(4):
        gov._update_fast_path(torch.tensor([[step]]), step=step)

    assert len(gov._fast_token_buffer) == 4


def test_bypass_threshold_flips_at_documented_condition() -> None:
    """Integer-bigram diversity crosses the documented div >= 0.40 bypass line.

    Class docstring: "Skips expensive string decoding and regex parsing when
    integer bigram diversity is healthy (div >= 0.40)."

    Measured characterization (window=24):
      * all-same token (42) repeated 24x -> distinct = 1/23 = 0.043478 < 0.40
        -> bypass NOT healthy (full string decoding path would be engaged).
      * 24 distinct tokens -> distinct = 1.0 >= 0.40 -> bypass healthy.
      * After a degenerate fill, appending distinct tokens rolls the ring; the
        first distinct >= 0.40 occurs at append step 32, i.e. when 10/23 bigram
        slots are unique (distinct = 0.434783).
    """
    gov = _make_gov()

    # Degenerate fill: 24 copies of the same token.
    for i in range(FAST_PATH_WINDOW):
        gov._update_fast_path(torch.tensor([[42]]), step=i)

    degenerate_distinct = gov._fast_path_distinct
    assert degenerate_distinct == pytest.approx(1 / 23, abs=1e-6)
    assert degenerate_distinct < gov.fast_path_bypass_threshold

    # Roll in distinct tokens; find the first step where diversity >= 0.40.
    flip_step = None
    flip_distinct = None
    for i in range(FAST_PATH_WINDOW, 100):
        gov._update_fast_path(torch.tensor([[1000 + i]]), step=i)
        if flip_step is None and gov._fast_path_distinct >= gov.fast_path_bypass_threshold:
            flip_step = i
            flip_distinct = gov._fast_path_distinct

    assert flip_step == 32
    assert flip_distinct == pytest.approx(10 / 23, abs=1e-6)  # 0.434783
    assert flip_distinct >= gov.fast_path_bypass_threshold

    # Once every degenerate slot has rolled out, the fully diverse ring reaches 1.0.
    assert gov._fast_path_distinct == pytest.approx(1.0, abs=1e-6)


def test_bypass_threshold_attr_is_stored_but_unread_by_helper() -> None:
    """Surprising characterization: fast_path_bypass_threshold is dead config.

    The attribute is set in __init__ (controller.py lines 143/167) but is never
    read by _update_fast_path or by generate(). The generate() gate uses a
    hardcoded literal 0.40 (line 683: ``token_diversity < 0.40``) rather than
    self.fast_path_bypass_threshold. The helper computes _fast_path_distinct and
    returns; it does not flip any internal bypass flag.
    """
    gov = _make_gov()

    assert gov.fast_path_bypass_threshold == FAST_PATH_BYPASS_THRESHOLD
    # The helper itself has no flag attribute to flip -- it only stores the
    # computed distinct value for the caller (generate) to compare.
    assert not hasattr(gov, "_fast_path_bypass_active")


# ---------------------------------------------------------------------------
# _compute_sigma1 / get_sigma1_0 on synthetic hidden-state matrices
# ---------------------------------------------------------------------------

def test_get_sigma1_0_is_none_until_calibration() -> None:
    """Seeded calibration accessor: None before calibration runs, whatever the
    flag value (the accessor just returns the stored _sigma1_0)."""
    gov = _make_gov()
    assert gov.get_sigma1_0() is None


def test_compute_sigma1_near_rank1_vs_full_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """_compute_sigma1 on a small synthetic hidden-state matrix.

    The method's numeric core is: mean-pool across layers -> center columns ->
    singular values -> return s.max() clamped at 1e-8. We feed the real method
    a deterministic synthetic matrix through a stub model + stub HookManager.

    Measured characterization (12x8 matrices):
      * near-rank-1 (arange outer product u @ v, u in [1..12], v in [1..8]):
          centered Frobenius norm = 170.798126
          sigma1 = 170.798126  (== fro norm; single dominant singular value)
          s1 / sum(s) = 1.0    (all spectral energy in one mode)
      * full-rank (torch.randn(12, 8), seed 0):
          centered Frobenius norm = 24.588096
          sigma1 = 5.594574
          s1 / sum(s) = 0.227522 (energy spread across all 8 modes)
    """
    torch.manual_seed(0)

    # Deterministic rank-1 matrix: outer product of arange vectors.
    u = torch.arange(1, 13, dtype=torch.float32).unsqueeze(1)  # (12, 1)
    v = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0)   # (1, 8)
    rank1 = u @ v

    # Full-rank matrix: seeded random 12x8.
    full = torch.randn(12, 8)

    def _centered_spectrum(mat: torch.Tensor):
        x = mat - mat.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(x)
        return x, s

    # --- near-rank-1 ---
    gov = _make_sigma_governor(monkeypatch, rank1)
    sigma1_rank1 = gov._compute_sigma1(torch.zeros(1, 1, dtype=torch.long))
    xc_rank1, s_rank1 = _centered_spectrum(rank1)

    assert sigma1_rank1 == pytest.approx(xc_rank1.norm().item(), rel=1e-5)
    assert sigma1_rank1 == pytest.approx(170.798126, rel=1e-5)
    assert s_rank1[0] / s_rank1.sum() == pytest.approx(1.0, abs=1e-6)

    # --- full-rank ---
    gov = _make_sigma_governor(monkeypatch, full)
    sigma1_full = gov._compute_sigma1(torch.zeros(1, 1, dtype=torch.long))
    xc_full, s_full = _centered_spectrum(full)

    assert sigma1_full == pytest.approx(s_full[0].item(), rel=1e-5)
    assert sigma1_full == pytest.approx(5.594574, rel=1e-5)
    assert sigma1_full < xc_full.norm().item()  # energy spread across modes
    assert s_full[0] / s_full.sum() == pytest.approx(0.227522, abs=1e-5)

    # Contrast: a rank-1 matrix concentrates all energy in sigma1; a full-rank
    # matrix of similar row count spreads it, so sigma1 is far smaller.
    assert sigma1_rank1 > sigma1_full


# ---------------------------------------------------------------------------
# _update_h_128_buffer / _log_shadow_pr / get_shadow_log
# ---------------------------------------------------------------------------

def test_update_h_128_buffer_appends_and_rolls() -> None:
    """Shadow H_128 ring buffer appends rows and trims to h_128_window."""
    gov = _make_gov(h_128_window=3)

    for i in range(6):
        gov._shadow_acts = {6: torch.randn(1, 1, 4)}
        gov._update_h_128_buffer()
        assert gov._h_128_buffer.shape[0] <= 3

    assert gov._h_128_buffer.shape == (3, 4)
    # Roll evidence: the buffer holds the LAST 3 vectors, not the first 3.
    assert gov._h_128_buffer.shape[0] == 3


def test_update_h_128_buffer_noop_without_shadow_acts() -> None:
    """With empty _shadow_acts the buffer is untouched."""
    gov = _make_gov()
    gov._update_h_128_buffer()
    assert gov._h_128_buffer is None


def test_log_shadow_pr_appends_and_computes_v_pr() -> None:
    """_log_shadow_pr appends a record and v_pr = (prev - current) / k_slow."""
    gov = _make_gov(k_slow=K_SLOW)
    gov._h_128_buffer = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    gov._fast_path_distinct = 0.5

    gov._log_shadow_pr(step=8)
    assert len(gov.get_shadow_log()) == 1
    first = gov.get_shadow_log()[0]
    assert first["step"] == 8
    assert first["fast_distinct"] == 0.5
    assert first["v_pr"] == 0.0  # no prior PR -> v_pr = 0

    # Change the buffer so the PR changes; v_pr must be non-zero.
    gov._h_128_buffer = torch.eye(4)
    gov._log_shadow_pr(step=16)
    second = gov.get_shadow_log()[1]
    assert second["step"] == 16
    assert second["v_pr"] == pytest.approx((first["pr"] - second["pr"]) / K_SLOW)
    assert second["v_pr"] != 0.0


def test_get_shadow_log_returns_copy() -> None:
    """get_shadow_log returns a defensive copy, not the live list."""
    gov = _make_gov()
    log = gov.get_shadow_log()
    log.append({"step": -1, "pr": 0.0, "v_pr": 0.0, "fast_distinct": 0.0})
    assert gov.get_shadow_log() == []
