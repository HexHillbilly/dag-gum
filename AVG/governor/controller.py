"""
Active Variety Governor (AVG) – Low-Latency Production Controller.

Features:
- Prompt-Isolated $N$-gram Loop Detection: Evaluates token repetition strictly
  over newly generated tokens to prevent prompt options/formatting contamination.
- Fast Integer Token Proxy: Skips expensive string decoding and regex parsing
  when integer bigram diversity is healthy (div >= 0.40).
- Sub-sampled CTR Evaluation: Restricts string CTR decoding strictly to every_k steps.
- Lazy Hook Registration: Kept unattached during dormant steps to eliminate forward-hook tax.
- Joint Conjunction Collapse Predicate: (token_diversity < 0.30 AND trailing_ctr < 0.30).
- Persistence Hysteresis & Calibrated L2 Impulse: Bounded [0.35, 0.75] L2.
"""

from __future__ import annotations

import re
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from AVG.core.hooks import HookManager
from AVG.core.sae_loader import BaseSAE
from AVG.core.causal_probe import CausalProbe
from AVG.core.intervention_log import (
    InterventionLogger,
    InterventionRecord,
)
from AVG.governor.profiler import VarietyProfiler, VarietyProfile
from AVG.core.metrics import (
    compute_coherent_token_ratio,
    compute_non_linguistic_density,
    is_code_syntax_context,
    participation_ratio,
    numeric_token_fraction,
)


# B1 numeric-immunity threshold (PROPOSAL value — owner signs into the
# protected smoke script; the agent does not author or change it).
NUMERIC_IMMUNITY_THRESHOLD = 0.80


class FailureMode(Enum):
    NONE = auto()
    HALLUCINATION_RISK = auto()
    ADVERSARIAL_SPOOF = auto()
    UNKNOWN = auto()


@dataclass
class BaselineStats:
    mean_attenuation: Dict[int, float] = field(default_factory=dict)
    std_attenuation: Dict[int, float] = field(default_factory=dict)
    mean_pr: Dict[int, float] = field(default_factory=dict)
    p10_attenuation: Dict[int, float] = field(default_factory=dict)
    p90_attenuation: Dict[int, float] = field(default_factory=dict)

    def is_calibrated(self) -> bool:
        return len(self.mean_attenuation) > 0


@dataclass
class InterventionDecision:
    layer_idx: int
    mode: FailureMode
    action: str
    strength: float
    message: str = ""


def compute_token_distinct_2_fast(
    input_ids: torch.Tensor,
    prompt_len: int = 0,
    window_len: int = 24
) -> Tuple[float, List[int]]:
    """
    Fast, vectorized bigram diversity calculator.
    Falls back gracefully to full trailing context if generated tokens < 4,
    eliminating cold-start blind spots on degenerate prompts.
    """
    # 1. Prefer generated tokens, but fall back to trailing input_ids if generated sequence is cold
    if input_ids.shape[-1] - prompt_len >= 4:
        tokens = input_ids[0, prompt_len:][-window_len:]
    else:
        tokens = input_ids[0, -window_len:]

    n = tokens.numel()
    if n < 4:
        return 1.0, []

    # 2. Vectorized bigram uniqueness via GPU stack
    t1 = tokens[:-1]
    t2 = tokens[1:]
    bigrams = torch.stack([t1, t2], dim=1)
    n_unique = torch.unique(bigrams, dim=0).shape[0]
    dist_2 = n_unique / float(n - 1)

    # 3. Vectorized repeated-token detection
    unique_vals, counts = torch.unique(tokens, return_counts=True)
    repeated_mask = counts >= 2
    repeated_ids = unique_vals[repeated_mask]
    repeated_ids_list = repeated_ids.tolist() if repeated_ids.numel() > 0 else []

    return dist_2, repeated_ids_list


def sample_top_p(logits: torch.Tensor, top_p: float = 0.85, filter_value: float = -1e4) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
    return logits


class ActiveVarietyGovernor:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        layer_indices: Optional[Sequence[int]] = None,
        every_k: int = 2,
        sae_map: Optional[Dict[int, BaseSAE]] = None,
        hallucination_z: float = -3.5,
        spoof_z: float = 2.0,
        base_strength: float = 0.45,
        max_strength: float = 0.75,
        projection_k: int = 32,
        noise_scale: float = 0.05,
        device: Optional[torch.device] = None,
        use_causal_filter: bool = True,
        causal_kl_threshold: float = 0.01,
        history_len: int = 16,
        cooldown_steps: int = 8,
        top_p: float = 0.85,
        suppression_strength: float = 5.0,
        eos_guard: bool = True,
        eos_release_after: int = 0,
        band_low: float = 8.216097,
        non_linguistic_density_threshold: float = 0.45,
        use_code_filter: bool = False,
        use_dual_resolution: bool = False,
        fast_path_window: int = 24,
        fast_path_stride: int = 1,
        fast_path_bypass_threshold: float = 0.40,
        k_slow: int = 8,
        h_128_window: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_indices = layer_indices
        self.every_k = every_k
        self.sae_map = sae_map or {}
        self.hallucination_z = hallucination_z
        self.spoof_z = spoof_z
        self.base_strength = base_strength
        self.max_strength = max_strength
        self.projection_k = projection_k
        self.noise_scale = noise_scale
        self.device = device or next(model.parameters()).device
        # PARKED per RFC-004 Amendment A4 — CausalProbe not
        # instantiated on this model. Retained for cross-model
        # transfer reactivation checklist.
        self.use_causal_filter = use_causal_filter
        self.history_len = history_len
        self.cooldown_steps = cooldown_steps
        self.top_p = top_p
        self.suppression_strength = suppression_strength
        self.eos_guard = eos_guard
        self.eos_release_after = eos_release_after
        self.band_low = band_low
        self.non_linguistic_density_threshold = non_linguistic_density_threshold
        self._eos_token_id = tokenizer.eos_token_id if tokenizer is not None else None
        self.use_code_filter = use_code_filter
        self.use_dual_resolution = use_dual_resolution
        self.fast_path_window = fast_path_window
        self.fast_path_stride = fast_path_stride
        self.fast_path_bypass_threshold = fast_path_bypass_threshold
        self.k_slow = k_slow
        self.h_128_window = h_128_window

        self.profiler = VarietyProfiler(
            model,
            layer_indices=layer_indices,
            every_k=every_k,
            sae_map=self.sae_map,
            device=self.device,
        )
        self.hook_manager = HookManager(model, detach=False, history_len=history_len)
        self.baseline = BaselineStats()
        self._last_profile: Optional[VarietyProfile] = None
        self._last_decisions: List[InterventionDecision] = []
        self._n_layers = self._count_layers()
        self._consecutive_interventions: Dict[int, int] = {}
        self._active_suppress_tokens: Dict[int, int] = {}
        self._kickstart_counter: int = 0
        self._eos_release_counter: int = 0
        self._eos_released: bool = False

        # RFC-003 Phase 1: dual-resolution state
        self._fast_token_buffer: List[int] = []
        self._fast_path_distinct: float = 1.0
        self._sigma1_0: Optional[float] = None

        # Dual-predicate persistence counters (RFC-004 Amendment A3)
        self._bigram_ctr: int = 0
        self._spectral_ctr: int = 0
        self._symbol_ctr: int = 0
        self._symbol_suppress_counter: int = 0

        # B2: which predicate fired last (populated by _evaluate_collapse;
        # feeds the per-generation "collapse" telemetry record).
        self._last_bigram_fire: bool = False
        self._last_spectral_fire: bool = False
        self._last_symbol_fire: bool = False

        # Layer-2 spectral PR hook state (DS-034b parity PASS, 0/93)
        self._layer2_pr: Optional[float] = None
        self._layer2_pr_buffer: List[torch.Tensor] = []
        self._layer2_pr_handle: Any = None

        # RFC-003 Phase 2a: shadow-mode PR-entropy logger state
        self._shadow_hook_handles: List[Any] = []
        self._shadow_acts: Dict[int, torch.Tensor] = {}
        self._h_128_buffer: Optional[torch.Tensor] = None
        self._pr_history: deque = deque(maxlen=8)
        self._shadow_log: List[Dict[str, Any]] = []

        self.logger = InterventionLogger()
        self._causal_probe: Optional[CausalProbe] = None

        self._prose_token_ids: Set[int] = set()
        self._non_prose_token_ids: Set[int] = set()
        self._non_ascii_token_ids: Set[int] = set()
        self._non_ascii_mask: Optional[torch.Tensor] = None
        if tokenizer is not None:
            self._cache_vocabulary_subsets()

    def _cache_vocabulary_subsets(self) -> None:
        """Cache prose/non-prose token ID subsets.

        Deprecated for kickstart targeting per night-009:
        94% of escape tokens (newlines, spaces, periods, EOS)
        are classified as non-prose and crushed by the -1e4
        penalty. Retained for diagnostic vocabulary analysis.
        Kickstart SHOULD retarget to active_loop_ids.
        """
        if self.tokenizer is None:
            return
        try:
            vocab_size = len(self.tokenizer)
        except TypeError:
            vocab_size = getattr(self.tokenizer, "vocab_size", 151646)
        vowels = set("aeiouyAEIOUY")

        for tid in range(vocab_size):
            try:
                tok_str = self.tokenizer.decode([tid])
            except Exception:
                continue

            clean_str = tok_str.strip()

            if not tok_str.isascii():
                self._non_ascii_token_ids.add(tid)

            if clean_str in ("a", "I", "A") or (
                len(clean_str) >= 2
                and all(c.isalpha() or c in ("'", "-") for c in clean_str)
                and any(c in vowels for c in clean_str)
            ):
                self._prose_token_ids.add(tid)
            else:
                self._non_prose_token_ids.add(tid)

        if self._non_ascii_token_ids:
            # Size the mask to the MODEL's output vocab (the logits dim), which
            # can exceed len(tokenizer) (e.g. Qwen2.5-1.5B: 151936 vs 151665).
            # Token ids are always < len(tokenizer), so the extra tail stays False.
            logits_vocab = vocab_size
            cfg = getattr(self.model, "config", None)
            if cfg is not None and getattr(cfg, "vocab_size", 0) > vocab_size:
                logits_vocab = cfg.vocab_size
            mask = torch.zeros(logits_vocab, dtype=torch.bool, device=self.device)
            idx = torch.tensor(
                sorted(self._non_ascii_token_ids), dtype=torch.long, device=self.device
            )
            mask[idx] = True
            self._non_ascii_mask = mask

    def _count_layers(self) -> int:
        for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return len(obj)
            except AttributeError:
                continue
        return 12

    def _get_layer_blocks(self):
        """Return the transformer layer ModuleList for hook registration."""
        candidates = [
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "model.decoder.layers",
            "transformer.layers",
        ]
        for path in candidates:
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 2:
                    return obj
            except AttributeError:
                continue

        for name, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and 4 < len(module) < 200:
                return module
        raise RuntimeError("Could not locate transformer layer stack.")

    def _dual_resolution_layer_span(self) -> List[int]:
        """Return layer indices covering the 55–85% depth span."""
        start = int(self._n_layers * 0.55)
        end = int(self._n_layers * 0.85)
        return list(range(start, max(start + 1, end + 1)))

    def _compute_sigma1(self, input_ids: torch.Tensor) -> float:
        """Compute the dominant singular value of the prompt hidden-state matrix."""
        layer_indices = self._dual_resolution_layer_span()
        hm = HookManager(self.model, detach=True, dtype=torch.float32)
        hm.register_residual_hooks(layer_indices=layer_indices, every_k=1)
        try:
            with torch.no_grad():
                _ = self.model(input_ids=input_ids)
            records = hm.get_records()
        finally:
            hm.remove_hooks()

        if not records:
            return 0.0

        acts = []
        for idx in sorted(records.keys()):
            act = records[idx].activations.float()
            if act.device != self.device:
                act = act.to(self.device)
            acts.append(act)

        # Mean-pool across the selected layers: (L, batch, seq, d) -> (batch, seq, d)
        stacked = torch.stack(acts, dim=0)
        pooled = stacked.mean(dim=0)

        # Reshape to (N, d_model)
        x = pooled.reshape(-1, pooled.shape[-1])
        n, d = x.shape
        compute_device = torch.device("cpu") if (n * d > 1_000_000) else x.device
        x = x.to(compute_device, dtype=torch.float32)
        x = x - x.mean(dim=0, keepdim=True)

        try:
            s = torch.linalg.svdvals(x)
        except RuntimeError:
            _, s, _ = torch.svd_lowrank(x, q=min(n, d, 64))
            s = s.clamp(min=0.0)

        s = s.clamp(min=1e-8)
        return s.max().item()

    # ------------------------------------------------------------------
    # RFC-003 Phase 2a: shadow-mode PR-entropy logger
    # ------------------------------------------------------------------
    def _register_shadow_hooks(self) -> None:
        """Register lightweight forward hooks on the 55-85% layer span.

        Shadow mode strictly: these hooks only record activations; they never
        modify the residual stream or trigger interventions.
        """
        blocks = self._get_layer_blocks()
        layer_indices = self._dual_resolution_layer_span()
        self._shadow_hook_handles = []
        self._shadow_acts = {}

        def make_hook(layer_idx: int):
            def hook_fn(module: nn.Module, input_args: Tuple[Any, ...], output: Any) -> Any:
                if isinstance(output, tuple):
                    residual = output[0]
                else:
                    residual = output
                # Record only the last-token hidden state to minimize overhead.
                self._shadow_acts[layer_idx] = residual[:, -1:, :].detach().float()
                return output
            return hook_fn

        for idx in layer_indices:
            if 0 <= idx < len(blocks):
                handle = blocks[idx].register_forward_hook(make_hook(idx))
                self._shadow_hook_handles.append(handle)

    def _remove_shadow_hooks(self) -> None:
        """Remove all registered shadow-mode hooks.

        DS-005: called from generate()'s finally block so shadow hooks are
        ALWAYS removed, even when generation throws mid-run. Must stay
        idempotent (safe to call with an empty handle list).
        """
        for handle in self._shadow_hook_handles:
            handle.remove()
        self._shadow_hook_handles = []
        self._shadow_acts = {}

    def _update_h_128_buffer(self) -> None:
        """Append the latest mean-pooled hidden state to the H_128 ring buffer."""
        if not self._shadow_acts:
            return

        acts = [self._shadow_acts[idx] for idx in sorted(self._shadow_acts.keys())]
        # (L, batch, 1, d_model) -> (batch, 1, d_model)
        stacked = torch.stack(acts, dim=0)
        pooled = stacked.mean(dim=0)
        # (1, d_model)
        vector = pooled.squeeze(1)

        if self._h_128_buffer is None:
            self._h_128_buffer = vector
        else:
            self._h_128_buffer = torch.cat([self._h_128_buffer, vector], dim=0)
            if self._h_128_buffer.shape[0] > self.h_128_window:
                self._h_128_buffer = self._h_128_buffer[-self.h_128_window :]

    def _log_shadow_pr(self, step: int) -> None:
        """Compute PR(H_128) and v_PR and append to the shadow log."""
        if self._h_128_buffer is None or self._h_128_buffer.shape[0] < 2:
            return

        h = self._h_128_buffer.unsqueeze(0)
        pr = participation_ratio(h).item()

        v_pr = 0.0
        if self._pr_history:
            prev_pr = self._pr_history[-1]
            v_pr = (prev_pr - pr) / float(self.k_slow)

        self._pr_history.append(pr)
        self._shadow_log.append(
            {
                "step": step,
                "pr": pr,
                "v_pr": v_pr,
                "fast_distinct": self._fast_path_distinct,
            }
        )

    def get_shadow_log(self) -> List[Dict[str, Any]]:
        return list(self._shadow_log)

    def calibrate(self, trusted_ids: Sequence[torch.Tensor]) -> BaselineStats:
        atten_lists: Dict[int, List[float]] = {}
        pr_lists: Dict[int, List[float]] = {}

        for ids in trusted_ids:
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            profile = self.profiler.profile(ids.to(self.device))
            for layer, stats in profile.stats.items():
                atten_lists.setdefault(layer, []).append(stats.attenuation)
                pr_lists.setdefault(layer, []).append(stats.participation_ratio)

        baseline = BaselineStats()
        for layer, vals in atten_lists.items():
            t = torch.tensor(vals, dtype=torch.float32)
            baseline.mean_attenuation[layer] = t.mean().item()
            baseline.std_attenuation[layer] = t.std(unbiased=False).item() + 1e-6
            baseline.p10_attenuation[layer] = torch.quantile(t, 0.10).item()
            baseline.p90_attenuation[layer] = torch.quantile(t, 0.90).item()
            if layer in pr_lists:
                baseline.mean_pr[layer] = torch.tensor(pr_lists[layer]).mean().item()

        self.baseline = baseline

        # RFC-003 Phase 1: seeded startup sigma_1^0 calibration
        if self.use_dual_resolution and trusted_ids:
            ids = trusted_ids[0]
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            self._sigma1_0 = self._compute_sigma1(ids.to(self.device))

        return baseline

    def diagnose(
        self,
        profile: VarietyProfile,
        input_ids: Optional[torch.Tensor] = None,
        token_diversity: float = 1.0,
        trailing_ctr: float = 1.0,
        is_code_context: bool = False,
    ) -> List[InterventionDecision]:
        decisions: List[InterventionDecision] = []
        if not self.baseline.is_calibrated():
            return decisions

        mid_start = int(self._n_layers * 0.55)

        # 1. DUAL-GATE IMMUNITY & STATE RESET
        if trailing_ctr >= 0.75 and token_diversity >= 0.35:
            self._consecutive_interventions = {}
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            return decisions

        # 2. PATH 2 STRUCTURAL CODE EXCLUSION
        if self.use_code_filter and is_code_context:
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            self._consecutive_interventions = {}
            return decisions

        # 3. DUAL-PREDICATE COLLAPSE DETECTION (RFC-004 Amendment A3)
        bigram_collapse = (
            token_diversity < 0.30
        ) and (
            trailing_ctr < 0.30
        )
        if bigram_collapse:
            self._bigram_ctr += 1
        else:
            self._bigram_ctr = 0
        bigram_fire = self._bigram_ctr >= 2

        spectral_collapse = (
            self._layer2_pr is not None
            and self._layer2_pr < 8.216097
        )
        if spectral_collapse:
            self._spectral_ctr += 1
        else:
            self._spectral_ctr = 0
        if self._bigram_ctr < 2:
            spectral_fire = (
                self._spectral_ctr >= 2
                and token_diversity < 0.40
            )
        else:
            spectral_fire = self._spectral_ctr >= 2

        if is_code_context:
            spectral_fire = False

        is_collapsed = bigram_fire or spectral_fire
        if not is_collapsed:
            self._consecutive_interventions = {}
            return decisions

        candidate_layers = []
        for layer in profile.monitored_layers:
            stats = profile.stats[layer]
            mean_a = self.baseline.mean_attenuation.get(layer)
            std_a = self.baseline.std_attenuation.get(layer)

            if mean_a is None or std_a is None:
                continue

            if layer >= mid_start:
                candidate_layers.append(layer)

        if candidate_layers:
            target_layer = max(candidate_layers)
            prev_steps = self._consecutive_interventions.get(target_layer, 0)

            target_impulse = min(self.max_strength, self.base_strength + (prev_steps * 0.05))
            self._consecutive_interventions[target_layer] = prev_steps + 1

            if self.use_code_filter:
                message = (
                    f"L{target_layer}: Fast Impulse (div={token_diversity:.2f}, "
                    f"ctr={trailing_ctr:.2f}, bigram={self._bigram_ctr}, spectral={self._spectral_ctr}, "
                    f"code_context={is_code_context})"
                )
            else:
                message = (
                    f"L{target_layer}: Fast Impulse (div={token_diversity:.2f}, "
                    f"ctr={trailing_ctr:.2f}, bigram={self._bigram_ctr}, spectral={self._spectral_ctr})"
                )

            decisions.append(
                InterventionDecision(
                    layer_idx=target_layer,
                    mode=FailureMode.HALLUCINATION_RISK,
                    action="sae_guided_reset",
                    strength=target_impulse,
                    message=message,
                )
            )

        self._last_decisions = decisions
        return decisions

    def _evaluate_collapse(
        self,
        token_diversity: float,
        trailing_ctr: float,
        is_code_context: bool,
    ) -> bool:
        """A3 dual-predicate collapse detection, extracted from diagnose().

        B2: returns ``is_collapsed`` as a bool WITHOUT touching residual
        actuation. Mirrors the A3 predicate section of diagnose() exactly.
        Records which predicate fired (``_last_bigram_fire`` /
        ``_last_spectral_fire``) to feed the "collapse" telemetry record.
        Cheap (no SVD); called every decode step, replacing the
        is_profile_step gate for collapse detection.
        """
        # 1. dual-gate immunity & state reset
        if trailing_ctr >= 0.75 and token_diversity >= 0.35:
            self._consecutive_interventions = {}
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            self._last_bigram_fire = False
            self._last_spectral_fire = False
            return False

        # 2. code exclusion
        if self.use_code_filter and is_code_context:
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            self._consecutive_interventions = {}
            self._last_bigram_fire = False
            self._last_spectral_fire = False
            return False

        # 3. bigram predicate
        if (token_diversity < 0.30) and (trailing_ctr < 0.30):
            self._bigram_ctr += 1
        else:
            self._bigram_ctr = 0
        bigram_fire = self._bigram_ctr >= 2

        # 4. spectral predicate (band_low — per-model valve, default 8.216097 = 1.5B-derived)
        spectral_collapse = (
            self._layer2_pr is not None
            and self._layer2_pr < self.band_low
        )
        if spectral_collapse:
            self._spectral_ctr += 1
        else:
            self._spectral_ctr = 0
        if self._bigram_ctr < 2:
            spectral_fire = (
                self._spectral_ctr >= 2
                and token_diversity < 0.40
            )
        else:
            spectral_fire = self._spectral_ctr >= 2

        if is_code_context:
            spectral_fire = False

        self._last_bigram_fire = bigram_fire
        self._last_spectral_fire = spectral_fire
        return bigram_fire or spectral_fire

    def _make_sae_guided_reset_fn(
        self,
        sae: Optional[BaseSAE],
        strength: float = 0.45,
        decision: Optional[InterventionDecision] = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        warnings.warn(
            "_make_sae_guided_reset_fn is deprecated per "
            "RFC-004 Amendment A4 — residual perturbation path "
            "inert under greedy decoding on Qwen2.5-1.5B. "
            "Retained for cross-model transfer reactivation "
            "checklist.",
            DeprecationWarning,
            stacklevel=2,
        )
        def inject_guided_reset(residual: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                original = residual.detach()
                orig_dtype, orig_device = residual.dtype, residual.device
                res_flat = original[:, -1, :].float()
                res_norm = res_flat.norm(dim=-1, keepdim=True) + 1e-8

                guided_direction = None

                if sae is not None and hasattr(sae, "W_dec"):
                    try:
                        W_dec = sae.W_dec.detach().float().to(device=res_flat.device)
                        if W_dec.shape[0] != res_flat.shape[-1]:
                            W_dec = W_dec.T

                        norm_x = res_flat / res_norm
                        feature_norms = W_dec.norm(dim=-1, keepdim=True) + 1e-8
                        norm_dict = W_dec / feature_norms

                        cos_sims = torch.matmul(norm_dict, norm_x.T).squeeze(-1)
                        ortho_mask = cos_sims.abs() < 0.10
                        ortho_indices = torch.nonzero(ortho_mask).squeeze(-1)

                        if ortho_indices.numel() > 0:
                            idx = ortho_indices[torch.randint(0, ortho_indices.numel(), (1,)).item()]
                            guided_direction = W_dec[idx : idx + 1]
                    except Exception:
                        guided_direction = None

                if guided_direction is None or guided_direction.norm() < 1e-6:
                    raw_noise = torch.randn_like(res_flat)
                    dot_prod = torch.sum(raw_noise * res_flat, dim=-1, keepdim=True)
                    proj = (dot_prod / (res_norm ** 2)) * res_flat
                    guided_direction = raw_noise - proj

                dot_p = torch.sum(guided_direction * res_flat, dim=-1, keepdim=True)
                ortho_vec = guided_direction - (dot_p / (res_norm ** 2)) * res_flat
                ortho_unit = ortho_vec / (ortho_vec.norm(dim=-1, keepdim=True) + 1e-8)

                clamped_l2_force = max(0.35, min(0.75, strength))

                intervened_flat = res_flat + (clamped_l2_force * ortho_unit)
                intervened = original.clone()
                intervened[:, -1, :] = intervened_flat.to(dtype=orig_dtype)

                orig_last = original[:, -1, :].float()
                inter_last = intervened[:, -1, :].float()
                delta_l2 = (inter_last - orig_last).norm(dim=-1).mean().item()
                cos_sim = F.cosine_similarity(orig_last, inter_last, dim=-1).mean().item()

                self.logger.log(
                    InterventionRecord(
                        layer_idx=decision.layer_idx if decision else -1,
                        action="sae_guided_reset",
                        strength=clamped_l2_force,
                        mode=decision.mode.name if decision else "UNKNOWN",
                        residual_delta_norm=delta_l2,
                        residual_cosine=cos_sim,
                        message=decision.message if decision else "",
                    )
                )
                return intervened.to(device=orig_device, dtype=orig_dtype)

        return inject_guided_reset

    def apply_interventions(self, decisions: List[InterventionDecision]) -> None:
        self.hook_manager.clear_interventions()
        for dec in decisions:
            if dec.action in ("sae_guided_reset", "ortho_reset"):
                sae = self.sae_map.get(dec.layer_idx, None)
                fn = self._make_sae_guided_reset_fn(sae=sae, strength=dec.strength, decision=dec)
                self.hook_manager.register_intervention(dec.layer_idx, fn)

    def _update_fast_path(self, next_token: torch.Tensor, step: int) -> None:
        """Update the CPU token ring buffer and compute host-side Distinct-2.

        RFC-003 audit finding: AVG's generate() never transfers next_token to the
        host during autoregressive decoding; it is concatenated on-device. Because
        no per-token host transfer exists, this Fast-Path uses a stride-based
        fallback: a device-to-host copy of the latest token id and a CPU Distinct-2
        evaluation. The cost is measured by Gate 1.2.
        """
        if step % self.fast_path_stride != 0:
            return

        # Device-to-host copy of the single latest token id.
        token_id = int(next_token.item())
        self._fast_token_buffer.append(token_id)
        if len(self._fast_token_buffer) > self.fast_path_window:
            self._fast_token_buffer.pop(0)

        n = len(self._fast_token_buffer)
        if n < 4:
            self._fast_path_distinct = 1.0
            return

        tokens = self._fast_token_buffer
        bigrams = {(tokens[i], tokens[i + 1]) for i in range(n - 1)}
        distinct = len(bigrams) / float(n - 1)
        self._fast_path_distinct = distinct

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 64,
        do_sample: bool = True,
        temperature: float = 0.8,
        intervene: bool = True,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        if isinstance(prompt, str):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
        else:
            input_ids = prompt.to(self.device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

        if input_ids.shape[0] != 1:
            raise ValueError(
                f"B2 KV-cache decode assumes batch == 1; got batch = {input_ids.shape[0]}"
            )

        prompt_len = input_ids.shape[-1]
        self._consecutive_interventions = {}
        self._active_suppress_tokens.clear()
        self._kickstart_counter = 0
        self._eos_release_counter = 0
        self._eos_released = False
        numeric_immunity_steps = 0
        self.logger.clear()

        # B2 collapse telemetry (replaces the residual-path fire counts)
        collapse_record: Dict[str, Any] = {
            "bigram_fires": 0,
            "spectral_fires": 0,
            "symbol_fires": 0,
            "steps": [],
        }

        # RFC-003 Phase 1: reset per-generation dual-resolution state
        if self.use_dual_resolution:
            self._fast_token_buffer = []
            self._fast_path_distinct = 1.0
            self._h_128_buffer = None
            self._pr_history.clear()
            self._shadow_log = []

        # RFC-004 Amendment A3: reset per-generation detection state
        self._bigram_ctr = 0
        self._spectral_ctr = 0
        self._layer2_pr = None
        self._layer2_pr_buffer = []
        self._last_bigram_fire = False
        self._last_spectral_fire = False
        self._symbol_ctr = 0
        self._symbol_suppress_counter = 0
        self._last_symbol_fire = False

        try:
            # RFC-003 Phase 2a: register shadow-mode hooks when dual-resolution is active.
            if self.use_dual_resolution:
                self._register_shadow_hooks()

            # ------------------------------------------------------------------
            # B2: PRE-FILL (once, full prompt) — fixes rigor-review BLOCKER #1.
            # The layer-2 spectral PR hook is registered AFTER pre-fill so it
            # only sees single-token decode steps (hs.shape[1] == 1).
            # ------------------------------------------------------------------
            attention_mask = torch.ones_like(input_ids, device=self.device)
            position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :].clone()

            def _make_layer2_pr_hook():
                def hook_fn(module, input_args, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output
                    if hs.shape[1] == 1:
                        self._layer2_pr_buffer.append(
                            hs[:, -1:, :].detach().to(dtype=torch.float32)
                        )
                        if len(self._layer2_pr_buffer) >= 24:
                            win = torch.cat(
                                self._layer2_pr_buffer[-24:], dim=1
                            )
                            self._layer2_pr = float(
                                participation_ratio(win).item()
                            )
                    return output
                return hook_fn

            self._layer2_pr_handle = (
                self._get_layer_blocks()[2]
                .register_forward_hook(_make_layer2_pr_hook())
            )

            for step in range(max_new_tokens):
                # 1. FAST INTEGER-ONLY TOKEN PROXY (SCOPED TO GENERATED TOKENS)
                token_diversity, active_loop_ids = compute_token_distinct_2_fast(
                    input_ids, prompt_len=prompt_len, window_len=24
                )
                trailing_ctr = 1.0

                # Suppression state is armed only when intervene=True, so that
                # intervene=False is a true dormant (no-actuation) baseline
                # while the collapse telemetry below still records what the
                # governor WOULD have detected. (Fixes the dead-flag trap.)
                if intervene and active_loop_ids:
                    for tid in active_loop_ids:
                        self._active_suppress_tokens[tid] = self.cooldown_steps

                # EOS-release debounce (bounded release): count consecutive
                # loop-free steps; once eos_release_after clean steps elapse,
                # permanently stop masking <eos> for the rest of this
                # generation. Default 0 = disabled (mask while suppression is
                # active, the pre-release behavior).
                if self.eos_release_after > 0:
                    if active_loop_ids:
                        self._eos_release_counter = 0
                    else:
                        self._eos_release_counter += 1
                        if self._eos_release_counter >= self.eos_release_after:
                            self._eos_released = True

                # 2. SUB-SAMPLED STRING DECODING (Every K steps OR when token diversity drops)
                is_code_context = False
                if token_diversity < 0.40 or step % self.every_k == 0:
                    if self.tokenizer is not None:
                        recent_gen = input_ids[0, prompt_len:] if input_ids.shape[-1] > prompt_len else input_ids[0]
                        recent_tokens = recent_gen[-16:] if recent_gen.numel() > 0 else input_ids[0, -16:]
                        trailing_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
                        trailing_ctr = compute_coherent_token_ratio(trailing_text)
                        if self.use_code_filter:
                            is_code_context = is_code_syntax_context(trailing_text)

                # ------------------------------------------------------------------
                # B1: NUMERIC-CONTEXT IMMUNITY (regime-independent, every step)
                # ------------------------------------------------------------------
                is_numeric_context = False
                if self.tokenizer is not None:
                    # night-019 finding: use the FULL-CONTEXT window
                    # (input_ids[0, -16:]), not generated-only.
                    trailing_text = self.tokenizer.decode(
                        input_ids[0, -16:], skip_special_tokens=True
                    )
                    is_numeric_context = (
                        numeric_token_fraction(trailing_text) >= NUMERIC_IMMUNITY_THRESHOLD
                    )
                if is_numeric_context:
                    numeric_immunity_steps += 1
                    self._kickstart_counter = 0

                # ------------------------------------------------------------------
                # B2: COLLAPSE DETECTION (every step) — replaces the profile gate.
                # Skipped on numeric context to preserve B1 dormancy.
                # ------------------------------------------------------------------
                is_collapsed = False
                if not is_numeric_context:
                    is_collapsed = self._evaluate_collapse(
                        token_diversity, trailing_ctr, is_code_context
                    )
                    if self._last_bigram_fire:
                        collapse_record["bigram_fires"] += 1
                    if self._last_spectral_fire:
                        collapse_record["spectral_fires"] += 1
                collapse_record["steps"].append(
                    (step, is_collapsed, self._bigram_ctr, self._spectral_ctr)
                )

                # ------------------------------------------------------------------
                # HIGH-DIVERSITY (non-linguistic) collapse: emoji/symbol spam.
                # Distinct-token symbol floods pass the LOW-diversity detectors
                # (every spam token is unique), so key on CHARACTER density of
                # non-linguistic content (non-alphanumeric, non-whitespace) over
                # the decoded window — catches both non-ASCII emoji floods AND
                # dense ASCII symbol loops (----, ====, !?!?).
                # ------------------------------------------------------------------
                non_ling_density = compute_non_linguistic_density(
                    input_ids[0, prompt_len:], self.tokenizer
                )
                if non_ling_density > self.non_linguistic_density_threshold:
                    self._symbol_ctr += 1
                else:
                    self._symbol_ctr = 0
                self._last_symbol_fire = self._symbol_ctr >= 2
                if self._last_symbol_fire:
                    collapse_record["symbol_fires"] += 1
                    if intervene:
                        self._symbol_suppress_counter = self.cooldown_steps

                # ------------------------------------------------------------------
                # B2 §3.4: kickstart trigger — regime-independent (experiment:
                # re-enable under sampling; numeric immunity guards Number Loop).
                # ------------------------------------------------------------------
                if intervene and (
                    not is_numeric_context
                    and active_loop_ids
                    and (trailing_ctr < 0.50 or is_collapsed)
                ):
                    self._kickstart_counter = max(self._kickstart_counter, 3)

                # 3. LOGIT PENALTIES (suppression)
                if self._active_suppress_tokens:
                    for tid, steps_left in list(self._active_suppress_tokens.items()):
                        if steps_left > 0:
                            next_logits[:, tid] -= self.suppression_strength
                            self._active_suppress_tokens[tid] -= 1
                        else:
                            del self._active_suppress_tokens[tid]
                    # PROACTIVE (root cause): while suppressing a loop token,
                    # also constrain the logit vacuum to prose by suppressing
                    # the non-ASCII class, so 4-bit noise can't fill it with
                    # emoji/symbol spam in the first place.
                    if self._non_ascii_mask is not None:
                        next_logits[:, self._non_ascii_mask] -= self.suppression_strength

                # High-diversity symbol-class suppression (reactive safety net):
                # penalize the whole non-ASCII class (emoji/symbol/unicode), not
                # individual ids, for any non-suppression-induced symbol flood.
                if self._symbol_suppress_counter > 0:
                    next_logits[:, self._non_ascii_mask] -= self.suppression_strength
                    self._symbol_suppress_counter -= 1

                # EOS-guard: while suppression is active, mask <eos> so the logit
                # vacuum left by penalizing the loop token cannot be filled by early
                # termination (EOS-death). Released automatically once suppression
                # disarms (cooldown expires), or earlier via eos_release_after.
                # Gated on eos_guard + intervene (the suppress dict is only
                # populated when intervene=True).
                if (
                    self.eos_guard
                    and not self._eos_released
                    and (self._active_suppress_tokens or self._symbol_suppress_counter > 0)
                    and self._eos_token_id is not None
                ):
                    next_logits[:, self._eos_token_id] = -1e9

                # kickstart penalty (experiment: regime-independent, active_loop_ids)
                if self._kickstart_counter > 0 and active_loop_ids:
                    loop_tensor = torch.tensor(list(active_loop_ids), device=self.device)
                    if self._kickstart_counter == 3:
                        penalty = -1e4
                    elif self._kickstart_counter == 2:
                        penalty = -5.0
                    else:
                        penalty = -2.0
                    next_logits[:, loop_tensor] += penalty
                    self._kickstart_counter -= 1
                elif self._kickstart_counter > 0:
                    self._kickstart_counter -= 1

                # Apply top_p on EVERY sampling step, not only while suppressing.
                # Full-vocab sampling (the prior behavior) is looser than standard
                # top_p and lets small models (Phi-3-mini) drift off-task when the
                # governor is idle. Greedy (do_sample=False) still skips top_p.
                if do_sample:
                    next_logits = sample_top_p(next_logits, top_p=self.top_p)

                current_temp = 0.70 if (self._active_suppress_tokens or self._symbol_suppress_counter > 0) else temperature
                next_logits = next_logits / max(current_temp, 1e-5)

                if do_sample:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if self.use_dual_resolution:
                    self._update_fast_path(next_token, step)

                # ------------------------------------------------------------------
                # B2: SINGLE-TOKEN DECODE FORWARD (KV-cache)
                # ------------------------------------------------------------------
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device)], dim=-1
                )
                position_ids = torch.tensor(
                    [[prompt_len + step]], device=self.device
                )
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :].clone()

                # RFC-003 Phase 2a: update shadow H_128 buffer every step; log PR at stride.
                if self.use_dual_resolution:
                    self._update_h_128_buffer()
                    if step % self.k_slow == 0:
                        self._log_shadow_pr(step)

        finally:
            # DS-005: shadow hooks MUST always be removed, even on exception.
            self._remove_shadow_hooks()
            # Remove layer-2 spectral PR hook
            if self._layer2_pr_handle is not None:
                self._layer2_pr_handle.remove()
                self._layer2_pr_handle = None
            self.hook_manager.remove_hooks()
            self.hook_manager.clear_interventions()

        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True) if self.tokenizer else None
        gen_only_text = self.tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True) if self.tokenizer else None

        return {
            "sequences": input_ids,
            "text": text,
            "gen_only_text": gen_only_text,
            "prompt_len": prompt_len,
            # B2: residual path detached from generate() (Amendment A4).
            "profile": None,
            "decisions": [],
            "intervened": False,
            "intervention_log": [],
            "intervention_summary": {"n_interventions": 0},
            "numeric_immunity_steps": numeric_immunity_steps,
            "collapse": collapse_record,
        }

    def get_last_profile(self) -> Optional[VarietyProfile]:
        return self._last_profile

    def get_last_decisions(self) -> List[InterventionDecision]:
        return list(self._last_decisions)

    def get_intervention_log(self) -> List[Dict[str, Any]]:
        return self.logger.as_list()

    def get_intervention_summary(self) -> Dict[str, Any]:
        return self.logger.summary()

    def get_sigma1_0(self) -> Optional[float]:
        return self._sigma1_0
