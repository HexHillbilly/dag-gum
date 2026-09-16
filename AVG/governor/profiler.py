"""
Variety Profiler – collects per-layer continuous entropy proxies and sparse
feature statistics. Falls back to the HookManager ring buffer when the
incoming activation has sequence length 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from AVG.core.hooks import HookManager, LayerActivationRecord
from AVG.core.metrics import (
    coherence_index,
    feature_sparsity_ratio,
    participation_ratio,
    residual_reconstruction_error,
    singular_value_spectrum_entropy,
    variety_attenuation_proxy,
)
from AVG.core.sae_loader import BaseSAE, SAEOutput

@dataclass
class LayerVarietyStats:
    layer_idx: int
    participation_ratio: float
    spectrum_entropy: float
    sparsity: float = 0.0
    reconstruction_error: float = 0.0
    coherence: float = 0.0
    attenuation: float = 0.0
    n_tokens: int = 0

@dataclass
class VarietyProfile:
    stats: Dict[int, LayerVarietyStats] = field(default_factory=dict)
    monitored_layers: List[int] = field(default_factory=list)

    def attenuation_vector(self) -> torch.Tensor:
        if not self.monitored_layers:
            return torch.tensor([])
        return torch.tensor(
            [self.stats[i].attenuation for i in self.monitored_layers],
            dtype=torch.float32,
        )

    def pr_vector(self) -> torch.Tensor:
        return torch.tensor(
            [self.stats[i].participation_ratio for i in self.monitored_layers],
            dtype=torch.float32,
        )

class VarietyProfiler:
    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[Sequence[int]] = None,
        every_k: int = 4,
        sae_map: Optional[Dict[int, BaseSAE]] = None,
        device: Optional[torch.device] = None,
        use_local_pr: bool = False,
    ):
        self.model = model
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.every_k = every_k
        self.sae_map = sae_map or {}
        self.device = device or torch.device("cpu")
        self.use_local_pr = use_local_pr

        self.hook_manager = HookManager(model, detach=True, dtype=torch.float32)

    def profile(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **forward_kwargs,
    ) -> VarietyProfile:
        self.hook_manager.register_residual_hooks(
            layer_indices=self.layer_indices,
            every_k=self.every_k,
        )

        try:
            with torch.no_grad():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **forward_kwargs,
                )
            records = self.hook_manager.get_records()
        finally:
            self.hook_manager.remove_hooks()

        return self._compute_profile(records)

    def _compute_profile(
        self, records: Dict[int, LayerActivationRecord]
    ) -> VarietyProfile:
        profile = VarietyProfile()
        sorted_layers = sorted(records.keys())
        profile.monitored_layers = sorted_layers

        for layer_idx in sorted_layers:
            act = records[layer_idx].activations

            if act.shape[1] == 1:
                hist = self.hook_manager.get_history(layer_idx)
                if hist is not None and hist.shape[1] >= 2:
                    act = hist

            if act.device != self.device:
                act = act.to(self.device)

            pr = participation_ratio(act).item()
            entropy = singular_value_spectrum_entropy(act, normalize=True).item()

            sparsity = 0.0
            recon_err = 0.0
            coh = 0.0

            if layer_idx in self.sae_map:
                sae = self.sae_map[layer_idx]
                try:
                    sae_param = next(sae.parameters())
                    sae_device = sae_param.device
                    sae_dtype = sae_param.dtype
                except StopIteration:
                    sae_device = act.device
                    sae_dtype = torch.float32

                act_sae = act.to(device=sae_device, dtype=sae_dtype)
                with torch.no_grad():
                    out: SAEOutput = sae(act_sae)
                sparsity = feature_sparsity_ratio(out.sparse_codes).item()
                recon_err = residual_reconstruction_error(
                    act_sae, out.reconstructed
                ).item()
                coh = coherence_index(
                    act_sae, out.reconstructed, out.sparse_codes
                ).item()

            atten = variety_attenuation_proxy(
                continuous_pr=torch.tensor(pr),
                sparse_sparsity=torch.tensor(sparsity),
                reconstruction_err=torch.tensor(recon_err),
            ).item()

            profile.stats[layer_idx] = LayerVarietyStats(
                layer_idx=layer_idx,
                participation_ratio=pr,
                spectrum_entropy=entropy,
                sparsity=sparsity,
                reconstruction_error=recon_err,
                coherence=coh,
                attenuation=atten,
                n_tokens=act.shape[0] * act.shape[1],
            )

        return profile

    def profile_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> VarietyProfile:
        return self.profile(input_ids, attention_mask=attention_mask, **kwargs)
