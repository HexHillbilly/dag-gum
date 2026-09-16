"""

PARKED per RFC-004 Amendment A4 — residual path deprecated
for Qwen2.5-1.5B under greedy decoding. Retained for
cross-model transfer reactivation checklist.

Causal Spectator Probe for AVG.

Verifies whether active SAE features causally impact output probabilities
before residual projection interventions are allowed to fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalEffectRecord:
    is_causal: bool
    kl_divergence: float
    max_logit_shift: float
    message: str = ""


class CausalProbe:
    def __init__(
        self,
        unembed: nn.Module,
        kl_threshold: float = 0.01,
        logit_shift_threshold: float = 0.5,
        topk: int = 16,
        device: Optional[torch.device] = None,
    ):
        self.unembed = unembed
        self.kl_threshold = kl_threshold
        self.logit_shift_threshold = logit_shift_threshold
        self.topk = topk
        self.device = device

    @torch.inference_mode()
    def probe(
        self,
        residual: torch.Tensor,
        sparse_codes: torch.Tensor,
        sae_decode_fn: Callable[[torch.Tensor], torch.Tensor],
        mode: str = "ablate",
    ) -> CausalEffectRecord:
        """
        Ablates or isolates top-k active sparse codes and computes KL divergence
        on the unembedded output logits to verify feature causality.
        """
        weight = getattr(self.unembed, "weight", None)
        target_dtype = weight.dtype if weight is not None else residual.dtype
        target_device = weight.device if weight is not None else residual.device

        res_flat = residual.detach()[:, -1, :].to(device=target_device, dtype=target_dtype)
        logits_orig = self.unembed(res_flat)
        probs_orig = F.softmax(logits_orig, dim=-1)

        codes_mutated = sparse_codes.clone().detach()

        # Zero out top-k features to measure output distribution movement (ablation test)
        if codes_mutated.shape[-1] > self.topk:
            topk_vals, topk_idx = torch.topk(codes_mutated.abs(), k=self.topk, dim=-1)
            mask = torch.zeros_like(codes_mutated)
            mask.scatter_(-1, topk_idx, 1.0)
            if mode == "ablate":
                codes_mutated = codes_mutated * (1.0 - mask)
            else:
                codes_mutated = codes_mutated * mask

        res_mutated = sae_decode_fn(codes_mutated).to(device=target_device, dtype=target_dtype)
        if res_mutated.ndim == 3:
            res_mutated = res_mutated[:, -1, :]

        logits_mutated = self.unembed(res_mutated)
        probs_mutated = F.softmax(logits_mutated, dim=-1)

        kl = F.kl_div(
            probs_mutated.log().clamp(min=-100),
            probs_orig,
            reduction="batchmean",
        ).item()

        logit_shift = (logits_orig - logits_mutated).abs().max().item()
        is_causal = (kl >= self.kl_threshold) or (logit_shift >= self.logit_shift_threshold)

        return CausalEffectRecord(
            is_causal=is_causal,
            kl_divergence=kl,
            max_logit_shift=logit_shift,
            message=f"Probe mode={mode}: KL={kl:.4f}, max_shift={logit_shift:.4f}, causal={is_causal}",
        )
