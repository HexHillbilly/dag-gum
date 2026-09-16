"""
Lightweight logging of AVG interventions.

Records residual change and next-token distribution shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InterventionRecord:
    layer_idx: int
    action: str
    strength: float
    mode: str
    residual_delta_norm: float
    residual_cosine: float
    kl_next_token: Optional[float] = None
    logit_shift_norm: Optional[float] = None
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class InterventionLogger:
    def __init__(self):
        self.records: List[InterventionRecord] = []

    def clear(self) -> None:
        self.records.clear()

    def log(self, record: InterventionRecord) -> None:
        self.records.append(record)

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"n_interventions": 0}

        deltas = [r.residual_delta_norm for r in self.records]
        kls = [r.kl_next_token for r in self.records if r.kl_next_token is not None]

        return {
            "n_interventions": len(self.records),
            "mean_residual_delta": sum(deltas) / len(deltas),
            "max_residual_delta": max(deltas),
            "mean_kl_next_token": (sum(kls) / len(kls)) if kls else None,
            "actions": [r.action for r in self.records],
            "layers": [r.layer_idx for r in self.records],
        }

    def as_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "layer": r.layer_idx,
                "action": r.action,
                "strength": r.strength,
                "mode": r.mode,
                "residual_delta_norm": r.residual_delta_norm,
                "residual_cosine": r.residual_cosine,
                "kl_next_token": r.kl_next_token,
                "logit_shift_norm": r.logit_shift_norm,
                "message": r.message,
            }
            for r in self.records
        ]


def measure_residual_change(
    original: torch.Tensor,
    intervened: torch.Tensor,
) -> tuple[float, float]:
    orig = original.float().reshape(-1, original.shape[-1])
    inter = intervened.float().reshape(-1, intervened.shape[-1])

    delta = (inter - orig).norm(dim=-1).mean().item()
    cos = F.cosine_similarity(orig, inter, dim=-1).mean().item()
    return delta, cos


def measure_next_token_shift(
    unembed: nn.Module,
    original_residual: torch.Tensor,
    intervened_residual: torch.Tensor,
) -> Tuple[float, float]:
    weight = getattr(unembed, "weight", None)
    target_dtype = weight.dtype if weight is not None else original_residual.dtype
    target_device = weight.device if weight is not None else original_residual.device

    orig = original_residual.detach()[:, -1, :].to(device=target_device, dtype=target_dtype)
    inter = intervened_residual.detach()[:, -1, :].to(device=target_device, dtype=target_dtype)

    logits_orig = unembed(orig)
    logits_inter = unembed(inter)

    probs_orig = F.softmax(logits_orig, dim=-1)
    probs_inter = F.softmax(logits_inter, dim=-1)

    kl = F.kl_div(
        probs_inter.log().clamp(min=-100),
        probs_orig,
        reduction="batchmean",
    ).item()

    shift = (logits_orig - logits_inter).norm(p=2, dim=-1).mean().item()

    return kl, shift
