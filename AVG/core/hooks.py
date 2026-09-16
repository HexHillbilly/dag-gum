"""
PyTorch forward-hook utilities for residual-stream extraction, ring-buffer
history (for auto-regressive seq_len=1 safety), and optional in-place intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerActivationRecord:
    layer_idx: int
    activations: torch.Tensor  # (batch, seq, hidden)
    modified: bool = False


@dataclass
class HookManager:
    model: nn.Module
    capture_input: bool = False
    detach: bool = True
    dtype: Optional[torch.dtype] = None
    history_len: int = 16
    _dormant: bool = field(default=False, repr=False)

    _handles: List[Any] = field(default_factory=list, init=False, repr=False)
    _records: Dict[int, LayerActivationRecord] = field(default_factory=dict, init=False, repr=False)
    _interventions: Dict[int, Callable[[torch.Tensor], torch.Tensor]] = field(default_factory=dict, init=False, repr=False)
    _history: Dict[int, List[torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    def set_dormant(self, dormant: bool) -> None:
        """Sleep or wake hook callbacks. When dormant, passthrough is ~free."""
        self._dormant = dormant

    def has_interventions(self) -> bool:
        return bool(self._interventions)

    def register_intervention(
        self, layer_idx: int, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self._interventions[layer_idx] = fn

    def clear_interventions(self) -> None:
        self._interventions.clear()

    def clear_records(self) -> None:
        self._records.clear()
        self._history.clear()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_records(self) -> Dict[int, LayerActivationRecord]:
        return self._records

    def get_history(self, layer_idx: int) -> Optional[torch.Tensor]:
        hist = self._history.get(layer_idx, [])
        if not hist:
            return None
        return torch.cat(hist, dim=1)

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook_fn(module: nn.Module, input_args: Tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                residual = output[0]
                rest = output[1:]
            else:
                residual = output
                rest = None

            # Interventions ALWAYS fire, even when dormant
            modified = False
            if layer_idx in self._interventions:
                residual = self._interventions[layer_idx](residual)
                modified = True
                if isinstance(output, tuple):
                    return (residual,) + rest
                return residual

            # TRACK B: Dormant passthrough — skip all recording/history overhead
            if self._dormant:
                return output

            act_to_store = residual.detach() if self.detach else residual
            if self.dtype is not None:
                act_to_store = act_to_store.to(dtype=self.dtype)

            self._records[layer_idx] = LayerActivationRecord(
                layer_idx=layer_idx,
                activations=act_to_store,
                modified=modified,
            )

            # Update auto-regressive ring buffer for seq_len=1 safety
            if layer_idx not in self._history:
                self._history[layer_idx] = []

            hist = self._history[layer_idx]
            last_token = act_to_store[:, -1:, :]

            # Defensive guard against intra-generation batch-size mismatches
            if hist and hist[0].shape[0] != last_token.shape[0]:
                hist.clear()

            hist.append(last_token)
            if len(hist) > self.history_len:
                hist.pop(0)

            if isinstance(output, tuple):
                return (residual,) + rest
            return residual

        return hook_fn

    def register_residual_hooks(
        self,
        layer_indices: Optional[List[int]] = None,
        every_k: int = 1,
        max_layers: Optional[int] = None,
    ) -> None:
        self.remove_hooks()
        self.clear_records()

        blocks = None
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
                    blocks = obj
                    break
            except AttributeError:
                continue

        if blocks is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and 4 < len(module) < 200:
                    blocks = module
                    break

        if blocks is None:
            raise RuntimeError(
                "Could not locate transformer layer stack. "
                "Extend discovery logic or pass modules manually."
            )

        n_layers = len(blocks)
        if max_layers is not None:
            n_layers = min(n_layers, max_layers)

        if layer_indices is None:
            layer_indices = list(range(0, n_layers, max(1, every_k)))

        for idx in layer_indices:
            if 0 <= idx < len(blocks):
                handle = blocks[idx].register_forward_hook(self._make_hook(idx))
                self._handles.append(handle)

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()
        self.clear_interventions()
