from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    sparse_codes: torch.Tensor
    reconstructed: torch.Tensor
    l0: float = 0.0
    recon_error: float = 0.0
    reconstruction_error: float = 0.0


class BaseSAE(nn.Module, abc.ABC):
    d_model: int
    dict_size: int
    W_dec: nn.Parameter

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decode(self, sparse_codes: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> SAEOutput:
        pass


class SAELensWrapper(BaseSAE):
    def __init__(self, sae_lens_obj: Any, device: torch.device):
        super().__init__()
        self.sae = sae_lens_obj.to(device)
        self.device = device

        w_dec = None
        if hasattr(self.sae, "W_dec"):
            w_dec = self.sae.W_dec
        elif hasattr(self.sae, "decoder") and hasattr(self.sae.decoder, "weight"):
            w_dec = self.sae.decoder.weight.T
        elif hasattr(self.sae, "_W_dec"):
            w_dec = self.sae._W_dec
        elif hasattr(self.sae, "cfg") and hasattr(self.sae.cfg, "W_dec"):
            w_dec = self.sae.cfg.W_dec

        if w_dec is None:
            raise AttributeError(
                "Could not locate W_dec decoder matrix. "
                "Available attrs: " + str([a for a in dir(self.sae) if "dec" in a.lower() or "w_" in a.lower()])
            )

        self.W_dec = w_dec.detach().float() if isinstance(w_dec, torch.Tensor) else nn.Parameter(torch.eye(1536))
        self.d_model = getattr(self.sae, "d_in", None) or getattr(self.sae.cfg, "d_in", self.W_dec.shape[1])
        self.dict_size = getattr(self.sae, "d_sae", None) or getattr(self.sae.cfg, "d_sae", self.W_dec.shape[0])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.sae, "encode"):
            return self.sae.encode(x)
        return torch.matmul(x.float(), self.W_dec.T)

    def decode(self, sparse_codes: torch.Tensor) -> torch.Tensor:
        if hasattr(self.sae, "decode"):
            return self.sae.decode(sparse_codes)
        return torch.matmul(sparse_codes.float(), self.W_dec)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        orig_dtype = x.dtype
        x_float = x.float()

        codes = self.encode(x)
        recon = self.decode(codes).to(dtype=orig_dtype)

        l0 = (codes.abs() > 1e-6).float().sum(dim=-1).mean().item()
        recon_err = (x_float - recon.float()).norm(dim=-1).mean() / (x_float.norm(dim=-1).mean() + 1e-8)

        return SAEOutput(
            sparse_codes=codes,
            reconstructed=recon,
            l0=l0,
            recon_error=recon_err,
            reconstruction_error=recon_err
        )


class DummySAE(BaseSAE):
    def __init__(self, d_model: int = 768, dict_size: int = 2048, device: Optional[torch.device] = None):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.device = device or torch.device("cpu")

        W = torch.randn(dict_size, d_model, device=self.device)
        self.W_dec = nn.Parameter(F.normalize(W, dim=-1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = F.normalize(x.float(), dim=-1)
        return torch.matmul(norm_x, self.W_dec.T)

    def decode(self, sparse_codes: torch.Tensor) -> torch.Tensor:
        return torch.matmul(sparse_codes.float(), self.W_dec)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        orig_dtype = x.dtype
        x_float = x.float()

        codes = self.encode(x_float)
        top_codes, _ = torch.topk(codes, k=min(32, codes.shape[-1]), dim=-1)
        thresh = top_codes[..., -1:]
        sparse_codes = torch.where(codes >= thresh, codes, torch.zeros_like(codes))

        recon = self.decode(sparse_codes).to(dtype=orig_dtype)
        l0 = (sparse_codes.abs() > 1e-6).float().sum(dim=-1).mean().item()
        recon_err = (x_float - recon.float()).norm(dim=-1).mean() / (x_float.norm(dim=-1).mean() + 1e-8)

        return SAEOutput(
            sparse_codes=sparse_codes,
            reconstructed=recon,
            l0=l0,
            recon_error=recon_err,
            reconstruction_error=recon_err
        )


def load_sae(
    release_or_path: str = "gpt2-small-res-jb",
    layer: int = 7,
    d_model: int = 768,
    device: Optional[torch.device] = None,
    hook_name: Optional[str] = None,
) -> BaseSAE:
    """
    Factory loader for SAELens or Dummy SAE models.

    Args:
        hook_name: SAE hook point. Defaults to 'hook_resid_post' for most models,
                   but 'hook_resid_pre' for GPT-2-small in SAELens.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if release_or_path == "dummy":
        return DummySAE(d_model=d_model, device=device)

    # Auto-detect hook name if not provided
    if hook_name is None:
        if "gpt2" in release_or_path.lower():
            hook_name = "hook_resid_pre"
        else:
            hook_name = "hook_resid_post"

    try:
        from sae_lens import SAE
        sae_obj = SAE.from_pretrained(
            release=release_or_path,
            sae_id=f"blocks.{layer}.{hook_name}",
            device=str(device),
        )
        return SAELensWrapper(sae_obj, device=device)
    except Exception as e:
        print(f"[sae_loader] Warning: Could not load SAELens model ({e}). Falling back to DummySAE.")
        return DummySAE(d_model=d_model, device=device)


def load_sae_from_hf_repo(
    repo_id: str,
    layer: int,
    d_model: int,
    device: Optional[torch.device] = None,
) -> BaseSAE:
    """Load SAE from a HuggingFace repo using SAELens v2+ API."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_id = f"layer_{layer}/final_sae"

    try:
        from sae_lens import SAE
        sae_obj = SAE.from_pretrained(
            release=repo_id,
            sae_id=sae_id,
            device=str(device),
        )
        print(f"[sae_loader] Loaded layer {layer} from {repo_id}")
        return SAELensWrapper(sae_obj, device=device)
    except Exception as e:
        print(f"[sae_loader] HF repo load failed for layer {layer}: {e}")
        return DummySAE(d_model=d_model, dict_size=16384, device=device)


def _normalize_model_key(name: str) -> str:
    return name.lower().strip()


def load_sae_map_for_model(
    model_name: str,
    model: nn.Module,
    layers: List[int],
    device: Optional[torch.device] = None,
) -> Dict[int, BaseSAE]:
    device = device or next(model.parameters()).device
    sae_map: Dict[int, BaseSAE] = {}

    d_model = getattr(model.config, "hidden_size", 1536)

    hf_repo_map = {
        _normalize_model_key("Qwen/Qwen2.5-1.5B"): "HuggingAnalist/sae-qwen2.5-1.5B-res",
        _normalize_model_key("Qwen/Qwen2.5-1.5B-Instruct"): "HuggingAnalist/sae-qwen2.5-1.5B-res",
    }

    release_map = {
        _normalize_model_key("gpt2"): "gpt2-small-res-jb",
    }

    model_key = _normalize_model_key(model_name)
    repo_id = hf_repo_map.get(model_key)
    release = release_map.get(model_key)

    if repo_id is not None:
        print(f"[sae_loader] Loading SAEs from HuggingFace repo: {repo_id}")
        print(f"[sae_loader] Target layers: {layers}")
        for layer in layers:
            sae = load_sae_from_hf_repo(repo_id, layer, d_model, device=device)
            sae_map[layer] = sae
        return sae_map

    if release is not None:
        print(f"[sae_loader] Loading SAEs from SAELens release: {release}")
        for layer in layers:
            sae = load_sae(release_or_path=release, layer=layer, d_model=d_model, device=device)
            sae_map[layer] = sae
        return sae_map

    print(f"[sae_loader] No pretrained SAE source for '{model_name}' (key='{model_key}'). Using DummySAE.")
    for layer in layers:
        sae_map[layer] = DummySAE(d_model=d_model, dict_size=16384, device=device)
    return sae_map
