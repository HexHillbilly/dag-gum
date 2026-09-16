"""
Dimensionality and variety metrics for Active Variety Governor.

All SVD paths force float32 for numerical stability under mixed-precision
(FP16/BF16) inference and fall back to CPU for large matrices.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import re

CODE_DELIMITER_PATTERNS = [
    r"//",          # C/C++/Java/JS single-line comments
    r"/\*",        # Multi-line comment open
    r"#\s*%%",     # Jupyter / VSCode code cell breaks
    r"\bdef\b",    # Python function definition
    r"\blet\b",    # JS variable declaration
    r"\bconst\b",  # JS constant declaration
    r"\bvar\b",    # JS/Go variable declaration
    r"\bstruct\b", # C/C++/Rust struct keyword
    r"\btypedef\b",# C/C++ typedef keyword
    r"```",         # Markdown code fences
]

CODE_REGEX = re.compile("|".join(CODE_DELIMITER_PATTERNS))


def is_code_syntax_context(text: str) -> bool:
    """
    Returns True if the trailing window contains structural code indicators,
    allowing the governor to bypass false-positive residual resets.
    """
    if not text:
        return False
    return bool(CODE_REGEX.search(text))


def numeric_token_fraction(text: str) -> float:
    """
    Fraction of whitespace-separated tokens containing at least one digit.

    Named numeric-immunity predicate (B1). Mirrors the smoke script's local
    _numeric_token_fraction. Kept separate from is_code_syntax_context on
    purpose — code context and numeric context are distinct signals.
    """
    tokens = text.strip().split()
    if not tokens:
        return 0.0
    numeric = sum(1 for tok in tokens if any(ch.isdigit() for ch in tok))
    return numeric / float(len(tokens))


def compute_coherent_token_ratio(text: str) -> float:
    """
    Language- and code-aware coherence metric (CTR).
    Long runs of single-character alphabetic tokens are treated as low coherence.
    """
    raw_tokens = text.strip().split()
    if not raw_tokens:
        return 0.0

    valid_count = 0
    single_char_count = 0

    for tok in raw_tokens:
        clean = tok.strip("(),;:{}[]\"'<>`#$%^&*=-+/")

        if len(clean) == 1 and clean.isalpha():
            single_char_count += 1
            continue

        if len(clean) >= 2 and all(c.isalpha() or c in ("'", "-") for c in clean):
            valid_count += 1
        elif any(
            "\u4e00" <= c <= "\u9fa5"
            or "\u3040" <= c <= "\u30ff"
            or "\uac00" <= c <= "\ud7af"
            for c in tok
        ):
            valid_count += 1
        elif re.match(r"^\$?[a-zA-Z_][a-zA-Z0-9_]*$", clean) and len(clean) >= 2:
            valid_count += 1
        elif tok in ("->", "==", "!=", "<=", ">=", "&&", "||", "::", "=>", "++", "--", "/*", "*/", "//"):
            valid_count += 1

    if single_char_count / float(len(raw_tokens)) > 0.5:
        return 0.15

    return min(1.0, valid_count / float(len(raw_tokens)))


def compute_non_linguistic_density(
    token_ids: torch.Tensor,
    tokenizer,
    window_size: int = 24,
) -> float:
    """Char-level density of non-linguistic content in the trailing window.

    Decodes the trailing `window_size` tokens and returns the fraction of
    characters that are neither alphanumeric nor whitespace (symbols, emoji,
    punctuation/divider runs) over the total decoded character count. Catches
    BOTH high-diversity failure modes: non-ASCII emoji floods AND dense ASCII
    symbol loops (----, ====, !?!?, markdown divider runs) whose bytes are all
    inside [0, 127] and therefore invisible to a token-class non-ASCII check.

    Whitespace is excluded from the numerator (it is neutral, not
    non-linguistic), so clean prose sits ~0.02-0.08 and symbol/emoji spam
    ~0.8-1.0 — a wide margin around the 0.45 firing threshold.
    """
    if token_ids is None or tokenizer is None:
        return 0.0
    ids = token_ids if token_ids.dim() == 1 else token_ids[0]
    trailing = ids[-window_size:]
    if trailing.numel() == 0:
        return 0.0
    text = tokenizer.decode(trailing.tolist(), skip_special_tokens=True)
    if not text:
        return 0.0
    # Defensive: a window that decodes to only whitespace/specials has no
    # linguistic content to measure — treat as zero density.
    if not text.strip():
        return 0.0
    non_ling = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return non_ling / float(len(text))


def participation_ratio(activations: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if activations.ndim < 2:
        raise ValueError("activations must have at least 2 dimensions")

    x = activations.reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
    n, d = x.shape

    if n < 2:
        return torch.tensor(float(d), device=activations.device, dtype=torch.float32)

    compute_device = torch.device("cpu") if (n * d > 1_000_000) else x.device
    x = x.to(compute_device)
    x = x - x.mean(dim=0, keepdim=True)

    try:
        s = torch.linalg.svdvals(x)
    except RuntimeError:
        _, s, _ = torch.svd_lowrank(x, q=min(n, d, 64))
        s = s.clamp(min=0.0)

    s = s.clamp(min=eps)
    pr = (s.sum() ** 2) / (s.pow(2).sum() + eps)
    return pr.to(activations.device)

def singular_value_spectrum_entropy(
    activations: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    if activations.ndim < 2:
        raise ValueError("activations must have at least 2 dimensions")

    x = activations.reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
    n, d = x.shape
    if n < 2:
        return torch.tensor(0.0 if normalize else math.log(max(d, 1)), device=activations.device)

    compute_device = torch.device("cpu") if (n * d > 1_000_000) else x.device
    x = x.to(compute_device)
    x = x - x.mean(dim=0, keepdim=True)

    try:
        s = torch.linalg.svdvals(x)
    except RuntimeError:
        _, s, _ = torch.svd_lowrank(x, q=min(n, d, 64))
        s = s.clamp(min=0.0)

    s = s.clamp(min=eps)
    p = s / (s.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()

    if normalize:
        rank = (s > eps * 10).sum().clamp(min=1)
        entropy = entropy / torch.log(rank.float() + eps)

    return entropy.to(activations.device)

def effective_rank(activations: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    if activations.ndim < 2:
        raise ValueError("activations must have at least 2 dimensions")

    x = activations.reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
    n, d = x.shape
    if n < 2:
        return torch.tensor(float(d), device=activations.device)

    compute_device = torch.device("cpu") if (n * d > 1_000_000) else x.device
    x = x.to(compute_device)
    x = x - x.mean(dim=0, keepdim=True)

    try:
        s = torch.linalg.svdvals(x)
    except RuntimeError:
        _, s, _ = torch.svd_lowrank(x, q=min(n, d, 64))
        s = s.clamp(min=0.0)

    energy = s.pow(2)
    total = energy.sum() + 1e-12
    cum = torch.cumsum(energy, dim=0) / total
    rank = (cum < (1.0 - threshold)).sum() + 1
    return rank.float().to(activations.device)

def feature_sparsity_ratio(sparse_codes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if sparse_codes.numel() == 0:
        return torch.tensor(0.0, device=sparse_codes.device)
    active = (sparse_codes.abs() > eps).float()
    return active.mean()

def residual_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    relative: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    diff = original.float() - reconstructed.float()
    err = torch.norm(diff, p=2, dim=-1).mean()
    if relative:
        denom = torch.norm(original.float(), p=2, dim=-1).mean() + eps
        err = err / denom
    return err

def coherence_index(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sparse_codes: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    residual = original.float() - reconstructed.float()
    var_orig = original.float().var(dim=-1, unbiased=False).mean() + eps
    var_res = residual.var(dim=-1, unbiased=False).mean()
    explained = 1.0 - (var_res / var_orig).clamp(0.0, 1.0)

    if sparse_codes is not None:
        sparsity = 1.0 - feature_sparsity_ratio(sparse_codes)
        explained = explained * sparsity

    return explained.clamp(0.0, 1.0)

def variety_attenuation_proxy(
    continuous_pr: torch.Tensor,
    sparse_sparsity: torch.Tensor,
    reconstruction_err: torch.Tensor,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> torch.Tensor:
    pr_norm = continuous_pr / (continuous_pr + 200.0)
    atten_from_pr = 1.0 - pr_norm
    atten_from_sparsity = 1.0 - sparse_sparsity
    atten_from_recon = 1.0 - reconstruction_err.clamp(0.0, 1.0)

    score = (
        alpha * atten_from_pr
        + beta * atten_from_sparsity
        + gamma * atten_from_recon
    )
    return score.clamp(0.0, 1.0)

def local_participation_ratio(
    activations: torch.Tensor,
    window: int = 8,
) -> torch.Tensor:
    if activations.ndim != 3:
        raise ValueError("Expected (batch, seq, hidden)")

    b, s, d = activations.shape
    if s < 2:
        return torch.full((b, s), float(d), device=activations.device, dtype=torch.float32)

    pad = window // 2
    padded = F.pad(activations, (0, 0, pad, pad), mode="replicate")

    prs = []
    for t in range(s):
        chunk = padded[:, t : t + window, :]
        pr = participation_ratio(chunk)
        prs.append(pr)
    pr_tensor = torch.stack(prs)
    return pr_tensor.unsqueeze(0).expand(b, -1)
