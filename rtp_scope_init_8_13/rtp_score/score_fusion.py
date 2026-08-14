"""Numerically stable score fusion used by encoder and decoder ranking."""

from typing import Optional

from torch import Tensor


def geometric_score_fusion(
    cls_score: Tensor,
    quality_score: Optional[Tensor] = None,
    unique_score: Optional[Tensor] = None,
    *,
    cls_exp: float = 1.0,
    quality_exp: float = 0.0,
    unique_exp: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """Fuse probability-like scores in log space.

    Component tensors only need to be broadcast-compatible. Disabled
    components must have exponent zero and may be ``None``.
    """
    exponents = (float(cls_exp), float(quality_exp), float(unique_exp))
    if min(exponents) < 0:
        raise ValueError("score-fusion exponents must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if quality_exp > 0 and quality_score is None:
        raise ValueError("quality_score is required when quality_exp > 0")
    if unique_exp > 0 and unique_score is None:
        raise ValueError("unique_score is required when unique_exp > 0")

    log_score = cls_score.clamp_min(eps).log() * cls_exp
    if quality_exp:
        log_score = log_score + quality_score.clamp_min(eps).log() * quality_exp
    if unique_exp:
        log_score = log_score + unique_score.clamp_min(eps).log() * unique_exp
    return log_score.exp()


__all__ = ["geometric_score_fusion"]
