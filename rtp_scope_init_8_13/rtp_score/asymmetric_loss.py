"""AMP-safe asymmetric multi-label classification loss."""

import torch.nn as nn
from torch import Tensor


class AsymmetricLoss(nn.Module):
    """Asymmetric BCE used for sparse image-level class presence targets."""

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if min(gamma_pos, gamma_neg, clip) < 0:
            raise ValueError("ASL gamma and clip values must be non-negative")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean, sum or none")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError("ASL logits and targets must have identical shapes")
        # The loss math remains in fp32 under autocast; gradients are cast back
        # by PyTorch at the input boundary.
        logits_f = logits.float()
        targets_f = targets.float()
        positive_prob = logits_f.sigmoid()
        negative_prob = 1.0 - positive_prob
        if self.clip:
            negative_prob = (negative_prob + self.clip).clamp(max=1.0)

        log_likelihood = (
            targets_f * positive_prob.clamp_min(self.eps).log()
            + (1.0 - targets_f) * negative_prob.clamp_min(self.eps).log()
        )
        if self.gamma_pos or self.gamma_neg:
            pt = (
                targets_f * positive_prob
                + (1.0 - targets_f) * negative_prob
            )
            gamma = (
                targets_f * self.gamma_pos
                + (1.0 - targets_f) * self.gamma_neg
            )
            log_likelihood = log_likelihood * (1.0 - pt).pow(gamma)

        loss = -log_likelihood
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


__all__ = ["AsymmetricLoss"]
