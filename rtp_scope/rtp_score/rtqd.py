"""Rotated Threshold Quality Distribution (RTQD)."""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils.tensor_utils import connected_zero


class RotatedThresholdQualityHead(nn.Module):
    """Predict probabilities of crossing several rotated-IoU thresholds."""

    def __init__(
        self,
        embed_dims: int = 256,
        thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
        tau: float = 0.05,
    ) -> None:
        super().__init__()
        thresholds = tuple(float(value) for value in thresholds)
        if not thresholds or tuple(sorted(thresholds)) != thresholds:
            raise ValueError("RTQD thresholds must be non-empty and ordered")
        if thresholds[0] < 0 or thresholds[-1] > 1:
            raise ValueError("RTQD thresholds must lie in [0, 1]")
        if tau <= 0:
            raise ValueError("RTQD tau must be positive")
        self.thresholds = thresholds
        self.tau = float(tau)
        self.net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, len(thresholds)),
        )

    def forward(self, query_features: Tensor) -> Tensor:
        return self.net(query_features)

    def soft_targets(self, rotated_iou: Tensor) -> Tensor:
        thresholds = rotated_iou.new_tensor(self.thresholds)
        return torch.sigmoid(
            (rotated_iou.unsqueeze(-1) - thresholds) / self.tau
        )

    def loss(
        self,
        quality_logits: Tensor,
        positive_indices: Sequence[Tensor],
        positive_ious: Sequence[Tensor],
        *,
        loss_weight: float = 1.0,
        monotonic_weight: float = 0.1,
    ) -> Tuple[Tensor, Tensor, dict]:
        """Compute positive-only distribution and monotonicity losses."""
        selected_logits = []
        selected_targets = []
        for image_index, (indices, ious) in enumerate(
            zip(positive_indices, positive_ious)
        ):
            if indices.numel():
                selected_logits.append(quality_logits[image_index, indices])
                selected_targets.append(self.soft_targets(ious.detach()))

        if selected_logits:
            logits = torch.cat(selected_logits, dim=0)
            targets = torch.cat(selected_targets, dim=0)
            raw_loss = F.binary_cross_entropy_with_logits(logits, targets)
            probabilities = logits.sigmoid()
            raw_mono = F.relu(
                probabilities[..., 1:] - probabilities[..., :-1]
            ).mean()
            violation = (
                probabilities[..., 1:] > probabilities[..., :-1]
            ).float().mean()
            positive_means = probabilities.mean(dim=0)
        else:
            raw_loss = connected_zero(quality_logits)
            raw_mono = connected_zero(quality_logits)
            violation = quality_logits.new_zeros(())
            positive_means = quality_logits.new_zeros(
                (quality_logits.size(-1),)
            )

        all_means = quality_logits.sigmoid().mean(dim=(0, 1))
        diagnostics = {
            "q50_mean_pos": positive_means[0].detach(),
            "q50_mean_all": all_means[0].detach(),
            "quality_monotonic_violation_rate": violation.detach(),
        }
        for index, threshold in enumerate(self.thresholds[1:], start=1):
            name = f"q{int(round(threshold * 100))}_mean_pos"
            diagnostics[name] = positive_means[index].detach()
        return (
            raw_loss * float(loss_weight),
            raw_mono * float(monotonic_weight),
            diagnostics,
        )


__all__ = ["RotatedThresholdQualityHead"]
