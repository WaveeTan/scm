"""Scale-dynamic regression loss for the 8/15 SCM + O²-RTDETR branch.

This is an OBB adaptation of the scale-dynamic supervision idea:
small objects receive relatively stronger center/location supervision,
while the balance gradually returns to 1:1 for larger objects.

Important:
- This is NOT a verbatim copy of SDIoU/SD Loss from another repository.
- It keeps the original O²-RTDETR KLD/GD loss untouched.
- It is intended to replace the normal matching-query L1 bbox term only.
- DN regression can still use the original L1 loss through the custom head.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RotatedScaleDynamicLoss(nn.Module):
    """Scale-dynamic L1-style regression loss for normalized OBB targets.

    Inputs are expected in normalized ``(cx, cy, w, h, angle)`` format.

    Let

        s = sqrt(w_gt * h_gt)

    and

        beta = delta * clamp((s / scale_ref) ** scale_power, 0, 1).

    We define

        lambda_loc   = 1 + delta - beta
        lambda_shape = 1 - delta + beta

    Therefore:
    - very small targets:
        lambda_loc   -> 1 + delta
        lambda_shape -> 1 - delta
    - targets with s >= scale_ref:
        lambda_loc   -> 1
        lambda_shape -> 1

    This preserves the overall O² regression structure while making the
    L1-style term explicitly scale sensitive.

    Args:
        delta:
            Maximum dynamic offset. ``0.5`` gives location/shape weights
            approximately ``1.5 / 0.5`` for extremely small targets.
        scale_ref:
            Normalized reference scale. ``0.12`` matches the upper boundary
            used by the 8/15 SCM scale grouping.
        scale_power:
            Controls how quickly the dynamic weights return to 1.
        angle_weight:
            Weight of the normalized angle L1 term inside the shape group.
            The default ``1.0`` preserves the original L1-style treatment.
        loss_weight:
            Global loss multiplier. Use ``5.0`` to match the original
            O²-RTDETR bbox L1 loss weight.
        reduction:
            ``'mean'``, ``'sum'`` or ``'none'``.
        eps:
            Numerical stability constant.
    """

    def __init__(
        self,
        delta: float = 0.5,
        scale_ref: float = 0.12,
        scale_power: float = 1.0,
        angle_weight: float = 1.0,
        loss_weight: float = 5.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not 0.0 <= float(delta) < 1.0:
            raise ValueError(f"delta must be in [0, 1), got {delta}.")
        if float(scale_ref) <= 0.0:
            raise ValueError(f"scale_ref must be positive, got {scale_ref}.")
        if float(scale_power) <= 0.0:
            raise ValueError(
                f"scale_power must be positive, got {scale_power}."
            )
        if float(angle_weight) < 0.0:
            raise ValueError(
                f"angle_weight must be non-negative, got {angle_weight}."
            )
        if float(loss_weight) < 0.0:
            raise ValueError(
                f"loss_weight must be non-negative, got {loss_weight}."
            )
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be one of {'none', 'mean', 'sum'}, "
                f"got {reduction!r}."
            )

        self.delta = float(delta)
        self.scale_ref = float(scale_ref)
        self.scale_power = float(scale_power)
        self.angle_weight = float(angle_weight)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.eps = float(eps)

    def _dynamic_weights(
        self, target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return target scale, location weight and shape weight."""
        if target.size(-1) != 5:
            raise ValueError(
                "RotatedScaleDynamicLoss expects target[..., 5] in "
                "(cx, cy, w, h, angle) format, "
                f"got shape {tuple(target.shape)}."
            )

        target_wh = target[..., 2:4].clamp_min(0.0)
        target_scale = torch.sqrt(
            (target_wh[..., 0] * target_wh[..., 1]).clamp_min(self.eps)
        )

        scale_ratio = (target_scale / self.scale_ref).clamp(0.0, 1.0)
        scale_ratio = scale_ratio.pow(self.scale_power)

        beta = self.delta * scale_ratio

        location_weight = 1.0 + self.delta - beta
        shape_weight = 1.0 - self.delta + beta

        return target_scale, location_weight, shape_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        """Compute the scale-dynamic regression loss.

        Args:
            pred:
                Normalized predicted OBBs, shape ``[..., 5]``.
            target:
                Normalized target OBBs, shape ``[..., 5]``.
            weight:
                Per-coordinate bbox weights with the same shape as ``pred``.
                In DETR this is 1 for matched positives and 0 otherwise.
            avg_factor:
                Normalization factor. O²-RTDETR passes the globally reduced
                number of positive samples here.
            reduction_override:
                Optional runtime reduction override.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have identical shape, got "
                f"{tuple(pred.shape)} vs {tuple(target.shape)}."
            )
        if pred.size(-1) != 5:
            raise ValueError(
                "RotatedScaleDynamicLoss expects pred[..., 5], "
                f"got shape {tuple(pred.shape)}."
            )

        reduction = (
            self.reduction if reduction_override is None else reduction_override
        )
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction_override must be one of {'none', 'mean', 'sum'}, "
                f"got {reduction!r}."
            )

        if weight is None:
            weight = torch.ones_like(pred)
        elif weight.shape != pred.shape:
            raise ValueError(
                f"weight must match pred shape, got "
                f"{tuple(weight.shape)} vs {tuple(pred.shape)}."
            )

        # Preserve the original normalized-coordinate L1 geometry.
        abs_error = (pred - target).abs()

        # Per-coordinate DETR bbox weights are retained.
        weighted_error = abs_error * weight

        # (cx, cy)
        location_error = weighted_error[..., 0:2].sum(dim=-1)

        # (w, h, angle)
        shape_error = weighted_error[..., 2:4].sum(dim=-1)
        shape_error = (
            shape_error
            + self.angle_weight * weighted_error[..., 4]
        )

        _, location_weight, shape_weight = self._dynamic_weights(target)

        per_box_loss = (
            location_weight * location_error
            + shape_weight * shape_error
        )

        if reduction == "none":
            return self.loss_weight * per_box_loss

        if reduction == "sum":
            return self.loss_weight * per_box_loss.sum()

        # Match MMDetection's avg_factor behavior used by the original
        # L1Loss: sum positive losses and divide by the positive count.
        if avg_factor is not None:
            if torch.is_tensor(avg_factor):
                denom = avg_factor.to(
                    device=pred.device, dtype=pred.dtype
                ).clamp_min(self.eps)
            else:
                denom = pred.new_tensor(float(avg_factor)).clamp_min(self.eps)

            return self.loss_weight * per_box_loss.sum() / denom

        return self.loss_weight * per_box_loss.mean()


__all__ = ["RotatedScaleDynamicLoss"]
