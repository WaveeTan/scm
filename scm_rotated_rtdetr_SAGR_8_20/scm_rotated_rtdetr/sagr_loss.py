"""Final-layer Scale-Aspect Geometry Residual (SAGR) loss.

Purpose
-------
This loss is an auxiliary residual term for SCM + O²-RTDETR.

It does NOT replace the original O²-RTDETR L1 regression loss. Instead:

1. small objects receive extra center supervision;
2. small + elongated objects additionally receive periodic-angle and
   short-side supervision;
3. ordinary / large objects are almost unchanged.

Expected box format
-------------------
Normalized rotated boxes:
    (cx, cy, w, h, angle)

For the current O²-RTDETR codebase, ``angle`` is normalized by ``pi``.
Therefore an angle period of ``1.0`` corresponds to a physical period
of ``pi`` radians.

The loss is intended to be attached only to the FINAL matching decoder
layer by ``SCMSAGRRotatedRTDETRHead``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ScaleAspectGeometryResidualLoss(nn.Module):
    """Scale- and aspect-ratio-aware residual OBB regression loss.

    The loss is

        L_sagr =
            lambda * [
                g_s(s) * w_c * L_center
                +
                g_s(s) * g_ar(r) *
                (w_a * L_angle + w_q * L_short)
            ]

    where

        s = sqrt(w_gt * h_gt)

        g_s(s) = clamp(1 - s / scale_ref, 0, 1)

        r = max(w_gt, h_gt) / min(w_gt, h_gt)

        g_ar(r) = clamp(
            (r - ar_ref) / (ar_full - ar_ref),
            0, 1
        )

    Thus:
    - large / ordinary targets: residual ~= 0;
    - small targets: center residual is active;
    - small + elongated targets: center + angle + short-side residuals
      are active.

    Args:
        scale_ref:
            Normalized scale at which the residual becomes zero.
            Default 0.04 corresponds to roughly 32 px for an 800 px image.
        ar_ref:
            Aspect ratio where elongated-object supervision begins.
        ar_full:
            Aspect ratio where the elongated gate reaches 1.
        center_weight:
            Weight inside the residual for center error.
        angle_weight:
            Weight inside the elongated residual for periodic angle error.
        short_side_weight:
            Weight inside the elongated residual for relative short-side error.
        loss_weight:
            Global residual coefficient. This is intentionally much smaller
            than the baseline L1 loss weight (5.0), because SAGR is auxiliary.
        short_side_floor:
            Lower bound for the denominator of relative short-side error.
        angle_period:
            Period of the normalized angle representation. In the current
            O²-RTDETR setup, theta is normalized by pi, hence period=1.
        eps:
            Numerical stability constant.
    """

    def __init__(
        self,
        scale_ref: float = 0.04,
        ar_ref: float = 4.0,
        ar_full: float = 8.0,
        center_weight: float = 1.0,
        angle_weight: float = 0.5,
        short_side_weight: float = 0.5,
        loss_weight: float = 0.5,
        short_side_floor: float = 0.01,
        angle_period: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if scale_ref <= 0:
            raise ValueError(f"scale_ref must be > 0, got {scale_ref}.")
        if ar_ref < 1:
            raise ValueError(f"ar_ref must be >= 1, got {ar_ref}.")
        if ar_full <= ar_ref:
            raise ValueError(
                f"ar_full must be > ar_ref, got ar_ref={ar_ref}, "
                f"ar_full={ar_full}."
            )
        if center_weight < 0:
            raise ValueError(
                f"center_weight must be >= 0, got {center_weight}."
            )
        if angle_weight < 0:
            raise ValueError(
                f"angle_weight must be >= 0, got {angle_weight}."
            )
        if short_side_weight < 0:
            raise ValueError(
                "short_side_weight must be >= 0, "
                f"got {short_side_weight}."
            )
        if loss_weight < 0:
            raise ValueError(
                f"loss_weight must be >= 0, got {loss_weight}."
            )
        if short_side_floor <= 0:
            raise ValueError(
                "short_side_floor must be > 0, "
                f"got {short_side_floor}."
            )
        if angle_period <= 0:
            raise ValueError(
                f"angle_period must be > 0, got {angle_period}."
            )

        self.scale_ref = float(scale_ref)
        self.ar_ref = float(ar_ref)
        self.ar_full = float(ar_full)

        self.center_weight = float(center_weight)
        self.angle_weight = float(angle_weight)
        self.short_side_weight = float(short_side_weight)

        self.loss_weight = float(loss_weight)
        self.short_side_floor = float(short_side_floor)
        self.angle_period = float(angle_period)
        self.eps = float(eps)

    def _scale_gate(self, target_wh: Tensor) -> Tensor:
        target_scale = torch.sqrt(
            (
                target_wh[..., 0]
                * target_wh[..., 1]
            ).clamp_min(self.eps)
        )
        return (
            1.0 - target_scale / self.scale_ref
        ).clamp(0.0, 1.0)

    def _aspect_gate(self, target_wh: Tensor) -> Tensor:
        target_long = target_wh.max(dim=-1).values
        target_short = target_wh.min(dim=-1).values.clamp_min(self.eps)
        target_ar = target_long / target_short

        return (
            (target_ar - self.ar_ref)
            / (self.ar_full - self.ar_ref)
        ).clamp(0.0, 1.0)

    def _periodic_angle_error(
        self,
        pred_angle: Tensor,
        target_angle: Tensor,
    ) -> Tensor:
        """Shortest absolute distance on a periodic normalized angle."""
        half_period = 0.5 * self.angle_period
        delta = torch.remainder(
            pred_angle - target_angle + half_period,
            self.angle_period,
        ) - half_period
        return delta.abs()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """Compute SAGR.

        Args:
            pred:
                Normalized predicted OBBs, shape ``[..., 5]``.
            target:
                Normalized target OBBs, shape ``[..., 5]``.
            weight:
                Coordinate-wise bbox weights. In the current DETR target
                builder, matched positives have active bbox weights and
                unmatched queries have zero bbox weights.
            avg_factor:
                Positive-sample normalization factor.
        """
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must have the same shape, "
                f"got pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}."
            )
        if pred.size(-1) != 5:
            raise ValueError(
                "SAGR expects normalized OBBs with last dimension 5, "
                f"got {tuple(pred.shape)}."
            )

        if weight is None:
            weight = torch.ones_like(pred)
        elif weight.shape != pred.shape:
            raise ValueError(
                "weight must have the same shape as pred, "
                f"got weight={tuple(weight.shape)}, "
                f"pred={tuple(pred.shape)}."
            )

        # A query is treated as a positive regression target when at least
        # one of its bbox coordinates is active.
        positive = (
            weight.abs().sum(dim=-1) > 0
        ).to(dtype=pred.dtype)

        target_wh = target[..., 2:4].clamp_min(self.eps)
        pred_wh = pred[..., 2:4].clamp_min(self.eps)

        scale_gate = self._scale_gate(target_wh)
        aspect_gate = self._aspect_gate(target_wh)

        # ------------------------------------------------------------
        # 1) Small-object center residual.
        # Preserve coordinate-wise bbox weights from the DETR target builder.
        # ------------------------------------------------------------
        center_error = (
            (pred[..., 0:2] - target[..., 0:2]).abs()
            * weight[..., 0:2]
        ).sum(dim=-1)

        # ------------------------------------------------------------
        # 2) Periodic orientation residual.
        # angle_period=1.0 because theta is normalized by pi in this project.
        # ------------------------------------------------------------
        angle_error = self._periodic_angle_error(
            pred[..., 4],
            target[..., 4],
        )
        angle_error = angle_error * weight[..., 4]

        # ------------------------------------------------------------
        # 3) Relative short-side residual.
        #
        # For highly elongated OBBs, rIoU is sensitive to the short side.
        # log1p limits extremely large relative errors on tiny boxes.
        # ------------------------------------------------------------
        target_short = target_wh.min(dim=-1).values
        pred_short = pred_wh.min(dim=-1).values

        short_denominator = target_short.clamp_min(
            self.short_side_floor
        )
        relative_short_error = (
            pred_short - target_short
        ).abs() / short_denominator

        short_error = torch.log1p(relative_short_error)

        # Respect whether width/height regression is active for this query.
        wh_active = (
            weight[..., 2:4].abs().sum(dim=-1) > 0
        ).to(dtype=pred.dtype)
        short_error = short_error * wh_active

        center_term = (
            scale_gate
            * self.center_weight
            * center_error
        )

        elongated_term = (
            scale_gate
            * aspect_gate
            * (
                self.angle_weight * angle_error
                + self.short_side_weight * short_error
            )
        )

        per_box_loss = (
            center_term + elongated_term
        ) * positive

        if avg_factor is None:
            denominator = positive.sum().clamp_min(1.0)
        elif torch.is_tensor(avg_factor):
            denominator = avg_factor.to(
                device=pred.device,
                dtype=pred.dtype,
            ).clamp_min(1.0)
        else:
            denominator = pred.new_tensor(
                float(avg_factor)
            ).clamp_min(1.0)

        return (
            self.loss_weight
            * per_box_loss.sum()
            / denominator
        )


__all__ = ["ScaleAspectGeometryResidualLoss"]
