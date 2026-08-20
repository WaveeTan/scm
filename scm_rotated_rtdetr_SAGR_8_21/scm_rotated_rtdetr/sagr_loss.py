"""Final-layer Scale-Aspect Geometry Residual (SAGR) loss.

SAGR is an auxiliary residual term for SCM + O²-RTDETR. It does not replace
original O² L1/KLD losses.

Fixed geometry convention
-------------------------
Detector targets are normalized independently by image width/height:

    w_n = w_px / img_w
    h_n = h_px / img_h

Directly computing max/min on (w_n, h_n) changes physical aspect ratio when
img_w != img_h. The fixed implementation therefore receives per-query factors:

    D = sqrt(img_w * img_h)
    f_w = img_w / D
    f_h = img_h / D

and uses isotropic geometry:

    w_iso = w_n * f_w = w_px / D
    h_iso = h_n * f_h = h_px / D

This restores physical aspect ratio while preserving the previous area/scale
normalization sqrt(w*h / image_area).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ScaleAspectGeometryResidualLoss(nn.Module):
    """Scale- and aspect-ratio-aware residual OBB regression loss."""

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
            raise ValueError(f'scale_ref must be > 0, got {scale_ref}.')
        if ar_ref < 1:
            raise ValueError(f'ar_ref must be >= 1, got {ar_ref}.')
        if ar_full <= ar_ref:
            raise ValueError(
                'ar_full must be > ar_ref, '
                f'got ar_ref={ar_ref}, ar_full={ar_full}.'
            )
        if center_weight < 0:
            raise ValueError('center_weight must be >= 0.')
        if angle_weight < 0:
            raise ValueError('angle_weight must be >= 0.')
        if short_side_weight < 0:
            raise ValueError('short_side_weight must be >= 0.')
        if loss_weight < 0:
            raise ValueError('loss_weight must be >= 0.')
        if short_side_floor <= 0:
            raise ValueError('short_side_floor must be > 0.')
        if angle_period <= 0:
            raise ValueError('angle_period must be > 0.')

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
            (target_wh[..., 0] * target_wh[..., 1]).clamp_min(self.eps)
        )
        return (1.0 - target_scale / self.scale_ref).clamp(0.0, 1.0)

    def _aspect_gate(self, target_wh: Tensor) -> Tensor:
        target_long = target_wh.max(dim=-1).values
        target_short = target_wh.min(dim=-1).values.clamp_min(self.eps)
        target_ar = target_long / target_short
        return (
            (target_ar - self.ar_ref) / (self.ar_full - self.ar_ref)
        ).clamp(0.0, 1.0)

    def _periodic_angle_error(
        self,
        pred_angle: Tensor,
        target_angle: Tensor,
    ) -> Tensor:
        half_period = 0.5 * self.angle_period
        delta = torch.remainder(
            pred_angle - target_angle + half_period,
            self.angle_period,
        ) - half_period
        return delta.abs()

    def _prepare_geometry_wh(
        self,
        pred_wh: Tensor,
        target_wh: Tensor,
        geometry_wh_factors: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if geometry_wh_factors is None:
            raise ValueError(
                'geometry_wh_factors is required for fixed SAGR. '
                'Build it from batch_img_metas in the head.'
            )
        if geometry_wh_factors.shape != target_wh.shape:
            raise ValueError(
                'geometry_wh_factors must match wh shape: '
                f'{tuple(geometry_wh_factors.shape)} vs '
                f'{tuple(target_wh.shape)}.'
            )

        geometry_wh_factors = geometry_wh_factors.to(
            device=target_wh.device,
            dtype=target_wh.dtype,
        )
        if not torch.isfinite(geometry_wh_factors).all():
            raise ValueError('geometry_wh_factors contains non-finite values.')
        if (geometry_wh_factors <= 0).any():
            raise ValueError('geometry_wh_factors must be positive.')

        target_wh_geom = target_wh * geometry_wh_factors
        pred_wh_geom = pred_wh * geometry_wh_factors
        return pred_wh_geom, target_wh_geom

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        geometry_wh_factors: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Compute fixed SAGR.

        Center error intentionally remains in native DETR normalized
        coordinates, so this patch changes only the problematic AR/short-side
        geometry and keeps the experiment controlled.
        """
        if pred.shape != target.shape:
            raise ValueError(
                'pred and target must have same shape, '
                f'got {tuple(pred.shape)} vs {tuple(target.shape)}.'
            )
        if pred.size(-1) != 5:
            raise ValueError(
                f'SAGR expects last dimension 5, got {tuple(pred.shape)}.'
            )

        if weight is None:
            weight = torch.ones_like(pred)
        elif weight.shape != pred.shape:
            raise ValueError(
                'weight must match pred shape, '
                f'got {tuple(weight.shape)} vs {tuple(pred.shape)}.'
            )

        positive = (
            weight.abs().sum(dim=-1) > 0
        ).to(dtype=pred.dtype)

        target_wh = target[..., 2:4].clamp_min(self.eps)
        pred_wh = pred[..., 2:4].clamp_min(self.eps)

        # FIX 2: physical/isotropic AR and short-side geometry.
        pred_wh_geom, target_wh_geom = self._prepare_geometry_wh(
            pred_wh=pred_wh,
            target_wh=target_wh,
            geometry_wh_factors=geometry_wh_factors,
        )

        # Isotropic conversion preserves sqrt(wh / image_area), so the
        # scale_ref semantics remain unchanged.
        scale_gate = self._scale_gate(target_wh_geom)
        aspect_gate = self._aspect_gate(target_wh_geom)

        center_error = (
            (pred[..., 0:2] - target[..., 0:2]).abs()
            * weight[..., 0:2]
        ).sum(dim=-1)

        angle_error = self._periodic_angle_error(
            pred[..., 4], target[..., 4]
        ) * weight[..., 4]

        target_short = target_wh_geom.min(dim=-1).values
        pred_short = pred_wh_geom.min(dim=-1).values
        short_denominator = target_short.clamp_min(self.short_side_floor)
        relative_short_error = (
            (pred_short - target_short).abs() / short_denominator
        )
        short_error = torch.log1p(relative_short_error)

        wh_active = (
            weight[..., 2:4].abs().sum(dim=-1) > 0
        ).to(dtype=pred.dtype)
        short_error = short_error * wh_active

        center_term = scale_gate * self.center_weight * center_error
        elongated_term = (
            scale_gate
            * aspect_gate
            * (
                self.angle_weight * angle_error
                + self.short_side_weight * short_error
            )
        )

        per_box_loss = (center_term + elongated_term) * positive

        if avg_factor is None:
            denominator = positive.sum().clamp_min(1.0)
        elif torch.is_tensor(avg_factor):
            denominator = avg_factor.to(
                device=pred.device,
                dtype=pred.dtype,
            ).clamp_min(1.0)
        else:
            denominator = pred.new_tensor(float(avg_factor)).clamp_min(1.0)

        return self.loss_weight * per_box_loss.sum() / denominator


__all__ = ['ScaleAspectGeometryResidualLoss']
