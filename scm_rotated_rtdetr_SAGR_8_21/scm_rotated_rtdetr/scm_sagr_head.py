"""SCM + O²-RTDETR head with FINAL-layer-only fixed SAGR."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean

from projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr_head import (
    RotatedRTDETRHead,
)
from .sagr_loss import ScaleAspectGeometryResidualLoss


class SCMSAGRRotatedRTDETRHead(RotatedRTDETRHead):
    """Original O² losses + final matching decoder SAGR only."""

    def __init__(
        self,
        *args,
        sagr_loss_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = {} if sagr_loss_cfg is None else dict(sagr_loss_cfg)
        self.loss_sagr = ScaleAspectGeometryResidualLoss(**cfg)

    @staticmethod
    def _build_geometry_wh_factors(
        bbox_targets_list: List[Tensor],
        batch_img_metas: List[dict],
        reference_tensor: Tensor,
    ) -> Tensor:
        """Return [sum_i Q_i,2] factors for isotropic wh geometry.

        Native target normalization:
            w_n = w_px / img_w
            h_n = h_px / img_h

        Isotropic geometry:
            D = sqrt(img_w * img_h)
            w_iso = w_n * img_w / D
            h_iso = h_n * img_h / D
        """
        if len(bbox_targets_list) != len(batch_img_metas):
            raise RuntimeError(
                'bbox target/meta batch mismatch: '
                f'{len(bbox_targets_list)} vs {len(batch_img_metas)}.'
            )

        factors = []
        for bbox_targets_i, img_meta in zip(
            bbox_targets_list, batch_img_metas
        ):
            if 'img_shape' not in img_meta:
                raise RuntimeError(
                    'img_shape is required to build SAGR geometry factors.'
                )
            img_shape = img_meta['img_shape']
            if len(img_shape) < 2:
                raise RuntimeError(
                    f'img_shape must contain (h,w), got {img_shape}.'
                )

            img_h = float(img_shape[0])
            img_w = float(img_shape[1])
            if img_h <= 0 or img_w <= 0:
                raise RuntimeError(f'Invalid img_shape: {img_shape}.')

            isotropic_size = (img_h * img_w) ** 0.5
            factor_i = reference_tensor.new_tensor([
                img_w / isotropic_size,
                img_h / isotropic_size,
            ])
            factors.append(
                factor_i.unsqueeze(0).expand(bbox_targets_i.size(0), -1)
            )

        if len(factors) == 0:
            return reference_tensor.new_zeros((0, 2))
        return torch.cat(factors, dim=0)

    def _loss_sagr_final(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tensor:
        if cls_scores.ndim != 3:
            raise RuntimeError(
                'final cls_scores must be [B,Q,C], '
                f'got {tuple(cls_scores.shape)}.'
            )
        if bbox_preds.ndim != 3 or bbox_preds.size(-1) != 5:
            raise RuntimeError(
                'final bbox_preds must be [B,Q,5], '
                f'got {tuple(bbox_preds.shape)}.'
            )

        num_imgs = cls_scores.size(0)
        if bbox_preds.size(0) != num_imgs:
            raise RuntimeError('cls/bbox batch mismatch.')

        # Assignment is discrete; detached predictions are sufficient for
        # rebuilding the target structure. Loss gradients still use bbox_preds.
        cls_scores_list = [
            cls_scores[i].detach() for i in range(num_imgs)
        ]
        bbox_preds_list = [
            bbox_preds[i].detach() for i in range(num_imgs)
        ]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            batch_gt_instances,
            batch_img_metas,
        )
        (
            _labels_list,
            _label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            _num_total_neg,
        ) = cls_reg_targets

        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)

        # FIX 2: factor rows are aligned with flattened per-image queries.
        geometry_wh_factors = self._build_geometry_wh_factors(
            bbox_targets_list=bbox_targets_list,
            batch_img_metas=batch_img_metas,
            reference_tensor=bbox_preds,
        )
        if geometry_wh_factors.shape != bbox_targets[:, 2:4].shape:
            raise RuntimeError(
                'geometry factor/target shape mismatch: '
                f'{tuple(geometry_wh_factors.shape)} vs '
                f'{tuple(bbox_targets[:, 2:4].shape)}.'
            )

        num_total_pos_tensor = bbox_preds.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(
            reduce_mean(num_total_pos_tensor), min=1
        ).item()

        return self.loss_sagr(
            bbox_preds.reshape(-1, 5),
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_pos,
            geometry_wh_factors=geometry_wh_factors,
        )

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        # Full original O² loss path.
        losses = super().loss_by_feat(
            all_layers_cls_scores=all_layers_cls_scores,
            all_layers_bbox_preds=all_layers_bbox_preds,
            enc_cls_scores=enc_cls_scores,
            enc_bbox_preds=enc_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )

        (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            _all_layers_denoising_cls_scores,
            _all_layers_denoising_bbox_preds,
        ) = self.split_outputs(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            dn_meta,
        )

        if len(all_layers_matching_cls_scores) == 0:
            raise RuntimeError('No matching decoder cls outputs.')
        if len(all_layers_matching_bbox_preds) == 0:
            raise RuntimeError('No matching decoder bbox outputs.')

        final_cls_scores = all_layers_matching_cls_scores[-1]
        final_bbox_preds = all_layers_matching_bbox_preds[-1]

        losses['loss_sagr'] = self._loss_sagr_final(
            final_cls_scores,
            final_bbox_preds,
            batch_gt_instances,
            batch_img_metas,
        )
        return losses


__all__ = ['SCMSAGRRotatedRTDETRHead']
