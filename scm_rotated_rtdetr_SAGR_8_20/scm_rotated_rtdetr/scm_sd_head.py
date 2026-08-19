"""O²-RTDETR head with scale-dynamic matching-query bbox supervision.

Controlled-experiment behavior:
- decoder matching-query bbox loss: RotatedScaleDynamicLoss
- encoder selected-query bbox loss: RotatedScaleDynamicLoss
- DN bbox loss: original O² L1Loss
- classification loss: unchanged
- KLD/GD IoU loss: unchanged
- Hungarian matching: unchanged

The separation is achieved by overriding only ``loss_by_feat_single``.
DINO's DN path calls ``_loss_dn_single`` instead, which remains inherited
from ``RotatedRTDETRHead`` and therefore still uses ``self.loss_bbox``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, reduce_mean

from ai4rs.registry import MODELS
from ai4rs.structures.bbox import rbbox_overlaps

from projects.rotated_rtdetr.rotated_rtdetr.rotated_rtdetr_head import (
    RotatedRTDETRHead,
)
from projects.rotated_rtdetr.rotated_rtdetr.varifocal_loss import VarifocalLoss
from projects.rotated_rtdetr.rotated_rtdetr.prob_iou import probiou

from .sd_loss import RotatedScaleDynamicLoss


@MODELS.register_module()
class SCMSDRotatedRTDETRHead(RotatedRTDETRHead):
    """RotatedRTDETRHead using scale-dynamic bbox loss for normal queries.

    ``self.loss_bbox`` remains the original L1 loss configured in the model.
    It is intentionally preserved because inherited ``_loss_dn_single`` uses
    it for denoising queries.

    ``self.loss_sd_bbox`` is used only by this class's
    ``loss_by_feat_single`` for decoder matching queries and encoder selected
    queries.
    """

    def __init__(
        self,
        *args,
        sd_loss_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        cfg = {} if sd_loss_cfg is None else dict(sd_loss_cfg)
        self.loss_sd_bbox = RotatedScaleDynamicLoss(**cfg)

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        """Loss for one decoder layer / encoder selected-query predictions.

        The implementation intentionally mirrors the original
        ``RotatedRTDETRHead.loss_by_feat_single``. The only optimization change
        is the final bbox term:

            original:
                self.loss_bbox(...)       # L1, weight 5

            this head:
                self.loss_sd_bbox(...)    # scale-dynamic L1-style loss

        The KLD/GD IoU term is unchanged.
        """
        num_imgs = cls_scores.size(0)

        cls_scores_list = [
            cls_scores[i] for i in range(num_imgs)
        ]
        bbox_preds_list = [
            bbox_preds[i] for i in range(num_imgs)
        ]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            batch_gt_instances,
            batch_img_metas,
        )

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ------------------------------------------------------------
        # Classification loss: identical to original O²-RTDETR.
        # ------------------------------------------------------------
        cls_scores = cls_scores.reshape(
            -1, self.cls_out_channels
        )

        cls_avg_factor = (
            num_total_pos * 1.0
            + num_total_neg * self.bg_cls_weight
        )

        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )

        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, VarifocalLoss):
            bg_class_ind = self.num_classes

            pos_inds = (
                (labels >= 0)
                & (labels < bg_class_ind)
            ).nonzero().squeeze(1)

            cls_iou_targets = label_weights.new_zeros(
                cls_scores.shape
            )

            if self.varifocal_loss_iou_type == "hbox_iou":
                pos_bbox_targets = (
                    bbox_targets[pos_inds][..., :4]
                )
                pos_decode_bbox_targets = (
                    bbox_cxcywh_to_xyxy(
                        pos_bbox_targets
                    )
                )

                pos_bbox_pred = (
                    bbox_preds
                    .reshape(-1, 5)[pos_inds][..., :4]
                )
                pos_decode_bbox_pred = (
                    bbox_cxcywh_to_xyxy(
                        pos_bbox_pred
                    )
                )

                pos_labels = labels[pos_inds]

                cls_iou_targets[
                    pos_inds, pos_labels
                ] = bbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True,
                )

            elif self.varifocal_loss_iou_type == "rbox_iou":
                img_h, img_w = batch_img_metas[0]["img_shape"]

                factor = torch.tensor(
                    [
                        img_w,
                        img_h,
                        img_w,
                        img_h,
                        self.angle_factor,
                    ],
                    device=pos_inds.device,
                )

                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = (
                    pos_bbox_targets * factor
                )

                pos_bbox_pred = (
                    bbox_preds.reshape(-1, 5)[pos_inds]
                )
                pos_decode_bbox_pred = (
                    pos_bbox_pred * factor
                )

                pos_labels = labels[pos_inds]

                cls_iou_targets[
                    pos_inds, pos_labels
                ] = rbbox_overlaps(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                    is_aligned=True,
                )

            elif self.varifocal_loss_iou_type == "prob_iou":
                img_h, img_w = batch_img_metas[0]["img_shape"]

                factor = torch.tensor(
                    [
                        img_w,
                        img_h,
                        img_w,
                        img_h,
                        self.angle_factor,
                    ],
                    device=pos_inds.device,
                )

                pos_bbox_targets = bbox_targets[pos_inds]
                pos_decode_bbox_targets = (
                    pos_bbox_targets * factor
                )

                pos_bbox_pred = (
                    bbox_preds.reshape(-1, 5)[pos_inds]
                )
                pos_decode_bbox_pred = (
                    pos_bbox_pred * factor
                )

                pos_labels = labels[pos_inds]

                cls_iou_targets[
                    pos_inds, pos_labels
                ] = probiou(
                    pos_decode_bbox_pred.detach(),
                    pos_decode_bbox_targets,
                )[:, 0]

            else:
                raise NotImplementedError(
                    "Unsupported varifocal_loss_iou_type: "
                    f"{self.varifocal_loss_iou_type}"
                )

            loss_cls = self.loss_cls(
                cls_scores,
                cls_iou_targets,
                avg_factor=cls_avg_factor,
            )

        else:
            loss_cls = self.loss_cls(
                cls_scores,
                labels,
                label_weights,
                avg_factor=cls_avg_factor,
            )

        # ------------------------------------------------------------
        # Same positive-count normalization as original O²-RTDETR.
        # ------------------------------------------------------------
        num_total_pos = loss_cls.new_tensor(
            [num_total_pos]
        )
        num_total_pos = torch.clamp(
            reduce_mean(num_total_pos),
            min=1,
        ).item()

        # ------------------------------------------------------------
        # Original O² factor construction for physical-space KLD/GD.
        # ------------------------------------------------------------
        factors = []

        for img_meta, bbox_pred in zip(
            batch_img_metas,
            bbox_preds,
        ):
            img_h, img_w = img_meta["img_shape"]

            factor = bbox_pred.new_tensor(
                [
                    img_w,
                    img_h,
                    img_w,
                    img_h,
                    self.angle_factor,
                ]
            ).unsqueeze(0).repeat(
                bbox_pred.size(0), 1
            )

            factors.append(factor)

        factors = torch.cat(factors, 0)

        bbox_preds = bbox_preds.reshape(-1, 5)

        bboxes = bbox_preds * factors
        bboxes_gt = bbox_targets * factors

        # ------------------------------------------------------------
        # KLD/GD IoU loss: unchanged.
        # ------------------------------------------------------------
        loss_iou = self.loss_iou(
            bboxes,
            bboxes_gt,
            bbox_weights,
            avg_factor=num_total_pos,
        )

        # ------------------------------------------------------------
        # ONLY changed term:
        # normal matching/encoder L1 -> scale-dynamic OBB regression.
        # ------------------------------------------------------------
        loss_bbox = self.loss_sd_bbox(
            bbox_preds,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_pos,
        )

        return loss_cls, loss_bbox, loss_iou


__all__ = ["SCMSDRotatedRTDETRHead"]
