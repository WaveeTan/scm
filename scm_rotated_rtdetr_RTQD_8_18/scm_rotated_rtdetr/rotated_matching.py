"""One-pass Hungarian targets shared by final RTP-Score objectives."""

from dataclasses import dataclass
from typing import List

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from ai4rs.structures.bbox import rbbox_overlaps


@dataclass
class RTPImageTargets:
    """Assignment and geometry targets for one image."""

    labels: Tensor
    label_weights: Tensor
    bbox_targets: Tensor
    bbox_weights: Tensor
    pos_inds: Tensor
    neg_inds: Tensor
    assigned_gt_inds: Tensor
    gt_labels: Tensor
    gt_bboxes: Tensor
    decoded_bbox_preds: Tensor
    rotated_iou: Tensor
    pairwise_iou: Tensor


class RotatedMatchingTargetBuilder:
    """Run the configured assigner exactly once for one prediction layer."""

    def __init__(
        self,
        assigner,
        num_classes: int,
        angle_cfg: dict,
        angle_factor: float,
    ) -> None:
        self.assigner = assigner
        self.num_classes = int(num_classes)
        self.angle_cfg = dict(angle_cfg)
        self.angle_factor = float(angle_factor)

    def build_image(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
    ) -> RTPImageTargets:
        img_h, img_w = img_meta["img_shape"][:2]
        factor = bbox_pred.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]
        ).unsqueeze(0)
        decoded_bbox_pred = bbox_pred.detach() * factor
        num_queries = bbox_pred.size(0)

        # regularize_boxes mutates in place, so never pass dataset-owned boxes.
        gt_boxes = gt_instances.bboxes.clone()
        gt_boxes.regularize_boxes(**self.angle_cfg)
        normalized_gt = InstanceData(
            bboxes=gt_boxes,
            labels=gt_instances.labels,
        )
        assign_result = self.assigner.assign(
            pred_instances=InstanceData(
                scores=cls_score.detach(),
                bboxes=decoded_bbox_pred,
            ),
            gt_instances=normalized_gt,
            img_meta=img_meta,
        )

        gt_bboxes = gt_boxes.tensor
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False
        ).squeeze(-1)
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False
        ).squeeze(-1)
        assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        labels = bbox_pred.new_full(
            (num_queries,), self.num_classes, dtype=torch.long
        )
        label_weights = bbox_pred.new_ones((num_queries,))
        bbox_targets = torch.zeros_like(decoded_bbox_pred)
        bbox_weights = torch.zeros_like(decoded_bbox_pred)
        if pos_inds.numel():
            labels[pos_inds] = gt_labels[assigned_gt_inds]
            bbox_weights[pos_inds] = 1.0
            bbox_targets[pos_inds] = gt_bboxes[assigned_gt_inds] / factor

        if gt_bboxes.numel():
            pairwise_iou = rbbox_overlaps(
                decoded_bbox_pred.detach(),
                gt_bboxes,
                mode="iou",
                is_aligned=False,
            ).clamp(0, 1)
        else:
            pairwise_iou = bbox_pred.new_zeros((num_queries, 0))
        if pos_inds.numel():
            rotated_iou = pairwise_iou[
                pos_inds, assigned_gt_inds
            ].clamp(0, 1)
        else:
            rotated_iou = bbox_pred.new_zeros((0,))

        return RTPImageTargets(
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            assigned_gt_inds=assigned_gt_inds,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            decoded_bbox_preds=decoded_bbox_pred,
            rotated_iou=rotated_iou,
            pairwise_iou=pairwise_iou,
        )

    def build_batch(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: List[InstanceData],
        batch_img_metas: List[dict],
    ) -> List[RTPImageTargets]:
        sizes = (
            len(cls_scores),
            len(bbox_preds),
            len(batch_gt_instances),
            len(batch_img_metas),
        )
        if len(set(sizes)) != 1:
            raise ValueError("matching-target batch inputs have different sizes")
        return [
            self.build_image(cls_score, bbox_pred, gt_instances, img_meta)
            for cls_score, bbox_pred, gt_instances, img_meta in zip(
                cls_scores,
                bbox_preds,
                batch_gt_instances,
                batch_img_metas,
            )
        ]


__all__ = ["RTPImageTargets", "RotatedMatchingTargetBuilder"]