"""BASD-Reg head: bounded reweighting of final matched regression only."""

from typing import Dict, List, Optional, Tuple

import torch
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, reduce_mean
from torch import Tensor

from ai4rs.registry import MODELS
from ai4rs.structures.bbox import rbbox_overlaps
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead
from projects.rotated_rtdetr.rotated_rtdetr.prob_iou import probiou
from projects.rotated_rtdetr.rotated_rtdetr.varifocal_loss import VarifocalLoss

from .basd_reg_grouping import BASDRegGroupState


@MODELS.register_module()
class BASDRegRotatedRTDETRHead(RotatedRTDETRHead):
    """Apply BASD-Reg only to final-decoder Hungarian-positive regression."""

    def __init__(self, *args, basd_reg_cfg: Optional[dict] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = {} if basd_reg_cfg is None else dict(basd_reg_cfg)
        apply_to = cfg.pop("apply_to", "final_decoder_regression")
        use_ambiguity = bool(cfg.pop("use_ambiguity", False))
        use_progress = bool(cfg.pop("use_progress", False))
        if apply_to != "final_decoder_regression":
            raise ValueError("BASD-Reg v1 only supports final decoder regression")
        if use_ambiguity or use_progress:
            raise ValueError("BASD-Reg v1 disables ambiguity and progress")
        self.basd_reg_state = BASDRegGroupState(**cfg)

    def set_basd_reg_alpha(self, value: float) -> float:
        return self.basd_reg_state.set_alpha(value)

    def _get_targets_with_membership_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
    ) -> tuple:
        """Run the original O2 matcher and also map GT group memberships."""
        img_h, img_w = img_meta["img_shape"][:2]
        factor = bbox_pred.new_tensor(
            [img_w, img_h, img_w, img_h, self.angle_factor]
        ).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        decoded_bbox_pred = bbox_pred * factor

        gt_instances.bboxes.regularize_boxes(**self.angle_cfg)
        pred_instances = InstanceData(scores=cls_score, bboxes=decoded_bbox_pred)
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta,
        )

        gt_bboxes = gt_instances.bboxes.tensor
        gt_labels = gt_instances.labels
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(decoded_bbox_pred)
        bbox_weights = torch.zeros_like(decoded_bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = gt_bboxes[pos_assigned_gt_inds.long()] / factor

        gt_membership = self.basd_reg_state.compute_membership(
            gt_bboxes, img_meta["img_shape"]
        )
        pos_membership = gt_membership[pos_assigned_gt_inds.long()]
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
            pos_membership,
        )

    def _build_final_targets(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> tuple:
        records = [
            self._get_targets_with_membership_single(
                cls_score, bbox_pred, gt_instances, img_meta
            )
            for cls_score, bbox_pred, gt_instances, img_meta in zip(
                cls_scores,
                bbox_preds,
                batch_gt_instances,
                batch_img_metas,
            )
        ]
        labels, label_weights, bbox_targets, bbox_weights = zip(
            *(record[:4] for record in records)
        )
        num_total_pos = sum(record[4].numel() for record in records)
        num_total_neg = sum(record[5].numel() for record in records)
        membership = torch.cat([record[6] for record in records], dim=0)
        return (
            torch.cat(labels),
            torch.cat(label_weights),
            torch.cat(bbox_targets),
            torch.cat(bbox_weights),
            num_total_pos,
            num_total_neg,
            membership,
        )

    @staticmethod
    def _as_per_instance(loss: Tensor, num_instances: int) -> Tensor:
        if num_instances == 0:
            return loss.reshape(-1)
        if loss.ndim == 0:
            return loss.reshape(1)
        return loss.reshape(num_instances, -1).sum(dim=-1)

    def _original_classification_loss(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        num_total_pos: int,
        num_total_neg: int,
        batch_img_metas: List[dict],
    ) -> Tensor:
        """Exact classification path of ``RotatedRTDETRHead``."""
        flat_cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        flat_bbox_preds = bbox_preds.reshape(-1, 5)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(flat_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if not isinstance(self.loss_cls, VarifocalLoss):
            return self.loss_cls(
                flat_cls_scores,
                labels,
                label_weights,
                avg_factor=cls_avg_factor,
            )

        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero().squeeze(1)
        cls_iou_targets = label_weights.new_zeros(flat_cls_scores.shape)
        if self.varifocal_loss_iou_type == "hbox_iou":
            target_xyxy = bbox_cxcywh_to_xyxy(bbox_targets[pos_inds, :4])
            pred_xyxy = bbox_cxcywh_to_xyxy(flat_bbox_preds[pos_inds, :4])
            cls_iou_targets[pos_inds, labels[pos_inds]] = bbox_overlaps(
                pred_xyxy.detach(), target_xyxy, is_aligned=True
            )
        elif self.varifocal_loss_iou_type in ("rbox_iou", "prob_iou"):
            factors = []
            for img_meta, image_bbox_preds in zip(batch_img_metas, bbox_preds):
                img_h, img_w = img_meta["img_shape"][:2]
                factor = (
                    image_bbox_preds.new_tensor(
                        [img_w, img_h, img_w, img_h, self.angle_factor]
                    )
                    .unsqueeze(0)
                    .repeat(image_bbox_preds.size(0), 1)
                )
                factors.append(factor)
            factors = torch.cat(factors)
            pred_rboxes = flat_bbox_preds[pos_inds] * factors[pos_inds]
            target_rboxes = bbox_targets[pos_inds] * factors[pos_inds]
            if self.varifocal_loss_iou_type == "rbox_iou":
                quality = rbbox_overlaps(
                    pred_rboxes.detach(), target_rboxes, is_aligned=True
                )
            else:
                quality = probiou(pred_rboxes.detach(), target_rboxes)[:, 0]
            cls_iou_targets[pos_inds, labels[pos_inds]] = quality
        else:
            raise NotImplementedError(self.varifocal_loss_iou_type)
        return self.loss_cls(
            flat_cls_scores, cls_iou_targets, avg_factor=cls_avg_factor
        )

    def _loss_final_matching(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        targets = self._build_final_targets(
            cls_scores, bbox_preds, batch_gt_instances, batch_img_metas
        )
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_pos,
            num_total_neg,
            membership,
        ) = targets
        loss_cls = self._original_classification_loss(
            cls_scores,
            bbox_preds,
            labels,
            label_weights,
            bbox_targets,
            num_total_pos,
            num_total_neg,
            batch_img_metas,
        )

        factors = []
        for img_meta, image_bbox_preds in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta["img_shape"][:2]
            factor = (
                image_bbox_preds.new_tensor(
                    [img_w, img_h, img_w, img_h, self.angle_factor]
                )
                .unsqueeze(0)
                .repeat(image_bbox_preds.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors)
        flat_bbox_preds = bbox_preds.reshape(-1, 5)
        pos_mask = bbox_weights[:, 0] > 0
        pred_pos = flat_bbox_preds[pos_mask]
        target_pos = bbox_targets[pos_mask]
        decoded_pred_pos = pred_pos * factors[pos_mask]
        decoded_target_pos = target_pos * factors[pos_mask]
        num_positive = pred_pos.size(0)

        if num_positive:
            per_l1 = self.loss_bbox(pred_pos, target_pos, reduction_override="none")
            per_l1 = self._as_per_instance(per_l1, num_positive)
            per_kld = self.loss_iou(
                decoded_pred_pos,
                decoded_target_pos,
                reduction_override="none",
            )
            per_kld = self._as_per_instance(per_kld, num_positive)
        else:
            connected_zero = flat_bbox_preds[pos_mask, 0]
            per_l1 = connected_zero
            per_kld = connected_zero

        # The current batch uses the state from the previous update.
        instance_weight, group_weight, mass = self.basd_reg_state.compute_weights(
            membership
        )
        loss_bbox = self.basd_reg_state.global_weighted_mean(per_l1, instance_weight)
        loss_iou = self.basd_reg_state.global_weighted_mean(per_kld, instance_weight)

        unweighted = torch.ones_like(instance_weight)
        unweighted_l1 = self.basd_reg_state.global_weighted_mean(
            per_l1.detach(), unweighted
        )
        unweighted_kld = self.basd_reg_state.global_weighted_mean(
            per_kld.detach(), unweighted
        )
        weight_stats = self.basd_reg_state.summarize_instance_weights(instance_weight)

        with torch.no_grad():
            if num_positive:
                rotated_iou = rbbox_overlaps(
                    decoded_pred_pos.detach(),
                    decoded_target_pos,
                    mode="iou",
                    is_aligned=True,
                ).clamp(0, 1)
                regression_error = 1 - rotated_iou
            else:
                regression_error = pred_pos.new_zeros((0,))
            state_info = self.basd_reg_state.update(regression_error, membership)

        diagnostics = dict(
            alpha=self.basd_reg_state.current_alpha.detach().clone(),
            mass=mass.detach(),
            error_ema=state_info["error_ema"].detach(),
            initialized=state_info["initialized"].detach(),
            group_weight=group_weight.detach(),
            instance_weight_min=weight_stats["min"].detach(),
            instance_weight_mean=weight_stats["mean"].detach(),
            instance_weight_max=weight_stats["max"].detach(),
            unweighted_l1=unweighted_l1.detach(),
            weighted_l1=loss_bbox.detach(),
            unweighted_kld=unweighted_kld.detach(),
            weighted_kld=loss_iou.detach(),
        )
        return loss_cls, loss_bbox, loss_iou, diagnostics

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Optional[Dict[str, int]],
        batch_gt_instances_ignore=None,
    ) -> Dict[str, Tensor]:
        if batch_gt_instances_ignore is not None:
            raise AssertionError("BASD-Reg does not support ignored GT")
        matching_cls, matching_bbox, dn_cls, dn_bbox = self.split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta
        )

        final_cls, final_bbox, final_iou, diagnostics = self._loss_final_matching(
            matching_cls[-1],
            matching_bbox[-1],
            batch_gt_instances,
            batch_img_metas,
        )
        loss_dict = dict(loss_cls=final_cls, loss_bbox=final_bbox, loss_iou=final_iou)

        # Auxiliary decoder layers retain the exact O2 loss path.
        for layer_id in range(len(matching_cls) - 1):
            layer_losses = super().loss_by_feat_single(
                matching_cls[layer_id],
                matching_bbox[layer_id],
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
            )
            loss_dict[f"d{layer_id}.loss_cls"] = layer_losses[0]
            loss_dict[f"d{layer_id}.loss_bbox"] = layer_losses[1]
            loss_dict[f"d{layer_id}.loss_iou"] = layer_losses[2]

        if enc_cls_scores is not None:
            enc_losses = super().loss_by_feat_single(
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
            )
            loss_dict["enc_loss_cls"] = enc_losses[0]
            loss_dict["enc_loss_bbox"] = enc_losses[1]
            loss_dict["enc_loss_iou"] = enc_losses[2]

        if dn_cls is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                dn_cls,
                dn_bbox,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta,
            )
            loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
            loss_dict["dn_loss_iou"] = dn_losses_iou[-1]
            for layer_id, (loss_cls, loss_bbox, loss_iou) in enumerate(
                zip(
                    dn_losses_cls[:-1],
                    dn_losses_bbox[:-1],
                    dn_losses_iou[:-1],
                )
            ):
                loss_dict[f"d{layer_id}.dn_loss_cls"] = loss_cls
                loss_dict[f"d{layer_id}.dn_loss_bbox"] = loss_bbox
                loss_dict[f"d{layer_id}.dn_loss_iou"] = loss_iou

        for group_id, group_name in enumerate(self.basd_reg_state.GROUP_NAMES):
            loss_dict[f"basd_reg_mass_{group_name}"] = diagnostics["mass"][group_id]
            loss_dict[f"basd_reg_error_ema_{group_name}"] = diagnostics["error_ema"][
                group_id
            ]
            loss_dict[f"basd_reg_group_weight_{group_name}"] = diagnostics[
                "group_weight"
            ][group_id]
            loss_dict[f"basd_reg_initialized_{group_name}"] = diagnostics[
                "initialized"
            ][group_id].float()
        for name in (
            "alpha",
            "instance_weight_min",
            "instance_weight_mean",
            "instance_weight_max",
            "unweighted_l1",
            "weighted_l1",
            "unweighted_kld",
            "weighted_kld",
        ):
            loss_dict[f"basd_reg_{name}"] = diagnostics[name]
        return loss_dict


__all__ = ["BASDRegRotatedRTDETRHead"]
