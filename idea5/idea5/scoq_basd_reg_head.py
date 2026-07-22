"""Idea3 BASD-Reg plus scene-conditioned oriented quality calibration."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, reduce_mean
from torch import Tensor

from ai4rs.registry import MODELS
from ai4rs.structures.bbox import rbbox_overlaps
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead
from projects.rotated_rtdetr.rotated_rtdetr.prob_iou import probiou
from projects.rotated_rtdetr.rotated_rtdetr.varifocal_loss import (
    VarifocalLoss,
    varifocal_loss,
)

from .basd_reg_grouping import BASDRegGroupState


@MODELS.register_module()
class SCOQBASDRegRotatedRTDETRHead(RotatedRTDETRHead):
    """BASD-Reg final regression with final-query rotated-quality scoring."""

    def __init__(
        self,
        *args,
        basd_reg_cfg: Optional[dict] = None,
        scoq_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        basd_cfg = {} if basd_reg_cfg is None else dict(basd_reg_cfg)
        apply_to = basd_cfg.pop("apply_to", "final_decoder_regression")
        use_ambiguity = bool(basd_cfg.pop("use_ambiguity", False))
        use_progress = bool(basd_cfg.pop("use_progress", False))
        if apply_to != "final_decoder_regression":
            raise ValueError("BASD-Reg v1 only supports final decoder regression")
        if use_ambiguity or use_progress:
            raise ValueError("BASD-Reg v1 disables ambiguity and progress")
        self.basd_reg_state = BASDRegGroupState(**basd_cfg)

        cfg = {} if scoq_cfg is None else dict(scoq_cfg)
        hidden_dims = int(cfg.pop("quality_hidden_dims", 64))
        self.quality_loss_weight = float(cfg.pop("quality_loss_weight", 0.25))
        self.quality_beta = float(cfg.pop("quality_beta", 0.25))
        self.quality_detach_inputs = bool(cfg.pop("quality_detach_inputs", True))
        self.use_scene_conditioning = bool(cfg.pop("use_scene_conditioning", True))
        self.quality_supervise_negatives = bool(
            cfg.pop("quality_supervise_negatives", True)
        )
        self.quality_vfl_alpha = float(cfg.pop("quality_vfl_alpha", 0.75))
        self.quality_vfl_gamma = float(cfg.pop("quality_vfl_gamma", 2.0))
        self.quality_loss_type = str(cfg.pop("quality_loss_type", "vfl"))
        self.quality_prior_prob = float(cfg.pop("quality_prior_prob", 0.01))
        self.quality_eps = float(cfg.pop("quality_eps", 1e-6))
        if cfg:
            raise ValueError(f"Unknown scoq_cfg keys: {sorted(cfg)}")
        if hidden_dims <= 0:
            raise ValueError("quality_hidden_dims must be positive")
        if self.quality_loss_weight < 0:
            raise ValueError("quality_loss_weight must be non-negative")
        if not 0 <= self.quality_beta < 1:
            raise ValueError("quality_beta must be in [0, 1)")
        if not 0 < self.quality_prior_prob < 1:
            raise ValueError("quality_prior_prob must be in (0, 1)")
        if self.quality_eps <= 0:
            raise ValueError("quality_eps must be positive")
        if self.quality_loss_type not in ("vfl", "mse"):
            raise ValueError("quality_loss_type must be 'vfl' or 'mse'")

        self.scene_gamma = nn.Parameter(
            torch.zeros(1), requires_grad=self.use_scene_conditioning
        )
        self.quality_head = nn.Sequential(
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
        )
        self.register_buffer("quality_loss_scale", torch.ones(()), persistent=False)

    def init_weights(self) -> None:
        super().init_weights()
        first_linear = self.quality_head[1]
        output_linear = self.quality_head[3]
        nn.init.normal_(first_linear.weight, std=0.01)
        nn.init.zeros_(first_linear.bias)
        # Keep q constant before the delayed loss is enabled.  Once the output
        # layer moves, gradients also reach the hidden projection.
        nn.init.zeros_(output_linear.weight)
        prior_bias = math.log(self.quality_prior_prob / (1.0 - self.quality_prior_prob))
        nn.init.constant_(output_linear.bias, prior_bias)

    def set_basd_reg_alpha(self, value: float) -> float:
        return self.basd_reg_state.set_alpha(value)

    def set_scoq_loss_scale(self, value: float) -> float:
        applied = min(max(float(value), 0.0), 1.0)
        self.quality_loss_scale.fill_(applied)
        return applied

    def set_quality_beta(self, value: float) -> float:
        if not 0 <= value < 1:
            raise ValueError("quality beta must be in [0, 1)")
        self.quality_beta = float(value)
        return self.quality_beta

    def _quality_logits(
        self,
        final_query: Tensor,
        scene_feature: Tensor,
        dn_meta: Optional[Dict[str, int]],
    ) -> Tensor:
        if final_query.ndim != 3:
            raise ValueError("final_query must have shape [B, N, C]")
        if scene_feature.ndim != 2:
            raise ValueError("scene_feature must have shape [B, C]")
        if final_query.size(0) != scene_feature.size(0):
            raise ValueError("final_query and scene_feature batch sizes disagree")
        if final_query.size(-1) != self.embed_dims:
            raise ValueError("final_query channel dimension is not embed_dims")
        if scene_feature.size(-1) != self.embed_dims:
            raise ValueError("scene_feature channel dimension is not embed_dims")

        num_dn = 0 if dn_meta is None else int(dn_meta["num_denoising_queries"])
        if not 0 <= num_dn <= final_query.size(1):
            raise ValueError("invalid number of denoising queries")
        query_feature = final_query[:, num_dn:, :]
        scene = scene_feature
        if self.quality_detach_inputs:
            query_feature = query_feature.detach()
            scene = scene.detach()
        quality_feature = query_feature
        if self.use_scene_conditioning:
            # Detach the two inputs, not the sum: scene_gamma must still learn.
            quality_feature = quality_feature + self.scene_gamma * scene[:, None, :]
        return self.quality_head(quality_feature)

    def forward(
        self,
        hidden_states: List[Tensor],
        references: List[Tensor],
        final_query: Optional[Tensor] = None,
        scene_feature: Optional[Tensor] = None,
        dn_meta: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> tuple:
        # Parent loss/predict helpers call ``self(hidden_states, references)``.
        # Preserve that two-output contract for the beta=0 parity path.
        if final_query is None and scene_feature is None:
            return hidden_states, references
        if final_query is None or scene_feature is None:
            raise ValueError("final_query and scene_feature must be provided together")
        return (
            hidden_states,
            references,
            self._quality_logits(final_query, scene_feature, dn_meta),
        )

    def loss(
        self,
        hidden_states: List[Tensor],
        references: List[Tensor],
        final_query: Tensor,
        scene_feature: Tensor,
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Optional[Dict[str, int]],
    ) -> Dict[str, Tensor]:
        batch_gt_instances = [
            data_sample.gt_instances for data_sample in batch_data_samples
        ]
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        outputs = self(
            hidden_states,
            references,
            final_query,
            scene_feature,
            dn_meta,
        )
        return self.loss_by_feat(
            *outputs,
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances,
            batch_img_metas,
            dn_meta,
        )

    def _get_targets_with_membership_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
    ) -> tuple:
        """Run one Hungarian assignment for BASD and SCOQ targets."""
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
        assigned = assign_result.gt_inds[pos_inds] - 1

        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[assigned]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(decoded_bbox_pred)
        bbox_weights = torch.zeros_like(decoded_bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = gt_bboxes[assigned.long()] / factor

        gt_membership = self.basd_reg_state.compute_membership(
            gt_bboxes, img_meta["img_shape"]
        )
        pos_membership = gt_membership[assigned.long()]
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
        """Exact classification path retained from Idea3."""
        flat_cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        flat_bbox_preds = bbox_preds.reshape(-1, 5)
        cls_avg_factor = num_total_pos + num_total_neg * self.bg_cls_weight
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
            quality = bbox_overlaps(pred_xyxy.detach(), target_xyxy, is_aligned=True)
        elif self.varifocal_loss_iou_type in ("rbox_iou", "prob_iou"):
            factors = self._flatten_factors(batch_img_metas, bbox_preds)
            pred_rboxes = flat_bbox_preds[pos_inds] * factors[pos_inds]
            target_rboxes = bbox_targets[pos_inds] * factors[pos_inds]
            if self.varifocal_loss_iou_type == "rbox_iou":
                quality = rbbox_overlaps(
                    pred_rboxes.detach(), target_rboxes, is_aligned=True
                )
            else:
                quality = probiou(pred_rboxes.detach(), target_rboxes)[:, 0]
        else:
            raise NotImplementedError(self.varifocal_loss_iou_type)
        cls_iou_targets[pos_inds, labels[pos_inds]] = quality
        return self.loss_cls(
            flat_cls_scores, cls_iou_targets, avg_factor=cls_avg_factor
        )

    def _flatten_factors(
        self, batch_img_metas: List[dict], bbox_preds: Tensor
    ) -> Tensor:
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
        return torch.cat(factors)

    def _oriented_quality_loss(
        self,
        quality_logits: Tensor,
        quality_targets: Tensor,
        positive_mask: Tensor,
        num_total_pos: int,
    ) -> Tensor:
        avg_factor = (
            reduce_mean(quality_logits.new_tensor([num_total_pos])).clamp_min(1).item()
        )
        weights = None
        if not self.quality_supervise_negatives:
            weights = positive_mask.to(quality_logits).unsqueeze(-1)
        if self.quality_loss_type == "vfl":
            raw_loss = varifocal_loss(
                quality_logits,
                quality_targets,
                weight=weights,
                alpha=self.quality_vfl_alpha,
                gamma=self.quality_vfl_gamma,
                iou_weighted=True,
                reduction="mean",
                avg_factor=avg_factor,
            )
        else:
            per_query = (quality_logits.sigmoid() - quality_targets).square()
            if weights is not None:
                per_query = per_query * weights
            raw_loss = per_query.sum() / avg_factor
        return raw_loss * self.quality_loss_weight * self.quality_loss_scale

    def _loss_final_matching(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        quality_logits: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
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

        flat_bbox_preds = bbox_preds.reshape(-1, 5)
        flat_quality_logits = quality_logits.reshape(-1, 1)
        if flat_quality_logits.size(0) != flat_bbox_preds.size(0):
            raise ValueError("SCOQ logits must cover final matching queries only")
        factors = self._flatten_factors(batch_img_metas, bbox_preds)
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
            rotated_iou = rbbox_overlaps(
                decoded_pred_pos.detach(),
                decoded_target_pos,
                mode="iou",
                is_aligned=True,
            ).clamp(0, 1)
        else:
            connected_zero = flat_bbox_preds[pos_mask, 0]
            per_l1 = connected_zero
            per_kld = connected_zero
            rotated_iou = pred_pos.new_zeros((0,))

        instance_weight, group_weight, mass = self.basd_reg_state.compute_weights(
            membership
        )
        loss_bbox = self.basd_reg_state.global_weighted_mean(per_l1, instance_weight)
        loss_iou = self.basd_reg_state.global_weighted_mean(per_kld, instance_weight)

        quality_targets = flat_quality_logits.new_zeros(flat_quality_logits.shape)
        quality_targets[pos_mask, 0] = rotated_iou.detach()
        loss_quality = self._oriented_quality_loss(
            flat_quality_logits,
            quality_targets,
            pos_mask,
            num_total_pos,
        )

        unweighted = torch.ones_like(instance_weight)
        unweighted_l1 = self.basd_reg_state.global_weighted_mean(
            per_l1.detach(), unweighted
        )
        unweighted_kld = self.basd_reg_state.global_weighted_mean(
            per_kld.detach(), unweighted
        )
        weight_stats = self.basd_reg_state.summarize_instance_weights(instance_weight)
        with torch.no_grad():
            state_info = self.basd_reg_state.update(1 - rotated_iou, membership)
            quality_pred_pos = flat_quality_logits[pos_mask].sigmoid().squeeze(-1)
            quality_count = self.basd_reg_state.global_sum(
                rotated_iou.new_tensor(float(num_positive))
            )
            quality_denominator = quality_count.clamp_min(1.0)
            quality_target_mean = (
                self.basd_reg_state.global_sum(rotated_iou.sum()) / quality_denominator
            )
            quality_pred_mean = (
                self.basd_reg_state.global_sum(quality_pred_pos.sum())
                / quality_denominator
            )
            quality_brier = (
                self.basd_reg_state.global_sum(
                    (quality_pred_pos - rotated_iou).square().sum()
                )
                / quality_denominator
            )

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
            quality_target_mean=quality_target_mean.detach(),
            quality_pred_mean=quality_pred_mean.detach(),
            quality_brier=quality_brier.detach(),
            quality_scale=self.quality_loss_scale.detach().clone(),
            scene_gamma=self.scene_gamma.detach().clone().squeeze(),
        )
        return loss_cls, loss_bbox, loss_iou, loss_quality, diagnostics

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        quality_logits: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Optional[Dict[str, int]],
        batch_gt_instances_ignore=None,
    ) -> Dict[str, Tensor]:
        if batch_gt_instances_ignore is not None:
            raise AssertionError("Idea5 does not support ignored GT")
        matching_cls, matching_bbox, dn_cls, dn_bbox = self.split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta
        )
        final_losses = self._loss_final_matching(
            matching_cls[-1],
            matching_bbox[-1],
            quality_logits,
            batch_gt_instances,
            batch_img_metas,
        )
        final_cls, final_bbox, final_iou, final_quality, diagnostics = final_losses
        loss_dict = dict(
            loss_cls=final_cls,
            loss_bbox=final_bbox,
            loss_iou=final_iou,
            loss_oriented_quality=final_quality,
        )

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
        for name in (
            "quality_target_mean",
            "quality_pred_mean",
            "quality_brier",
            "quality_scale",
            "scene_gamma",
        ):
            loss_dict[f"scoq_{name}"] = diagnostics[name]
        return loss_dict

    def predict(
        self,
        hidden_states: List[Tensor],
        references: List[Tensor],
        final_query: Tensor,
        scene_feature: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
        **kwargs,
    ) -> InstanceList:
        # This exact branch is the equivalence control for Idea3.
        if self.quality_beta == 0:
            return super().predict(
                hidden_states,
                references,
                batch_data_samples,
                rescale=rescale,
            )
        quality_logits = self._quality_logits(final_query, scene_feature, dn_meta=None)
        cls_scores = hidden_states[-1]
        bbox_preds = references[-1]
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        return [
            self._predict_single_with_quality(
                cls_score,
                bbox_pred,
                quality_logit,
                img_meta,
                rescale,
            )
            for cls_score, bbox_pred, quality_logit, img_meta in zip(
                cls_scores, bbox_preds, quality_logits, batch_img_metas
            )
        ]

    def _fuse_probabilities(self, class_prob: Tensor, quality_prob: Tensor) -> Tensor:
        if self.quality_beta == 0:
            return class_prob
        log_class = class_prob.clamp_min(self.quality_eps).log()
        log_quality = quality_prob.clamp_min(self.quality_eps).log()
        return torch.exp(
            (1.0 - self.quality_beta) * log_class + self.quality_beta * log_quality
        )

    def _predict_single_with_quality(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        quality_logit: Tensor,
        img_meta: dict,
        rescale: bool,
    ) -> InstanceData:
        if not (len(cls_score) == len(bbox_pred) == len(quality_logit)):
            raise ValueError("classification, box and quality queries disagree")
        max_per_img = self.test_cfg.get("max_per_img", len(cls_score))
        img_h, img_w = img_meta["img_shape"][:2]
        quality_prob = quality_logit.sigmoid()

        if self.loss_cls.use_sigmoid:
            class_prob = cls_score.sigmoid()
            fused = self._fuse_probabilities(class_prob, quality_prob)
            scores, indexes = fused.reshape(-1).topk(max_per_img)
            labels = indexes % self.num_classes
            query_indices = indexes // self.num_classes
        else:
            class_prob = F.softmax(cls_score, dim=-1)[..., :-1]
            fused = self._fuse_probabilities(class_prob, quality_prob)
            query_scores, query_labels = fused.max(-1)
            scores, query_indices = query_scores.topk(max_per_img)
            labels = query_labels[query_indices]

        det_bboxes = bbox_pred[query_indices].clone()
        det_quality = quality_prob[query_indices].reshape(-1)
        det_bboxes[:, 0:4:2] *= img_w
        det_bboxes[:, 1:4:2] *= img_h
        det_bboxes[:, 4] *= self.angle_factor
        det_bboxes[:, 0:4:2].clamp_(min=0, max=img_w)
        det_bboxes[:, 1:4:2].clamp_(min=0, max=img_h)
        if rescale:
            if img_meta.get("scale_factor") is None:
                raise ValueError("scale_factor is required when rescale=True")
            scale_factor = np.asarray(img_meta["scale_factor"], dtype=np.float32)
            scale_factor = scale_factor.reshape(-1)
            if scale_factor.size == 2:
                scale_factor = np.tile(scale_factor, 2)
            elif scale_factor.size != 4:
                raise ValueError("scale_factor must contain 2 or 4 values")
            scale_factor = np.append(scale_factor, 1.0)
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

        results = InstanceData(
            bboxes=det_bboxes,
            scores=scores,
            labels=labels,
            quality_scores=det_quality,
        )
        nms_cfg = self.test_cfg.get("nms", None)
        if nms_cfg is not None:
            _, keep = batched_nms(
                boxes=results.bboxes,
                scores=results.scores,
                idxs=results.labels,
                nms_cfg=nms_cfg,
            )
            results = results[keep]
        return results


__all__ = ["SCOQBASDRegRotatedRTDETRHead"]
