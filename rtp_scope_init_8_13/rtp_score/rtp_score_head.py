"""RTP-Score head for rotated RT-DETR."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
from projects.rotated_rtdetr.rotated_rtdetr.varifocal_loss import VarifocalLoss

from .asymmetric_loss import AsymmetricLoss
from .config import build_rtp_score_cfg
from .rsu import RotatedSetUniquenessHead, RotatedSetUniquenessLoss
from .rtqd import RotatedThresholdQualityHead
from .scne import (
    build_presence_targets,
    calibrate_class_logits,
    negative_evidence_bias,
)
from .score_fusion import geometric_score_fusion
from .utils.rotated_matching import (
    RTPImageTargets,
    RotatedMatchingTargetBuilder,
)
from .utils.tensor_utils import image_factors


@MODELS.register_module()
class RTPScoreRotatedRTDETRHead(RotatedRTDETRHead):
    """Factor true-positive ranking into scene, quality and uniqueness terms."""

    def __init__(
        self,
        *args,
        rtp_score_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rtp_score_cfg = build_rtp_score_cfg(rtp_score_cfg)
        self.rtp_enabled = bool(self.rtp_score_cfg["enabled"])
        self.eps = float(self.rtp_score_cfg["eps"])
        scne_cfg = self.rtp_score_cfg["scne"]
        rtqd_cfg = self.rtp_score_cfg["rtqd"]
        rsu_cfg = self.rtp_score_cfg["rsu"]
        if int(scne_cfg["num_classes"]) != self.num_classes:
            raise ValueError("SCNE and bbox-head class counts must match")

        self.scne_enabled = self.rtp_enabled and bool(scne_cfg["enabled"])
        self.rtqd_enabled = self.rtp_enabled and bool(rtqd_cfg["enabled"])
        self.rsu_enabled = self.rtp_enabled and bool(rsu_cfg["enabled"])
        self.final_rtqd_enabled = (
            self.rtqd_enabled and bool(rtqd_cfg["use_final_decoder"])
        )
        self.encoder_rtqd_enabled = (
            self.rtqd_enabled and bool(rtqd_cfg["use_encoder"])
        )
        self.unique_head_enabled = (
            self.rsu_enabled and bool(rsu_cfg["use_unique_head"])
        )

        self.scene_loss = AsymmetricLoss(
            gamma_pos=scne_cfg["gamma_pos"],
            gamma_neg=scne_cfg["gamma_neg"],
            clip=scne_cfg["asl_clip"],
        )
        self.quality_head = (
            RotatedThresholdQualityHead(
                embed_dims=self.embed_dims,
                thresholds=rtqd_cfg["thresholds"],
                tau=rtqd_cfg["tau"],
            )
            if self.final_rtqd_enabled
            else None
        )
        self.encoder_quality_head = (
            RotatedThresholdQualityHead(
                embed_dims=self.embed_dims,
                thresholds=rtqd_cfg["thresholds"],
                tau=rtqd_cfg["tau"],
            )
            if self.encoder_rtqd_enabled
            else None
        )
        self.rsu_loss = (
            RotatedSetUniquenessLoss(
                rival_iou_thr=rsu_cfg["rival_iou_thr"],
                max_rivals_per_gt=rsu_cfg["max_rivals_per_gt"],
                margin=rsu_cfg["margin"],
                loss_weight=rsu_cfg["loss_weight"],
                require_argmax_class_match=rsu_cfg[
                    "require_argmax_class_match"
                ],
            )
            if self.rsu_enabled
            else None
        )
        self.unique_head = (
            RotatedSetUniquenessHead(embed_dims=self.embed_dims)
            if self.unique_head_enabled
            else None
        )
        self.target_builder = RotatedMatchingTargetBuilder(
            assigner=self.assigner,
            num_classes=self.num_classes,
            angle_cfg=self.angle_cfg,
            angle_factor=self.angle_factor,
        )
        self.register_buffer(
            "scene_calibration_enabled",
            torch.tensor(False),
            persistent=True,
        )

    def set_rtp_epoch(self, epoch: int) -> None:
        scne_cfg = self.rtp_score_cfg["scne"]
        enabled = (
            self.scne_enabled
            and int(epoch) >= int(scne_cfg["warmup_epochs"])
        )
        self.scene_calibration_enabled.fill_(enabled)

    def _matching_query_features(
        self,
        final_query_feats: Tensor,
        dn_meta: Optional[Dict[str, int]],
    ) -> Tensor:
        if final_query_feats.ndim != 3:
            raise ValueError("final_query_feats must have shape [B, Q, C]")
        num_dn = 0 if dn_meta is None else int(
            dn_meta["num_denoising_queries"]
        )
        if not 0 <= num_dn <= final_query_feats.size(1):
            raise ValueError("invalid denoising-query count")
        return final_query_feats[:, num_dn:, :]

    def forward(
        self,
        hidden_states: List[Tensor],
        references: List[Tensor],
        final_query_feats: Optional[Tensor] = None,
        dn_meta: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> dict:
        """Return a structured output while retaining baseline score tensors."""
        output = dict(
            all_layers_cls_scores=hidden_states,
            all_layers_bbox_preds=references,
            final_quality_logits=None,
            final_unique_logits=None,
            final_query_feats=None,
        )
        if final_query_feats is None:
            return output
        matching_features = self._matching_query_features(
            final_query_feats, dn_meta
        )
        output["final_query_feats"] = matching_features
        if self.quality_head is not None:
            output["final_quality_logits"] = self.quality_head(
                matching_features
            )
        return output

    def loss(
        self,
        hidden_states: List[Tensor],
        references: List[Tensor],
        final_query_feats: Tensor,
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Optional[Dict[str, int]],
        enc_outputs_quality: Optional[Tensor] = None,
        scene_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        batch_gt_instances = [
            sample.gt_instances for sample in batch_data_samples
        ]
        batch_img_metas = [
            sample.metainfo for sample in batch_data_samples
        ]
        outputs = self(
            hidden_states,
            references,
            final_query_feats=final_query_feats,
            dn_meta=dn_meta,
        )
        return self.loss_by_feat(
            **outputs,
            enc_cls_scores=enc_outputs_class,
            enc_bbox_preds=enc_outputs_coord,
            enc_quality_logits=enc_outputs_quality,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta,
            scene_logits=scene_logits,
        )

    def _classification_loss_from_targets(
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
        flat_cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        flat_bbox_preds = bbox_preds.reshape(-1, 5)
        avg_factor = num_total_pos + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(
                flat_cls_scores.new_tensor([avg_factor])
            )
        avg_factor = max(avg_factor, 1)
        if not isinstance(self.loss_cls, VarifocalLoss):
            return self.loss_cls(
                flat_cls_scores,
                labels,
                label_weights,
                avg_factor=avg_factor,
            )

        pos_inds = (
            (labels >= 0) & (labels < self.num_classes)
        ).nonzero().squeeze(1)
        cls_iou_targets = label_weights.new_zeros(flat_cls_scores.shape)
        if pos_inds.numel():
            if self.varifocal_loss_iou_type == "hbox_iou":
                target_xyxy = bbox_cxcywh_to_xyxy(
                    bbox_targets[pos_inds, :4]
                )
                pred_xyxy = bbox_cxcywh_to_xyxy(
                    flat_bbox_preds[pos_inds, :4]
                )
                quality = bbox_overlaps(
                    pred_xyxy.detach(), target_xyxy, is_aligned=True
                )
            elif self.varifocal_loss_iou_type in (
                "rbox_iou",
                "prob_iou",
            ):
                # Build factors per image, not from batch_img_metas[0].
                factors = image_factors(
                    batch_img_metas, bbox_preds, self.angle_factor
                )
                pred_rboxes = flat_bbox_preds[pos_inds] * factors[pos_inds]
                target_rboxes = bbox_targets[pos_inds] * factors[pos_inds]
                if self.varifocal_loss_iou_type == "rbox_iou":
                    quality = rbbox_overlaps(
                        pred_rboxes.detach(),
                        target_rboxes,
                        mode="iou",
                        is_aligned=True,
                    )
                else:
                    quality = probiou(
                        pred_rboxes.detach(), target_rboxes
                    )[:, 0]
            else:
                raise NotImplementedError(self.varifocal_loss_iou_type)
            cls_iou_targets[pos_inds, labels[pos_inds]] = quality.clamp(0, 1)
        return self.loss_cls(
            flat_cls_scores,
            cls_iou_targets,
            avg_factor=avg_factor,
        )

    def _loss_from_targets(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        num_total_pos: int,
        num_total_neg: int,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        loss_cls = self._classification_loss_from_targets(
            cls_scores,
            bbox_preds,
            labels,
            label_weights,
            bbox_targets,
            num_total_pos,
            num_total_neg,
            batch_img_metas,
        )
        avg_pos = reduce_mean(
            loss_cls.new_tensor([num_total_pos])
        ).clamp_min(1).item()
        factors = image_factors(
            batch_img_metas, bbox_preds, self.angle_factor
        )
        flat_preds = bbox_preds.reshape(-1, 5)
        loss_iou = self.loss_iou(
            flat_preds * factors,
            bbox_targets * factors,
            bbox_weights,
            avg_factor=avg_pos,
        )
        loss_bbox = self.loss_bbox(
            flat_preds,
            bbox_targets,
            bbox_weights,
            avg_factor=avg_pos,
        )
        return loss_cls, loss_bbox, loss_iou

    def _loss_from_records(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        records: Sequence[RTPImageTargets],
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self._loss_from_targets(
            cls_scores,
            bbox_preds,
            torch.cat([record.labels for record in records]),
            torch.cat([record.label_weights for record in records]),
            torch.cat([record.bbox_targets for record in records]),
            torch.cat([record.bbox_weights for record in records]),
            sum(record.pos_inds.numel() for record in records),
            sum(record.neg_inds.numel() for record in records),
            batch_img_metas,
        )

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        targets = self.get_targets(
            list(cls_scores),
            list(bbox_preds),
            batch_gt_instances,
            batch_img_metas,
        )
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_pos,
            num_total_neg,
        ) = targets
        return self._loss_from_targets(
            cls_scores,
            bbox_preds,
            torch.cat(labels),
            torch.cat(label_weights),
            torch.cat(bbox_targets),
            torch.cat(bbox_weights),
            num_total_pos,
            num_total_neg,
            batch_img_metas,
        )

    def _loss_dn_single(
        self,
        dn_cls_scores: Tensor,
        dn_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        targets = self.get_dn_targets(
            batch_gt_instances, batch_img_metas, dn_meta
        )
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_pos,
            num_total_neg,
        ) = targets
        return self._loss_from_targets(
            dn_cls_scores,
            dn_bbox_preds,
            torch.cat(labels),
            torch.cat(label_weights),
            torch.cat(bbox_targets),
            torch.cat(bbox_weights),
            num_total_pos,
            num_total_neg,
            batch_img_metas,
        )

    def _scene_bias(
        self,
        scene_logits: Optional[Tensor],
        *,
        decoder: bool,
    ) -> Optional[Tensor]:
        scne_cfg = self.rtp_score_cfg["scne"]
        apply = (
            scne_cfg["apply_to_decoder"]
            if decoder
            else scne_cfg["apply_to_encoder"]
        )
        if scene_logits is None or not self.scne_enabled or not apply:
            return None
        return negative_evidence_bias(
            scene_logits,
            calibration_lambda=scne_cfg["calibration_lambda"],
            presence_threshold=scne_cfg["presence_threshold"],
            min_bias=scne_cfg["min_bias"],
            eps=self.eps,
            detach=scne_cfg["detach_calibration"],
            enabled=bool(self.scene_calibration_enabled.item()),
        )

    def _calibrated_logits(
        self,
        cls_scores: Tensor,
        scene_logits: Optional[Tensor],
    ) -> Tensor:
        bias = self._scene_bias(scene_logits, decoder=True)
        if bias is None:
            return cls_scores
        return calibrate_class_logits(cls_scores, bias)

    def _current_detection_scores(
        self,
        cls_scores: Tensor,
        quality_logits: Optional[Tensor],
        scene_logits: Optional[Tensor],
    ) -> Tensor:
        class_prob = self._calibrated_logits(
            cls_scores, scene_logits
        ).sigmoid()
        if quality_logits is None:
            return class_prob
        cfg = self.rtp_score_cfg["rtqd"]
        return geometric_score_fusion(
            class_prob,
            quality_logits.sigmoid()[..., :1],
            cls_exp=cfg["final_cls_exp"],
            quality_exp=cfg["final_quality_exp"],
            eps=self.eps,
        )

    def _decoded_box_list(
        self,
        bbox_preds: Tensor,
        batch_img_metas: List[dict],
    ) -> List[Tensor]:
        decoded = []
        for boxes, img_meta in zip(bbox_preds, batch_img_metas):
            img_h, img_w = img_meta["img_shape"][:2]
            factor = boxes.new_tensor(
                [img_w, img_h, img_w, img_h, self.angle_factor]
            )
            decoded.append(boxes * factor)
        return decoded

    def _scene_loss_and_metrics(
        self,
        scene_logits: Optional[Tensor],
        batch_gt_instances: InstanceList,
    ) -> Dict[str, Tensor]:
        if not self.scne_enabled:
            return {}
        if scene_logits is None:
            raise ValueError("SCNE is enabled but scene_logits were not routed")
        cfg = self.rtp_score_cfg["scne"]
        targets = build_presence_targets(
            [instances.labels for instances in batch_gt_instances],
            self.num_classes,
            reference=scene_logits,
        )
        probabilities = scene_logits.sigmoid()
        predictions = probabilities >= 0.5
        true_positive = (predictions & targets.bool()).sum().float()
        precision = true_positive / predictions.sum().clamp_min(1)
        recall = true_positive / targets.sum().clamp_min(1)
        bias = self._scene_bias(scene_logits, decoder=True)
        if bias is None:
            bias = torch.zeros_like(scene_logits)
        return {
            "loss_scene_presence": self.scene_loss(
                scene_logits, targets
            ) * float(cfg["loss_weight"]),
            "scene_presence_precision": precision.detach(),
            "scene_presence_recall": recall.detach(),
            "scene_bias_mean": bias.mean().detach(),
            "scene_bias_min": bias.min().detach(),
            "scene_suppressed_class_ratio": (
                (bias < 0).float().mean().detach()
            ),
        }

    def loss_by_feat(
        self,
        all_layers_cls_scores: List[Tensor],
        all_layers_bbox_preds: List[Tensor],
        final_quality_logits: Optional[Tensor],
        final_unique_logits: Optional[Tensor],
        final_query_feats: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_quality_logits: Optional[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Optional[Dict[str, int]],
        scene_logits: Optional[Tensor] = None,
        batch_gt_instances_ignore=None,
    ) -> Dict[str, Tensor]:
        del final_unique_logits
        if batch_gt_instances_ignore is not None:
            raise AssertionError("RTP-Score does not support ignored GT")
        matching_cls, matching_bbox, dn_cls, dn_bbox = self.split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta
        )
        final_cls = matching_cls[-1]
        final_bbox = matching_bbox[-1]
        final_records = self.target_builder.build_batch(
            final_cls,
            final_bbox,
            batch_gt_instances,
            batch_img_metas,
        )
        final_losses = self._loss_from_records(
            final_cls, final_bbox, final_records, batch_img_metas
        )
        losses = dict(
            loss_cls=final_losses[0],
            loss_bbox=final_losses[1],
            loss_iou=final_losses[2],
        )

        for layer_id in range(len(matching_cls) - 1):
            layer_loss = self.loss_by_feat_single(
                matching_cls[layer_id],
                matching_bbox[layer_id],
                batch_gt_instances,
                batch_img_metas,
            )
            losses[f"d{layer_id}.loss_cls"] = layer_loss[0]
            losses[f"d{layer_id}.loss_bbox"] = layer_loss[1]
            losses[f"d{layer_id}.loss_iou"] = layer_loss[2]

        if enc_cls_scores is not None:
            encoder_records = self.target_builder.build_batch(
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances,
                batch_img_metas,
            )
            encoder_loss = self._loss_from_records(
                enc_cls_scores,
                enc_bbox_preds,
                encoder_records,
                batch_img_metas,
            )
            losses["enc_loss_cls"] = encoder_loss[0]
            losses["enc_loss_bbox"] = encoder_loss[1]
            losses["enc_loss_iou"] = encoder_loss[2]
            if self.encoder_quality_head is not None:
                if enc_quality_logits is None:
                    raise ValueError(
                        "encoder RTQD is enabled but logits are missing"
                    )
                enc_rtqd = self.encoder_quality_head.loss(
                    enc_quality_logits,
                    [record.pos_inds for record in encoder_records],
                    [record.rotated_iou for record in encoder_records],
                    loss_weight=self.rtp_score_cfg["rtqd"]["loss_weight"],
                    monotonic_weight=self.rtp_score_cfg["rtqd"][
                        "monotonic_weight"
                    ],
                )
                losses["enc_loss_rtqd"] = enc_rtqd[0]
                losses["enc_loss_rtqd_mono"] = enc_rtqd[1]
                for name, value in enc_rtqd[2].items():
                    losses[f"enc_{name}"] = value

        if dn_cls is not None:
            dn_loss_cls, dn_loss_bbox, dn_loss_iou = self.loss_dn(
                dn_cls,
                dn_bbox,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta,
            )
            losses["dn_loss_cls"] = dn_loss_cls[-1]
            losses["dn_loss_bbox"] = dn_loss_bbox[-1]
            losses["dn_loss_iou"] = dn_loss_iou[-1]
            for layer_id, layer_loss in enumerate(
                zip(
                    dn_loss_cls[:-1],
                    dn_loss_bbox[:-1],
                    dn_loss_iou[:-1],
                )
            ):
                losses[f"d{layer_id}.dn_loss_cls"] = layer_loss[0]
                losses[f"d{layer_id}.dn_loss_bbox"] = layer_loss[1]
                losses[f"d{layer_id}.dn_loss_iou"] = layer_loss[2]

        losses.update(
            self._scene_loss_and_metrics(
                scene_logits, batch_gt_instances
            )
        )

        if self.quality_head is not None:
            if final_quality_logits is None:
                raise ValueError("final RTQD is enabled but logits are missing")
            rtqd_loss = self.quality_head.loss(
                final_quality_logits,
                [record.pos_inds for record in final_records],
                [record.rotated_iou for record in final_records],
                loss_weight=self.rtp_score_cfg["rtqd"]["loss_weight"],
                monotonic_weight=self.rtp_score_cfg["rtqd"][
                    "monotonic_weight"
                ],
            )
            losses["loss_rtqd"] = rtqd_loss[0]
            losses["loss_rtqd_mono"] = rtqd_loss[1]
            losses.update(rtqd_loss[2])

        current_scores = self._current_detection_scores(
            final_cls, final_quality_logits, scene_logits
        )
        selections = None
        if self.rsu_loss is not None:
            loss_rsu, diagnostics, selections = self.rsu_loss(
                current_scores, final_records
            )
            losses["loss_rsu_pairwise"] = loss_rsu
            losses.update(diagnostics)

        if self.unique_head is not None:
            if selections is None:
                raise RuntimeError("unique head requires RSU selections")
            unique_logits = self.unique_head(
                final_query_feats,
                self._decoded_box_list(final_bbox, batch_img_metas),
                final_cls,
            )
            losses["loss_unique_head"] = self.unique_head.loss(
                unique_logits,
                selections,
                self.rtp_score_cfg["rsu"]["unique_head_loss_weight"],
            )
        return losses

    def _fusion_exponents(self) -> Tuple[float, float, float]:
        if self.unique_head_enabled:
            cfg = self.rtp_score_cfg["score_fusion"]
            return cfg["cls_exp"], cfg["quality_exp"], cfg["unique_exp"]
        if self.final_rtqd_enabled:
            cfg = self.rtp_score_cfg["rtqd"]
            return cfg["final_cls_exp"], cfg["final_quality_exp"], 0.0
        return 1.0, 0.0, 0.0

    def predict(
        self,
        hidden_states: List[Tensor],
        references: List[Tensor],
        final_query_feats: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
        scene_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> InstanceList:
        ranking_active = (
            self.final_rtqd_enabled
            or self.unique_head_enabled
            or (
                self.scne_enabled
                and self.rtp_score_cfg["scne"]["apply_to_decoder"]
                and bool(self.scene_calibration_enabled.item())
            )
            or bool(self.test_cfg.get("export_query_details", False))
        )
        if not ranking_active:
            return super().predict(
                hidden_states,
                references,
                batch_data_samples,
                rescale=rescale,
            )

        outputs = self(
            hidden_states,
            references,
            final_query_feats=final_query_feats,
            dn_meta=None,
        )
        cls_scores = outputs["all_layers_cls_scores"][-1]
        bbox_preds = outputs["all_layers_bbox_preds"][-1]
        quality_logits = outputs["final_quality_logits"]
        batch_img_metas = [
            sample.metainfo for sample in batch_data_samples
        ]
        unique_logits = None
        if self.unique_head is not None:
            unique_logits = self.unique_head(
                outputs["final_query_feats"],
                self._decoded_box_list(bbox_preds, batch_img_metas),
                cls_scores,
            )
        if scene_logits is None:
            scene_parts = [None] * len(batch_img_metas)
        else:
            scene_parts = list(scene_logits)
        quality_parts = (
            [None] * len(batch_img_metas)
            if quality_logits is None
            else list(quality_logits)
        )
        unique_parts = (
            [None] * len(batch_img_metas)
            if unique_logits is None
            else list(unique_logits)
        )
        return [
            self._predict_by_feat_single_rtp(
                cls_score,
                bbox_pred,
                quality_logit,
                unique_logit,
                scene_logit,
                img_meta,
                rescale,
            )
            for (
                cls_score,
                bbox_pred,
                quality_logit,
                unique_logit,
                scene_logit,
                img_meta,
            ) in zip(
                cls_scores,
                bbox_preds,
                quality_parts,
                unique_parts,
                scene_parts,
                batch_img_metas,
            )
        ]

    def _rescale_boxes(
        self,
        boxes: Tensor,
        img_meta: dict,
        rescale: bool,
    ) -> Tensor:
        img_h, img_w = img_meta["img_shape"][:2]
        decoded = boxes.clone()
        decoded[:, 0:4:2] *= img_w
        decoded[:, 1:4:2] *= img_h
        decoded[:, 4] *= self.angle_factor
        decoded[:, 0:4:2].clamp_(min=0, max=img_w)
        decoded[:, 1:4:2].clamp_(min=0, max=img_h)
        if rescale:
            scale = np.asarray(
                img_meta["scale_factor"], dtype=np.float32
            ).reshape(-1)
            if scale.size == 2:
                scale = np.tile(scale, 2)
            if scale.size != 4:
                raise ValueError("scale_factor must contain 2 or 4 values")
            decoded = decoded / decoded.new_tensor(
                np.append(scale, 1.0)
            )
        return decoded

    def _predict_by_feat_single_rtp(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        quality_logit: Optional[Tensor],
        unique_logit: Optional[Tensor],
        scene_logit: Optional[Tensor],
        img_meta: dict,
        rescale: bool,
    ) -> InstanceData:
        if self.loss_cls.use_sigmoid:
            class_prob = cls_score.sigmoid()
        else:
            class_prob = F.softmax(cls_score, dim=-1)[..., :-1]
        if scene_logit is not None:
            bias = self._scene_bias(
                scene_logit.unsqueeze(0), decoder=True
            )
            if bias is not None:
                if self.loss_cls.use_sigmoid:
                    class_prob = (
                        cls_score + bias.squeeze(0).unsqueeze(0)
                    ).sigmoid()
                else:
                    raise ValueError(
                        "SCNE decoder calibration requires sigmoid classes"
                    )

        quality_prob = (
            None
            if quality_logit is None
            else quality_logit.sigmoid()[..., :1]
        )
        unique_prob = (
            None if unique_logit is None else unique_logit.sigmoid()
        )
        cls_exp, quality_exp, unique_exp = self._fusion_exponents()
        fused = geometric_score_fusion(
            class_prob,
            quality_prob,
            unique_prob,
            cls_exp=cls_exp,
            quality_exp=quality_exp,
            unique_exp=unique_exp,
            eps=self.eps,
        )
        max_per_img = int(
            self.test_cfg.get("max_per_img", bbox_pred.size(0))
        )
        topk = min(max_per_img, fused.numel())
        scores, flat_indices = fused.reshape(-1).topk(topk)
        labels = flat_indices % self.num_classes
        query_indices = flat_indices // self.num_classes
        decoded_all = self._rescale_boxes(bbox_pred, img_meta, rescale)
        results = InstanceData(
            bboxes=decoded_all[query_indices],
            scores=scores,
            labels=labels,
            query_indices=query_indices,
            class_scores=class_prob[query_indices, labels],
        )
        if quality_prob is not None:
            results.quality_scores = quality_prob[
                query_indices, 0
            ]
        if unique_prob is not None:
            results.unique_scores = unique_prob[query_indices, 0]
        if scene_logit is not None:
            results.scene_scores = scene_logit.sigmoid()[labels]

        if self.test_cfg.get("export_query_details", False):
            metadata = dict(
                rtp_query_bboxes=decoded_all.detach().cpu().numpy(),
                rtp_query_class_probabilities=(
                    class_prob.detach().cpu().numpy()
                ),
                rtp_query_fused_scores=fused.detach().cpu().numpy(),
            )
            if quality_logit is not None:
                metadata["rtp_query_quality_probabilities"] = (
                    quality_logit.sigmoid().detach().cpu().numpy()
                )
            if unique_logit is not None:
                metadata["rtp_query_unique_probabilities"] = (
                    unique_logit.sigmoid().detach().cpu().numpy()
                )
            if scene_logit is not None:
                metadata["rtp_scene_probabilities"] = (
                    scene_logit.sigmoid().detach().cpu().numpy()
                )
            results.set_metainfo(metadata)

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


__all__ = ["RTPScoreRotatedRTDETRHead"]
