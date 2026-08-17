from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mmcv.ops import batched_nms
from mmengine.structures import InstanceData

from mmdet.structures import SampleList
from mmdet.utils import InstanceList

from torch import Tensor

from ai4rs.registry import MODELS

from projects.rotated_rtdetr.rotated_rtdetr import (
    RotatedRTDETRHead,
)

from .rtqd import RotatedThresholdQualityHead
from .rotated_matching import (
    RotatedMatchingTargetBuilder,
)


@MODELS.register_module()
class SCMRTQDRotatedRTDETRHead(
    RotatedRTDETRHead
):
    """Rotated RT-DETR head with final-only RTQD.

    SCM is responsible for encoder query selection.
    RTQD is used only for final decoder ranking.

    RTQD feature gradients are detached from the decoder.
    """

    def __init__(
        self,
        *args,
        rtqd_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:

        super().__init__(
            *args,
            **kwargs,
        )

        # ---------------------------------------------------------
        # Minimal RTQD configuration.
        # ---------------------------------------------------------
        cfg = dict(
            enabled=True,

            thresholds=(
                0.5,
                0.6,
                0.7,
                0.8,
            ),

            tau=0.05,

            loss_weight=1.0,
            monotonic_weight=0.10,

            # Current implementation:
            # Hungarian-aware unique TP.
            unmatched_policy='unique_tp',

            # Final score:
            # p_cls^1.0 * q50^0.2
            final_cls_exp=1.0,
            final_quality_exp=0.20,

            eps=1e-6,
        )

        if rtqd_cfg is not None:
            cfg.update(
                dict(rtqd_cfg)
            )

        if cfg[
            'unmatched_policy'
        ] != 'unique_tp':
            raise ValueError(
                'SCM-RTQD currently supports only '
                'unmatched_policy="unique_tp".'
            )

        self.rtqd_cfg = cfg
        self.rtqd_enabled = bool(
            cfg['enabled']
        )

        self.eps = float(
            cfg['eps']
        )

        # ---------------------------------------------------------
        # Final decoder RTQD head.
        # ---------------------------------------------------------
        self.quality_head = (
            RotatedThresholdQualityHead(
                embed_dims=self.embed_dims,
                thresholds=cfg[
                    'thresholds'
                ],
                tau=cfg['tau'],
            )
            if self.rtqd_enabled
            else None
        )

        # ---------------------------------------------------------
        # One Hungarian assignment provides:
        #
        # pos_inds
        # matched rIoU
        # full pairwise rIoU
        #
        # Used by Unique-TP RTQD.
        # ---------------------------------------------------------
        self.target_builder = (
            RotatedMatchingTargetBuilder(
                assigner=self.assigner,
                num_classes=self.num_classes,
                angle_cfg=self.angle_cfg,
                angle_factor=self.angle_factor,
            )
            if self.rtqd_enabled
            else None
        )

    # =============================================================
    # Helper: remove denoising queries
    # =============================================================

    def _matching_query_features(
        self,
        final_query_feats: Tensor,
        dn_meta: Optional[
            Dict[str, int]
        ],
    ) -> Tensor:

        if final_query_feats.ndim != 3:
            raise ValueError(
                'final_query_feats must have '
                'shape [B, Q, C]'
            )

        num_dn = (
            0
            if dn_meta is None
            else int(
                dn_meta[
                    'num_denoising_queries'
                ]
            )
        )

        if not (
            0
            <= num_dn
            <= final_query_feats.size(1)
        ):
            raise ValueError(
                'Invalid number of '
                'denoising queries.'
            )

        return final_query_feats[
            :,
            num_dn:,
            :
        ]

    # =============================================================
    # Loss
    # =============================================================

    def loss(
        self,
        hidden_states,
        references,
        final_query_feats: Tensor,
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Optional[
            Dict[str, int]
        ],
        **kwargs,
    ) -> Dict[str, Tensor]:

        # ---------------------------------------------------------
        # 1. Original O2 / Rotated RT-DETR losses.
        #
        # Nothing about bbox/classification training is changed.
        # ---------------------------------------------------------
        losses = super().loss(
            hidden_states=hidden_states,
            references=references,
            enc_outputs_class=(
                enc_outputs_class
            ),
            enc_outputs_coord=(
                enc_outputs_coord
            ),
            batch_data_samples=(
                batch_data_samples
            ),
            dn_meta=dn_meta,
        )

        if not self.rtqd_enabled:
            return losses

        # ---------------------------------------------------------
        # 2. Recover original decoder prediction tensors.
        #
        # RotatedRTDETRHead.forward() returns:
        # (
        #   all_layers_cls_scores,
        #   all_layers_bbox_preds
        # )
        # ---------------------------------------------------------
        (
            all_layers_cls_scores,
            all_layers_bbox_preds,
        ) = self(
            hidden_states,
            references,
        )

        (
            matching_cls,
            matching_bbox,
            _,
            _,
        ) = self.split_outputs(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            dn_meta,
        )

        final_cls = matching_cls[-1]
        final_bbox = matching_bbox[-1]

        # ---------------------------------------------------------
        # 3. Final decoder query features.
        #
        # IMPORTANT:
        # detach prevents RTQD loss from modifying decoder.
        # ---------------------------------------------------------
        matching_features = (
            self._matching_query_features(
                final_query_feats,
                dn_meta,
            )
        )

        quality_features = (
            matching_features.detach()
        )

        final_quality_logits = (
            self.quality_head(
                quality_features
            )
        )

        # ---------------------------------------------------------
        # 4. Build Hungarian-aware Unique-TP targets.
        # ---------------------------------------------------------
        batch_gt_instances = [
            sample.gt_instances
            for sample
            in batch_data_samples
        ]

        batch_img_metas = [
            sample.metainfo
            for sample
            in batch_data_samples
        ]

        final_records = (
            self.target_builder.build_batch(
                final_cls,
                final_bbox,
                batch_gt_instances,
                batch_img_metas,
            )
        )

        # ---------------------------------------------------------
        # 5. Unique-TP RTQD.
        #
        # positive:
        #   matched rIoU
        #
        # unmatched:
        #   target IoU = 0
        #
        # pairwise IoU is still supplied so that RTQD can
        # distinguish easy background from high-IoU duplicates.
        # ---------------------------------------------------------
        rtqd_loss = (
            self.quality_head.loss(
                final_quality_logits,
                [
                    record.pos_inds
                    for record
                    in final_records
                ],
                [
                    record.rotated_iou
                    for record
                    in final_records
                ],

                pairwise_ious=[
                    record.pairwise_iou
                    for record
                    in final_records
                ],

                loss_weight=float(
                    self.rtqd_cfg[
                        'loss_weight'
                    ]
                ),

                monotonic_weight=float(
                    self.rtqd_cfg[
                        'monotonic_weight'
                    ]
                ),
            )
        )

        losses['loss_rtqd'] = (
            rtqd_loss[0]
        )

        losses[
            'loss_rtqd_mono'
        ] = rtqd_loss[1]

        # q50 diagnostics etc.
        losses.update(
            rtqd_loss[2]
        )

        return losses

    # =============================================================
    # Prediction
    # =============================================================

    def predict(
        self,
        hidden_states,
        references,
        final_query_feats: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
        **kwargs,
    ) -> InstanceList:

        quality_exp = float(
            self.rtqd_cfg[
                'final_quality_exp'
            ]
        )

        # ---------------------------------------------------------
        # R0:
        #
        # quality exponent = 0
        # -> exact baseline classification ranking path.
        # ---------------------------------------------------------
        if (
            not self.rtqd_enabled
            or quality_exp <= 0.0
        ):
            return super().predict(
                hidden_states,
                references,
                batch_data_samples,
                rescale=rescale,
            )

        (
            all_layers_cls_scores,
            all_layers_bbox_preds,
        ) = self(
            hidden_states,
            references,
        )

        final_cls = (
            all_layers_cls_scores[-1]
        )

        final_bbox = (
            all_layers_bbox_preds[-1]
        )

        # In inference there are no DN queries.
        quality_features = (
            final_query_feats.detach()
        )

        quality_logits = (
            self.quality_head(
                quality_features
            )
        )

        batch_img_metas = [
            sample.metainfo
            for sample
            in batch_data_samples
        ]

        return [
            self._predict_by_feat_single_rtqd(
                cls_score,
                bbox_pred,
                quality_logit,
                img_meta,
                rescale,
            )
            for (
                cls_score,
                bbox_pred,
                quality_logit,
                img_meta,
            ) in zip(
                final_cls,
                final_bbox,
                quality_logits,
                batch_img_metas,
            )
        ]

    # =============================================================
    # Score fusion
    # =============================================================

    def _predict_by_feat_single_rtqd(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        quality_logit: Tensor,
        img_meta: dict,
        rescale: bool,
    ) -> InstanceData:

        # ---------------------------------------------------------
        # Classification probability.
        # ---------------------------------------------------------
        if self.loss_cls.use_sigmoid:
            class_prob = (
                cls_score.sigmoid()
            )
        else:
            class_prob = (
                F.softmax(
                    cls_score,
                    dim=-1,
                )[..., :-1]
            )

        # RTQD uses q50 for AP50-oriented ranking.
        quality_prob = (
            quality_logit
            .sigmoid()[..., :1]
        )

        cls_exp = float(
            self.rtqd_cfg[
                'final_cls_exp'
            ]
        )

        quality_exp = float(
            self.rtqd_cfg[
                'final_quality_exp'
            ]
        )

        # ---------------------------------------------------------
        # Final ranking:
        #
        # S = p_cls^1.0 * q50^0.2
        # ---------------------------------------------------------
        fused_score = (
            class_prob
            .clamp_min(self.eps)
            .pow(cls_exp)
            *
            quality_prob
            .clamp_min(self.eps)
            .pow(quality_exp)
        )

        max_per_img = int(
            self.test_cfg.get(
                'max_per_img',
                bbox_pred.size(0),
            )
        )

        topk = min(
            max_per_img,
            fused_score.numel(),
        )

        (
            scores,
            flat_indices,
        ) = (
            fused_score
            .reshape(-1)
            .topk(topk)
        )

        labels = (
            flat_indices
            % self.num_classes
        )

        query_indices = (
            flat_indices
            // self.num_classes
        )

        # ---------------------------------------------------------
        # Decode normalized OBB.
        # ---------------------------------------------------------
        decoded_all = (
            self._decode_boxes(
                bbox_pred,
                img_meta,
                rescale,
            )
        )

        results = InstanceData(
            bboxes=decoded_all[
                query_indices
            ],
            scores=scores,
            labels=labels,
        )

        # Useful for diagnostics.
        results.query_indices = (
            query_indices
        )

        results.class_scores = (
            class_prob[
                query_indices,
                labels,
            ]
        )

        results.quality_scores = (
            quality_prob[
                query_indices,
                0,
            ]
        )

        # Optional NMS.
        nms_cfg = self.test_cfg.get(
            'nms',
            None,
        )

        if nms_cfg is not None:
            _, keep = batched_nms(
                boxes=results.bboxes,
                scores=results.scores,
                idxs=results.labels,
                nms_cfg=nms_cfg,
            )

            results = results[keep]

        return results

    # =============================================================
    # Decode / rescale OBB
    # =============================================================

    def _decode_boxes(
        self,
        bbox_pred: Tensor,
        img_meta: dict,
        rescale: bool,
    ) -> Tensor:

        img_h, img_w = (
            img_meta['img_shape'][:2]
        )

        decoded = bbox_pred.clone()

        decoded[:, 0] *= img_w
        decoded[:, 1] *= img_h
        decoded[:, 2] *= img_w
        decoded[:, 3] *= img_h
        decoded[:, 4] *= (
            self.angle_factor
        )

        decoded[:, 0].clamp_(
            min=0,
            max=img_w,
        )

        decoded[:, 1].clamp_(
            min=0,
            max=img_h,
        )

        if rescale:
            scale = np.asarray(
                img_meta[
                    'scale_factor'
                ],
                dtype=np.float32,
            ).reshape(-1)

            if scale.size == 2:
                scale = np.tile(
                    scale,
                    2,
                )

            if scale.size != 4:
                raise ValueError(
                    'scale_factor must '
                    'contain 2 or 4 values'
                )

            scale_tensor = (
                decoded.new_tensor(
                    [
                        scale[0],
                        scale[1],
                        scale[2],
                        scale[3],
                        1.0,
                    ]
                )
            )

            decoded = (
                decoded
                / scale_tensor
            )

        return decoded


__all__ = [
    'SCMRTQDRotatedRTDETRHead'
]