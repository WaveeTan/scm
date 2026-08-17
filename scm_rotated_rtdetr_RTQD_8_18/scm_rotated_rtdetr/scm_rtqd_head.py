from typing import Dict, Optional

import torch
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.utils import InstanceList

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
    """Final-only detached Unique-TP RTQD head."""

    def __init__(
        self,
        *args,
        rtqd_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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

            # final score
            final_cls_exp=1.0,
            final_quality_exp=0.20,

            eps=1e-6,
        )

        if rtqd_cfg is not None:
            cfg.update(dict(rtqd_cfg))

        self.rtqd_cfg = cfg
        self.rtqd_enabled = bool(
            cfg['enabled']
        )
        self.eps = float(
            cfg['eps']
        )

        if self.rtqd_enabled:
            self.quality_head = (
                RotatedThresholdQualityHead(
                    embed_dims=self.embed_dims,
                    thresholds=cfg['thresholds'],
                    tau=cfg['tau'],
                )
            )

            self.rtqd_target_builder = (
                RotatedMatchingTargetBuilder(
                    assigner=self.assigner,
                    num_classes=self.num_classes,
                    angle_cfg=self.angle_cfg,
                    angle_factor=self.angle_factor,
                )
            )
        else:
            self.quality_head = None
            self.rtqd_target_builder = None

    def _matching_query_features(
        self,
        final_query_feats: Tensor,
        dn_meta: Optional[
            Dict[str, int]
        ],
    ) -> Tensor:
        """Remove DN queries from terminal decoder features."""

        if final_query_feats.ndim != 3:
            raise ValueError(
                'final_query_feats must be [B, Q, C]'
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
            0 <= num_dn
            <= final_query_feats.size(1)
        ):
            raise ValueError(
                'Invalid number of DN queries.'
            )

        return final_query_feats[
            :,
            num_dn:,
            :
        ]

    def loss(
        self,
        hidden_states,
        references,
        final_query_feats: Tensor,
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta,
        **kwargs,
    ) -> Dict[str, Tensor]:

        # ========================================================
        # 1. EXACT original O2-RTDETR losses
        # ========================================================

        losses = super().loss(
            hidden_states=hidden_states,
            references=references,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            batch_data_samples=batch_data_samples,
            dn_meta=dn_meta,
        )

        if not self.rtqd_enabled:
            return losses

        # ========================================================
        # 2. Recover decoder cls / bbox predictions
        # ========================================================

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

        # ========================================================
        # 3. Matching decoder features only
        # ========================================================

        matching_features = (
            self._matching_query_features(
                final_query_feats,
                dn_meta,
            )
        )

        if (
            matching_features.size(1)
            != final_cls.size(1)
        ):
            raise RuntimeError(
                'RTQD feature/query count mismatch: '
                f'{matching_features.size(1)} vs '
                f'{final_cls.size(1)}'
            )

        # ========================================================
        # 4. IMPORTANT: detach from decoder
        # ========================================================

        quality_features = (
            matching_features.detach()
        )

        final_quality_logits = (
            self.quality_head(
                quality_features
            )
        )

        # ========================================================
        # 5. Hungarian-aware Unique-TP target
        # ========================================================

        batch_gt_instances = [
            sample.gt_instances
            for sample in batch_data_samples
        ]

        batch_img_metas = [
            sample.metainfo
            for sample in batch_data_samples
        ]

        records = (
            self.rtqd_target_builder.build_batch(
                final_cls,
                final_bbox,
                batch_gt_instances,
                batch_img_metas,
            )
        )

        rtqd_loss = self.quality_head.loss(
            final_quality_logits,

            positive_indices=[
                record.pos_inds
                for record in records
            ],

            positive_ious=[
                record.rotated_iou
                for record in records
            ],

            pairwise_ious=[
                record.pairwise_iou
                for record in records
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

        losses['loss_rtqd'] = (
            rtqd_loss[0]
        )

        losses['loss_rtqd_mono'] = (
            rtqd_loss[1]
        )

        # RTQD mechanism diagnostics
        losses.update(
            rtqd_loss[2]
        )

        return losses

    def predict(
        self,
        hidden_states,
        references,
        final_query_feats: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
        **kwargs,
    ) -> InstanceList:

        if not self.rtqd_enabled:
            return super().predict(
                hidden_states,
                references,
                batch_data_samples,
                rescale=rescale,
            )

        quality_exp = float(
            self.rtqd_cfg[
                'final_quality_exp'
            ]
        )

        # ========================================================
        # R0:
        # q_exp = 0 means EXACT baseline ranking
        # ========================================================

        if quality_exp <= 0.0:
            return super().predict(
                hidden_states,
                references,
                batch_data_samples,
                rescale=rescale,
            )

        if not self.loss_cls.use_sigmoid:
            raise ValueError(
                'Current RTQD score fusion assumes '
                'sigmoid classification.'
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

        # No DN queries during inference.
        quality_logits = (
            self.quality_head(
                final_query_feats.detach()
            )
        )

        batch_img_metas = [
            sample.metainfo
            for sample in batch_data_samples
        ]

        results = []

        cls_exp = float(
            self.rtqd_cfg[
                'final_cls_exp'
            ]
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
        ):

            class_prob = (
                cls_score.sigmoid()
            )

            # q50
            quality_prob = (
                quality_logit
                .sigmoid()[:, :1]
            )

            # ====================================================
            # S = p_cls^1.0 * q50^beta
            # ====================================================

            fused_prob = (
                class_prob
                .clamp_min(self.eps)
                .pow(cls_exp)
                *
                quality_prob
                .clamp_min(self.eps)
                .pow(quality_exp)
            )

            # Reuse EXACT original RT-DETR post-processing.
            fused_logits = torch.logit(
                fused_prob.clamp(
                    min=self.eps,
                    max=1.0 - self.eps,
                )
            )

            pred = (
                super()
                ._predict_by_feat_single(
                    fused_logits,
                    bbox_pred,
                    img_meta,
                    rescale=rescale,
                )
            )

            results.append(pred)

        return results


__all__ = [
    'SCMRTQDRotatedRTDETRHead'
]