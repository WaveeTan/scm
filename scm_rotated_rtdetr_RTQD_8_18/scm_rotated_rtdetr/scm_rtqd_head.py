from typing import Dict, Optional

import torch
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.utils import InstanceList

from ai4rs.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETRHead

from .rotated_matching import RotatedMatchingTargetBuilder
from .rtqd import RotatedThresholdQualityHead


@MODELS.register_module()
class SCMRTQDRotatedRTDETRHead(RotatedRTDETRHead):
    """Final-only detached Unique-TP RTQD head.

    Detection losses remain the original O2-RTDETR losses. RTQD receives only
    ``final_query_feats`` explicitly exported by ``SCMRotatedRTDETR`` and is
    detached from the decoder before the quality MLP.
    """

    def __init__(
        self,
        *args,
        rtqd_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        cfg = dict(
            enabled=True,
            thresholds=(0.5, 0.6, 0.7, 0.8),
            tau=0.05,
            loss_weight=1.0,
            monotonic_weight=0.10,
            final_cls_exp=1.0,
            final_quality_exp=0.20,
            eps=1e-6,
        )
        if rtqd_cfg is not None:
            cfg.update(dict(rtqd_cfg))

        self.rtqd_cfg = cfg
        self.rtqd_enabled = bool(cfg['enabled'])
        self.eps = float(cfg['eps'])

        if self.rtqd_enabled:
            self.quality_head = RotatedThresholdQualityHead(
                embed_dims=self.embed_dims,
                thresholds=cfg['thresholds'],
                tau=cfg['tau'],
            )
            self.rtqd_target_builder = RotatedMatchingTargetBuilder(
                assigner=self.assigner,
                num_classes=self.num_classes,
                angle_cfg=self.angle_cfg,
                angle_factor=self.angle_factor,
            )
        else:
            self.quality_head = None
            self.rtqd_target_builder = None

    def _matching_query_features(
        self,
        final_query_feats: Tensor,
        dn_meta: Optional[Dict[str, int]],
    ) -> Tensor:
        """Remove DN queries from terminal decoder query embeddings."""
        if not torch.is_tensor(final_query_feats):
            raise TypeError(
                'final_query_feats must be a Tensor, got '
                f'{type(final_query_feats)}'
            )
        if final_query_feats.ndim != 3:
            raise RuntimeError(
                'final_query_feats must be [B, Q_total, C], got '
                f'{tuple(final_query_feats.shape)}'
            )
        if final_query_feats.size(-1) != self.embed_dims:
            raise RuntimeError(
                'RTQD requires decoder query embeddings with dim '
                f'{self.embed_dims}, but got {final_query_feats.size(-1)}. '
                f'Full shape={tuple(final_query_feats.shape)}. '
                'Do not pass classification logits to RTQD.'
            )

        num_dn = (
            0
            if dn_meta is None
            else int(dn_meta['num_denoising_queries'])
        )
        if not 0 <= num_dn <= final_query_feats.size(1):
            raise RuntimeError(
                f'Invalid num_denoising_queries={num_dn} for '
                f'Q_total={final_query_feats.size(1)}.'
            )

        matching_features = final_query_feats[:, num_dn:, :]

        if matching_features.size(-1) != self.embed_dims:
            raise RuntimeError(
                'RTQD matching feature dimension mismatch: '
                f'{matching_features.size(-1)} vs {self.embed_dims}'
            )

        return matching_features

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
        """Original O2 losses + final-only detached Unique-TP RTQD loss."""

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
        # 2. Recover final matching-query cls / bbox predictions
        # ========================================================
        all_layers_cls_scores, all_layers_bbox_preds = self(
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
        # 3. RTQD feature comes ONLY from explicit final_query_feats
        # ========================================================
        matching_features = self._matching_query_features(
            final_query_feats,
            dn_meta,
        )

        if final_cls.ndim != 3:
            raise RuntimeError(
                'final_cls must be [B, Q, num_classes], got '
                f'{tuple(final_cls.shape)}'
            )
        if final_bbox.ndim != 3:
            raise RuntimeError(
                'final_bbox must be [B, Q, 5], got '
                f'{tuple(final_bbox.shape)}'
            )
        if matching_features.ndim != 3:
            raise RuntimeError(
                'matching_features must be [B, Q, C], got '
                f'{tuple(matching_features.shape)}'
            )
        if matching_features.size(0) != final_cls.size(0):
            raise RuntimeError(
                'RTQD batch mismatch: feature batch '
                f'{matching_features.size(0)} vs cls batch {final_cls.size(0)}'
            )
        if matching_features.size(1) != final_cls.size(1):
            raise RuntimeError(
                'RTQD query-count mismatch: feature queries '
                f'{matching_features.size(1)} vs cls queries '
                f'{final_cls.size(1)}'
            )
        if matching_features.size(-1) != self.embed_dims:
            raise RuntimeError(
                'RTQD feature dimension mismatch: '
                f'{matching_features.size(-1)} vs {self.embed_dims}'
            )

        # ========================================================
        # 4. Detach RTQD from decoder
        # ========================================================
        quality_features = matching_features.detach()
        final_quality_logits = self.quality_head(quality_features)

        expected_quality_dim = len(self.quality_head.thresholds)
        if final_quality_logits.ndim != 3:
            raise RuntimeError(
                'final_quality_logits must be [B, Q, T], got '
                f'{tuple(final_quality_logits.shape)}'
            )
        if final_quality_logits.shape[:2] != final_cls.shape[:2]:
            raise RuntimeError(
                'RTQD output/query shape mismatch: '
                f'{tuple(final_quality_logits.shape)} vs '
                f'{tuple(final_cls.shape)}'
            )
        if final_quality_logits.size(-1) != expected_quality_dim:
            raise RuntimeError(
                'RTQD threshold dimension mismatch: '
                f'{final_quality_logits.size(-1)} vs {expected_quality_dim}'
            )

        # ========================================================
        # 5. Hungarian-aware Unique-TP targets
        # ========================================================
        batch_gt_instances = [
            sample.gt_instances for sample in batch_data_samples
        ]
        batch_img_metas = [
            sample.metainfo for sample in batch_data_samples
        ]

        records = self.rtqd_target_builder.build_batch(
            final_cls,
            final_bbox,
            batch_gt_instances,
            batch_img_metas,
        )

        loss_rtqd, loss_rtqd_mono, rtqd_diagnostics = self.quality_head.loss(
            final_quality_logits,
            positive_indices=[record.pos_inds for record in records],
            positive_ious=[record.rotated_iou for record in records],
            pairwise_ious=[record.pairwise_iou for record in records],
            loss_weight=float(self.rtqd_cfg['loss_weight']),
            monotonic_weight=float(self.rtqd_cfg['monotonic_weight']),
        )

        losses['loss_rtqd'] = loss_rtqd
        losses['loss_rtqd_mono'] = loss_rtqd_mono
        losses.update(rtqd_diagnostics)
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
        """Predict with p_cls^alpha * q50^beta final ranking."""

        if not self.rtqd_enabled:
            return super().predict(
                hidden_states,
                references,
                batch_data_samples,
                rescale=rescale,
            )

        quality_exp = float(self.rtqd_cfg['final_quality_exp'])

        # R0: exact baseline ranking path from the same RTQD-trained checkpoint.
        if quality_exp <= 0.0:
            return super().predict(
                hidden_states,
                references,
                batch_data_samples,
                rescale=rescale,
            )

        if not self.loss_cls.use_sigmoid:
            raise ValueError(
                'Current RTQD score fusion assumes sigmoid classification.'
            )

        all_layers_cls_scores, all_layers_bbox_preds = self(
            hidden_states,
            references,
        )
        final_cls = all_layers_cls_scores[-1]
        final_bbox = all_layers_bbox_preds[-1]

        # Inference has no DN queries, so final_query_feats should contain
        # exactly the same Q matching queries as final_cls/final_bbox.
        if not torch.is_tensor(final_query_feats):
            raise TypeError(
                'final_query_feats must be a Tensor during inference, got '
                f'{type(final_query_feats)}'
            )
        if final_query_feats.ndim != 3:
            raise RuntimeError(
                'Inference final_query_feats must be [B, Q, C], got '
                f'{tuple(final_query_feats.shape)}'
            )
        if final_query_feats.size(-1) != self.embed_dims:
            raise RuntimeError(
                'Inference RTQD feature dim mismatch: '
                f'{final_query_feats.size(-1)} vs {self.embed_dims}. '
                f'Full shape={tuple(final_query_feats.shape)}'
            )
        if final_query_feats.shape[:2] != final_cls.shape[:2]:
            raise RuntimeError(
                'Inference RTQD query shape mismatch: '
                f'feature={tuple(final_query_feats.shape)}, '
                f'cls={tuple(final_cls.shape)}'
            )

        quality_logits = self.quality_head(final_query_feats.detach())

        batch_img_metas = [
            sample.metainfo for sample in batch_data_samples
        ]

        cls_exp = float(self.rtqd_cfg['final_cls_exp'])
        results = []

        for cls_score, bbox_pred, quality_logit, img_meta in zip(
            final_cls,
            final_bbox,
            quality_logits,
            batch_img_metas,
        ):
            class_prob = cls_score.sigmoid()
            quality_prob = quality_logit.sigmoid()[:, :1]  # q50

            fused_prob = (
                class_prob.clamp_min(self.eps).pow(cls_exp)
                * quality_prob.clamp_min(self.eps).pow(quality_exp)
            )

            # Reuse the exact O2-RTDETR post-processing by converting the
            # fused probability back to logits.
            fused_logits = torch.logit(
                fused_prob.clamp(
                    min=self.eps,
                    max=1.0 - self.eps,
                )
            )

            pred = super()._predict_by_feat_single(
                fused_logits,
                bbox_pred,
                img_meta,
                rescale=rescale,
            )
            results.append(pred)

        return results


__all__ = ['SCMRTQDRotatedRTDETRHead']
