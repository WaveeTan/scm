"""SCM + O²-RTDETR head with FINAL-layer-only SAGR supervision.

Design goal
-----------
The previous SD version overrode ``loss_by_feat_single``. In DINO/O²-RTDETR,
that method is used by:
- every matching decoder layer, and
- the selected encoder proposals.

Therefore the old SD formulation changed both the decoder regression objective
and the encoder box regression that indirectly participates in SCM scale-based
query selection.

This head instead:
1. runs the COMPLETE original O²-RTDETR loss path first;
2. keeps original VFL, L1, KLD/GD, encoder losses and DN losses unchanged;
3. adds one auxiliary ``loss_sagr`` only on the FINAL matching decoder layer.

This removes the direct replacement of encoder / auxiliary-decoder L1 losses.
"""

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
    """RotatedRTDETRHead with final-layer-only residual geometry supervision."""

    def __init__(
        self,
        *args,
        sagr_loss_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        cfg = {} if sagr_loss_cfg is None else dict(sagr_loss_cfg)
        self.loss_sagr = ScaleAspectGeometryResidualLoss(**cfg)

    def _loss_sagr_final(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tensor:
        """Compute SAGR on FINAL matching queries only.

        Hungarian targets are rebuilt for the final layer using detached
        predictions. The detached tensors are used only to decide assignments;
        the SAGR value itself is computed from the original ``bbox_preds`` so
        gradients flow through the final decoder regression path.
        """
        if cls_scores.ndim != 3:
            raise RuntimeError(
                "final cls_scores must be [B, Q, C], "
                f"got {tuple(cls_scores.shape)}."
            )
        if bbox_preds.ndim != 3 or bbox_preds.size(-1) != 5:
            raise RuntimeError(
                "final bbox_preds must be [B, Q, 5], "
                f"got {tuple(bbox_preds.shape)}."
            )

        num_imgs = cls_scores.size(0)
        if bbox_preds.size(0) != num_imgs:
            raise RuntimeError(
                "cls/bbox batch mismatch: "
                f"{tuple(cls_scores.shape)} vs {tuple(bbox_preds.shape)}."
            )

        # Rebuild exactly the same type of matching targets used by the
        # original O² head, but without retaining an unnecessary graph
        # through the Hungarian assignment path.
        cls_scores_list = [
            cls_scores[i].detach()
            for i in range(num_imgs)
        ]
        bbox_preds_list = [
            bbox_preds[i].detach()
            for i in range(num_imgs)
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

        bbox_targets = torch.cat(
            bbox_targets_list,
            dim=0,
        )
        bbox_weights = torch.cat(
            bbox_weights_list,
            dim=0,
        )

        # Same distributed positive-count normalization convention as the
        # original O²-RTDETR head.
        num_total_pos_tensor = bbox_preds.new_tensor(
            [num_total_pos]
        )
        num_total_pos = torch.clamp(
            reduce_mean(num_total_pos_tensor),
            min=1,
        ).item()

        return self.loss_sagr(
            bbox_preds.reshape(-1, 5),
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_pos,
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
        """Keep all original O² losses, then add final-layer SAGR.

        Crucially, this class does NOT override ``loss_by_feat_single``.
        Therefore:
        - encoder selected-query bbox loss -> original L1;
        - decoder d0..d4 bbox losses       -> original L1;
        - final decoder bbox loss          -> original L1;
        - DN bbox losses                    -> original L1;
        - KLD/GD losses                     -> unchanged.

        The only new optimization term is:
            ``loss_sagr`` on final matching decoder queries.
        """
        # ------------------------------------------------------------
        # 1. Run the full original DINO/O²-RTDETR loss pipeline.
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # 2. Split DN queries away and select only the FINAL matching layer.
        # RotatedRTDETRHead.split_outputs works for either per-layer lists
        # or a 4-D tensor whose first dimension is decoder layer.
        # ------------------------------------------------------------
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
            raise RuntimeError(
                "No matching decoder classification outputs were produced."
            )
        if len(all_layers_matching_bbox_preds) == 0:
            raise RuntimeError(
                "No matching decoder bbox outputs were produced."
            )

        final_cls_scores = all_layers_matching_cls_scores[-1]
        final_bbox_preds = all_layers_matching_bbox_preds[-1]

        # ------------------------------------------------------------
        # 3. Add ONLY the final-layer residual geometry loss.
        # ------------------------------------------------------------
        losses["loss_sagr"] = self._loss_sagr_final(
            final_cls_scores,
            final_bbox_preds,
            batch_gt_instances,
            batch_img_metas,
        )

        return losses


__all__ = ["SCMSAGRRotatedRTDETRHead"]
