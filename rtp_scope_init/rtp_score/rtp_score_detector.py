"""O2-RT-DETR detector integration for RTP-Score."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from mmdet.structures import OptSampleList
from torch import Tensor

from ai4rs.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR
from projects.rotated_rtdetr.rotated_rtdetr.rtdetr_layers import (
    RTDETRHybridEncoder,
)

from .config import build_rtp_score_cfg
from .rtp_score_decoder import RTPScoreRotatedRTDETRTransformerDecoder
from .scne import (
    SceneNegativeEvidenceHead,
    calibrate_class_logits,
    negative_evidence_bias,
)
from .score_fusion import geometric_score_fusion
from .utils.tensor_utils import batched_gather


@MODELS.register_module()
class RTPScoreRotatedRTDETR(RotatedRTDETR):
    """Rotated RT-DETR with explicit SCNE and encoder-RTQD routing."""

    def __init__(
        self,
        *args,
        rtp_score_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rtp_score_cfg = build_rtp_score_cfg(rtp_score_cfg)
        self.rtp_enabled = bool(self.rtp_score_cfg["enabled"])
        scne_cfg = self.rtp_score_cfg["scne"]
        self.scne_enabled = self.rtp_enabled and bool(scne_cfg["enabled"])
        if self.scne_enabled:
            if int(scne_cfg["num_classes"]) != self.bbox_head.num_classes:
                raise ValueError(
                    "SCNE num_classes must match bbox_head.num_classes"
                )
            self.scene_head = SceneNegativeEvidenceHead(
                embed_dims=self.embed_dims,
                num_classes=self.bbox_head.num_classes,
                hidden_dim=int(scne_cfg["hidden_dim"]),
                topk_ratio=float(scne_cfg["topk_ratio"]),
            )
        else:
            self.scene_head = None
        self.register_buffer(
            "encoder_rerank_enabled",
            torch.tensor(False),
            persistent=True,
        )

    def _init_layers(self) -> None:
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = RTPScoreRotatedRTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def set_rtp_epoch(self, epoch: int) -> None:
        """Apply warmup gates; called by :class:`RTPScoreScheduleHook`."""
        rtqd_cfg = self.rtp_score_cfg["rtqd"]
        rerank = (
            self.rtp_enabled
            and rtqd_cfg["enabled"]
            and rtqd_cfg["use_encoder"]
            and int(epoch) >= int(rtqd_cfg["encoder_rerank_start_epoch"])
        )
        self.encoder_rerank_enabled.fill_(rerank)
        if hasattr(self.bbox_head, "set_rtp_epoch"):
            self.bbox_head.set_rtp_epoch(epoch)

    def forward_encoder(
        self,
        mlvl_feats: Tuple[Tensor],
        spatial_shapes: Tensor,
    ) -> Dict:
        outputs = super().forward_encoder(mlvl_feats, spatial_shapes)
        scene_logits = None
        if self.scene_head is not None:
            scene_logits = self.scene_head(
                outputs["memory"], outputs["memory_mask"]
            )
        outputs["scene_logits"] = scene_logits
        return outputs

    def _encoder_scene_logits(
        self,
        cls_logits: Tensor,
        scene_logits: Optional[Tensor],
    ) -> Tensor:
        scne_cfg = self.rtp_score_cfg["scne"]
        if (
            scene_logits is None
            or not self.scne_enabled
            or not scne_cfg["apply_to_encoder"]
            or not bool(self.bbox_head.scene_calibration_enabled.item())
        ):
            return cls_logits
        bias = negative_evidence_bias(
            scene_logits,
            calibration_lambda=scne_cfg["calibration_lambda"],
            presence_threshold=scne_cfg["presence_threshold"],
            min_bias=scne_cfg["min_bias"],
            eps=self.rtp_score_cfg["eps"],
            detach=scne_cfg["detach_calibration"],
            enabled=True,
        )
        return calibrate_class_logits(cls_logits, bias)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        scene_logits: Optional[Tensor] = None,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        _, token_count, channels = memory.shape
        prediction_layer = self.decoder.num_layers
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        raw_enc_cls = self.bbox_head.cls_branches[prediction_layer](
            output_memory
        )
        selection_cls = self._encoder_scene_logits(
            raw_enc_cls, scene_logits
        )
        cls_rank = selection_cls.sigmoid().max(dim=-1).values

        rtqd_cfg = self.rtp_score_cfg["rtqd"]
        use_encoder_rtqd = (
            self.rtp_enabled
            and rtqd_cfg["enabled"]
            and rtqd_cfg["use_encoder"]
        )
        enc_quality = None
        if use_encoder_rtqd:
            preselect_k = min(
                int(rtqd_cfg["encoder_preselect_k"]), token_count
            )
            pre_idx = cls_rank.topk(preselect_k, dim=1).indices
            pre_memory = batched_gather(output_memory, pre_idx)
            pre_proposals = batched_gather(output_proposals, pre_idx)
            pre_cls = batched_gather(raw_enc_cls, pre_idx)
            pre_selection_cls = batched_gather(selection_cls, pre_idx)
            pre_coords_unact = (
                self.bbox_head.reg_branches[prediction_layer](pre_memory)
                + pre_proposals
            )
            pre_quality = self.bbox_head.encoder_quality_head(pre_memory)
            pre_cls_rank = pre_selection_cls.sigmoid().max(dim=-1).values
            if bool(self.encoder_rerank_enabled.item()):
                pre_rank = geometric_score_fusion(
                    pre_cls_rank,
                    pre_quality.sigmoid()[..., 0],
                    cls_exp=rtqd_cfg["encoder_cls_exp"],
                    quality_exp=rtqd_cfg["encoder_quality_exp"],
                    eps=self.rtp_score_cfg["eps"],
                )
            else:
                pre_rank = pre_cls_rank
            select_count = min(
                int(rtqd_cfg["encoder_num_select"]),
                self.num_queries,
                preselect_k,
            )
            if select_count != self.num_queries:
                raise ValueError(
                    "encoder_num_select must equal detector num_queries"
                )
            local_idx = pre_rank.topk(select_count, dim=1).indices
            query = batched_gather(pre_memory, local_idx)
            topk_coords_unact = batched_gather(
                pre_coords_unact, local_idx
            )
            topk_score = batched_gather(pre_cls, local_idx)
            enc_quality = batched_gather(pre_quality, local_idx)
        else:
            topk_idx = cls_rank.topk(self.num_queries, dim=1).indices
            query = batched_gather(output_memory, topk_idx)
            selected_proposals = batched_gather(
                output_proposals, topk_idx
            )
            topk_coords_unact = (
                self.bbox_head.reg_branches[prediction_layer](query)
                + selected_proposals
            )
            topk_score = batched_gather(raw_enc_cls, topk_idx)

        if self.training:
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()
            (
                dn_label_query,
                dn_bbox_query,
                dn_mask,
                dn_meta,
            ) = self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query.detach()], dim=1)
            reference_points = torch.cat(
                [
                    dn_bbox_query.type_as(topk_coords_unact),
                    topk_coords_unact,
                ],
                dim=1,
            )
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            cls_branches=self.bbox_head.cls_branches,
            eval_idx=self.eval_idx,
        )
        head_inputs = dict(scene_logits=scene_logits)
        if self.training:
            head_inputs.update(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                enc_outputs_quality=enc_quality,
                dn_meta=dn_meta,
            )
        return decoder_inputs, head_inputs

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        cls_branches=None,
        eval_idx: int = -1,
    ) -> Dict:
        classes, coordinates, final_query = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            cls_branches=cls_branches,
            eval_idx=eval_idx,
        )
        if query.size(1) == self.num_queries:
            classes[0] = classes[0] + (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )
        return dict(
            hidden_states=classes,
            references=coordinates,
            final_query_feats=final_query,
        )


__all__ = ["RTPScoreRotatedRTDETR"]
