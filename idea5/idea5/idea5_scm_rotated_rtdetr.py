"""SCM Rotated RT-DETR detector that routes scene and query features to SCOQ."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ai4rs.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr.rtdetr_layers import (
    RTDETRHybridEncoder,
)
from projects.scm_rotated_rtdetr.scm_rotated_rtdetr import SCMRotatedRTDETR

from .scoq_decoder import SCOQRotatedRTDETRTransformerDecoder


@MODELS.register_module()
class Idea5SCMRotatedRTDETR(SCMRotatedRTDETR):
    """Idea3 SCM+BASD detector with final-query SCOQ feature routing."""

    def _init_layers(self) -> None:
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = SCOQRotatedRTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples=None,
    ) -> Tuple[Dict, Dict]:
        _, _, channels = memory.shape
        prediction_layer = self.decoder.num_layers
        cls_out_features = self.bbox_head.cls_branches[prediction_layer].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        enc_outputs_class = self.bbox_head.cls_branches[prediction_layer](output_memory)
        scene_outputs = self.scene_context(
            memory=output_memory,
            spatial_shapes=spatial_shapes,
            memory_mask=memory_mask,
        )
        selection_outputs_class = enc_outputs_class
        if self.use_scene_class_bias:
            selection_outputs_class = (
                selection_outputs_class + scene_outputs["class_bias"][:, None, :]
            )

        topk_indices = torch.topk(
            selection_outputs_class.max(-1)[0], k=self.num_queries, dim=1
        )[1]
        query = torch.gather(
            output_memory,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, channels),
        )
        selected_proposals = torch.gather(
            output_proposals,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 5),
        )
        topk_coords_unact = (
            self.bbox_head.reg_branches[prediction_layer](query) + selected_proposals
        )

        if self.training:
            topk_score = torch.gather(
                selection_outputs_class,
                1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
            )
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples
            )
            query = torch.cat([dn_label_query, query.detach()], dim=1)
            dn_bbox_query = dn_bbox_query.type_as(topk_coords_unact)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
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
        # Scene features are needed during both loss and prediction.  Keeping
        # them in head inputs also avoids mutable detector-side caches at test.
        head_inputs = dict(scene_feature=scene_outputs["scene_feature"])
        if self.training:
            self._latest_scene_outputs = scene_outputs
            head_inputs.update(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
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
            final_query=final_query,
        )


__all__ = ["Idea5SCMRotatedRTDETR"]
