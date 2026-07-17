from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmdet.structures import OptSampleList
from torch import Tensor

from ai4rs.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR

from .scene_context_module import SceneContextModule
from .utils import build_scene_class_targets


@MODELS.register_module()
class SCMRotatedRTDETR(RotatedRTDETR):
    """RotatedRTDETR with a bounded scene-conditioned class bias.

    The detector keeps the original RotatedRTDETR query selection, regression
    proposal, decoder, matcher and detection losses. The only selection change
    is adding a bounded image-level scene class bias to encoder class logits
    before the original global top-k query selection.
    """

    def __init__(self,
                 *args,
                 scene_cfg: dict = None,
                 use_scene_class_bias: bool = True,
                 loss_scene_cls_weight: float = 0.05,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_scene_class_bias = bool(use_scene_class_bias)
        self.loss_scene_cls_weight = float(loss_scene_cls_weight)
        self._latest_scene_outputs = None

        scene_cfg = {} if scene_cfg is None else dict(scene_cfg)
        self.scene_context = SceneContextModule(
            embed_dims=self.embed_dims,
            num_classes=self.bbox_head.num_classes,
            **scene_cfg)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)

        scene_outputs = self.scene_context(
            memory=output_memory,
            spatial_shapes=spatial_shapes,
            memory_mask=memory_mask)
        if self.use_scene_class_bias:
            selection_outputs_class = (
                enc_outputs_class + scene_outputs['class_bias'][:, None, :])
        else:
            selection_outputs_class = enc_outputs_class

        # Keep the original RotatedRTDETR global top-k policy.
        topk_indices = torch.topk(
            selection_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))
        topk_output_proposals = torch.gather(
            output_proposals, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 5))
        topk_coords_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](query) + topk_output_proposals

        if self.training:
            topk_score = torch.gather(
                selection_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()

            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = query.detach()
            query = torch.cat([dn_label_query, query], dim=1)
            dn_bbox_query = dn_bbox_query.type_as(topk_coords_unact)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            cls_branches=self.bbox_head.cls_branches,
            eval_idx=self.eval_idx)

        head_inputs_dict = dict()
        if self.training:
            self._latest_scene_outputs = scene_outputs
            head_inputs_dict = dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta)
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: OptSampleList) -> dict:
        self._latest_scene_outputs = None
        losses = super().loss(batch_inputs, batch_data_samples)

        scene_outputs = self._latest_scene_outputs
        if scene_outputs is not None and self.loss_scene_cls_weight > 0:
            raw_class_logits = scene_outputs.get('raw_class_logits', None)
            if raw_class_logits is None:
                raw_class_logits = scene_outputs.get('raw_class_bias', None)
            if raw_class_logits is not None:
                scene_targets = build_scene_class_targets(
                    batch_data_samples,
                    self.bbox_head.num_classes,
                    device=raw_class_logits.device,
                    dtype=raw_class_logits.dtype)
                loss_scene_cls = F.binary_cross_entropy_with_logits(
                    raw_class_logits, scene_targets, reduction='mean')
                losses['loss_scene_cls'] = (
                    loss_scene_cls * self.loss_scene_cls_weight)
        return losses
