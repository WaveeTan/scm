from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures import OptSampleList
from torch import Tensor

from ai4rs.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR
from projects.rotated_rtdetr.rotated_rtdetr.rtdetr_layers import RTDETRHybridEncoder
from .scene_context_module import SceneContextModule
from .utils import build_scene_class_targets,build_scene_scale_targets,build_scene_ar_targets,soft_scale_membership,soft_ar_membership
from .rtqd_decoder import SCMRTQDRotatedRTDETRTransformerDecoder

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
                 use_scene_scale_bias: bool = False,
                 loss_scene_scale_weight: float = 0.05,
                 scale_boundaries=(0.02, 0.04, 0.12),
                 scale_temperature: float = 0.20,
                 use_scene_ar_bias: bool = False,
                 loss_scene_ar_weight: float = 0.05,
                 ar_boundaries=(2.0, 4.0, 8.0),
                 ar_temperature: float = 0.20,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_scene_class_bias = bool(use_scene_class_bias)
        self.loss_scene_cls_weight = float(loss_scene_cls_weight)
        self._latest_scene_outputs = None
        self.use_scene_scale_bias = bool(
            use_scene_scale_bias
        )

        self.loss_scene_scale_weight = float(
            loss_scene_scale_weight
        )

        self.scale_boundaries = tuple(
            float(x) for x in scale_boundaries
        )

        self.scale_temperature = float(
            scale_temperature
        )
        self.use_scene_ar_bias = bool(
            use_scene_ar_bias
        )

        self.loss_scene_ar_weight = float(
            loss_scene_ar_weight
        )

        self.ar_boundaries = tuple(
            float(x) for x in ar_boundaries
        )

        self.ar_temperature = float(
            ar_temperature
        )

        if len(self.scale_boundaries) != 3:
            raise ValueError(
                'scale_boundaries must contain 3 values.'
            )

        if not (
            0.0
            < self.scale_boundaries[0]
            < self.scale_boundaries[1]
            < self.scale_boundaries[2]
        ):
            raise ValueError(
                f'Invalid scale boundaries: '
                f'{self.scale_boundaries}'
            )

        scene_cfg = {} if scene_cfg is None else dict(scene_cfg)
        self.scene_context = SceneContextModule(
            embed_dims=self.embed_dims,
            num_classes=self.bbox_head.num_classes,
            **scene_cfg)
    def _init_layers(self) -> None:

        self.encoder = RTDETRHybridEncoder(
            **self.encoder
        )

        self.decoder = (
            SCMRTQDRotatedRTDETRTransformerDecoder(
                **self.decoder
            )
        )

        self.embed_dims = self.decoder.embed_dims

        self.memory_trans_fc = nn.Linear(
            self.embed_dims,
            self.embed_dims,
        )

        self.memory_trans_norm = nn.LayerNorm(
            self.embed_dims
        )
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
        # ------------------------------------------------------------
        # Coarse OBB prediction for every encoder candidate.
        # Needed only for geometry-aware query selection.
        # ------------------------------------------------------------

        enc_outputs_coord_unact_all = (
            self.bbox_head.reg_branches[
                self.decoder.num_layers
            ](output_memory)
            + output_proposals
        )

        # [B, N, 5]
        # cx, cy, w, h, theta, all normalized after sigmoid.
        enc_outputs_coord_all = (
            enc_outputs_coord_unact_all.sigmoid()
        )

        scene_outputs = self.scene_context(
            memory=output_memory,
            spatial_shapes=spatial_shapes,
            memory_mask=memory_mask)

        candidate_wh = (
            enc_outputs_coord_all[..., 2:4]
            .detach()
            .clamp_min(1e-6)
        )

        candidate_scale = torch.sqrt(
            (
                candidate_wh[..., 0]
                * candidate_wh[..., 1]
            ).clamp_min(1e-8)
        )
        
        # ============================================================
        # 1. Existing scene-conditioned CLASS prior
        # ============================================================

        if self.use_scene_class_bias:
            selection_outputs_class = (
                enc_outputs_class
                + scene_outputs[
                    'class_bias'
                ][:, None, :]
            )
        else:
            selection_outputs_class = enc_outputs_class


        # Original classification-based query score.
        selection_score = (
            selection_outputs_class
            .max(dim=-1)
            .values
        )


        # ============================================================
        # 2. NEW: scene-conditioned SCALE prior
        # ============================================================

        if self.use_scene_scale_bias:

            candidate_wh = (
                enc_outputs_coord_all[
                    ..., 2:4
                ]
                .detach()
                .clamp_min(1e-6)
            )

            candidate_scale = torch.sqrt(
                (
                    candidate_wh[..., 0]
                    * candidate_wh[..., 1]
                ).clamp_min(1e-8)
            )

            # [B, N, 4]
            scale_membership = (
                soft_scale_membership(
                    candidate_scale,
                    boundaries=self.scale_boundaries,
                    temperature=self.scale_temperature,
                )
            )

            # Scene scale prior:
            # [B, 4] -> [B, 1, 4]
            scene_scale_bias = (
                scene_outputs[
                    'scale_bias'
                ][:, None, :]
            )

            # [B, N]
            scale_bonus = (
                scale_membership
                * scene_scale_bias
            ).sum(dim=-1)

            selection_score = (
                selection_score
                + scale_bonus
            )
        if self.use_scene_ar_bias:

            candidate_w = candidate_wh[..., 0]
            candidate_h = candidate_wh[..., 1]

            candidate_ar = (
                torch.maximum(
                    candidate_w,
                    candidate_h,
                )
                /
                torch.minimum(
                    candidate_w,
                    candidate_h,
                ).clamp_min(1e-6)
            )

            ar_membership = soft_ar_membership(
                candidate_ar,
                boundaries=self.ar_boundaries,
                temperature=self.ar_temperature,
            )

            scene_ar_bias = (
                scene_outputs['ar_bias'][:, None, :]
            )

            ar_bonus = (
                ar_membership
                * scene_ar_bias
            ).sum(dim=-1)

            # 直接在当前 selection_score 上累加
            selection_score = (
                selection_score
                + ar_bonus
            )

            scene_outputs[
                'candidate_ar_membership'
            ] = ar_membership.detach()

            scene_outputs[
                'candidate_ar_bonus'
            ] = ar_bonus.detach()
            # # Diagnostics
            # scene_outputs[
            #     'candidate_scale_membership'
            # ] = scale_membership.detach()

            # scene_outputs[
            #     'candidate_scale_bonus'
            # ] = scale_bonus.detach()


        # ============================================================
        # 3. Same Top-300 policy as baseline
        # ============================================================

        topk_indices = torch.topk(
            selection_score,
            k=self.num_queries,
            dim=1,
        )[1]

        query = torch.gather(output_memory, 1,
                             topk_indices.unsqueeze(-1).repeat(1, 1, c))
        # topk_output_proposals = torch.gather(
        #     output_proposals, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, 5))
        # topk_coords_unact = self.bbox_head.reg_branches[
        #     self.decoder.num_layers](query) + topk_output_proposals
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact_all,
            1,
            topk_indices.unsqueeze(-1).repeat(
                1, 1, 5
            ),
        )

        if self.training:
            topk_score = torch.gather(
                enc_outputs_class, 1,
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
    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask=None,
        cls_branches=None,
        eval_idx: int = -1,
    ) -> Dict:

        (
            hidden_states,
            references,
            final_query,
        ) = self.decoder(
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

        # references:
        # (
        #     all_layers_cls_scores,
        #     all_layers_bbox_preds
        # )
        classes, coordinates = references

        # Preserve the zero-connected label embedding behavior.
        if query.size(1) == self.num_queries:
            classes[0] = classes[0] + (
                self.dn_query_generator
                .label_embedding.weight[0, 0]
                * 0.0
            )

        return dict(
            # Keep baseline bbox-head interface.
            hidden_states=hidden_states,
            references=(
                classes,
                coordinates,
            ),

            # Extra RTQD input.
            final_query_feats=final_query,
        )

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
        if (scene_outputs is not None and self.loss_scene_scale_weight > 0):
            raw_scale_logits = scene_outputs.get(
                'raw_scale_logits',
                None,
            )

            if raw_scale_logits is not None:

                scene_scale_targets = (
                    build_scene_scale_targets(
                        batch_data_samples,
                        boundaries=self.scale_boundaries,
                        device=raw_scale_logits.device,
                        dtype=raw_scale_logits.dtype,
                        temperature=self.scale_temperature,
                        count_tau=1.0,
                    )
                )

                loss_scene_scale = (
                    F.binary_cross_entropy_with_logits(
                        raw_scale_logits,
                        scene_scale_targets,
                        reduction='mean',
                    )
                )

                losses[
                    'loss_scene_scale'
                ] = (
                    loss_scene_scale
                    * self.loss_scene_scale_weight
                )
        if scene_outputs is not None and self.loss_scene_ar_weight > 0:

            raw_ar_logits = scene_outputs.get(
                'raw_ar_logits',
                None,
            )

            if raw_ar_logits is not None:
                # Target is generated from GT OBB width / height.
                scene_ar_targets = (
                    build_scene_ar_targets(
                        batch_data_samples,
                        boundaries=self.ar_boundaries,
                        device=raw_ar_logits.device,
                        dtype=raw_ar_logits.dtype,
                        temperature=self.ar_temperature,
                        count_tau=1.0,
                    )
                )

                loss_scene_ar = (
                    F.binary_cross_entropy_with_logits(
                        raw_ar_logits,
                        scene_ar_targets,
                        reduction='mean',
                    )
                )

                losses['loss_scene_ar'] = (
                    loss_scene_ar
                    * self.loss_scene_ar_weight
                )
        return losses
