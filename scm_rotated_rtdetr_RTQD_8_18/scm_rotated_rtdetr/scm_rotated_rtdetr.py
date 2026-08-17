from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmdet.structures import OptSampleList
from torch import Tensor, nn

from ai4rs.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR
from projects.rotated_rtdetr.rotated_rtdetr.rtdetr_layers import (
    RTDETRHybridEncoder,
)

from .rtqd_decoder import SCMRTQDRotatedRTDETRTransformerDecoder
from .scene_context_module import SceneContextModule
from .utils import (
    build_scene_class_targets,
    build_scene_scale_targets,
    soft_scale_membership,
)


@MODELS.register_module()
class SCMRotatedRTDETR(RotatedRTDETR):
    """O2-RTDETR with SCM Class+Scale query selection and final RTQD export.

    SCM modifies only encoder query selection:
        class prior + scale prior -> Top-K matching queries.

    RTQD does not participate in query selection. The only decoder-side change
    is exposing the terminal decoder query feature as ``final_query_feats`` for
    the detached quality head.
    """

    def __init__(
        self,
        *args,
        scene_cfg: dict = None,
        use_scene_class_bias: bool = True,
        loss_scene_cls_weight: float = 0.05,
        use_scene_scale_bias: bool = False,
        loss_scene_scale_weight: float = 0.05,
        scale_boundaries=(0.02, 0.04, 0.12),
        scale_temperature: float = 0.20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.use_scene_class_bias = bool(use_scene_class_bias)
        self.loss_scene_cls_weight = float(loss_scene_cls_weight)
        self.use_scene_scale_bias = bool(use_scene_scale_bias)
        self.loss_scene_scale_weight = float(loss_scene_scale_weight)
        self.scale_boundaries = tuple(float(x) for x in scale_boundaries)
        self.scale_temperature = float(scale_temperature)
        self._latest_scene_outputs = None

        if len(self.scale_boundaries) != 3:
            raise ValueError('scale_boundaries must contain 3 values.')
        if not (
            0.0
            < self.scale_boundaries[0]
            < self.scale_boundaries[1]
            < self.scale_boundaries[2]
        ):
            raise ValueError(f'Invalid scale boundaries: {self.scale_boundaries}')

        scene_cfg = {} if scene_cfg is None else dict(scene_cfg)
        self.scene_context = SceneContextModule(
            embed_dims=self.embed_dims,
            num_classes=self.bbox_head.num_classes,
            **scene_cfg,
        )

    def _init_layers(self) -> None:
        """Use baseline RT-DETR encoder and RTQD-aware decoder."""
        self.encoder = RTDETRHybridEncoder(**self.encoder)
        self.decoder = SCMRTQDRotatedRTDETRTransformerDecoder(**self.decoder)
        self.embed_dims = self.decoder.embed_dims
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """SCM-8_15 query selection: Class prior + Scale prior -> Top-K."""
        _, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )

        # Raw encoder classification logits. These are also kept for the
        # original encoder classification loss after Top-K gathering.
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ](output_memory)

        # Coarse OBB for every encoder candidate. Used by Scale Prior and later
        # gathered as the selected encoder box output.
        enc_outputs_coord_unact_all = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )
        enc_outputs_coord_all = enc_outputs_coord_unact_all.sigmoid()

        scene_outputs = self.scene_context(
            memory=output_memory,
            spatial_shapes=spatial_shapes,
            memory_mask=memory_mask,
        )

        # ============================================================
        # 1. Scene-conditioned CLASS prior
        # ============================================================
        if self.use_scene_class_bias:
            selection_outputs_class = (
                enc_outputs_class + scene_outputs['class_bias'][:, None, :]
            )
        else:
            selection_outputs_class = enc_outputs_class

        selection_score = selection_outputs_class.max(dim=-1).values

        # ============================================================
        # 2. Scene-conditioned SCALE prior
        # ============================================================
        if self.use_scene_scale_bias:
            candidate_wh = (
                enc_outputs_coord_all[..., 2:4].detach().clamp_min(1e-6)
            )
            candidate_scale = torch.sqrt(
                (
                    candidate_wh[..., 0] * candidate_wh[..., 1]
                ).clamp_min(1e-8)
            )

            scale_membership = soft_scale_membership(
                candidate_scale,
                boundaries=self.scale_boundaries,
                temperature=self.scale_temperature,
            )

            scene_scale_bias = scene_outputs['scale_bias'][:, None, :]
            scale_bonus = (scale_membership * scene_scale_bias).sum(dim=-1)
            selection_score = selection_score + scale_bonus

            # Diagnostics only.
            scene_outputs['candidate_scale_membership'] = (
                scale_membership.detach()
            )
            scene_outputs['candidate_scale_bonus'] = scale_bonus.detach()

        # ============================================================
        # 3. Same global Top-K policy as SCM-8_15
        # ============================================================
        topk_indices = torch.topk(
            selection_score,
            k=self.num_queries,
            dim=1,
        )[1]

        query = torch.gather(
            output_memory,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, c),
        )

        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact_all,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 5),
        )

        if self.training:
            # IMPORTANT: encoder classification loss still uses raw encoder
            # logits, not scene-biased logits.
            topk_score = torch.gather(
                enc_outputs_class,
                1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
            )
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()

            dn_label_query, dn_bbox_query, dn_mask, dn_meta = (
                self.dn_query_generator(batch_data_samples)
            )

            query = query.detach()
            query = torch.cat([dn_label_query, query], dim=1)
            dn_bbox_query = dn_bbox_query.type_as(topk_coords_unact)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1
            )
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
            eval_idx=self.eval_idx,
        )

        head_inputs_dict = dict()
        if self.training:
            self._latest_scene_outputs = scene_outputs
            head_inputs_dict = dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta,
            )

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
        """Run decoder and explicitly export the terminal query to RTQD."""

        all_classes, all_coords, final_query = self.decoder(
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

        if not isinstance(all_classes, (list, tuple)):
            raise RuntimeError('all_classes must be a list/tuple of tensors.')
        if not isinstance(all_coords, (list, tuple)):
            raise RuntimeError('all_coords must be a list/tuple of tensors.')
        if len(all_classes) != len(all_coords):
            raise RuntimeError(
                'Decoder class/bbox layer counts differ: '
                f'{len(all_classes)} vs {len(all_coords)}.'
            )

        if final_query.ndim != 3:
            raise RuntimeError(
                'final_query must be [B, Q_total, C], got '
                f'{tuple(final_query.shape)}'
            )
        if final_query.size(-1) != self.embed_dims:
            raise RuntimeError(
                'final_query feature dimension mismatch: '
                f'{final_query.size(-1)} vs expected {self.embed_dims}. '
                f'Full shape={tuple(final_query.shape)}'
            )

        # Keep the DINO zero-connected label-embedding dependency when there
        # are no denoising queries in this GPU.
        if query.size(1) == self.num_queries:
            all_classes[0] = (
                all_classes[0]
                + self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        return dict(
            # RotatedRTDETRHead uses these names for decoder class/box lists.
            hidden_states=all_classes,
            references=all_coords,
            # RTQD-only input; this is [B, Q_total, 256], never class logits.
            final_query_feats=final_query,
        )

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList,
    ) -> dict:
        """Original detector losses plus SCM scene auxiliary losses."""
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
                    dtype=raw_class_logits.dtype,
                )
                loss_scene_cls = F.binary_cross_entropy_with_logits(
                    raw_class_logits,
                    scene_targets,
                    reduction='mean',
                )
                losses['loss_scene_cls'] = (
                    loss_scene_cls * self.loss_scene_cls_weight
                )

        if scene_outputs is not None and self.loss_scene_scale_weight > 0:
            raw_scale_logits = scene_outputs.get('raw_scale_logits', None)

            if raw_scale_logits is not None:
                scene_scale_targets = build_scene_scale_targets(
                    batch_data_samples,
                    boundaries=self.scale_boundaries,
                    device=raw_scale_logits.device,
                    dtype=raw_scale_logits.dtype,
                    temperature=self.scale_temperature,
                    count_tau=1.0,
                )

                loss_scene_scale = F.binary_cross_entropy_with_logits(
                    raw_scale_logits,
                    scene_scale_targets,
                    reduction='mean',
                )
                losses['loss_scene_scale'] = (
                    loss_scene_scale * self.loss_scene_scale_weight
                )

        return losses
