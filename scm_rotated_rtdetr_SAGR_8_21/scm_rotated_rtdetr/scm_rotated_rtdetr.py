from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmdet.structures import OptSampleList
from torch import Tensor

from ai4rs.registry import MODELS
from projects.rotated_rtdetr.rotated_rtdetr import RotatedRTDETR

from .scene_context_module import SceneContextModule
from .utils import (
    build_scene_class_targets,
    build_scene_scale_targets,
    soft_scale_membership,
)


@MODELS.register_module()
class SCMRotatedRTDETR(RotatedRTDETR):
    """RotatedRTDETR with bounded scene-conditioned class/scale priors.

    Fixed implementation relative to the previous SCM/SAGR branch:
    1. invalid encoder proposals are excluded from SceneContext pooling;
    2. invalid encoder proposals are excluded from scale-prior diagnostics;
    3. invalid encoder proposals are hard-masked before Top-K selection.
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
        self._latest_scene_outputs = None

        self.use_scene_scale_bias = bool(use_scene_scale_bias)
        self.loss_scene_scale_weight = float(loss_scene_scale_weight)
        self.scale_boundaries = tuple(float(x) for x in scale_boundaries)
        self.scale_temperature = float(scale_temperature)

        if len(self.scale_boundaries) != 3:
            raise ValueError('scale_boundaries must contain 3 values.')
        if not (
            0.0 < self.scale_boundaries[0]
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

    @staticmethod
    def _build_proposal_invalid_mask(
        output_proposals: Tensor,
        memory_mask: Tensor = None,
    ) -> Tensor:
        """Return [B,N] bool mask where True means invalid proposal."""
        if output_proposals.ndim != 3:
            raise RuntimeError(
                'output_proposals must be [B,N,D], '
                f'got {tuple(output_proposals.shape)}.'
            )

        # Parent gen_proposals() fills invalid proposal logits with +inf.
        proposal_invalid_mask = ~torch.isfinite(output_proposals).all(dim=-1)

        if memory_mask is not None:
            if memory_mask.shape != proposal_invalid_mask.shape:
                raise RuntimeError(
                    'memory_mask/proposal mask shape mismatch: '
                    f'{tuple(memory_mask.shape)} vs '
                    f'{tuple(proposal_invalid_mask.shape)}.'
                )
            proposal_invalid_mask = (
                proposal_invalid_mask | memory_mask.bool()
            )

        return proposal_invalid_mask

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        _, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )

        # FIX 1: recover proposal validity from the unactivated proposals.
        proposal_invalid_mask = self._build_proposal_invalid_mask(
            output_proposals=output_proposals,
            memory_mask=memory_mask,
        )

        num_valid_per_image = (~proposal_invalid_mask).sum(dim=1)
        if (num_valid_per_image < self.num_queries).any():
            raise RuntimeError(
                'Not enough valid encoder proposals for SCM Top-K: '
                f'min_valid={int(num_valid_per_image.min().item())}, '
                f'num_queries={self.num_queries}.'
            )

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ](output_memory)

        enc_outputs_coord_unact_all = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )
        enc_outputs_coord_all = enc_outputs_coord_unact_all.sigmoid()

        # FIX 1A: invalid proposal tokens do not participate in scene pooling.
        scene_outputs = self.scene_context(
            memory=output_memory,
            spatial_shapes=spatial_shapes,
            memory_mask=proposal_invalid_mask,
        )

        # 1. Scene-conditioned class prior.
        if self.use_scene_class_bias:
            selection_outputs_class = (
                enc_outputs_class
                + scene_outputs['class_bias'][:, None, :]
            )
        else:
            selection_outputs_class = enc_outputs_class

        selection_score = selection_outputs_class.max(dim=-1).values

        # 2. Scene-conditioned scale prior.
        if self.use_scene_scale_bias:
            candidate_wh = (
                enc_outputs_coord_all[..., 2:4]
                .detach()
                .clamp_min(1e-6)
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

            # FIX 1B: invalid sigmoid(+inf)=1 geometry is excluded from
            # both diagnostics and scale-prior scoring.
            scale_membership = scale_membership.masked_fill(
                proposal_invalid_mask.unsqueeze(-1),
                0.0,
            )

            scene_scale_bias = scene_outputs['scale_bias'][:, None, :]
            scale_bonus = (scale_membership * scene_scale_bias).sum(dim=-1)
            scale_bonus = scale_bonus.masked_fill(
                proposal_invalid_mask,
                0.0,
            )
            selection_score = selection_score + scale_bonus

            scene_outputs['candidate_scale_membership'] = (
                scale_membership.detach()
            )
            scene_outputs['candidate_scale_bonus'] = scale_bonus.detach()

        # FIX 1C: functional hard guarantee before Top-K.
        selection_score = selection_score.masked_fill(
            proposal_invalid_mask,
            float('-inf'),
        )

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
            topk_score = torch.gather(
                enc_outputs_class,
                1,
                topk_indices.unsqueeze(-1).repeat(
                    1, 1, cls_out_features
                ),
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

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList,
    ) -> dict:
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
