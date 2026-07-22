"""RT-DETR decoder that exposes the terminal query feature to SCOQ."""

from typing import Tuple

from mmdet.models.layers.transformer import inverse_sigmoid
from torch import Tensor, nn

from projects.rotated_rtdetr.rotated_rtdetr import (
    RotatedRTDETRTransformerDecoder,
)


class SCOQRotatedRTDETRTransformerDecoder(RotatedRTDETRTransformerDecoder):
    """Preserve O2 outputs and additionally return the scored query feature."""

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
        cls_branches: nn.ModuleList,
        **kwargs,
    ) -> Tuple[Tensor]:
        assert self.return_intermediate
        assert reg_branches is not None
        assert reference_points.shape[-1] == 5

        eval_idx = kwargs.pop("eval_idx", -1)
        if eval_idx < 0:
            eval_idx += self.num_layers
        if not 0 <= eval_idx < self.num_layers:
            raise ValueError(f"Invalid eval_idx={eval_idx}.")

        all_classes = []
        all_coords = []
        final_query = None
        for layer_id, layer in enumerate(self.layers):
            num_levels = layer.cross_attn_cfg.num_levels
            reference_points_input = reference_points.unsqueeze(2).repeat(
                1, 1, num_levels, 1
            )
            reference_points_input[..., -1] *= self.angle_factor
            query_pos = self.ref_point_head(reference_points)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs,
            )
            bbox_delta = reg_branches[layer_id](query)

            if self.training or layer_id == eval_idx:
                all_classes.append(cls_branches[layer_id](query))
                all_coords.append(
                    (bbox_delta + inverse_sigmoid(reference_points, eps=1e-3)).sigmoid()
                )
                terminal = not self.training or layer_id == self.num_layers - 1
                if terminal:
                    final_query = query
                    break

            unactivated_reference = (
                bbox_delta + inverse_sigmoid(reference_points, eps=1e-3).detach()
            )
            reference_points = unactivated_reference.sigmoid().detach()

        if final_query is None:
            raise RuntimeError("SCOQ decoder did not produce a terminal query")
        return all_classes, all_coords, final_query


__all__ = ["SCOQRotatedRTDETRTransformerDecoder"]
