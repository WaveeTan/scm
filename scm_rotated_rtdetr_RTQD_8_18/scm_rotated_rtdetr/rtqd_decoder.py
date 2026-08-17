"""RT-DETR decoder variant that exposes the terminal query feature."""

from typing import Tuple

from mmdet.models.layers.transformer import inverse_sigmoid
from torch import Tensor, nn

from projects.rotated_rtdetr.rotated_rtdetr import (
    RotatedRTDETRTransformerDecoder,
)


class SCMRTQDRotatedRTDETRTransformerDecoder(RotatedRTDETRTransformerDecoder):
    """Preserve baseline outputs and additionally return final query features."""

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
    ):
        if not self.return_intermediate:
            raise ValueError(
                'RTQD requires return_intermediate=True'
            )

        if reg_branches is None or cls_branches is None:
            raise ValueError(
                'RTQD decoder requires prediction branches'
            )

        if reference_points.shape[-1] != 5:
            raise ValueError(
                'RTQD decoder expects rotated references'
            )

        eval_idx = int(
            kwargs.pop('eval_idx', -1)
        )

        if eval_idx < 0:
            eval_idx += self.num_layers

        if not 0 <= eval_idx < self.num_layers:
            raise ValueError(
                f'Invalid decoder eval_idx={eval_idx}'
            )

        # ---------------------------------------------------------
        # Preserve original RT-DETR decoder outputs.
        # ---------------------------------------------------------
        hidden_states = []
        all_classes = []
        all_coords = []

        # Extra output required only by RTQD.
        final_query = None

        for layer_id, layer in enumerate(
            self.layers
        ):
            num_levels = (
                layer.cross_attn_cfg.num_levels
            )

            reference_input = (
                reference_points
                .unsqueeze(2)
                .repeat(
                    1,
                    1,
                    num_levels,
                    1,
                )
            )

            reference_input[..., -1] *= (
                self.angle_factor
            )

            query_pos = self.ref_point_head(
                reference_points
            )

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_input,
                **kwargs,
            )

            bbox_delta = reg_branches[
                layer_id
            ](query)

            if (
                self.training
                or layer_id == eval_idx
            ):
                # Keep original hidden-state output.
                hidden_states.append(query)

                all_classes.append(
                    cls_branches[layer_id](
                        query
                    )
                )

                all_coords.append(
                    (
                        bbox_delta
                        + inverse_sigmoid(
                            reference_points,
                            eps=1e-3,
                        )
                    ).sigmoid()
                )

                if (
                    not self.training
                    or layer_id
                    == self.num_layers - 1
                ):
                    # Terminal decoder feature for RTQD.
                    final_query = query
                    break

            reference_points = (
                bbox_delta
                + inverse_sigmoid(
                    reference_points,
                    eps=1e-3,
                ).detach()
            ).sigmoid().detach()

        if final_query is None:
            raise RuntimeError(
                'Decoder did not produce '
                'terminal query features.'
            )

        # Original contract + one RTQD output.
        return (
            hidden_states,
            (
                all_classes,
                all_coords,
            ),
            final_query,
        )


__all__ = ["SCMRTQDRotatedRTDETRTransformerDecoder"]