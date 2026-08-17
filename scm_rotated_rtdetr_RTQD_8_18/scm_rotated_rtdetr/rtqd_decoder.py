"""RT-DETR decoder variant that exposes the terminal query feature for RTQD."""

from mmdet.models.layers.transformer import inverse_sigmoid
from torch import Tensor, nn

from projects.rotated_rtdetr.rotated_rtdetr import (
    RotatedRTDETRTransformerDecoder,
)


class SCMRTQDRotatedRTDETRTransformerDecoder(
    RotatedRTDETRTransformerDecoder
):
    """Preserve O2-RTDETR decoder outputs and expose final query features.

    Returns:
        tuple:
            - all_classes: list[Tensor], each [B, Q_total, num_classes]
            - all_coords: list[Tensor], each [B, Q_total, 5]
            - final_query: Tensor [B, Q_total, C]

        The first two values intentionally keep the exact return contract of
        :class:`RotatedRTDETRTransformerDecoder`. ``final_query`` is the only
        additional value and is consumed exclusively by RTQD.
    """

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
            raise ValueError('RTQD requires return_intermediate=True')
        if reg_branches is None or cls_branches is None:
            raise ValueError('RTQD decoder requires reg_branches and cls_branches')
        if reference_points.shape[-1] != 5:
            raise ValueError(
                'RTQD decoder expects rotated reference points with dim=5, '
                f'got {reference_points.shape[-1]}'
            )

        eval_idx = int(kwargs.pop('eval_idx', -1))
        if eval_idx < 0:
            eval_idx += self.num_layers
        if not 0 <= eval_idx < self.num_layers:
            raise ValueError(f'Invalid decoder eval_idx={eval_idx}')

        all_classes = []
        all_coords = []
        final_query = None

        for layer_id, layer in enumerate(self.layers):
            num_levels = layer.cross_attn_cfg.num_levels

            reference_input = (
                reference_points.unsqueeze(2).repeat(1, 1, num_levels, 1)
            )
            reference_input[..., -1] *= self.angle_factor

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
                reference_points=reference_input,
                **kwargs,
            )

            bbox_delta = reg_branches[layer_id](query)

            if self.training or layer_id == eval_idx:
                all_classes.append(cls_branches[layer_id](query))
                all_coords.append(
                    (
                        bbox_delta
                        + inverse_sigmoid(reference_points, eps=1e-3)
                    ).sigmoid()
                )

                if not self.training or layer_id == self.num_layers - 1:
                    # This is the only additional RTQD output.
                    final_query = query
                    break

            # Same iterative reference-point update as O2-RTDETR.
            reference_points = (
                bbox_delta
                + inverse_sigmoid(reference_points, eps=1e-3).detach()
            ).sigmoid().detach()

        if final_query is None:
            raise RuntimeError('Decoder did not produce terminal query features.')
        if final_query.ndim != 3:
            raise RuntimeError(
                'final_query must be [B, Q_total, C], got '
                f'{tuple(final_query.shape)}'
            )
        if final_query.size(-1) != self.embed_dims:
            raise RuntimeError(
                'final_query has wrong embedding dimension: '
                f'{final_query.size(-1)} vs expected {self.embed_dims}'
            )

        return all_classes, all_coords, final_query


__all__ = ['SCMRTQDRotatedRTDETRTransformerDecoder']
