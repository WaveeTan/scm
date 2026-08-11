"""Small tensor helpers shared by RTP-Score modules."""

from typing import List

import torch
from torch import Tensor


def batched_gather(values: Tensor, indices: Tensor) -> Tensor:
    """Gather the second dimension of a batched tensor.

    Args:
        values: Tensor shaped ``[B, N, ...]``.
        indices: Long tensor shaped ``[B, K]``.
    """
    if values.ndim < 2 or indices.ndim != 2:
        raise ValueError("values must be [B, N, ...] and indices must be [B, K]")
    if values.size(0) != indices.size(0):
        raise ValueError("values and indices have different batch sizes")
    view_shape = list(indices.shape) + [1] * (values.ndim - 2)
    expand_shape = list(indices.shape) + list(values.shape[2:])
    gather_index = indices.view(view_shape).expand(expand_shape)
    return torch.gather(values, 1, gather_index)


def image_factors(
    batch_img_metas: List[dict],
    bbox_preds: Tensor,
    angle_factor: float,
) -> Tensor:
    """Return per-image factors for normalized ``cx, cy, w, h, angle`` boxes."""
    if len(batch_img_metas) != bbox_preds.size(0):
        raise ValueError("image metadata and prediction batch sizes differ")
    factors = []
    for img_meta, image_preds in zip(batch_img_metas, bbox_preds):
        img_h, img_w = img_meta["img_shape"][:2]
        factor = image_preds.new_tensor(
            [img_w, img_h, img_w, img_h, angle_factor]
        )
        factors.append(factor.unsqueeze(0).expand(image_preds.size(0), -1))
    return torch.cat(factors, dim=0)


def connected_zero(*tensors: Tensor) -> Tensor:
    """Return a scalar zero connected to every supplied tensor."""
    if not tensors:
        return torch.tensor(0.0)
    result = tensors[0].sum() * 0.0
    for tensor in tensors[1:]:
        result = result + tensor.sum() * 0.0
    return result


__all__ = ["batched_gather", "connected_zero", "image_factors"]
