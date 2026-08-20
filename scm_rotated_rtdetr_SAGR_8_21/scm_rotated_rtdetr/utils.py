from typing import List, Sequence, Tuple

import torch
from torch import Tensor


def split_by_spatial_shapes(tensor: Tensor,
                            spatial_shapes: Tensor
                            ) -> Tuple[Tuple[Tensor, ...], List[int]]:
    """Split flattened multi-level features by spatial shapes."""
    level_sizes = spatial_shapes.prod(dim=1).tolist()
    level_sizes = [int(size) for size in level_sizes]
    return tensor.split(level_sizes, dim=1), level_sizes


def masked_mean(feature: Tensor, mask: Tensor = None) -> Tensor:
    """Mean-pool flattened tokens with an optional invalid-position mask."""
    if mask is None:
        return feature.mean(dim=1)

    valid = (~mask.bool()).to(feature.dtype).unsqueeze(-1)
    return (feature * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def build_scene_class_targets(batch_data_samples,
                              num_classes: int,
                              device,
                              dtype=torch.float32) -> Tensor:
    """Build image-level class-presence targets from GT labels."""
    targets = torch.zeros(
        (len(batch_data_samples), num_classes), device=device, dtype=dtype)
    for sample_id, data_sample in enumerate(batch_data_samples):
        labels = data_sample.gt_instances.labels
        if labels.numel() == 0:
            continue
        labels = labels.long().unique()
        labels = labels[(labels >= 0) & (labels < num_classes)]
        if labels.numel() > 0:
            targets[sample_id, labels] = 1.0
    return targets


def soft_scale_membership(
    scale: Tensor,
    boundaries: Sequence[float] = (0.02, 0.04, 0.12),
    temperature: float = 0.20,
    eps: float = 1e-6,
) -> Tensor:
    """Soft membership for tiny/small/medium/large objects.

    Args:
        scale:
            Normalized equivalent side length:
                sqrt(w * h)
            where w,h are normalized by image width/height.

            Shape can be [...].

        boundaries:
            Three boundaries:
                tiny / small / medium / large

        temperature:
            Smoothness in log-scale space.

    Returns:
        membership:
            [..., 4], and memberships sum to 1.
    """

    if len(boundaries) != 3:
        raise ValueError(
            'scale boundaries must contain exactly 3 values.'
        )

    b1, b2, b3 = [float(x) for x in boundaries]

    if not (0.0 < b1 < b2 < b3):
        raise ValueError(
            f'Invalid scale boundaries: {boundaries}'
        )

    if temperature <= 0:
        raise ValueError(
            'scale temperature must be positive.'
        )

    # Work in log-scale because remote-sensing object sizes
    # are highly long-tailed.
    log_scale = torch.log(
        scale.clamp_min(eps)
    )

    log_boundaries = torch.log(
        scale.new_tensor([b1, b2, b3]).clamp_min(eps)
    )

    transitions = torch.sigmoid(
        (
            log_scale.unsqueeze(-1)
            - log_boundaries
        ) / float(temperature)
    )

    t1 = transitions[..., 0]
    t2 = transitions[..., 1]
    t3 = transitions[..., 2]

    membership = torch.stack(
        [
            1.0 - t1,   # tiny
            t1 - t2,    # small
            t2 - t3,    # medium
            t3,         # large
        ],
        dim=-1,
    )

    membership = membership.clamp_min(0.0)

    membership = (
        membership
        / membership.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
    )

    return membership


def build_scene_scale_targets(
    batch_data_samples,
    boundaries: Sequence[float],
    device,
    dtype=torch.float32,
    temperature: float = 0.20,
    count_tau: float = 1.0,
) -> Tensor:
    """Build image-level soft scale-presence targets.

    Output:
        [B, 4]:
        tiny / small / medium / large
    """

    batch_size = len(batch_data_samples)

    targets = torch.zeros(
        (batch_size, 4),
        device=device,
        dtype=dtype,
    )

    for sample_id, data_sample in enumerate(
        batch_data_samples
    ):
        gt_bboxes = data_sample.gt_instances.bboxes

        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor

        gt_bboxes = torch.as_tensor(
            gt_bboxes,
            device=device,
            dtype=dtype,
        )

        if gt_bboxes.numel() == 0:
            continue

        # rbox:
        # [cx, cy, w, h, theta]
        wh = gt_bboxes[:, 2:4].abs()

        img_shape = data_sample.metainfo.get(
            'img_shape',
            data_sample.metainfo.get('ori_shape')
        )

        if img_shape is None:
            raise RuntimeError(
                'img_shape/ori_shape is required '
                'for scene scale targets.'
            )

        img_h = float(img_shape[0])
        img_w = float(img_shape[1])

        # Normalized equivalent side length.
        scale = torch.sqrt(
            (
                wh[:, 0] * wh[:, 1]
            ).clamp_min(1e-8)
            / max(img_h * img_w, 1.0)
        )

        membership = soft_scale_membership(
            scale,
            boundaries=boundaries,
            temperature=temperature,
        )

        # Soft number of instances per scale group.
        counts = membership.sum(dim=0)

        # Presence-like target rather than normalized histogram.
        #
        # One rare tiny object should still produce a useful signal.
        scale_target = (
            1.0
            - torch.exp(
                -counts / float(count_tau)
            )
        )

        targets[sample_id] = scale_target

    return targets