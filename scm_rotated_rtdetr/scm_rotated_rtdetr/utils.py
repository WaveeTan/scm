from typing import List, Tuple

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
