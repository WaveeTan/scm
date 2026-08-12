"""Scene-Conditioned Negative Evidence (SCNE)."""

from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .utils.tensor_utils import batched_gather


class SceneNegativeEvidenceHead(nn.Module):
    """Pool encoder tokens into image-level class-presence logits."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_classes: int = 20,
        hidden_dim: int = 256,
        topk_ratio: float = 0.02,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dims <= 0 or hidden_dim <= 0 or num_classes <= 0:
            raise ValueError("SCNE dimensions must be positive")
        if not 0 < topk_ratio <= 1:
            raise ValueError("topk_ratio must be in (0, 1]")
        self.topk_ratio = float(topk_ratio)
        self.num_classes = int(num_classes)
        self.token_score = nn.Linear(embed_dims, 1)
        self.scene_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if memory.ndim != 3:
            raise ValueError("SCNE memory must have shape [B, N, C]")
        if memory.size(1) == 0:
            raise ValueError("SCNE cannot pool an empty token sequence")

        if memory_mask is None:
            mean_feat = memory.mean(dim=1)
            token_scores = self.token_score(memory).squeeze(-1)
        else:
            if memory_mask.shape != memory.shape[:2]:
                raise ValueError("SCNE memory mask has an invalid shape")
            valid = (~memory_mask).to(memory).unsqueeze(-1)
            mean_feat = (memory * valid).sum(dim=1)
            mean_feat = mean_feat / valid.sum(dim=1).clamp_min(1.0)
            token_scores = self.token_score(memory).squeeze(-1)
            token_scores = token_scores.masked_fill(
                memory_mask, float("-inf")
            )

        k = max(1, min(memory.size(1), int(memory.size(1) * self.topk_ratio)))
        topk_idx = token_scores.topk(k, dim=1).indices
        topk_memory = batched_gather(memory, topk_idx)
        # Hard top-k alone only uses token_score to create integer indices,
        # which leaves its weight and bias unused by autograd under DDP.
        # A differentiable weighting inside the selected set preserves sparse
        # top-k pooling while allowing the scoring layer to learn.
        selected_scores = batched_gather(token_scores, topk_idx)
        selected_weights = selected_scores.float().softmax(dim=1).to(memory)
        topk_feat = (
            topk_memory * selected_weights.unsqueeze(-1)
        ).sum(dim=1)
        return self.scene_mlp(torch.cat([mean_feat, topk_feat], dim=-1))


def build_presence_targets(
    labels_per_image: Iterable[Tensor],
    num_classes: int,
    *,
    reference: Tensor,
) -> Tensor:
    """Build image-level multi-hot targets, including all-zero empty images."""
    labels_per_image = list(labels_per_image)
    targets = reference.new_zeros((len(labels_per_image), num_classes))
    for image_index, labels in enumerate(labels_per_image):
        if labels.numel():
            unique = labels.unique()
            if ((unique < 0) | (unique >= num_classes)).any():
                raise ValueError("ground-truth label is outside SCNE classes")
            targets[image_index, unique.long()] = 1.0
    return targets


def negative_evidence_bias(
    scene_logits: Tensor,
    *,
    calibration_lambda: float = 0.5,
    presence_threshold: float = 0.2,
    min_bias: float = -1.5,
    eps: float = 1e-6,
    detach: bool = True,
    enabled: bool = True,
) -> Tensor:
    """Return a non-positive class bias from scene-presence evidence."""
    if calibration_lambda < 0:
        raise ValueError("calibration_lambda must be non-negative")
    if not 0 <= presence_threshold <= 1:
        raise ValueError("presence_threshold must be in [0, 1]")
    if min_bias > 0:
        raise ValueError("min_bias must not be positive")
    if not enabled:
        return torch.zeros_like(scene_logits)

    presence = scene_logits.sigmoid()
    if detach:
        presence = presence.detach()
    raw_bias = calibration_lambda * (presence + eps).log()
    raw_bias = raw_bias.clamp(min=min_bias, max=0.0)
    return torch.where(
        presence < presence_threshold,
        raw_bias,
        torch.zeros_like(raw_bias),
    )


def calibrate_class_logits(cls_logits: Tensor, scene_bias: Tensor) -> Tensor:
    """Broadcast the SCNE class bias over queries or encoder tokens."""
    if cls_logits.size(0) != scene_bias.size(0):
        raise ValueError("classification and scene batches differ")
    if cls_logits.size(-1) != scene_bias.size(-1):
        raise ValueError("classification and scene class dimensions differ")
    return cls_logits + scene_bias.unsqueeze(1)


__all__ = [
    "SceneNegativeEvidenceHead",
    "build_presence_targets",
    "calibrate_class_logits",
    "negative_evidence_bias",
]
