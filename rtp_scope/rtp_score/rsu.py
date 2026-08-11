"""Rotated Set Uniqueness (RSU) losses and optional graph head."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ai4rs.structures.bbox import rbbox_overlaps

from .utils.rotated_matching import RTPImageTargets
from .utils.tensor_utils import connected_zero


@dataclass
class RSUSelection:
    """Winner/rival pairs and optional uniqueness targets for one image."""

    pairs: Tensor
    labels: Tensor
    unique_targets: Tensor


class RotatedSetUniquenessLoss(nn.Module):
    """Rank each Hungarian winner above nearby unmatched competitors."""

    def __init__(
        self,
        rival_iou_thr: float = 0.3,
        max_rivals_per_gt: int = 3,
        margin: float = 0.2,
        loss_weight: float = 0.2,
        require_argmax_class_match: bool = True,
    ) -> None:
        super().__init__()
        if not 0 <= rival_iou_thr <= 1:
            raise ValueError("rival_iou_thr must be in [0, 1]")
        if max_rivals_per_gt <= 0:
            raise ValueError("max_rivals_per_gt must be positive")
        if margin < 0 or loss_weight < 0:
            raise ValueError("RSU margin and loss weight must be non-negative")
        self.rival_iou_thr = float(rival_iou_thr)
        self.max_rivals_per_gt = int(max_rivals_per_gt)
        self.margin = float(margin)
        self.loss_weight = float(loss_weight)
        self.require_argmax_class_match = bool(require_argmax_class_match)

    def select(
        self,
        current_scores: Tensor,
        records: Sequence[RTPImageTargets],
    ) -> List[RSUSelection]:
        """Select discrete rivals with detached scores and geometry."""
        selections = []
        for image_scores, record in zip(current_scores, records):
            pairs = []
            pair_labels = []
            unique_targets = image_scores.new_full(
                (image_scores.size(0),), -1.0
            )
            unique_targets[record.pos_inds] = 1.0
            argmax_labels = image_scores.detach().argmax(dim=-1)

            # Only unmatched matching queries are valid rivals. This avoids
            # contradictory targets when one query is the winner of another GT.
            for local_index, winner in enumerate(record.pos_inds):
                gt_index = record.assigned_gt_inds[local_index]
                gt_label = record.gt_labels[gt_index]
                candidates = record.neg_inds
                if candidates.numel() == 0:
                    continue
                keep = (
                    record.pairwise_iou[candidates, gt_index]
                    > self.rival_iou_thr
                )
                if self.require_argmax_class_match:
                    keep = keep & (argmax_labels[candidates] == gt_label)
                candidates = candidates[keep]
                if candidates.numel() == 0:
                    continue
                candidate_scores = image_scores[
                    candidates, gt_label
                ].detach()
                topk = min(self.max_rivals_per_gt, candidates.numel())
                rivals = candidates[candidate_scores.topk(topk).indices]
                pairs.append(
                    torch.stack([winner.expand_as(rivals), rivals], dim=-1)
                )
                pair_labels.append(gt_label.expand(topk))
                unique_targets[rivals] = 0.0

            if pairs:
                pair_tensor = torch.cat(pairs, dim=0)
                label_tensor = torch.cat(pair_labels, dim=0)
            else:
                pair_tensor = image_scores.new_zeros((0, 2), dtype=torch.long)
                label_tensor = image_scores.new_zeros((0,), dtype=torch.long)
            selections.append(
                RSUSelection(
                    pairs=pair_tensor,
                    labels=label_tensor,
                    unique_targets=unique_targets,
                )
            )
        return selections

    def forward(
        self,
        current_scores: Tensor,
        records: Sequence[RTPImageTargets],
    ) -> Tuple[Tensor, dict, List[RSUSelection]]:
        selections = self.select(current_scores, records)
        losses = []
        winner_scores = []
        rival_scores = []
        for image_index, selection in enumerate(selections):
            if selection.pairs.numel() == 0:
                continue
            winner = selection.pairs[:, 0]
            rival = selection.pairs[:, 1]
            labels = selection.labels
            winners = current_scores[image_index, winner, labels]
            rivals = current_scores[image_index, rival, labels]
            losses.append(F.softplus(self.margin - winners + rivals))
            winner_scores.append(winners.detach())
            rival_scores.append(rivals.detach())

        if losses:
            all_losses = torch.cat(losses)
            loss = all_losses.mean() * self.loss_weight
            winners = torch.cat(winner_scores)
            rivals = torch.cat(rival_scores)
            satisfied = ((winners - rivals) >= self.margin).float().mean()
            winner_mean = winners.mean()
            rival_mean = rivals.mean()
        else:
            loss = connected_zero(current_scores)
            satisfied = current_scores.new_zeros(())
            winner_mean = current_scores.new_zeros(())
            rival_mean = current_scores.new_zeros(())

        num_winners = sum(record.pos_inds.numel() for record in records)
        num_rivals = sum(item.pairs.size(0) for item in selections)
        diagnostics = {
            "rsu_num_winners": current_scores.new_tensor(
                float(num_winners)
            ),
            "rsu_num_rivals": current_scores.new_tensor(float(num_rivals)),
            "rsu_winner_score_mean": winner_mean,
            "rsu_rival_score_mean": rival_mean,
            "rsu_margin_satisfied_ratio": satisfied,
        }
        return loss, diagnostics, selections


class RotatedSetUniquenessHead(nn.Module):
    """One-layer edge-aware query graph for the optional RSU upgrade."""

    def __init__(self, embed_dims: int = 256, edge_hidden_dims: int = 64) -> None:
        super().__init__()
        self.value = nn.Linear(embed_dims, embed_dims)
        self.edge_attention = nn.Sequential(
            nn.Linear(7, edge_hidden_dims),
            nn.GELU(),
            nn.Linear(edge_hidden_dims, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(embed_dims * 2),
            nn.Linear(embed_dims * 2, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, 1),
        )

    def forward(
        self,
        query_features: Tensor,
        decoded_boxes: Sequence[Tensor],
        cls_scores: Tensor,
    ) -> Tensor:
        outputs = []
        for features, boxes, logits in zip(
            query_features, decoded_boxes, cls_scores
        ):
            num_queries = features.size(0)
            if num_queries == 0:
                outputs.append(features.new_zeros((0, 1)))
                continue
            with torch.no_grad():
                riou = rbbox_overlaps(
                    boxes.detach(),
                    boxes.detach(),
                    mode="iou",
                    is_aligned=False,
                ).clamp(0, 1)
                same_class = (
                    logits.detach().argmax(dim=-1)[:, None]
                    == logits.detach().argmax(dim=-1)[None, :]
                )
                same_class.fill_diagonal_(False)
                delta = boxes.detach()[:, None, :] - boxes.detach()[None, :, :]
            normalized = F.normalize(features.float(), dim=-1)
            cosine = normalized @ normalized.transpose(0, 1)
            edge_features = torch.cat(
                [riou[..., None], cosine[..., None], delta], dim=-1
            ).to(features)
            attention_logits = self.edge_attention(edge_features).squeeze(-1)
            attention_logits = attention_logits.masked_fill(
                ~same_class, float("-inf")
            )
            has_neighbor = same_class.any(dim=-1)
            attention = features.new_zeros((num_queries, num_queries))
            if has_neighbor.any():
                attention[has_neighbor] = attention_logits[
                    has_neighbor
                ].softmax(dim=-1)
            message = attention @ self.value(features)
            outputs.append(self.output(torch.cat([features, message], dim=-1)))
        return torch.stack(outputs, dim=0)

    @staticmethod
    def loss(
        unique_logits: Tensor,
        selections: Sequence[RSUSelection],
        loss_weight: float,
    ) -> Tensor:
        targets = torch.stack(
            [selection.unique_targets for selection in selections], dim=0
        )
        valid = targets >= 0
        if valid.any():
            raw = F.binary_cross_entropy_with_logits(
                unique_logits.squeeze(-1)[valid], targets[valid]
            )
        else:
            raw = connected_zero(unique_logits)
        return raw * float(loss_weight)


__all__ = [
    "RSUSelection",
    "RotatedSetUniquenessHead",
    "RotatedSetUniquenessLoss",
]
