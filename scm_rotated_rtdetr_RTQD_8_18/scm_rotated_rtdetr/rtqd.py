"""Rotated Threshold Quality Distribution (RTQD)."""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def connected_zero(tensor: Tensor) -> Tensor:
    """Return a differentiable zero connected to tensor."""
    return tensor.sum() * 0.0

class RotatedThresholdQualityHead(nn.Module):
    """Predict probabilities of crossing several rotated-IoU thresholds."""

    def __init__(
        self,
        embed_dims: int = 256,
        thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8),
        tau: float = 0.05,
    ) -> None:
        super().__init__()
        thresholds = tuple(float(value) for value in thresholds)
        if not thresholds or tuple(sorted(thresholds)) != thresholds:
            raise ValueError("RTQD thresholds must be non-empty and ordered")
        if thresholds[0] < 0 or thresholds[-1] > 1:
            raise ValueError("RTQD thresholds must lie in [0, 1]")
        if tau <= 0:
            raise ValueError("RTQD tau must be positive")
        self.thresholds = thresholds
        self.tau = float(tau)
        self.net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, len(thresholds)),
        )

    def forward(self, query_features: Tensor) -> Tensor:
        return self.net(query_features)

    def soft_targets(self, rotated_iou: Tensor) -> Tensor:
        thresholds = rotated_iou.new_tensor(self.thresholds)
        return torch.sigmoid(
            (rotated_iou.unsqueeze(-1) - thresholds) / self.tau
        )

    def loss(
        self,
        quality_logits: Tensor,
        positive_indices: Sequence[Tensor],
        positive_ious: Sequence[Tensor],
        *,
        pairwise_ious: Optional[Sequence[Tensor]] = None,
        loss_weight: float = 1.0,
        monotonic_weight: float = 0.1,
    ) -> Tuple[Tensor, Tensor, dict]:
        """RTQD loss.

        If pairwise_ious is supplied, use all-query supervision:
            target_i = max_j rIoU(query_i, gt_j)

        Otherwise retain the original positive-only behavior.
        """

        # ================================================================
        # All-query RTQD
        # ================================================================
        if pairwise_ious is not None:
            if len(pairwise_ious) != quality_logits.size(0):
                raise ValueError(
                    "pairwise_ious batch size must match quality_logits"
                )

            all_logits = []
            all_targets = []
            all_weights = []
            all_max_ious = []

            for image_index, pairwise_iou in enumerate(pairwise_ious):
                num_queries = quality_logits.size(1)

                if pairwise_iou.ndim != 2:
                    raise ValueError(
                        "pairwise_iou must have shape [num_queries, num_gt]"
                    )

                if pairwise_iou.size(0) != num_queries:
                    raise ValueError(
                        "pairwise_iou query count does not match quality logits"
                    )

                # --------------------------------------------------------
                # Every query receives a localization-quality target.
                #
                # pairwise_iou: [Q, num_gt]
                # max_iou:      [Q]
                # --------------------------------------------------------
                if pairwise_iou.size(1) == 0:
                    max_iou = quality_logits.new_zeros((num_queries,))
                else:
                    max_iou = (
                        pairwise_iou.detach()
                        .max(dim=1)
                        .values
                        .clamp(0.0, 1.0)
                    )
                # --------------------------------------------------------
                # Unique-TP target
                # --------------------------------------------------------
                target_iou = quality_logits.new_zeros((num_queries,))
                pos_inds = positive_indices[image_index]
                pos_ious = positive_ious[image_index].detach()
                if pos_inds.numel() > 0:
                    target_iou[pos_inds] = pos_ious.clamp(0.0, 1.0)


                logits_i = quality_logits[image_index]   # [Q, T]
                targets_i = self.soft_targets(target_iou)  # [Q, T]

                # --------------------------------------------------------
                # Do NOT weight all 300 queries equally.
                #
                # Otherwise the large number of pure-background queries
                # makes predicting q -> 0 an easy shortcut.
                #
                # IoU < 0.1       : pure/background-like query
                # 0.1 <= IoU < .3 : weak candidate
                # 0.3 <= IoU < .5 : localization-near candidate
                # IoU >= .5       : AP50-capable candidate
                # --------------------------------------------------------
                weights_i = torch.ones_like(max_iou)
                is_positive = torch.zeros_like(max_iou,dtype=torch.bool,)
                if pos_inds.numel() > 0:
                    is_positive[pos_inds] = True
                # Very easy background
                weights_i[
                    (~is_positive)
                    & (max_iou < 0.10)
                ] = 0.10

                # Weak background candidate
                weights_i[
                    (~is_positive)
                    & (max_iou >= 0.10)
                    & (max_iou < 0.30)
                ] = 0.25

                # Near-object unmatched query
                weights_i[
                    (~is_positive)
                    & (max_iou >= 0.30)
                    & (max_iou < 0.50)
                ] = 0.50

                # --------------------------------------------------------
                # Important:
                # high-IoU unmatched query = duplicate hard negative
                # --------------------------------------------------------
                weights_i[
                    (~is_positive)
                    & (max_iou >= 0.50)
                ] = 1.00

                # Hungarian positives
                weights_i[
                    is_positive
                ] = 1.00
                # weights_i = torch.where(
                #     max_iou < 0.10,
                #     weights_i.new_full(weights_i.shape, 0.10),
                #     weights_i,
                # )

                # weights_i = torch.where(
                #     (max_iou >= 0.10) & (max_iou < 0.30),
                #     weights_i.new_full(weights_i.shape, 0.25),
                #     weights_i,
                # )

                # weights_i = torch.where(
                #     (max_iou >= 0.30) & (max_iou < 0.50),
                #     weights_i.new_full(weights_i.shape, 0.50),
                #     weights_i,
                # )

                # IoU >= 0.5 remains weight = 1.0

                all_logits.append(logits_i)
                all_targets.append(targets_i)
                all_weights.append(weights_i)
                all_max_ious.append(max_iou)

            logits = torch.cat(all_logits, dim=0)       # [B*Q, T]
            targets = torch.cat(all_targets, dim=0)     # [B*Q, T]
            query_weights = torch.cat(all_weights, 0)   # [B*Q]
            max_ious = torch.cat(all_max_ious, 0)       # [B*Q]

            # BCE for every threshold, then average thresholds per query.
            per_query_loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            ).mean(dim=-1)

            raw_loss = (
                per_query_loss * query_weights
            ).sum() / query_weights.sum().clamp_min(1.0)

            probabilities = logits.sigmoid()

            # q50 >= q60 >= q70 >= q80
            raw_mono = F.relu(
                probabilities[..., 1:]
                - probabilities[..., :-1]
            ).mean()

            violation = (
                probabilities[..., 1:]
                > probabilities[..., :-1]
            ).float().mean()

            # ------------------------------------------------------------
            # Keep the original positive diagnostics so old logs remain
            # comparable with positive-only RTQD.
            # ------------------------------------------------------------
            positive_probs = []

            for image_index, indices in enumerate(positive_indices):
                if indices.numel():
                    positive_probs.append(
                        quality_logits[
                            image_index, indices
                        ].sigmoid()
                    )

            if positive_probs:
                positive_means = torch.cat(
                    positive_probs, dim=0
                ).mean(dim=0)
            else:
                positive_means = quality_logits.new_zeros(
                    (quality_logits.size(-1),)
                )

            all_means = probabilities.mean(dim=0)

            q50 = probabilities[:, 0]

            def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
                if mask.any():
                    return values[mask].mean()
                return values.new_zeros(())

            diagnostics = {
                "q50_mean_pos": positive_means[0].detach(),
                "q50_mean_all": all_means[0].detach(),

                # New diagnostics: these are important for checking whether
                # all-query RTQD actually separates good and bad candidates.
                "q50_mean_iou_lt10": masked_mean(
                    q50, max_ious < 0.10
                ).detach(),

                "q50_mean_iou_10_30": masked_mean(
                    q50,
                    (max_ious >= 0.10) & (max_ious < 0.30),
                ).detach(),

                "q50_mean_iou_30_50": masked_mean(
                    q50,
                    (max_ious >= 0.30) & (max_ious < 0.50),
                ).detach(),

                "q50_mean_iou_ge50": masked_mean(
                    q50, max_ious >= 0.50
                ).detach(),

                "quality_monotonic_violation_rate":
                    violation.detach(),

                "rtqd_query_weight_mean":
                    query_weights.mean().detach(),
            }

            for index, threshold in enumerate(
                self.thresholds[1:], start=1
            ):
                diagnostics[
                    f"q{int(round(threshold * 100))}_mean_pos"
                ] = positive_means[index].detach()

            return (
                raw_loss * float(loss_weight),
                raw_mono * float(monotonic_weight),
                diagnostics,
            )

        # ================================================================
        # Original positive-only RTQD fallback
        # ================================================================
        selected_logits = []
        selected_targets = []

        for image_index, (indices, ious) in enumerate(
            zip(positive_indices, positive_ious)
        ):
            if indices.numel():
                selected_logits.append(
                    quality_logits[image_index, indices]
                )
                selected_targets.append(
                    self.soft_targets(ious.detach())
                )

        if selected_logits:
            logits = torch.cat(selected_logits, dim=0)
            targets = torch.cat(selected_targets, dim=0)

            raw_loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
            )

            probabilities = logits.sigmoid()

            raw_mono = F.relu(
                probabilities[..., 1:]
                - probabilities[..., :-1]
            ).mean()

            violation = (
                probabilities[..., 1:]
                > probabilities[..., :-1]
            ).float().mean()

            positive_means = probabilities.mean(dim=0)

        else:
            raw_loss = connected_zero(quality_logits)
            raw_mono = connected_zero(quality_logits)
            violation = quality_logits.new_zeros(())

            positive_means = quality_logits.new_zeros(
                (quality_logits.size(-1),)
            )

        all_means = quality_logits.sigmoid().mean(dim=(0, 1))

        diagnostics = {
            "q50_mean_pos": positive_means[0].detach(),
            "q50_mean_all": all_means[0].detach(),
            "quality_monotonic_violation_rate":
                violation.detach(),
        }

        for index, threshold in enumerate(
            self.thresholds[1:], start=1
        ):
            diagnostics[
                f"q{int(round(threshold * 100))}_mean_pos"
            ] = positive_means[index].detach()

        return (
            raw_loss * float(loss_weight),
            raw_mono * float(monotonic_weight),
            diagnostics,
        )


__all__ = ["RotatedThresholdQualityHead"]