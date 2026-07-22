"""Scale-density groups and lagged regression-error state for Idea5."""

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn


class BASDRegGroupState(nn.Module):
    """Checkpointable six-group state used by BASD-Reg.

    The implementation is intentionally identical to Idea3.  Idea5 changes
    final-query score calibration, not the BASD-Reg grouping or update rule.
    """

    GROUP_NAMES = (
        "tiny_sparse",
        "tiny_dense",
        "small_sparse",
        "small_dense",
        "medium",
        "large",
    )

    def __init__(
        self,
        scale_boundaries: Sequence[float],
        scale_temperature: float = 0.20,
        density_k: int = 3,
        density_boundary: float = 4.0,
        density_temperature: float = 0.50,
        ema_momentum: float = 0.99,
        active_mass_threshold: float = 2.0,
        difficulty_temperature: float = 0.05,
        max_alpha: float = 0.20,
        weight_min: float = 0.80,
        weight_max: float = 1.20,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(scale_boundaries) != 3:
            raise ValueError("scale_boundaries must contain q25/q50/q75")
        boundaries = tuple(float(value) for value in scale_boundaries)
        if any(value <= 0 for value in boundaries):
            raise ValueError("scale_boundaries must be positive")
        if tuple(sorted(boundaries)) != boundaries:
            raise ValueError("scale_boundaries must be strictly increasing")
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("scale_boundaries must be strictly increasing")
        if scale_temperature <= 0 or density_temperature <= 0:
            raise ValueError("membership temperatures must be positive")
        if density_k < 1:
            raise ValueError("density_k must be positive")
        if not 0 <= ema_momentum < 1:
            raise ValueError("ema_momentum must be in [0, 1)")
        if active_mass_threshold <= 0:
            raise ValueError("active_mass_threshold must be positive")
        if difficulty_temperature <= 0:
            raise ValueError("difficulty_temperature must be positive")
        if not 0 <= max_alpha < 1:
            raise ValueError("max_alpha must be in [0, 1)")
        if not 0 < weight_min <= 1 <= weight_max:
            raise ValueError("weights must satisfy 0 < min <= 1 <= max")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.scale_boundaries = boundaries
        self.scale_temperature = float(scale_temperature)
        self.density_k = int(density_k)
        self.density_boundary = float(density_boundary)
        self.density_temperature = float(density_temperature)
        self.ema_momentum = float(ema_momentum)
        self.active_mass_threshold = float(active_mass_threshold)
        self.difficulty_temperature = float(difficulty_temperature)
        self.max_alpha = float(max_alpha)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.eps = float(eps)
        self.num_groups = len(self.GROUP_NAMES)

        self.register_buffer("error_ema", torch.zeros(self.num_groups))
        self.register_buffer(
            "error_initialized", torch.zeros(self.num_groups, dtype=torch.bool)
        )
        self.register_buffer("last_mass", torch.zeros(self.num_groups))
        self.register_buffer("last_group_weight", torch.ones(self.num_groups))
        self.register_buffer("current_alpha", torch.zeros(()))

    @staticmethod
    def world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def global_sum(value: Tensor) -> Tensor:
        result = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    @torch.no_grad()
    def set_alpha(self, value: float) -> float:
        applied = min(max(float(value), 0.0), self.max_alpha)
        self.current_alpha.fill_(applied)
        return applied

    def _local_density(self, centers: Tensor) -> Tensor:
        num_gt = centers.size(0)
        if num_gt <= 1:
            return centers.new_zeros((num_gt,))
        k_eff = min(self.density_k, num_gt - 1)
        distances = torch.cdist(centers.float(), centers.float(), p=2)
        distances.fill_diagonal_(float("inf"))
        kth_distance = distances.kthvalue(k_eff, dim=1).values
        density = torch.log1p(float(k_eff) / (kth_distance.square() + self.eps))
        return density.to(dtype=centers.dtype)

    @torch.no_grad()
    def compute_membership(self, gt_bboxes: Tensor, img_shape: Sequence[int]) -> Tensor:
        """Return ``[N, 6]`` non-negative memberships summing to one."""
        num_gt = gt_bboxes.size(0)
        if num_gt == 0:
            return gt_bboxes.new_zeros((0, self.num_groups))

        img_h, img_w = img_shape[:2]
        image_area = max(float(img_h * img_w), self.eps)
        box_area = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).abs()
        scale = torch.sqrt((box_area / image_area).clamp_min(0))
        log_scale = torch.log(scale.clamp_min(self.eps))
        log_boundaries = log_scale.new_tensor(
            [math.log(value) for value in self.scale_boundaries]
        )
        gates = torch.sigmoid(
            (log_scale[:, None] - log_boundaries[None, :]) / self.scale_temperature
        )
        scale_membership = torch.stack(
            (
                1 - gates[:, 0],
                gates[:, 0] - gates[:, 1],
                gates[:, 1] - gates[:, 2],
                gates[:, 2],
            ),
            dim=1,
        ).clamp_min(0)
        scale_membership = scale_membership / scale_membership.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)

        normalizer = gt_bboxes.new_tensor([img_w, img_h])
        centers = gt_bboxes[:, :2] / normalizer
        density = self._local_density(centers)
        dense = torch.sigmoid(
            (density - self.density_boundary) / self.density_temperature
        )
        sparse = 1 - dense
        tiny, small, medium, large = scale_membership.unbind(dim=1)
        membership = torch.stack(
            (
                tiny * sparse,
                tiny * dense,
                small * sparse,
                small * dense,
                medium,
                large,
            ),
            dim=1,
        )
        return membership / membership.sum(dim=1, keepdim=True).clamp_min(self.eps)

    @torch.no_grad()
    def compute_weights(self, membership: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Use lagged EMA values to compute bounded instance weights."""
        if membership.ndim != 2 or membership.size(1) != self.num_groups:
            raise ValueError("membership must have shape [N, 6]")
        membership = membership.detach().float()
        mass = self.global_sum(membership.sum(dim=0))
        valid = self.error_initialized & (mass > 0)

        group_weight = torch.ones_like(self.error_ema)
        if valid.any() and float(self.current_alpha.item()) > 0:
            reference_error = (mass[valid] * self.error_ema[valid]).sum() / mass[
                valid
            ].sum().clamp_min(self.eps)
            delta = (self.error_ema[valid] - reference_error) / (
                self.difficulty_temperature
            )
            group_weight[valid] = 1 + self.current_alpha * torch.tanh(delta)

        instance_weight = 1 + membership @ (group_weight.to(membership) - 1)
        instance_weight = instance_weight.clamp(self.weight_min, self.weight_max)
        self.last_mass.copy_(mass)
        self.last_group_weight.copy_(group_weight)
        return instance_weight, group_weight.clone(), mass.clone()

    def global_weighted_mean(
        self, per_instance_loss: Tensor, instance_weight: Tensor
    ) -> Tensor:
        """DDP-equivalent global weighted mean with a detached denominator."""
        if per_instance_loss.ndim != 1 or instance_weight.ndim != 1:
            raise ValueError("loss and weight must both have shape [N]")
        if per_instance_loss.numel() != instance_weight.numel():
            raise ValueError("loss and weight must contain the same N")
        weight = instance_weight.detach().to(per_instance_loss)
        local_numerator = (per_instance_loss * weight).sum()
        global_denominator = self.global_sum(weight.sum().detach())
        return (
            local_numerator * self.world_size() / global_denominator.clamp_min(self.eps)
        )

    @torch.no_grad()
    def summarize_instance_weights(self, instance_weight: Tensor) -> Dict[str, Tensor]:
        """Return DDP-global min/mean/max diagnostics."""
        weight = instance_weight.detach().float()
        count = weight.new_tensor(float(weight.numel()))
        total = weight.sum()
        minimum = weight.min() if weight.numel() else weight.new_tensor(float("inf"))
        maximum = weight.max() if weight.numel() else weight.new_tensor(float("-inf"))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if count.item() == 0:
            one = weight.new_ones(())
            return dict(min=one, mean=one, max=one)
        return dict(min=minimum, mean=total / count, max=maximum)

    @torch.no_grad()
    def update(self, regression_error: Tensor, membership: Tensor) -> Dict[str, Tensor]:
        """Update group error EMA after the current loss has been formed."""
        if regression_error.ndim != 1:
            raise ValueError("regression_error must have shape [N]")
        if membership.ndim != 2 or membership.size(1) != self.num_groups:
            raise ValueError("membership must have shape [N, 6]")
        if regression_error.numel() != membership.size(0):
            raise ValueError("regression_error and membership disagree on N")

        error = regression_error.detach().float()
        membership = membership.detach().float()
        mass = self.global_sum(membership.sum(dim=0))
        error_sum = self.global_sum((membership * error[:, None]).sum(dim=0))
        batch_error = error_sum / mass.clamp_min(self.eps)
        active = mass >= self.active_mass_threshold
        initialized_before = self.error_initialized.clone()
        smoothed = torch.where(
            initialized_before,
            self.ema_momentum * self.error_ema + (1 - self.ema_momentum) * batch_error,
            batch_error,
        )
        self.error_ema[active] = smoothed[active]
        self.error_initialized[active] = True
        self.last_mass.copy_(mass)
        return dict(
            mass=mass.clone(),
            error_ema=self.error_ema.clone(),
            initialized=self.error_initialized.clone(),
            active=active.clone(),
            group_weight=self.last_group_weight.clone(),
        )


__all__ = ["BASDRegGroupState"]
