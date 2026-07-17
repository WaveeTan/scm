import math
from typing import Dict, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor


class BASDGroupState(nn.Module):
    """Scale-density soft groups and checkpointable online BASD state."""

    GROUP_NAMES = (
        'tiny_sparse',
        'tiny_dense',
        'small_sparse',
        'small_dense',
        'medium',
        'large',
    )

    def __init__(self,
                 scale_boundaries: Sequence[float] = (0.02, 0.04, 0.12),
                 scale_temperature: float = 0.20,
                 density_k: int = 3,
                 density_boundary: float = 4.0,
                 density_temperature: float = 0.50,
                 ema_momentum: float = 0.99,
                 progress_delta: int = 50,
                 progress_weight: float = 0.5,
                 budget_temperature: float = 0.20,
                 active_mass_threshold: float = 0.05,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if len(scale_boundaries) != 3:
            raise ValueError('Exactly three scale boundaries are required.')
        if any(value <= 0 for value in scale_boundaries):
            raise ValueError('Scale boundaries must be positive.')
        if list(scale_boundaries) != sorted(scale_boundaries):
            raise ValueError('Scale boundaries must be increasing.')
        if scale_temperature <= 0 or density_temperature <= 0:
            raise ValueError('Soft-group temperatures must be positive.')
        if density_k < 1 or progress_delta < 1:
            raise ValueError('density_k and progress_delta must be positive.')
        if not 0 <= ema_momentum < 1:
            raise ValueError('ema_momentum must be in [0, 1).')
        if budget_temperature <= 0:
            raise ValueError('budget_temperature must be positive.')

        self.scale_boundaries = tuple(float(v) for v in scale_boundaries)
        self.scale_temperature = float(scale_temperature)
        self.density_k = int(density_k)
        self.density_boundary = float(density_boundary)
        self.density_temperature = float(density_temperature)
        self.ema_momentum = float(ema_momentum)
        self.progress_delta = int(progress_delta)
        self.progress_weight = float(progress_weight)
        self.budget_temperature = float(budget_temperature)
        self.active_mass_threshold = float(active_mass_threshold)
        self.eps = float(eps)
        self.num_groups = len(self.GROUP_NAMES)

        self.register_buffer('quality_ema', torch.zeros(self.num_groups))
        self.register_buffer('quality_initialized',
                             torch.zeros(self.num_groups, dtype=torch.bool))
        self.register_buffer('quality_history',
                             torch.zeros(self.progress_delta, self.num_groups))
        self.register_buffer(
            'history_initialized',
            torch.zeros(
                self.progress_delta, self.num_groups, dtype=torch.bool))
        self.register_buffer('history_pointer',
                             torch.zeros((), dtype=torch.long))
        self.register_buffer('update_steps', torch.zeros((), dtype=torch.long))
        self.register_buffer(
            'last_budget',
            torch.full((self.num_groups, ), 1.0 / self.num_groups))
        self.register_buffer('last_difficulty', torch.zeros(self.num_groups))
        self.register_buffer('last_progress', torch.zeros(self.num_groups))
        self.register_buffer('last_mass', torch.zeros(self.num_groups))

    @staticmethod
    def world_size() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @staticmethod
    def global_sum(value: Tensor) -> Tensor:
        value = value.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def _local_density(self, centers: Tensor) -> Tensor:
        num_gt = centers.size(0)
        if num_gt <= 1:
            return centers.new_zeros((num_gt, ))

        k_eff = min(self.density_k, num_gt - 1)
        distances = torch.cdist(centers.float(), centers.float(), p=2)
        distances.fill_diagonal_(float('inf'))
        kth_distance = distances.kthvalue(k_eff, dim=1).values
        density = torch.log1p(
            float(k_eff) / (kth_distance.square() + self.eps))
        return density.to(dtype=centers.dtype)

    @torch.no_grad()
    def compute_membership(self, gt_bboxes: Tensor,
                           img_shape: Sequence[int]) -> Tensor:
        """Return six soft memberships whose rows sum to one."""
        num_gt = gt_bboxes.size(0)
        if num_gt == 0:
            return gt_bboxes.new_zeros((0, self.num_groups))

        img_h, img_w = img_shape[:2]
        image_area = max(float(img_h * img_w), self.eps)
        scale = torch.sqrt(gt_bboxes[:, 2].abs() * gt_bboxes[:, 3].abs() +
                           self.eps)
        scale = scale / math.sqrt(image_area)
        log_scale = torch.log(scale.clamp_min(self.eps))

        boundary = log_scale.new_tensor(
            [math.log(v) for v in self.scale_boundaries])
        gates = torch.sigmoid(
            (log_scale[:, None] - boundary[None, :]) / self.scale_temperature)
        scale_membership = torch.stack(
            (1 - gates[:, 0], gates[:, 0] - gates[:, 1],
             gates[:, 1] - gates[:, 2], gates[:, 2]),
            dim=1).clamp_min(0)
        scale_membership = scale_membership / scale_membership.sum(
            dim=1, keepdim=True).clamp_min(self.eps)

        normalizer = gt_bboxes.new_tensor([img_w, img_h])
        centers = gt_bboxes[:, :2] / normalizer
        density = self._local_density(centers)
        dense = torch.sigmoid(
            (density - self.density_boundary) / self.density_temperature)
        sparse = 1 - dense

        tiny, small, medium, large = scale_membership.unbind(dim=1)
        membership = torch.stack((tiny * sparse, tiny * dense, small * sparse,
                                  small * dense, medium, large),
                                 dim=1)
        return membership / membership.sum(
            dim=1, keepdim=True).clamp_min(self.eps)

    @torch.no_grad()
    def update(self, quality: Tensor, membership: Tensor) -> Dict[str, Tensor]:
        if quality.ndim != 1 or membership.ndim != 2:
            raise ValueError(
                'quality must be [N] and membership must be [N,G].')
        if membership.size(0) != quality.size(0):
            raise ValueError('quality and membership must have the same N.')
        if membership.size(1) != self.num_groups:
            raise ValueError('Unexpected BASD group dimension.')

        quality = quality.detach().float()
        membership = membership.detach().float()
        local_mass = membership.sum(dim=0)
        local_quality_sum = (membership * quality[:, None]).sum(dim=0)
        mass = self.global_sum(local_mass)
        quality_sum = self.global_sum(local_quality_sum)
        batch_quality = quality_sum / mass.clamp_min(self.eps)
        active = mass > self.active_mass_threshold

        initialized_before = self.quality_initialized.clone()
        smoothed = torch.where(
            initialized_before, self.ema_momentum * self.quality_ema +
            (1 - self.ema_momentum) * batch_quality, batch_quality)
        self.quality_ema[active] = smoothed[active]
        self.quality_initialized[active] = True

        pointer = int(self.history_pointer.item())
        old_quality = self.quality_history[pointer].clone()
        old_initialized = self.history_initialized[pointer].clone()
        progress_valid = active & old_initialized
        progress = torch.zeros_like(self.quality_ema)
        progress[progress_valid] = (
            self.quality_ema[progress_valid] - old_quality[progress_valid])

        valid = active & self.quality_initialized
        difficulty = torch.zeros_like(self.quality_ema)
        if valid.any():
            quality_median = self.quality_ema[valid].median()
            difficulty[valid] = torch.relu(quality_median -
                                           self.quality_ema[valid])
        if progress_valid.any():
            progress_median = progress[progress_valid].median()
            difficulty[progress_valid] += self.progress_weight * torch.relu(
                progress_median - progress[progress_valid])

        budget = torch.zeros_like(self.quality_ema)
        if valid.any():
            budget[valid] = torch.softmax(
                difficulty[valid] / self.budget_temperature, dim=0)
        elif self.quality_initialized.any():
            initialized = self.quality_initialized
            budget[initialized] = 1.0 / initialized.sum().float()
        else:
            budget.fill_(1.0 / self.num_groups)

        self.quality_history[pointer].copy_(self.quality_ema)
        self.history_initialized[pointer].copy_(self.quality_initialized)
        self.history_pointer.fill_((pointer + 1) % self.progress_delta)
        self.update_steps.add_(1)
        self.last_budget.copy_(budget)
        self.last_difficulty.copy_(difficulty)
        self.last_progress.copy_(progress)
        self.last_mass.copy_(mass)

        return dict(
            budget=budget.clone(),
            difficulty=difficulty.clone(),
            progress=progress.clone(),
            quality=self.quality_ema.clone(),
            mass=mass.clone(),
            active=valid.clone())


__all__ = ['BASDGroupState']
