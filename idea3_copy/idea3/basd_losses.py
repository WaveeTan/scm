from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmdet.structures.bbox import BaseBoxes
from torch import Tensor

from ai4rs.registry import MODELS, TASK_UTILS


def _reduce(loss: Tensor, reduction: str,
            avg_factor: Optional[Union[float, Tensor]]) -> Tensor:
    if reduction not in ('none', 'mean', 'sum'):
        raise ValueError(f'Unsupported reduction: {reduction}')
    if reduction == 'none':
        return loss
    if avg_factor is not None:
        factor = loss.new_tensor(float(avg_factor)) if not isinstance(
            avg_factor, Tensor) else avg_factor.to(loss).clamp_min(1)
        return loss.sum() / factor
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


@MODELS.register_module()
class BASDXYWHL1Loss(nn.Module):
    """L1 loss on center and size only, leaving angle to its own term."""

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[float, Tensor]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        reduction = reduction_override or self.reduction
        loss = F.l1_loss(pred[..., :4], target[..., :4], reduction='none')
        if weight is not None:
            loss = loss * weight[..., :4]
        return _reduce(loss, reduction, avg_factor) * self.loss_weight


@MODELS.register_module()
class BASDPeriodicAngleLoss(nn.Module):
    """Pi-periodic angle loss for long-edge rotated boxes."""

    def __init__(self,
                 period_factor: float = 2.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.period_factor = float(period_factor)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[Union[float, Tensor]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        reduction = reduction_override or self.reduction
        delta = pred[..., 4] - target[..., 4]
        loss = 0.5 * (1 - torch.cos(self.period_factor * delta))
        if weight is not None:
            loss = loss * weight[..., 4]
        return _reduce(loss, reduction, avg_factor) * self.loss_weight


@TASK_UTILS.register_module()
class BASDXYWHL1Cost(BaseMatchCost):
    """Normalized center-size L1 matching cost, excluding angle."""

    def __init__(self, weight: Union[float, int] = 1.0) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        img_h, img_w = img_meta['img_shape'][:2]
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h])
        return torch.cdist(
            pred_bboxes[..., :4] / factor, gt_bboxes[..., :4] / factor,
            p=1) * self.weight


@TASK_UTILS.register_module()
class BASDPeriodicAngleCost(BaseMatchCost):
    """Pi-periodic angle cost used by the BASD Hungarian matcher."""

    def __init__(self,
                 period_factor: float = 2.0,
                 weight: Union[float, int] = 1.0) -> None:
        super().__init__(weight=weight)
        self.period_factor = float(period_factor)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_angle = pred_instances.bboxes[:, 4, None]
        gt_bboxes = gt_instances.bboxes
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_angle = gt_bboxes[None, :, 4]
        return 0.5 * (1 - torch.cos(self.period_factor *
                                    (pred_angle - gt_angle))) * self.weight


__all__ = [
    'BASDXYWHL1Loss', 'BASDPeriodicAngleLoss', 'BASDXYWHL1Cost',
    'BASDPeriodicAngleCost'
]
