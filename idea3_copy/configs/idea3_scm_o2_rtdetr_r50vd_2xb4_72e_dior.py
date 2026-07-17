# ruff: noqa: F405

from mmengine.config import read_base
from mmdet.models.task_modules import FocalLossCost

from ai4rs.models.losses import RotatedIoULoss
from projects.idea3.idea3 import (BASDAmbiguityWarmupHook,
                                  BASDHungarianAssigner, BASDPeriodicAngleCost,
                                  BASDPeriodicAngleLoss, BASDRotatedRTDETRHead,
                                  BASDXYWHL1Cost, BASDXYWHL1Loss,
                                  Idea3SCMRotatedRTDETR)
from projects.rotated_dino.rotated_dino.match_cost import RotatedIoUCost

with read_base():
    from projects.scm_rotated_rtdetr.configs.scm_o2_rtdetr_r50vd_2xb4_72e_dior import *  # noqa: F401,F403,E501

# Keep SCM's scene-conditioned encoder selection and replace only its detector
# type, matching head, matcher and positive-instance loss reduction.
model['type'] = Idea3SCMRotatedRTDETR
model['bbox_head']['type'] = BASDRotatedRTDETRHead
model['bbox_head']['loss_cls']['varifocal_loss_iou_type'] = 'rbox_iou'
model['bbox_head']['loss_bbox'] = dict(type=BASDXYWHL1Loss, loss_weight=5.0)
model['bbox_head']['loss_iou'] = dict(
    type=RotatedIoULoss, mode='linear', loss_weight=1.5)
model['bbox_head']['loss_angle'] = dict(
    type=BASDPeriodicAngleLoss, period_factor=2.0, loss_weight=1.0)
model['bbox_head']['basd_cfg'] = dict(
    # COCO-compatible 16/32/96-pixel boundaries at an 800-pixel reference,
    # expressed through the resolution-invariant sqrt(area / image_area).
    scale_boundaries=(0.02, 0.04, 0.12),
    scale_temperature=0.20,
    density_k=3,
    density_boundary=4.0,
    density_temperature=0.50,
    ema_momentum=0.99,
    progress_delta=50,
    progress_weight=0.5,
    budget_temperature=0.20,
    active_mass_threshold=0.05,
    match_topk=5,
    match_temperature=1.0,
    ambiguity_rho=0.0,
    angle_iou_gamma=2.0,
    eps=1e-6)

model['train_cfg']['assigner'] = dict(
    type=BASDHungarianAssigner,
    match_costs=[
        dict(type=FocalLossCost, weight=2.0),
        dict(type=BASDXYWHL1Cost, weight=5.0),
        dict(type=RotatedIoUCost, iou_mode='iou', weight=2.0),
        dict(type=BASDPeriodicAngleCost, period_factor=2.0, weight=1.0)
    ])

custom_hooks.append(
    dict(
        type=BASDAmbiguityWarmupHook,
        start_epoch=3,
        full_epoch=12,
        max_rho=0.5,
        verbose=True,
        priority='NORMAL'))
