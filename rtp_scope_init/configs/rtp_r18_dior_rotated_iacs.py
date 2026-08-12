"""Phase 1: rotated-IoU-aware classification score, no RTP modules."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

with read_base():
    from ._base_rtp_score import *


model["bbox_head"]["loss_cls"]["varifocal_loss_iou_type"] = "rbox_iou"
work_dir = "./work_dirs/rtp_r18_dior_rotated_iacs"
