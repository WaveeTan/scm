"""Phase 6: RIACS plus negative-only SCNE."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

with read_base():
    from .rtp_r18_dior_rotated_iacs import *


rtp_score_cfg["scne"].update(
    enabled=True,
    warmup_epochs=8,
    apply_to_encoder=False,
    apply_to_decoder=True,
)
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg
work_dir = "./work_dirs/rtp_r18_dior_scne"
