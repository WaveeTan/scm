"""Phase 3: RIACS plus final-decoder RTQD."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

with read_base():
    from .rtp_r18_dior_rotated_iacs import *


rtp_score_cfg["rtqd"].update(
    enabled=True,
    use_final_decoder=True,
    use_encoder=False,
)
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg
work_dir = "./work_dirs/rtp_r18_dior_rtqd_final"
