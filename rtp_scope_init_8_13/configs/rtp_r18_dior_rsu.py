"""Phase 5: RIACS plus final matching-query pairwise RSU."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

with read_base():
    from .rtp_r18_dior_rotated_iacs import *


rtp_score_cfg["rsu"].update(
    enabled=True,
    mode="pairwise",
    use_unique_head=False,
)
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg
work_dir = "./work_dirs/rtp_r18_dior_rsu"
