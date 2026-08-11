"""Phase 7: SCNE + final/encoder RTQD + pairwise RSU."""

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
rtp_score_cfg["rtqd"].update(
    enabled=True,
    use_final_decoder=True,
    use_encoder=True,
    encoder_preselect_k=900,
    encoder_num_select=300,
    encoder_rerank_start_epoch=8,
)
rtp_score_cfg["rsu"].update(
    enabled=True,
    mode="pairwise",
    use_unique_head=False,
)
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg
work_dir = "./work_dirs/rtp_r18_dior_all"
