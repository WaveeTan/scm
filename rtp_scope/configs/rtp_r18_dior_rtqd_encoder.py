"""Phase 4: decoder RTQD plus warmed encoder preselection/reranking."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

with read_base():
    from .rtp_r18_dior_rtqd_final import *


rtp_score_cfg["rtqd"].update(
    use_encoder=True,
    encoder_preselect_k=900,
    encoder_num_select=300,
    encoder_rerank_start_epoch=8,
)
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg
work_dir = "./work_dirs/rtp_r18_dior_rtqd_encoder"
