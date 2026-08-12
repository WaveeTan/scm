"""DIOR-R ResNet-50 joint RTP-Score experiment."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

from projects.rtp_scope.rtp_score import (
    RTPScoreDiagnosticsHook,
    RTPScoreRotatedRTDETR,
    RTPScoreRotatedRTDETRHead,
    RTPScoreScheduleHook,
)

with read_base():
    from projects.rotated_rtdetr.configs.o2_rtdetr_r50vd_2xb4_72e_dior import *


rtp_score_cfg = dict(
    enabled=True,
    scne=dict(
        enabled=True,
        num_classes=20,
        warmup_epochs=8,
        apply_to_encoder=False,
        apply_to_decoder=True,
    ),
    rtqd=dict(
        enabled=True,
        use_final_decoder=True,
        use_encoder=True,
        encoder_preselect_k=900,
        encoder_num_select=300,
        encoder_rerank_start_epoch=8,
    ),
    rsu=dict(
        enabled=True,
        mode="pairwise",
        use_unique_head=False,
    ),
)
model["type"] = RTPScoreRotatedRTDETR
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["type"] = RTPScoreRotatedRTDETRHead
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["loss_cls"]["varifocal_loss_iou_type"] = "rbox_iou"

custom_hooks.extend(
    [
        dict(
            type=RTPScoreScheduleHook,
            verbose=True,
            priority="NORMAL",
        ),
        dict(
            type=RTPScoreDiagnosticsHook,
            require_nms_free=True,
            priority="VERY_HIGH",
        ),
    ]
)

randomness = dict(seed=42, deterministic=False)
work_dir = "./work_dirs/rtp_r50_dior_all"
