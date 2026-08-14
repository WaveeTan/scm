"""DIOR-R ResNet-50 joint RTP-Score experiment."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

from projects.rtp_scope_init_8_13.rtp_score import (
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
        enabled=False,
    ),
    rtqd=dict(
        enabled=True,

        thresholds=(0.5, 0.6, 0.7, 0.8),
        tau=0.05,

        loss_weight=1.0,
        monotonic_weight=0.10,

        # Only final decoder RTQD.
        use_final_decoder=True,
        use_encoder=False,

        # Keep these although encoder RTQD is disabled.
        encoder_preselect_k=900,
        encoder_num_select=300,
        encoder_rerank_start_epoch=8,

        # IMPORTANT:
        # final score = p_cls * q50^0.2
        final_cls_exp=1.0,
        final_quality_exp=0.20,
    ),
    rsu=dict(
        enabled=False,
        mode="pairwise",
        use_unique_head=False,
    ),
)
model["type"] = RTPScoreRotatedRTDETR
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["type"] = RTPScoreRotatedRTDETRHead
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg
# model["bbox_head"]["loss_cls"]["varifocal_loss_iou_type"] = "rbox_iou"

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
