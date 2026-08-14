"""RTP-Score R18 DIOR-R base: all new ranking modules disabled."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

from projects.rtp_scope_init_8_13.rtp_score import (
    RTPScoreDiagnosticsHook,
    RTPScoreRotatedRTDETR,
    RTPScoreRotatedRTDETRHead,
    RTPScoreScheduleHook,
)

with read_base():
    from projects.rotated_rtdetr.configs.o2_rtdetr_r18vd_2xb4_72e_dior import *


rtp_score_cfg = dict(
    enabled=True,
    eps=1e-6,
    score_fusion=dict(
        mode="geometric_mean",
        cls_exp=0.65,
        quality_exp=0.20,
        unique_exp=0.15,
    ),
    scne=dict(
        enabled=False,
        num_classes=20,
        hidden_dim=256,
        topk_ratio=0.02,
        loss_weight=0.20,
        gamma_pos=0.0,
        gamma_neg=4.0,
        asl_clip=0.05,
        calibration_lambda=0.50,
        warmup_epochs=8,
        presence_threshold=0.20,
        min_bias=-1.50,
        detach_calibration=True,
        apply_to_encoder=False,
        apply_to_decoder=True,
    ),
    rtqd=dict(
        enabled=False,
        thresholds=(0.5, 0.6, 0.7, 0.8),
        tau=0.05,
        loss_weight=1.0,
        monotonic_weight=0.10,
        unmatched_policy="ignore",
        detach_boxes=True,
        use_final_decoder=True,
        use_encoder=False,
        encoder_preselect_k=900,
        encoder_num_select=300,
        encoder_cls_exp=0.70,
        encoder_quality_exp=0.30,
        encoder_rerank_start_epoch=8,
        final_cls_exp=0.70,
        final_quality_exp=0.30,
    ),
    rsu=dict(
        enabled=False,
        mode="pairwise",
        rival_iou_thr=0.30,
        max_rivals_per_gt=3,
        margin=0.20,
        loss_weight=0.20,
        final_decoder_only=True,
        exclude_dn=True,
        require_argmax_class_match=True,
        use_unique_head=False,
        unique_head_loss_weight=0.50,
    ),
)
model["type"] = RTPScoreRotatedRTDETR
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["type"] = RTPScoreRotatedRTDETRHead
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg

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
