"""Validated configuration defaults for RTP-Score."""

from copy import deepcopy
from typing import Mapping, Optional


DEFAULT_RTP_SCORE_CFG = dict(
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
        num_pool_heads=1,
        use_layer_norm=False,
        bottleneck_dim=None,
        dropout=0.10,
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


def _merge_known(target: dict, update: Mapping, path: str) -> None:
    for key, value in update.items():
        if key not in target:
            raise ValueError(f"Unknown RTP-Score option: {path}{key}")
        if isinstance(target[key], dict):
            if not isinstance(value, Mapping):
                raise TypeError(f"{path}{key} must be a mapping")
            _merge_known(target[key], value, f"{path}{key}.")
        else:
            target[key] = value


def build_rtp_score_cfg(cfg: Optional[Mapping]) -> dict:
    """Merge a user config into the documented schema and validate it."""
    merged = deepcopy(DEFAULT_RTP_SCORE_CFG)
    if cfg is not None:
        _merge_known(merged, dict(cfg), "")

    if merged["eps"] <= 0:
        raise ValueError("RTP-Score eps must be positive")
    if merged["score_fusion"]["mode"] != "geometric_mean":
        raise ValueError("Only geometric_mean score fusion is supported")
    scne_cfg = merged["scne"]
    if min(
        int(scne_cfg[name])
        for name in ("num_classes", "hidden_dim", "num_pool_heads")
    ) <= 0:
        raise ValueError(
            "SCNE class, hidden and pooling dimensions must be positive"
        )
    if scne_cfg["bottleneck_dim"] is not None and int(
        scne_cfg["bottleneck_dim"]
    ) <= 0:
        raise ValueError("SCNE bottleneck_dim must be positive or None")
    if not 0 < float(scne_cfg["topk_ratio"]) <= 1:
        raise ValueError("SCNE topk_ratio must be in (0, 1]")
    if not 0 <= float(scne_cfg["dropout"]) < 1:
        raise ValueError("SCNE dropout must be in [0, 1)")
    fusion_exponents = [
        merged["score_fusion"][name]
        for name in ("cls_exp", "quality_exp", "unique_exp")
    ]
    if min(fusion_exponents) < 0:
        raise ValueError("score-fusion exponents must be non-negative")
    if merged["rtqd"]["unmatched_policy"] != "ignore":
        raise ValueError("RTQD unmatched_policy must remain 'ignore'")
    if not merged["rtqd"]["detach_boxes"]:
        raise ValueError(
            "RTQD targets must detach predicted boxes in this implementation"
        )
    if merged["rsu"]["mode"] != "pairwise":
        raise ValueError("The implemented RSU mode is 'pairwise'")
    if not merged["rsu"]["final_decoder_only"]:
        raise ValueError("RSU must remain final_decoder_only")
    if not merged["rsu"]["exclude_dn"]:
        raise ValueError("RSU must exclude denoising queries")
    if merged["rsu"]["use_unique_head"] and not merged["rsu"]["enabled"]:
        raise ValueError("use_unique_head requires RSU to be enabled")
    if merged["rsu"]["use_unique_head"] and not (
        merged["rtqd"]["enabled"]
        and merged["rtqd"]["use_final_decoder"]
    ):
        raise ValueError("use_unique_head requires final-decoder RTQD")
    return merged


__all__ = ["DEFAULT_RTP_SCORE_CFG", "build_rtp_score_cfg"]
