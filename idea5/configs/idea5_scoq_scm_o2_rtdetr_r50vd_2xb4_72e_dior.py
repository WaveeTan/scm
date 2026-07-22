# ruff: noqa: F405

from mmengine.config import read_base
from mmdet.models.losses import L1Loss
from mmdet.models.task_modules import FocalLossCost, HungarianAssigner

from ai4rs.models.losses import GDLoss
from projects.idea5.idea5 import (
    Idea5BASDRegWarmupHook,
    Idea5SCMRotatedRTDETR,
    SCOQBASDRegRotatedRTDETRHead,
    SCOQWarmupHook,
)
from projects.rotated_dino.rotated_dino.match_cost import ChamferCost, GDCost

with read_base():
    from projects.scm_rotated_rtdetr.configs.scm_o2_rtdetr_r50vd_2xb4_72e_dior import *  # noqa: E501,F401,F403

# Idea3 keeps the same SCM, matcher and BASD-Reg objectives/routing.  Idea5 adds
# only a final matching-query quality loss and pre-top-k score calibration.
model["type"] = Idea5SCMRotatedRTDETR
model["bbox_head"]["type"] = SCOQBASDRegRotatedRTDETRHead
model["bbox_head"]["loss_cls"]["varifocal_loss_iou_type"] = "hbox_iou"
model["bbox_head"]["loss_bbox"] = dict(type=L1Loss, loss_weight=5.0)
model["bbox_head"]["loss_iou"] = dict(
    type=GDLoss,
    loss_type="kld",
    fun="log1p",
    tau=1,
    sqrt=False,
    loss_weight=2.0,
)

# These are Idea3's checked-in bootstrap geometry values.  Replace them with
# traced train-split measurements before a formal run.
model["bbox_head"]["basd_reg_cfg"] = dict(
    scale_boundaries=(0.02404707, 0.04185167, 0.12871358),
    scale_temperature=0.20,
    density_k=3,
    density_boundary=6.38047266,
    density_temperature=0.50,
    ema_momentum=0.99,
    active_mass_threshold=2.0,
    difficulty_temperature=0.05,
    max_alpha=0.20,
    weight_min=0.80,
    weight_max=1.20,
    apply_to="final_decoder_regression",
    use_ambiguity=False,
    use_progress=False,
    eps=1e-6,
)
model["bbox_head"]["scoq_cfg"] = dict(
    quality_hidden_dims=64,
    quality_loss_weight=0.25,
    quality_beta=0.25,
    quality_detach_inputs=True,
    use_scene_conditioning=True,
    # True follows the proposed joint objectness-quality target.  Set False
    # for the Cascade-DETR-style matched-positive-only control.
    quality_supervise_negatives=True,
    quality_loss_type="vfl",
    quality_vfl_alpha=0.75,
    quality_vfl_gamma=2.0,
    quality_prior_prob=0.01,
    quality_eps=1e-6,
)

model["train_cfg"]["assigner"] = dict(
    type=HungarianAssigner,
    match_costs=[
        dict(type=FocalLossCost, weight=2.0),
        dict(type=ChamferCost, weight=5.0, box_format="xywha"),
        dict(
            type=GDCost,
            loss_type="kld",
            fun="log1p",
            tau=1,
            sqrt=False,
            weight=2.0,
        ),
    ],
)

custom_hooks.extend(
    [
        dict(
            type=Idea5BASDRegWarmupHook,
            start_epoch=6,
            full_epoch=18,
            max_alpha=0.20,
            verbose=True,
            priority="NORMAL",
        ),
        dict(
            type=SCOQWarmupHook,
            start_epoch=6,
            verbose=True,
            priority="NORMAL",
        ),
    ]
)

# Report both the legacy AP50 target and the localization-sensitive AP75.  The
# inherited SCM val split currently aliases test.txt; do not tune beta on it.
val_evaluator["iou_thrs"] = [0.5, 0.75]
test_evaluator["iou_thrs"] = [0.5, 0.75]

# The inherited ``val_dataloader`` points at test.txt.  Disable periodic
# validation by default so training cannot repeatedly inspect the test set.
# Formal work must create a held-out split, remove it from training, recompute
# BASD statistics, and then restore a meaningful validation interval.
train_cfg["val_interval"] = max_epochs + 1

randomness = dict(seed=42, deterministic=False)
work_dir = "./work_dirs/idea5_scoq_basd_reg_seed_42"
