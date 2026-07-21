# ruff: noqa: F405

from mmengine.config import read_base
from mmdet.models.losses import L1Loss
from mmdet.models.task_modules import FocalLossCost, HungarianAssigner

from ai4rs.models.losses import GDLoss
from projects.idea3.idea3 import (
    BASDRegRotatedRTDETRHead,
    BASDRegWarmupHook,
    Idea3SCMRotatedRTDETR,
)
from projects.rotated_dino.rotated_dino.match_cost import ChamferCost, GDCost

with read_base():
    from projects.scm_rotated_rtdetr.configs.scm_o2_rtdetr_r50vd_2xb4_72e_dior import *  # noqa: F401,F403,E501

# Keep SCM and all original O2 objectives. BASD-Reg replaces only the final
# decoder layer's matched-positive L1/KLD reduction.
model["type"] = Idea3SCMRotatedRTDETR
model["bbox_head"]["type"] = BASDRegRotatedRTDETRHead
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

# The repository does not contain data/DIOR, so these bootstrap values could
# not be replaced by measured train+val quantiles here. Before a formal run,
# execute projects/idea3/tools/compute_basd_reg_stats.py on the training machine
# and paste its q25/q50/q75 and lower-scale density median below. Never include
# test.txt.
model["bbox_head"]["basd_reg_cfg"] = dict(
    scale_boundaries=(0.02, 0.04, 0.12),
    scale_temperature=0.20,
    density_k=3,
    density_boundary=4.0,
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

# Explicitly restore the unmodified O2 Hungarian matcher.
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

custom_hooks.append(
    dict(
        type=BASDRegWarmupHook,
        start_epoch=6,
        full_epoch=18,
        max_alpha=0.20,
        verbose=True,
        priority="NORMAL",
    )
)

randomness = dict(seed=42, deterministic=False)
work_dir = "./work_dirs/idea3_basd_reg_seed_42"
