from mmengine.config import read_base

from projects.scm_rotated_rtdetr_SAGR_8_21.scm_rotated_rtdetr import (
    SCMRotatedRTDETR,
    SCMSAGRRotatedRTDETRHead,
    SceneConditionWarmupHook,
)

with read_base():
    from projects.rotated_rtdetr.configs.o2_rtdetr_r50vd_2xb4_72e_dota import *


# =========================================================================
# 1. Clean DOTA train / validation split
# =========================================================================

data_root = "/root/autodl-tmp/datasets/dota/dotav1/split_ss_dota/"

train_dataloader["dataset"].update(
    data_root=data_root,
    ann_file="train/annfiles/",
    data_prefix=dict(
        img_path="train/images/",
    ),
)

val_dataloader["dataset"].update(
    data_root=data_root,
    ann_file="val/annfiles/",
    data_prefix=dict(
        img_path="val/images/",
    ),
    test_mode=True,
)

test_dataloader["dataset"].update(
    data_root=data_root,
    data_prefix=dict(
        img_path="test/images/",
    ),
    test_mode=True,
)


# =========================================================================
# 2. SCM detector
# =========================================================================

model["type"] = SCMRotatedRTDETR

model.update(
    use_scene_class_bias=True,
    loss_scene_cls_weight=0.05,

    use_scene_scale_bias=True,
    loss_scene_scale_weight=0.05,

    scale_boundaries=(0.02, 0.04, 0.12),
    scale_temperature=0.20,

    scene_cfg=dict(
        num_levels=3,
        num_scene_prototypes=8,
        temperature=0.1,
        scene_bias_scale=0.10,
        num_scale_groups=4,
        scale_bias_scale=0.05,
    ),
)


# =========================================================================
# 3. E2-SAGR head
# =========================================================================

model["bbox_head"]["type"] = SCMSAGRRotatedRTDETRHead

model["bbox_head"]["sagr_loss_cfg"] = dict(
    scale_ref=0.04,

    ar_ref=4.0,
    ar_full=8.0,

    center_weight=1.0,
    angle_weight=0.0,
    short_side_weight=0.5,

    loss_weight=0.5,

    short_side_floor=0.01,

    angle_period=1.0,
)


# =========================================================================
# 4. Scene-condition warm-up
# =========================================================================

custom_hooks.append(
    dict(
        type=SceneConditionWarmupHook,
        start_epoch=0,
        full_epoch=24,
        min_mix=0.05,
        verbose=True,
        priority="NORMAL",
    )
)


# =========================================================================
# 5. Distributed wrapper
# =========================================================================

model_wrapper_cfg = dict(
    type="MMDistributedDataParallel",
    find_unused_parameters=False,
)


# =========================================================================
# 6. Experiment output
# =========================================================================

work_dir = (
    "./work_dirs/"
    "scm_o2_rtdetr_r50vd_2xb4_72e_dota_sagr_e2_fixed"
)


# =========================================================================
# 7. Validation / test evaluation
# =========================================================================

val_evaluator = dict(
    type="DOTAMetric",
    metric="mAP",
)

test_evaluator = dict(
    type="DOTAMetric",
    format_only=True,
    merge_patches=True,
    outfile_prefix=f"{work_dir}/Task1",
)
