"""FAIR1M joint RTP-Score config using the repository FAIR split layout."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

with read_base():
    from .rtp_r18_dior_all import *


# FAIR1M has 37 categories in ai4rs.datasets.FAIRDataset. The filename keeps
# the spec's R34 experiment name; replace the inherited R18 backbone with the
# repository's R34 layout.
model["backbone"]["depth"] = 34
model["backbone"]["init_cfg"]["checkpoint"] = (
    "https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/"
    "master/rtdetr/resnet34vd_pretrained_f6a72dc5.pth"
)
model["decoder"]["num_layers"] = 4
model["bbox_head"]["num_classes"] = 37
rtp_score_cfg["scne"]["num_classes"] = 37
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg

num_blocks_list = (3, 4, 6, 3)
downsample_norm_idx_list = (2, 3, 3, 3)
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = {"backbone": dict(lr_mult=0.1)}
custom_keys.update(
    {
        f"backbone.layer{stage_id + 1}.{block_id}.bn":
        backbone_norm_multi
        for stage_id, num_blocks in enumerate(num_blocks_list)
        for block_id in range(num_blocks)
    }
)
custom_keys.update(
    {
        (
            f"backbone.layer{stage_id + 1}.{block_id}.downsample."
            f"{downsample_norm_idx - 1}"
        ): backbone_norm_multi
        for stage_id, (num_blocks, downsample_norm_idx) in enumerate(
            zip(num_blocks_list, downsample_norm_idx_list)
        )
        for block_id in range(num_blocks)
    }
)
optim_wrapper["paramwise_cfg"]["custom_keys"] = custom_keys

dataset_type = "FAIRDataset"
data_root = "data/split_ss_fair1m1.0/"
backend_args = None
train_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="mmdet.RandomFlip",
        prob=0.75,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(type="mmdet.PackDetInputs"),
]
val_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]
test_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/annfiles/",
        data_prefix=dict(img_path="train/images/"),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/annfiles/",
        data_prefix=dict(img_path="train/images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="test/images/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
val_evaluator = dict(type="FAIRMetric", metric="mAP")
test_evaluator = dict(
    type="FAIRMetric",
    format_only=True,
    merge_patches=True,
    outfile_prefix="./work_dirs/rtp_r34_fair1m_all/Task1",
)
work_dir = "./work_dirs/rtp_r34_fair1m_all"
