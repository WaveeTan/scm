"""DroneVehicle template for DOTA-format five-class split annotations."""

# ruff: noqa: F401,F403,F405

from mmengine.config import read_base

with read_base():
    from .rtp_r18_dior_all import *


dronevehicle_classes = ("car", "truck", "bus", "van", "freight-car")
dronevehicle_metainfo = dict(classes=dronevehicle_classes)
model["bbox_head"]["num_classes"] = len(dronevehicle_classes)
rtp_score_cfg["scne"]["num_classes"] = len(dronevehicle_classes)
model["rtp_score_cfg"] = rtp_score_cfg
model["bbox_head"]["rtp_score_cfg"] = rtp_score_cfg

# This repository has no native DroneVehicle dataset class. The config expects
# preprocessing into the same DOTA text format used elsewhere in the repo.
dataset_type = "DOTADataset"
data_root = "data/split_ss_dronevehicle/"
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
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/annfiles/",
        data_prefix=dict(img_path="train/images/"),
        metainfo=dronevehicle_metainfo,
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
        ann_file="val/annfiles/",
        data_prefix=dict(img_path="val/images/"),
        metainfo=dronevehicle_metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="DOTAMetric", metric="mAP")
test_evaluator = val_evaluator
work_dir = "./work_dirs/rtp_r18_dronevehicle_rgb_all"
