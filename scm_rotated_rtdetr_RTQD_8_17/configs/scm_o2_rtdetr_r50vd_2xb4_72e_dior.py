from torch.optim.adamw import AdamW

from ai4rs.models.losses import GDLoss
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.layers.ema import ExpMomentumEMA
from mmdet.models.losses import L1Loss
from mmdet.models.necks import ChannelMapper
from mmdet.models.task_modules import FocalLossCost, HungarianAssigner
from mmengine.hooks.ema_hook import EMAHook
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from projects.rotated_dino.rotated_dino.match_cost import ChamferCost, GDCost
from projects.rotated_rtdetr.rotated_rtdetr import (
    RTDETRFPN, RTDETRVarifocalLoss, ResNetV1dPaddle, RotatedRTDETRHead)
from projects.scm_rotated_rtdetr_RTQD_8_17.scm_rotated_rtdetr import (
    SCMRotatedRTDETR, SceneConditionWarmupHook)


default_scope = 'ai4rs'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=99999),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False


# dataset settings
dataset_type = 'DIORDataset'
data_root = 'data/DIOR/'
backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['DATASET_TYPE'],
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='ImageSets/Main/train.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='ImageSets/Main/val.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline,
                backend_args=backend_args)
        ]))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(img_path='JPEGImages-test'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.5])
test_evaluator = val_evaluator


pretrained = ('https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/'
              'master/rtdetr/resnet50vd_ssld_v2_pretrained_d037e232.pth')

angle_cfg = dict(width_longer=True, start_angle=0)
angle_factor = 3.1415926535897932384626433832795

model = dict(
    type=SCMRotatedRTDETR,
    num_queries=300,
    with_box_refine=True,
    as_two_stage=True,
    use_scene_class_bias=True,
    loss_scene_cls_weight=0.05,
    use_scene_scale_bias=True,
    loss_scene_scale_weight=0.05,
    use_scene_ar_bias=True,
    loss_scene_ar_weight=0.05,
    ar_boundaries=(
        2.0,
        4.0,
        8.0,
    ),

    ar_temperature=0.20,
    scale_boundaries=(
        0.02,
        0.04,
        0.12,
    ),
    # Soft boundary in log-scale.
    scale_temperature=0.20,
    scene_cfg=dict(
        num_levels=3,
        num_scene_prototypes=8,
        temperature=0.1,

        scene_bias_scale=0.10,
        num_scale_groups=4,
        scale_bias_scale=0.05,
        num_ar_groups=4,
        ar_bias_scale=0.05),
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type=ResNetV1dPaddle,
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type=ChannelMapper,
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=3,
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=5**0.5,
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')),
    encoder=dict(
        use_encoder_idx=[-1],
        num_encoder_layers=1,
        in_channels=[256, 256, 256],
        fpn_cfg=dict(
            type=RTDETRFPN,
            in_channels=[256, 256, 256],
            out_channels=256,
            expansion=1.0,
            norm_cfg=dict(type='BN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        angle_factor=angle_factor,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=3,
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        type=RotatedRTDETRHead,
        num_classes=20,
        angle_cfg=angle_cfg,
        angle_factor=angle_factor,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type=RTDETRVarifocalLoss,
            varifocal_loss_iou_type='hbox_iou',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type=L1Loss, loss_weight=5.0),
        loss_iou=dict(
            type=GDLoss,
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        angle_cfg=angle_cfg,
        angle_factor=angle_factor,
        noise_mode='only_xyxy',
        group_cfg=dict(
            dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                dict(type=FocalLossCost, weight=2.0),
                dict(type=ChamferCost, weight=5.0, box_format='xywha'),
                dict(
                    type=GDCost,
                    loss_type='kld',
                    fun='log1p',
                    tau=1,
                    sqrt=False,
                    weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))


optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1)},
        norm_decay_mult=0,
        bypass_duplicate=True))

max_epochs = 72
train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=6)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=2000)
]

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0001,
        update_buffers=True,
        priority=49),
    dict(
        type=SceneConditionWarmupHook,
        start_epoch=0,
        full_epoch=24,
        min_mix=0.05,
        verbose=True,
        priority='NORMAL')
]

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel', find_unused_parameters=False)

auto_scale_lr = dict(enable=False, base_batch_size=8)
