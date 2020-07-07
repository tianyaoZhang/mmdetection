"""
@author Tianyao Zhang
@desc   更新mmdetection==2.2.0以后，该文件可以直接调用

        version log:
        1） output/work_dirs/gfl_r50_1x/
            学习率按照atss进行调整
        2) output/work_dirs/gfl_r50_1x_v2/
            更改学习律为原始版本
        3) 'output/work_dirs/gfl_r50_ms3'
            小样本train=1000,val=500

@date   2020/07/06
说明：
parameters:

function：

Data：
"""
# model settings
model = dict(
    type='GFL',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetV1d',
        depth=50,
        base_channels=32,
        stem_channels=32,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=64,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=1,
        in_channels=64,
        stacked_convs=2,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)

# dataset settings
dataset_type = 'CocoDataset'
data_root = "/home/tianyao/Documents/DeeCamp/data/"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(640, 320), (640, 400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/home/tianyao/Documents/DeeCamp/output/sampleddata/train_1000.json',
        # ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/valid.json',
        ann_file='/home/tianyao/Documents/DeeCamp/output/sampleddata/valid_500.json',
        img_prefix=data_root + 'valid/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/testA_image_info.json',
        ann_file='/home/tianyao/Documents/DeeCamp/output/sampleddata/valid_1000.json',
        img_prefix=data_root + 'testA/',
        pipeline=test_pipeline))

total_epochs = 3
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[1,2])
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[2])

# runtime settings
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/tianyao/Documents/DeeCamp/output/work_dirs/gfl_r50_1x_v2/'
load_from = None
resume_from = None
workflow = [('train', 1)]
