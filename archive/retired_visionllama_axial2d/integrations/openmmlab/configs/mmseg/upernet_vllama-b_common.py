_base_ = [
    'mmseg::_base_/datasets/ade20k.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py',
]

custom_imports = dict(imports=['integrations.openmmlab'], allow_failed_imports=False)

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='LSSODenseVisionLLaMA',
        scale='base',
        mixer='rrlsso',
        rank=32,
        checkpoint=None,
        image_size=224,
        patch_size=16,
        window_size=16,
        global_block_indices=(2, 5, 8, 11)),
    neck=dict(
        type='LSSOSimpleFPN',
        in_channels=768,
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=4, num_workers=8, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.0),
            'cls_token': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'decode_head': dict(lr_mult=10.0),
            'auxiliary_head': dict(lr_mult=10.0),
        }))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        save_best='mIoU',
        rule='greater',
        interval=8000,
        max_keep_ckpts=5))
