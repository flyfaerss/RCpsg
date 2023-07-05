_base_ = [
    '../_base_/datasets/psg.py',
    '../_base_/custom_runtime.py'
]

custom_imports = dict(imports=[
    'openpsg.models.frameworks.psgtr', 'openpsg.models.losses.seg_losses',
    'openpsg.models.relation_heads.psgtr_head', 'openpsg.datasets',
    'openpsg.datasets.pipelines.loading',
    'openpsg.datasets.pipelines.rel_randomcrop',
    'openpsg.models.relation_heads.approaches.matcher', 'openpsg.utils'
],
                      allow_failed_imports=False)

dataset_type = 'PanopticSceneGraphDataset'

# HACK:
object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged'
]

predicate_classes = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]

'''pretrained='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/checkpoints/pvt_v2_b5_22k.pth',
    backbone=dict(type='pvt_v2_b5',
                  depth=50,
                  num_stages=4,
                  out_indices=(1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=False),
                  norm_eval=True,
                  style='pytorch',),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 320, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),'''
# pretrained='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/checkpoints/pvt_v2_b5_22k.pth',
#
model = dict(
    type='PSGTr',
    backbone=dict(type='pvt_v2_b5',
                  depth=50,
                  num_stages=4,
                  out_indices=(1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=False),
                  norm_eval=True,
                  style='pytorch',
                  pretrained='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/checkpoints/pvt_v2_b5_22k.pth'),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 320, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(type='PSGTrHead',
                    num_classes=133,
                    num_relations=56,
                    object_classes=object_classes,
                    predicate_classes=predicate_classes,
                    use_mask=True,
                    num_query=100,
                   in_channels=2048,
               transformer=dict(
                   type='RelDeformable_Transformer',
                   encoder=dict(
                       type='DetrTransformerEncoder',
                       num_layers=4,
                       transformerlayers=dict(
                           type='BaseTransformerLayer',
                           attn_cfgs=dict(
                               type='MultiScaleDeformableAttention',
                               embed_dims=256,
                               num_levels=4,
                           ),
                           feedforward_channels=1024,
                           ffn_dropout=0.1,
                           operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                   decoder=dict(
                       type='RelDeformableDetrTransformerDecoder',
                       num_layers=4,
                       return_intermediate=True,
                       transformerlayers=dict(
                           type='DetrTransformerDecoderLayer',
                           attn_cfgs=[
                               dict(
                                   type='MultiheadAttention',
                                   embed_dims=256,
                                   num_heads=8,
                                   dropout=0.1),
                               dict(
                                   type='MultiScaleDeformableAttention',
                                   embed_dims=256,
                                   num_levels=4,
                               )
                           ],
                           feedforward_channels=1024,
                           ffn_dropout=0.1,
                           operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                            'ffn', 'norm')))),
                   positional_encoding=dict(type='SinePositionalEncoding',
                                            num_feats=128,
                                            normalize=True),
                   sub_loss_cls=dict(type='CrossEntropyLoss',
                                     use_sigmoid=False,
                                     loss_weight=1.0,
                                     class_weight=1.0),
                   sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                   sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                   sub_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                   sub_dice_loss=dict(type='psgtrDiceLoss', loss_weight=1.0),
                   obj_loss_cls=dict(type='CrossEntropyLoss',
                                     use_sigmoid=False,
                                     loss_weight=1.0,
                                     class_weight=1.0),
                   obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                   obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                   obj_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                   obj_dice_loss=dict(type='psgtrDiceLoss', loss_weight=1.0),
                   rel_loss_cls=dict(type='CrossEntropyLoss',
                                     use_sigmoid=False,
                                     loss_weight=2.0,
                                     class_weight=1.0),
               dataset_config=dict(cache='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/data/psg/statistics.cache',
                                   predicate_frequency='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/data/psg/predicate_frequency.txt',
                                   object_frequency='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/data/psg/object_frequency.txt'),
               sub_transformer_head=dict(type='MaskHead', d_model=256, nhead=8, num_decoder_layers=1),
               obj_transformer_head=dict(type='MaskHead', d_model=256, nhead=8, num_decoder_layers=1)),
    train_cfg=dict(assigner=dict(
        type='HTriMatcher',
        s_cls_cost=dict(type='ClassificationCost', weight=1.),
        s_reg_cost=dict(type='BBoxL1Cost', weight=5.0),
        s_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
        o_cls_cost=dict(type='ClassificationCost', weight=1.),
        o_reg_cost=dict(type='BBoxL1Cost', weight=5.0),
        o_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
        r_cls_cost=dict(type='ClassificationCost', weight=2.))),
    test_cfg=dict(max_per_img=100))

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanopticSceneGraphAnnotations',
         with_bbox=True,
         with_rel=True,
         with_mask=True,
         with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(type='Resize',
                     img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                (576, 1333), (608, 1333), (640, 1333),
                                (672, 1333), (704, 1333), (736, 1333),
                                (768, 1333), (800, 1333)],
                     multiscale_mode='value',
                     keep_ratio=True)
            ],
            [
                dict(type='Resize',
                     img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                     multiscale_mode='value',
                     keep_ratio=True),
                dict(type='RelRandomCrop',
                     crop_type='absolute_range',
                     crop_size=(384, 600),
                     allow_negative_crop=False),  # no empty relations
                dict(type='Resize',
                     img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                (576, 1333), (608, 1333), (640, 1333),
                                (672, 1333), (704, 1333), (736, 1333),
                                (768, 1333), (800, 1333)],
                     multiscale_mode='value',
                     override=True,
                     keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='RelsFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels', 'gt_masks'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            # dict(type='ToDataContainer', fields=(dict(key='gt_bboxes'), dict(key='gt_labels'))),
            dict(type='Collect', keys=['img']),
        ])
]

evaluation = dict(
    interval=1,
    metric='sgdet',
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method='pan_seg', # bbox
)

data = dict(samples_per_gpu=1,
            workers_per_gpu=0,
            train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00014,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'sampling_offsets': dict(lr_mult=0.1),
        'reference_points': dict(lr_mult=0.1)
    }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=18)
runner = dict(type='EpochBasedRunner', max_epochs=60)
custom_hooks = [dict(type='SetEpochInfoHook')]#[dict(type='GradChecker',priority='HIGHEST')]


project_name = 'psgpanformer'
expt_name = 'psgpanformer'
work_dir = f'./work_dirs/{expt_name}'
checkpoint_config = dict(interval=1, max_keep_ckpts=60)

'''log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=project_name,
                name=expt_name,
            ),
        )
    ],
)'''
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

# load_from = 'work_dirs/checkpoints/detr_pan_r101.pth'
# load_from = '/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/checkpoints/pvt_v2_b5_22k.pth'
resume_from = '/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/psgpanformer/20221008_154518/epoch_13.pth'
'''    neck=dict(
            type='ChannelMapper',
            in_channels=[128, 320, 512],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),'''
