_base_ = [
    './base.py',
    # '../_base_/datasets/psg.py',
    # '../_base_/custom_runtime.py',
]

find_unused_parameters = True
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

_dim_ = 256
_dim_half_ = _dim_//2
_feed_ratio_ = 4
_feed_dim_ = _feed_ratio_*_dim_
_num_levels_ = 4
model = dict(
    type='PanFormerPanoptic',
    pretrained='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/checkpoints/pvt_v2_b5_22k.pth',
    backbone=dict(
        type='pvt_v2_b5',
        out_indices=(1, 2, 3),
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 320, 512],
    ),
    bbox_head=dict(
        quality_threshold_things=0.3,
        quality_threshold_stuff=0.3,
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ),
        assigner_with_mask=dict(
            type='HungarianAssigner_multi_info',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            mask_cost=dict(type='DiceCost', weight=2.0),
            ),
        rcnn=dict(assigner=dict(type='MaxIoUAssigner',
                                pos_iou_thr=0.5,
                                neg_iou_thr=0.5,
                                min_pos_iou=0.5,
                                match_low_quality=True,
                                ignore_iof_thr=-1),
                  sampler=dict(type='RandomSampler',
                               num=512,
                               pos_fraction=0.25,
                               neg_pos_ub=-1,
                               add_gt_as_proposals=True),
                  mask_size=28,
                  pos_weight=-1,
                  debug=False),
        sampler =dict(type='PseudoSampler'),
        sampler_with_mask =dict(type='PseudoSampler_segformer'),
        ),
    test_cfg=dict(max_per_img=100),
    relation_head=dict(
        type='VCTreeHead',
        object_classes=object_classes,
        predicate_classes=predicate_classes,
        num_classes=len(object_classes) + 1,  # with background class
        num_predicates=len(predicate_classes) + 1,
        use_bias=True,  # NOTE: whether to use frequency bias
        glove_dir='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/data/glove',
        dataset_config=dict(cache='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/data/psg/statistics.cache',
                            predicate_frequency='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/data/psg/predicate_frequency.txt',
                            object_frequency='/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/data/psg/object_frequency.txt'),
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
            use_vision=True,
            embed_dim=200,
            hidden_dim=512,
            roi_dim=1024,
            context_pooling_dim=4096,
            dropout_rate=0.2,
            context_object_layer=1,
            context_edge_layer=1,
            glove_dir='data/glove/',
            causal_effect_analysis=False,
        ),
        bbox_roi_extractor=dict(
            type='VisualSpatialExtractor',
            bbox_roi_layer=dict(type='RoIAlign',
                                output_size=7,
                                sampling_ratio=2),
            with_visual_bbox=True,
            with_visual_mask=False,
            with_visual_point=False,
            with_spatial=False,
            in_channels=256,
            fc_out_channels=1024,
            featmap_strides=[4, 8, 16, 32],
        ),
        relation_roi_extractor=dict(
            type='VisualSpatialExtractor',
            bbox_roi_layer=dict(type='RoIAlign',
                                output_size=7,
                                sampling_ratio=2),
            with_visual_bbox=True,
            with_visual_mask=False,
            with_visual_point=False,
            with_spatial=True,
            separate_spatial=False,
            in_channels=256,
            fc_out_channels=1024,
            featmap_strides=[4, 8, 16, 32],
        ),
        relation_sampler=dict(
            type='Motif',
            pos_iou_thr=0.5,
            require_overlap=False,  # for sgdet training, not require
            num_sample_per_gt_rel=4,
            num_rel_per_image=1024,
            pos_fraction=0.25,
            # NOTE: To only include overlapping bboxes?
            test_overlap=False,  # for testing
        ),
        loss_object=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.0),
        # loss_object=dict(type='DynamicReweightCrossEntropy'),
        loss_relation=dict(type='DynamicReweightCrossEntropy'),
        loss_feature=dict(type='FeatureLoss'),
    ),
    # roi_head=dict(bbox_head=dict(type='SceneGraphBBoxHead'), ),
)

custom_hooks = [dict(type='SetEpochInfoHook')]# custom hooks

# To freeze modules
freeze_modules = [
    'backbone',
    'neck',
    'bbox_head',
]

evaluation = dict(interval=1,
                  metric='sgdet',
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg'
                  )

# Change batch size and learning rate
# data = dict(samples_per_gpu=16, workers_per_gpu=0)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticSceneGraphAnnotations',
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True,
    ),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    # dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='SceneGraphFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_bboxes',
            'gt_labels',
            'gt_rels',
            'gt_relmaps',
            'gt_masks',
            'gt_semantic_seg',
        ],
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # Since the forward process may need gt info, annos must be loaded.
    dict(type='LoadPanopticSceneGraphAnnotations',
         with_bbox=True,
         with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            # NOTE: Do not change the img to DC.
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            dict(
                type='ToDataContainer',
                fields=(dict(key='gt_bboxes'), dict(key='gt_labels')),
            ),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
    ),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    train=dict(
        type='PanopticSceneGraphDataset',
        pipeline=train_pipeline,
        split='train',
        all_bboxes=True,
        resample=None),
    val=dict(
        type='PanopticSceneGraphDataset',
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True),
    test=dict(
        type='PanopticSceneGraphDataset',
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True))
# optimizer = dict(lr=0.003)
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 step=[7, 10])

# Log config
project_name = 'openpsg'
expt_name = 'panformer_panoptic_pvtb5_sgdet_psg'
work_dir = f'./work_dirs/{expt_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ],
)

load_from = '/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/checkpoints/panoptic_segformer_pvtv2b5_2x.pth'
