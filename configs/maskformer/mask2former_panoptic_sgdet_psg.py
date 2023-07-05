_base_ = [
    '../_base_/datasets/psg.py',
    '../_base_/custom_runtime.py',
    '../_base_/schedules/schedule_1x.py',
]

find_unused_parameters = True
dataset_type = 'PanopticSceneGraphDataset'

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
    'beside', # 3
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off', # 9
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over', # 18
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at', # 24
    'guiding', # 25
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing', # 34
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to', # 43
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit', # 50
    'kicking', # 51
    'swinging',
    'entering', # 53
    'exiting', # 54
    'enclosing',
    'leaning on',
]

num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='Mask2FormerPanoptic',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler'),
        mask=dict(assigner=dict(type='MaskHungarianAssigner_PLUS',
                                cls_cost=dict(type='ClassificationCost', weight=1.0),
                                mask_cost=dict(type='FocalLossCost', weight=0),
                                dice_cost=dict(type='DiceCost', weight=1.0),
                                stuff_thing_flag=True)
                        ),
        rest_mask=dict(assigner=dict(type='MaskHungarianAssigner_PLUS',
                                cls_cost=dict(type='ClassificationCost', weight=1.0),
                                mask_cost=dict(type='FocalLossCost', weight=0),
                                dice_cost=dict(type='DiceCost', weight=2.0),
                                stuff_thing_flag=True)
                        ),
        use_full_train_samples=False,
        use_proposal_matching=True,
    ),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=False,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True,
        use_full_train_samples=False,
        use_peseudo_inference=True
    ),
    relation_head=dict(
        type='RelTransformerHead', # 'VCTreeHead_PLUS', RelTransformerHead
        object_classes=object_classes,
        predicate_classes=predicate_classes,
        num_classes=len(object_classes) + 1,  # with background class
        num_predicates=len(predicate_classes) + 1,
        use_bias=True,  # NOTE: whether to use frequency bias
        glove_dir='./data/glove',
        dataset_config=dict(cache='./data/psg/statistics.cache',
                            predicate_frequency='./data/psg/predicate_frequency.txt',
                            object_frequency='./data/psg/object_frequency.txt'),
        roi_neck=dict(
            type='ChannelMapper',
            in_channels=[256, 512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        pixel_extractor=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=0,
            in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
            feat_channels=256,
            out_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=2,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        head_config=dict(
            # NOTE: Evaluation type
            feature_extract_method='query', # 'roi'
            init_denoising=False,
            use_query_mode=True,
            num_thing_class=80,
            num_stuff_class=53,
            use_gt_box=False,
            use_gt_label=False,
            use_vision=True,
            classifier_apart=False,
            embed_dim=200,
            hidden_dim=256, # 256 512
            roi_dim=256, # 256 1024
            context_pooling_dim=512, # 512 4096
            dropout_rate=0.2,
            context_object_layer=1,
            context_edge_layer=1,
            glove_dir='./data/glove/',
            causal_effect_analysis=False,
            use_triplet_obj_loss=True,
            use_peseudo_labels=False,
            hard_index=[11, 19, 25, 29, 31, 32, 34, 35, 36, 42, 53, 54],
            transformer=dict(
                dropout_rate=0.1,
                obj_layer=4,
                rel_layer=4,
                num_head=8,
                key_dim=64,
                val_dim=64,
                inner_dim=1024,
            ),
            bias_module=dict(
                use_penalty=False,
                thing_stuff_apart=False,
                epsilon=0.001,
                bg_default_value=0.02,
                cb_cls_fusion_type='sum',
                cb_cls_fusion_weight=0.8,
                fusion_weight=1.0,
                cls_trans=None,
                eval_with_penalty=True,
                penalty_epsilon=0.001,
                penalty_fusion_weights=[1.0, 1.0],
                penalty_k=10,
                penalty_threshold=0.1,
                penalty_type='count_bias',
                penalty_weight=0.5,
                possible_bias_default_value=1.0,
                possible_bias_threshold=100.0,
                scale_weight=1.0,
                use_neg_penalty=False,
                use_curriculum=True,
            )
        ),
        bbox_roi_extractor=dict(
            type='VisualSpatialExtractor',
            bbox_roi_layer=dict(type='RoIAlign', # RoIAlign
                                output_size=7,
                                sampling_ratio=2),
            with_visual_bbox=True, # True
            with_visual_mask=False, # False
            with_visual_point=False,
            with_spatial=False,
            in_channels=256,
            fc_out_channels=1024,
            featmap_strides=[4, 8, 16, 32],
        ),
        relation_roi_extractor=dict(
            type='VisualSpatialExtractor',
            bbox_roi_layer=dict(type='RoIAlign', # RoIAlign
                                output_size=7,
                                sampling_ratio=2),
            with_visual_bbox=True, # True
            with_visual_mask=False, # False
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
        loss_object=dict(type='CrossEntropyLoss',use_sigmoid=False,
                         loss_weight=1.0),
        # loss_object=dict(type='DynamicReweightCrossEntropy'),
        loss_relation=dict(type='CrossEntropyLoss',use_sigmoid=False,
                           loss_weight=1.0,),
                           # class_weight=[0.02] + [1.0] * len(predicate_classes)),
        # loss_relation=dict(type='RelLabelSmoothingLoss',
        #                   classes=57, smoothing=0.1, use_peseudo_labels=False),
        # loss_relation=dict(type='DynamicReweightCrossEntropy'),
        # loss_feature=dict(type='FeatureLoss'),
        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True,
                       reduction='mean', loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', use_sigmoid=True, activate=True,
                       reduction='mean', naive_dice=True, eps=1.0, loss_weight=1.0)
    ),
    init_cfg=None)

freeze_modules = [
    'backbone',
    'panoptic_head',
    'panoptic_fusion_head'
]

evaluation = dict(interval=1,
                  metric='sgdet',
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg'
                  )

# dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
'''train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticSceneGraphAnnotations',
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]'''

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
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
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
        # samples_per_gpu=12,
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True))

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
'''optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))'''
# optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
'''lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)'''

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 step=[9, 12])

project_name = 'openpsg'
expt_name = 'maskformer_panoptic_psg_sgdet'
work_dir = f'./work_dirs/{expt_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ],
)

runner = dict(type='EpochBasedRunner', max_epochs=15)
# custom_hooks = [dict(type='CacheCleaner',priority='HIGHEST')]
custom_hooks = [dict(type='SetEpochInfoHook')]

load_from = './work_dirs/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'
# resume_from = './work_dirs/maskformer_panoptic_psg_sgdet/20221226_232008/epoch_1.pth'
