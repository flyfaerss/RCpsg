_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/psg.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/custom_runtime.py',
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

model = dict(
    type='SceneGraphPanopticFPN',
    semantic_head=dict(
        type='PanopticFPNHead',
        num_things_classes=80,
        num_stuff_classes=53,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(type='CrossEntropyLoss',
                      ignore_index=255,
                      loss_weight=0.5),
    ),
    panoptic_fusion_head=dict(type='HeuristicFusionHead',
                              num_things_classes=80,
                              num_stuff_classes=53),
    train_cfg=dict(rpn=dict(assigner=dict(type='MaxIoUAssigner',
                                          pos_iou_thr=0.7,
                                          neg_iou_thr=0.3,
                                          min_pos_iou=0.3,
                                          match_low_quality=True,
                                          ignore_iof_thr=-1),
                            sampler=dict(type='RandomSampler',
                                         num=256,
                                         pos_fraction=0.5,
                                         neg_pos_ub=-1,
                                         add_gt_as_proposals=False),
                            allowed_border=-1,
                            pos_weight=-1,
                            debug=False),
                   rpn_proposal=dict(nms_pre=2000,
                                     max_per_img=1000,
                                     nms=dict(type='nms', iou_threshold=0.7),
                                     min_bbox_size=0),
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
                    use_proposal_matching=False,
                   ),
    test_cfg=dict(
        use_full_train_samples=False,
        use_peseudo_inference=True,
        panoptic=dict(
            score_thr=0.6,
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_overlap=0.5,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            stuff_area_limit=4096,
    )),
    roi_head=dict(bbox_head=dict(type='SceneGraphBBoxHead'), ),
    relation_head=dict(
        type='RelTransformerHead',  # 'VCTreeHead_PLUS', RelTransformerHead, MotifHead
        object_classes=object_classes,
        predicate_classes=predicate_classes,
        num_classes=len(object_classes) + 1,  # with background class
        num_predicates=len(predicate_classes) + 1,
        use_bias=True,  # NOTE: whether to use frequency bias
        glove_dir='/home/sylvia/yjr/sgg/OpenPSG/data/glove',
        dataset_config=dict(cache='/home/sylvia/yjr/sgg/OpenPSG/data/psg/statistics.cache',
                            predicate_frequency='/home/sylvia/yjr/sgg/OpenPSG/data/psg/predicate_frequency.txt',
                            object_frequency='/home/sylvia/yjr/sgg/OpenPSG/data/psg/object_frequency.txt'),
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
            hidden_dim=256,
            roi_dim=256,
            context_pooling_dim=512,
            dropout_rate=0.2,
            context_object_layer=1,
            context_edge_layer=1,
            glove_dir='data/glove/',
            causal_effect_analysis=False,
            use_triplet_obj_loss=True,
            use_peseudo_labels=False,
            hard_index=[11, 19, 25, 29, 31, 32, 34, 35, 36, 42, 53, 54],
            transformer=dict(
                dropout_rate=0.1,
                obj_layer=4,
                rel_layer=4,
                num_head=8,
                key_dim=32,
                val_dim=32,
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
        loss_relation=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # loss_relation=dict(type='DynamicReweightCrossEntropy'),
        # loss_feature=dict(type='FeatureLoss'),
    ),
)

custom_hooks = [dict(type='SetEpochInfoHook')]# custom hooks

# To freeze modules
freeze_modules = [
    'backbone',
    'neck',
    'rpn_head',
    'roi_head',
    'semantic_head',
    'panoptic_fusion_head',
]

evaluation = dict(interval=1,
                  metric='sgdet',
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')

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
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 4),
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
    # dict(type='LoadPanopticSceneGraphAnnotations',
    #     with_bbox=True,
    #     with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # NOTE: Do not change the img to DC.
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            # dict(
            #     type='ToDataContainer',
            #     fields=(dict(key='gt_bboxes'), dict(key='gt_labels')),
            # ),
            dict(type='Collect', keys=['img']), # , 'gt_bboxes', 'gt_labels']),
            # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
    ),
]

# Change batch size and learning rate
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
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
# optimizer = dict(lr=0.003)
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 step=[9, 12])

# Log config
project_name = 'openpsg'
expt_name = 'fpn_panoptic_sgdet_psg'
work_dir = f'./work_dirs/{expt_name}'

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
        ),
    ],
)'''

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ],
)

load_from = 'work_dirs/checkpoints/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth'

