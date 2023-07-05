_base_ = [
    '../motifs/panoptic_fpn_r50_fpn_1x_predcls_psg.py',
]

model = dict(
    relation_head=dict(
        type='AMPHead',
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=False,
            use_gt_label=False,
            use_obj_refine=True,
            hidden_dim=1024,

        ),
    ),
    roi_head=dict(bbox_head=dict(type='SceneGraphBBoxHead'), ),
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'))
)

evaluation = dict(interval=1,
                  metric='sgdet',
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')

# Change batch size and learning rate
data = dict(samples_per_gpu=16,
            workers_per_gpu=0)  # FIXME: Is this the problem?
# optimizer = dict(lr=0.001)

# Log config
project_name = 'openpsg'
expt_name = 'vctree_panoptic_fpn_r50_fpn_1x_predcls_psg'
work_dir = f'./work_dirs/{expt_name}'

# Log config
project_name = 'openpsg'
expt_name = 'amp_panoptic_fpn_r101_fpn_1x_predcls_psg'
work_dir = f'./work_dirs/{expt_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ],
)

load_from = 'work_dirs/checkpoints/panoptic_fpn_r101_fpn_1x_coco_20210820_193950-ab9157a2.pth'

