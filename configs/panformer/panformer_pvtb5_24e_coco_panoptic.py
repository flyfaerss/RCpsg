
_base_ = './base.py'

model = dict(
    # get pvt_v2_b5_22k
    # wget https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5_22k.pth
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
    )
)

expt_name = 'panformer_pvtb5'

load_from = '/home/sylvia/yjr/sgg/PSG/PSG/OpenPSG/work_dirs/checkpoints/panoptic_segformer_pvtv2b5_2x.pth'
