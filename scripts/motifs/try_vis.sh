#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=1
export CUDA_VISIBLE_DEVICES="2"


# python tools/test.py \
# --config configs/panformer/relformer_panoptiv_pvtb5_24e_sgdet_psg.py \
# --checkpoint work_dirs/relformer_panoptic_pvtb5_sgdet_psg/20221007_192725/epoch_4.pth \
# --submit

# python tools/test.py \
#   --config configs/maskformer/mask2former_panoptic_sgdet_psg.py \
#   --checkpoint work_dirs/maskformer_panoptic_psg_sgdet/20221229_094528/epoch_9.pth \
#   --out work_dirs/maskformer_panoptic_psg_sgdet/epoch_9.pkl

python tools/test.py \
  --config configs/maskformer/mask2former_panoptic_sgdet_psg.py \
  --checkpoint work_dirs/maskformer_panoptic_psg_sgdet/20221223_005321/epoch_4.pth \
  --out work_dirs/maskformer_panoptic_psg_sgdet/epoch_4.pkl