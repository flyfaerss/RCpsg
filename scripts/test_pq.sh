#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=1
export CUDA_VISIBLE_DEVICES="0"

python tools/test.py \
  --config configs/panformer/panformer_pvtb5_24e_coco_panoptic.py \
  --checkpoint work_dirs/checkpoints/panoptic_segformer_pvtv2b5_2x.pth \
  --out work_dirs/panoptic_fpn_r101_fpn/result.pkl \
  --eval PQ