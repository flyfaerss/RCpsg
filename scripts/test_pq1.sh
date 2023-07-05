#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=1
export CUDA_VISIBLE_DEVICES="0"

python tools/test.py \
  --config configs/_base_/models/panoptic_fpn_r101_fpn_psg.py \
  --checkpoint work_dirs/checkpoints/panoptic_fpn_r101_fpn_1x_coco_20210820_193950-ab9157a2.pth \
  --out work_dirs/panoptic_fpn_r101_fpn/result.pkl \
  --eval PQ