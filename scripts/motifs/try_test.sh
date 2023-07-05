#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=1
export CUDA_VISIBLE_DEVICES="1"


# python tools/test.py \
# --config configs/panformer/relformer_panoptiv_pvtb5_24e_sgdet_psg.py \
# --checkpoint work_dirs/relformer_panoptic_pvtb5_sgdet_psg/20230421_121107/epoch_10.pth \
# --eval sgdet

python tools/test.py \
 --config configs/maskformer/mask2former_panoptic_sgdet_psg.py \
 --checkpoint work_dirs/maskformer_panoptic_psg_sgdet/20230421_230903/epoch_12.pth \
 --eval sgdet

# python tools/test.py \
# --config configs/psgtr/psgtr_pvtb5_psg.py \
# --checkpoint work_dirs/psgpanformer/20221008_154518/epoch_13.pth \
# --submit
# --eval sgdet

# python tools/test.py \
# --config configs/psgtr/psgtr_r50_psg.py \
# --checkpoint work_dirs/psgtr_epoch_60.pth \
# --submit
# --eval sgdet
