#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=2
export CUDA_VISIBLE_DEVICES="1, 2"


python tools/train.py --config "configs/panformer/panformer_panoptic_pvtb5_24e_sgdet_psg.py"
python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
          tools/train.py \
          --gpus ${gpu_num} \
          --config "configs/psgtr/psgtr_pvtb5_psg.py" \
          --launch pytorch
# python tools/train.py --config "configs/vctree/panoptic_fpn_r101_fpn_1x_sgdet_psg.py"
# python tools/train.py --config "configs/psgtr/psgtr_pvtb5_psg.py"