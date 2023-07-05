# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import numpy as np

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import replace_ImageToTensor
from openpsg.datasets import build_dataset

from openpsg.utils.utils import show_result



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('--config',
                        default='./configs/maskformer/mask2former_panoptic_sgdet_psg.py',
                        help='test config file path')
    parser.add_argument('--prediction_path',
                        default='./work_dirs/maskformer_panoptic_psg_sgdet/epoch_9.pkl',
                        help='prediction path where test pkl result')
    parser.add_argument('--show_dir',
                        default='./work_dirs/maskformer_panoptic_psg_sgdet/analyze_viz',
                        help='directory where painted images will be saved')
    parser.add_argument('--prediction_path_1',
                        default='./work_dirs/maskformer_panoptic_psg_sgdet/epoch_4.pkl',
                        help='prediction path where test pkl result')
    parser.add_argument('--show_dir_1',
                        default='./work_dirs/maskformer_panoptic_psg_sgdet/analyze_viz_1',
                        help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--img_idx',
                        default=[442],
                        nargs='+',
                        type=int,
                        help='which image to show')
    parser.add_argument('--wait-time',
                        type=float,
                        default=0,
                        help='the interval of show (s), 0 is block')
    parser.add_argument('--topk',
                        default=1,
                        type=int,
                        help='saved Number of the highest topk '
                        'and lowest topk after index sorting')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0,
                        help='score threshold (default: 0.)')
    parser.add_argument('--one_stage', default=False, action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # print(cfg.data.test)
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)
    outputs_1 = mmcv.load(args.prediction_path_1)

    inst_classes = dataset.CLASSES
    pred_classes = dataset.PREDICATES

    for idx in args.img_idx:
        print(idx, flush=True)
        img = dataset[idx]['img_metas'][0].data['filename']
        # img_id = dataset[idx]['img_metas'][0].data['image_id']
        inst_labels = [x['category_id'] for x in dataset.data[idx]['annotations']]
        inst_bbox = [x['bbox'] for x in dataset.data[idx]['annotations']]
        print(len(inst_labels), img)
        print(inst_bbox)
        print(inst_labels)
        for item in inst_labels:
            print(inst_classes[item], end=' ')
        print('\n')
        relations = dataset.data[idx]['relations']
        for rel in relations:
            print(inst_classes[inst_labels[rel[0]]] + '--' + pred_classes[rel[2] - 1] + '--' + inst_classes[inst_labels[rel[1]]])
        print('--------------------------------------------------------------------')
        '''result = outputs[idx]
        out_filepath = osp.join(args.show_dir, f'{idx}.png')
        show_result(img,
                    result,
                    is_one_stage=args.one_stage,
                    num_rel=args.topk,
                    out_dir=args.show_dir,
                    out_file=out_filepath)'''
        result_1 = outputs_1[idx]
        out_filepath_1 = osp.join(args.show_dir_1, f'{idx}.png')
        show_result(img,
                    result_1,
                    is_one_stage=args.one_stage,
                    num_rel=args.topk,
                    out_dir=args.show_dir_1,
                    out_file=out_filepath_1)


if __name__ == '__main__':
    main()
