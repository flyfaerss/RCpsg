# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from openpsg.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # "./configs/motifs/panoptic_fpn_r50_fpn_1x_sgdet_psg.py"
    # parser.add_argument('--config', default="./configs/diffusion/mask2former_diffusion_sgdet_temp.py", help='train config file path')
    parser.add_argument('--config', default="./configs/maskformer/mask2former_panoptic_sgdet_psg.py", help='train config file path')
    # parser.add_argument('--config', default="./configs/panformer/relformer_panoptiv_pvtb5_24e_sgdet_psg.py", help='train config file path')  # default="/home/sylvia/yjr/sgg/PSG/OpenPSG/configs/motifs/panoptic_fpn_r50_fpn_1x_sgdet_psg.py"
    # parser.add_argument('--config', default="./configs/panformer/fpn_panoptic_sgdet_psg.py", help='train config file path')
    # parser.add_argument('--config',
    #                 default="./configs/psgtr/psgtr_r101_psg.py",
    #                   help='train config file path')  # default="./configs/motifs/panoptic_fpn_r50_fpn_1x_sgdet_psg.py"
    # parser.add_argument('--config',
       #            default="./configs/panformer/onestage_panformer_pvtb5_panoptic_sgdet_psg.py",
          #              help='train config file path')
    # parser.add_argument('--config', default="./configs/panformer/panformer_panoptic_pvtb5_24e_sgdet_psg.py", help='train config file path') # default="./configs/motifs/panoptic_fpn_r50_fpn_1x_sgdet_psg.py"
    # parser.add_argument('--config',
    #                     default="./configs/vctree/panoptic_fpn_r101_fpn_1x_sgdet_psg.py",
    #                     help='train config file path')  # default="./configs/motifs/panoptic_fpn_r50_fpn_1x_sgdet_psg.py"

    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training',
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)',
    )
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)',
    )
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.',
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        a = str(timestamp)
        cfg.work_dir = osp.join(args.work_dir, str(timestamp))
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        a = str(timestamp)
        cfg.work_dir = osp.join(osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0]), str(timestamp))
    cfg.work_dir = osp.join(cfg.work_dir, timestamp)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    def is_mlu_available():
        """Returns a bool indicating if MLU is currently available."""
        return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()

    def get_device():
        """Returns an available device, cpu, cuda or mlu."""
        is_device_available = {
            'cuda': torch.cuda.is_available(),
            'mlu': is_mlu_available()
        }
        device_list = [k for k, v in is_device_available.items() if v]
        return device_list[0] if len(device_list) == 1 else 'cpu'

    cfg.device = get_device()

    # create work_dir
    print(osp.abspath(cfg.work_dir))
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    dataset = build_dataset(cfg.data.train)

    if hasattr(cfg, 'dataset_config'):
        cache_dir = cfg.dataset_config['cache']
        predicate_dir = cfg.dataset_config['predicate_frequency']
        object_dir = cfg.dataset_config['object_frequency']
        print('Loading Statistics...')
        if cache_dir is None:
            raise FileNotFoundError(
                'The cache_dir for caching the statistics is not provided.')
        if (not os.path.exists(cache_dir)) or (not os.path.exists(predicate_dir)) or (not os.path.exists(object_dir)):
            result, predicate_frequency, object_frequency = dataset.get_statistics()
            statistics = {
                'freq_matrix': result['freq_matrix'],
                'pred_dist': result['pred_dist'],
            }
            if not os.path.exists(cache_dir):
                torch.save(statistics, cache_dir)
            if not os.path.exists(predicate_dir):
                with open(predicate_dir, 'w') as f:
                    f.write(str(predicate_frequency))
                f.close()
            if not os.path.exists(object_dir):
                with open(object_dir, 'w') as f1:
                    f1.write(str(object_frequency))
                f1.close()
        print('\n Statistics created!')

    # dataset.apply_resampling()

    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # NOTE: Freeze weights here
    if hasattr(cfg, 'freeze_modules'):
        if cfg.freeze_modules is not None:
            for module_name in cfg.freeze_modules:
                for name, p in model.named_parameters():
                    if name.startswith(module_name):
                        p.requires_grad = False
    # Unfreeze weights here
    if hasattr(cfg, 'required_grad_modules'):
        if cfg.required_grad_modules is not None:
            for module_name in cfg.required_grad_modules:
                for name, p in model.named_parameters():
                    if name.startswith(module_name):
                        p.requires_grad = True

    datasets = [dataset]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset)),
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ +
                                          get_git_hash()[:7],
                                          CLASSES=datasets[0].CLASSES)

    cfg.find_unused_parameters = False

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=False, # (not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == '__main__':
    main()
