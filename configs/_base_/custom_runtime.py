checkpoint_config = dict(interval=1, max_keep_ckpts=50)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='SetEpochInfoHook')]# custom hooks

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

# workflow = [('train', 1), ('val', 1)]

workflow = [('train', 1)]
