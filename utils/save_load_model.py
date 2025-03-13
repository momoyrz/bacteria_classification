import os
from pathlib import Path

import torch

from utils.distributed_util import get_rank


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:

        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, fold_output_dir):
    output_dir = Path(fold_output_dir)
    epoch_name = str(epoch)
    output_dir.mkdir(exist_ok=True)
    checkpoint_paths = [output_dir / 'checkpoint-{}.pth'.format(epoch_name)]

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'epoch': epoch,
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_scaler': loss_scaler.state_dict(),
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / 'checkpoint-{}.pth'.format(to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_main_process():
    return get_rank() == 0
