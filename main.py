import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import timm
import torch
from torch.backends import cudnn
from torchvision import transforms

from timm import create_model, model_entrypoint
from my_datasets.my_datasets import BacteriaDataset
from utils.distributed_util import init_distributed_mode, get_rank, get_world_size
from utils.logger import create_logger
from utils.loss import FocalLoss
from utils.lr_factory import cosine_scheduler
from utils.optim_factory import create_optimizer
from utils.save_load_model import load_state_dict, save_model
from utils.amp_util import NativeScalerWithGradNormCount as NativeScaler
from utils.train_and_eval import train_one_epoch, evaluate
from my_datasets.split_data import split_dataset
from utils.config import model_paths_dict

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('Set arguments for training and evaluation', add_help=False)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpu_id', default='2', type=str)

    parser.add_argument('--model',
                        default='resnet34',
                        type=str)
    parser.add_argument('--model_key',
                        default='model|module',
                        type=str)

    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', default=1e-8, type=float)
    parser.add_argument('--opt_betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup_epochs', default=4, type=int)
    parser.add_argument('--start_warmup_value', default=0, type=float)
    parser.add_argument('--epochs', default=40, type=int)

    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--nb_classes', default=33, type=int)
    parser.add_argument('--data_dir', default='/home/ubuntu/qujunlong/data/bacteria', type=str)
    parser.add_argument('--jsonl_path', default='/home/ubuntu/qujunlong/bacteria/bacteria_classification/my_datasets/bacteria_dataset.jsonl', type=str)

    parser.add_argument('--loss', default='cross', type=str)
    parser.add_argument('--gamma', default=2.0, type=float)
    parser.add_argument('--alpha', default=0.25, type=float)

    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=2, type=int)
    parser.add_argument('--save_ckpt_num', default=5, type=int)

    parser.add_argument('--eval', type=str2bool, default=False)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--use_amp', type=str2bool, default=True)
    parser.add_argument('--s', default=1, type=int)

    return parser


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    image_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data, val_data, test_data = split_dataset(args.data_dir, args.jsonl_path, train_ratio=0.7, val_ratio=0.1)

    num_tasks = get_world_size()
    global_rank = get_rank()

    train_dataset = BacteriaDataset(train_data, transform=image_transform)
    val_dataset = BacteriaDataset(val_data, transform=test_transform)
    test_dataset = BacteriaDataset(test_data, transform=test_transform)

    sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_tasks, global_rank,
                                                        shuffle=True, seed=args.seed)
    sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    if global_rank == 0:  # 如果是主进程，创造日志文件
        # 暂停args.s秒
        args.prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_dir = os.path.join(args.output_dir, 'other', args.model, args.prefix)
        args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoint', args.model, args.prefix)
        # 如果'train_output/other/' + args.model + '_' + args.prefix这个文件夹不存在，则创建
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        logger = create_logger(args.log_dir, '')
        # 将args以json格式保存到args.log_dir下的args.json文件中
        with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_test,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    create_fn = model_entrypoint(args.model)
    model = create_fn(num_classes=args.nb_classes)
    if args.finetune:
        checkpoint = torch.load(model_paths_dict[args.model], map_location='cpu')
        logger.info(f"Loading model from {model_paths_dict[args.model]}")
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                break

        if checkpoint_model is None:
            checkpoint_model = checkpoint

        state_dict = model.state_dict()
        for k in ['fc.weight', 'fc.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        load_state_dict(model, checkpoint_model, prefix='')
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model = %s' % str(model_without_ddp))
    logger.info(f"Number of params: {n_parameters}")

    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    logger.info(f"LR = {args.lr}")
    logger.info(f"Total batch size = {total_batch_size}")
    logger.info(f"Total training examples = {len(train_dataset)}")
    logger.info(f"Total training steps = {num_training_steps_per_epoch * args.epochs}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()  # if args.use_amp is False, this won't be used

    logger.info('Use Cosine LR scheduler')
    lr_scheduler_values = cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
                                           args.warmup_epochs, args.start_warmup_value)

    if args.loss == 'focal':
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.eval:
        logger.info('Start evaluating...')
        # evaluate(model, criterion, device, data_loader_val, 0, logger)
        return

    max_accuracy = 0.0
    test_accuracy = 0.0
    logger.info('Start training...')

    # 在args.log_dir下创建csv文件
    train_csv_file = os.path.join(args.log_dir, 'train_stats.csv')
    val_csv_file = os.path.join(args.log_dir, 'val_stats.csv')
    test_csv_file = os.path.join(args.log_dir, 'test_stats.csv')
    # 将列名写入csv文件
    pd.DataFrame(columns=['epoch', 'train_loss', 'acc', 'pre', 'sen', 'f1', 'spec', 'kappa', 'auc', 'qwk']).to_csv(
        train_csv_file, index=False)
    pd.DataFrame(columns=['epoch', 'val_loss', 'acc', 'pre', 'sen', 'f1', 'spec', 'kappa', 'auc', 'qwk']).to_csv(
        val_csv_file, index=False)
    pd.DataFrame(columns=['epoch', 'test_loss', 'acc', 'pre', 'sen', 'f1', 'spec', 'kappa', 'auc', 'qwk']).to_csv(
        test_csv_file, index=False)

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                      loss_scaler=loss_scaler, start_epoch=epoch, logger=logger,
                                      lr_scheduler=lr_scheduler_values,
                                      num_training_steps_per_epoch=num_training_steps_per_epoch,
                                      use_amp=args.use_amp)
        with open(train_csv_file, 'a+') as f:
            row = [epoch] + list(train_stats.values())
            f.write(','.join(map(str, row)) + '\n')
        if args.checkpoint_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch == args.epochs - 1:
                save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler)

        val_state = evaluate(model, criterion, device, data_loader_val, epoch, use_amp=args.use_amp, logger=logger,
                             header='Val')
        with open(val_csv_file, 'a+') as f:
            row = [epoch] + list(val_state.values())
            f.write(','.join(map(str, row)) + '\n')

        test_state = evaluate(model, criterion, device, data_loader_test, epoch, use_amp=args.use_amp, logger=logger,
                              header='Test')
        with open(test_csv_file, 'a+') as f:
            row = [epoch] + list(test_state.values())
            f.write(','.join(map(str, row)) + '\n')

        logger.info(f"Max accuracy in Val_set: {max_accuracy}")

        if val_state['acc'] > max_accuracy:
            max_accuracy = val_state['acc']
            test_accuracy = test_state['acc']
            save_model(args=args, epoch='best', model=model, model_without_ddp=model_without_ddp,
                       optimizer=optimizer, loss_scaler=loss_scaler)
        logger.info(f"Max accuracy in Val_set: {max_accuracy}, accuracy in Test_set: {test_accuracy}")

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('All_models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # 从json文件加载配置
    def parse_config(args, config_file):
        # 从json文件加载配置
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # 更新args
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            elif getattr(args, key) != value:
                print(f"Warning: {key} in config file ({value}) differs from command line argument ({getattr(args, key)})")
        return args

    args = parse_config(args, 'config.json')
    main(args)  