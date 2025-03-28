import argparse
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import timm
import torch
from torch.backends import cudnn
from torchvision import transforms

from timm import create_model
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
from visu.results_analysis import plot_curve, process_results, test_and_visualize


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
    parser.add_argument('--output_dir', default='../output', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpu_id', default='2', type=str)

    parser.add_argument('--model',
                        default='resnet34',
                        type=str)
    parser.add_argument('--pretrained', type=str2bool, default=True)

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
    parser.add_argument('--category_to_idx_path', default='/home/ubuntu/qujunlong/bacteria/bacteria_classification/my_datasets/category_to_idx.json', type=str)

    parser.add_argument('--loss', default='cross', type=str)
    parser.add_argument('--gamma', default=2.0, type=float)
    parser.add_argument('--alpha', default=0.25, type=float)

    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=2, type=int)
    parser.add_argument('--save_ckpt_num', default=5, type=int)

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

    parser.add_argument('--fold', default=5, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--fold_index', default=-1, type=int, help='Current fold index to use for training (0 to fold-1)')

    return parser


def main(args):
    os.environ['TORCH_HOME'] = '/home/ubuntu/qujunlong/timm_checkpoint'
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

    # Get all cross-validation folds
    folds_data = split_dataset(
        args.data_dir,
        args.jsonl_path,
        fold=args.fold,
        val_size=0.1,
        random_state=args.seed
    )

    # Set timestamp prefix for all folds
    args.output_dir = os.path.join(args.output_dir, args.model, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = create_logger(args.output_dir, 'main')
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Train on the specified fold or all folds
    if args.fold_index >= 0:
        # Train on a specific fold
        train_one_fold(args, folds_data[args.fold_index], image_transform, test_transform, device)
    else:
        # Train on all folds
        all_results = []
        for fold_idx in range(args.fold):
            logger.info(f"\n=== Training on Fold {fold_idx} ===\n")
            args.fold_index = fold_idx
            fold_results = train_one_fold(args, folds_data[fold_idx], image_transform, test_transform, device)
            all_results.append(fold_results)

        # Calculate and log average results across all folds
        if get_rank() == 0:
            # 计算每一个 epoch 的平均 acc
            epoch_acc = {i: 0 for i in range(args.epochs)}
            for fold_results in all_results:
                for epoch, acc in enumerate(fold_results['val_acc']):
                    epoch_acc[epoch] += acc
            for epoch in epoch_acc:
                epoch_acc[epoch] /= len(all_results)

            # 打印结果
            logger.info("\n=== Cross-Validation Results ===")
            for epoch, acc in epoch_acc.items():
                logger.info(f"Epoch {epoch}: {acc:.4f}")

            # 使用平均 acc 最高的 epoch 的模型，进行测试
            best_epoch = max(epoch_acc, key=epoch_acc.get)
            logger.info(f"\n=== Using Epoch {best_epoch} Model ===")

            # 拿到所有 fold 的 best_epoch 的模型路径
            best_model_paths = []
            for results in all_results:
                best_model_paths.append(os.path.join(results['checkpoint_dir'], 'checkpoint-{}.pth'.format(best_epoch)))

            # test
            category_to_idx = json.load(open(args.category_to_idx_path))
            all_acc = []
            all_pre = []
            all_sen = []
            all_f1 = []
            all_spec = []
            all_kappa = []
            all_my_auc = []
            all_qwk = []
            for fold_idx, model_path in enumerate(best_model_paths):
                model = create_model(args.model, num_classes=args.nb_classes)
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
                model.eval()
                test_dataset = BacteriaDataset(folds_data[fold_idx]['test'], transform=test_transform, category_to_idx=category_to_idx)
                sampler_test = torch.utils.data.SequentialSampler(test_dataset)
                data_loader_test = torch.utils.data.DataLoader(
                    test_dataset, sampler=sampler_test,
                    batch_size=int(2 * args.batch_size),
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                    drop_last=False
                )
                fold_dir = os.path.join(args.output_dir, f'fold_{fold_idx}')
                fold_logger = create_logger(fold_dir, f'fold_{fold_idx}')
                acc, pre, sen, f1, spec, kappa, my_auc, qwk = test_and_visualize(args, model, device, data_loader_test, logger=fold_logger, fold_dir=fold_dir)
                all_acc.append(acc)
                all_pre.append(pre)
                all_sen.append(sen)
                all_f1.append(f1)
                all_spec.append(spec)
                all_kappa.append(kappa)
                all_my_auc.append(my_auc)
                all_qwk.append(qwk)
                logger.info(f"Fold {fold_idx} acc: {acc} pre: {pre} sen: {sen} f1: {f1} spec: {spec} kappa: {kappa} my_auc: {my_auc} qwk: {qwk}")           
            process_results(all_acc, all_pre, all_sen, all_f1, all_spec, all_kappa, all_my_auc, all_qwk, logger, args)
            delete_other_models(args.output_dir, best_epoch)
            logger.info("All unnecessary models have been deleted.")

            # 输出模型参数量
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of params: {n_parameters}")
            

def delete_other_models(output_dir, best_epoch):
    for subdir in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, subdir)):
            for file in os.listdir(os.path.join(output_dir, subdir)):
                if file.endswith('.pth') and file != f'checkpoint-{best_epoch}.pth':
                    os.remove(os.path.join(output_dir, subdir, file))

def train_one_fold(args, fold_data, image_transform, test_transform, device):
    train_data = fold_data['train']
    val_data = fold_data['val']

    num_tasks = get_world_size()
    global_rank = get_rank()

    category_to_idx = json.load(open(args.category_to_idx_path))
    train_dataset = BacteriaDataset(train_data, transform=image_transform, category_to_idx=category_to_idx)
    val_dataset = BacteriaDataset(val_data, transform=test_transform, category_to_idx=category_to_idx)

    sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_tasks, global_rank,
                                                        shuffle=True, seed=args.seed)
    sampler_val = torch.utils.data.DistributedSampler(val_dataset, num_tasks, global_rank,
                                                      shuffle=False, seed=args.seed)

    if global_rank == 0:  # 如果是主进程，创造日志文件
        fold_output_dir = os.path.join(args.output_dir, f'fold_{args.fold_index}')
        if not os.path.exists(fold_output_dir):
            os.makedirs(fold_output_dir, exist_ok=True)
        logger = create_logger(fold_output_dir, f'fold_{args.fold_index}')

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

    
    model = create_model(args.model, pretrained=args.pretrained, num_classes=args.nb_classes)
    
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

    logger.info('Start training...')

    # 在args.log_dir下创建csv文件
    train_csv_file = os.path.join(fold_output_dir, 'train_stats.csv')
    val_csv_file = os.path.join(fold_output_dir, 'val_stats.csv')
    # 将列名写入csv文件
    pd.DataFrame(columns=['epoch', 'loss', 'acc', 'pre', 'sen', 'f1', 'spec', 'kappa', 'auc', 'qwk']).to_csv(
        train_csv_file, index=False)
    pd.DataFrame(columns=['epoch', 'loss', 'acc', 'pre', 'sen', 'f1', 'spec', 'kappa', 'auc', 'qwk']).to_csv(
        val_csv_file, index=False)

    val_acc = []
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                      loss_scaler=loss_scaler, start_epoch=epoch, logger=logger,
                                      lr_scheduler=lr_scheduler_values, num_training_steps_per_epoch=num_training_steps_per_epoch,
                                      use_amp=args.use_amp)
        with open(train_csv_file, 'a+') as f:
            row = [epoch] + list(train_stats.values())
            f.write(','.join(map(str, row)) + '\n')
        if args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch == args.epochs - 1:
                save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, fold_output_dir)

        val_state = evaluate(model, criterion, device, data_loader_val, use_amp=args.use_amp, logger=logger,
                             header='Val')
        with open(val_csv_file, 'a+') as f:
            row = [epoch] + list(val_state.values())
            f.write(','.join(map(str, row)) + '\n')
        val_acc.append(val_state['acc'])
    
    # 绘制loss曲线，acc曲线
    plot_curve(train_csv_file, val_csv_file, fold_output_dir)
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # Return the best validation and test accuracies for cross-validation aggregation
    return {
        'val_acc': val_acc,
        'fold_index': args.fold_index,
        'checkpoint_dir': fold_output_dir
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('All_models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    def parse_config(args, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        for key, value in config.items():
            setattr(args, key, value)
        return args

    args = parse_config(args, 'config.json')
    main(args)
