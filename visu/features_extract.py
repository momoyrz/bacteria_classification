import argparse
import json
import sys
import os
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

import timm
import torch

from torch.backends import cudnn
from torchvision import transforms

from models.convnext import convnext_tiny, convnext_small
from models.crossvit import crossvit_tiny_224, crossvit_small_224
from models.densenet import densenet121
from models.efficientnet import efficientnet_b0
from models.resnet import resnet50, resnet101, resnet34
from models.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
from my_datasets.my_datasets import TryyDatasets
from utils.distributed_util import init_distributed_mode, get_rank

from utils.save_load_model import load_state_dict

from my_datasets.split_data import split_dataset


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
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpu_id', default='0', type=str)

    parser.add_argument('--model',
                        default='resnet34',
                        type=str)
    parser.add_argument('--finetune',
                        default='/home/ubuntu/qujunlong/txyy/output/checkpoint/resnet34/20241204_204030/checkpoint-11.pth',
                        type=str)
    parser.add_argument('--model_key',
                        default='model|module',
                        type=str)


    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--nb_classes', default=3, type=int)
    parser.add_argument('--data_dir', default='/home/ubuntu/qujunlong/data/tryy/pt_dir', type=str)
    parser.add_argument('--jsonl_path', default='/home/ubuntu/qujunlong/txyy/my_classification/my_datasets/tryy_with_normal.jsonl', type=str)
    parser.add_argument('--modality', default='all', type=str)


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
    lables = ['BCD', 'PR', 'Normal']

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    image_transform = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    _, _, test_data = split_dataset(args.jsonl_path, args.modality, train_ratio=0.7, val_ratio=0.1)

    global_rank = get_rank()


    test_dataset = TryyDatasets(args, test_data, transform=image_transform)


    sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    if global_rank == 0:  # 如果是主进程，创造日志文件
        args.prefix = args.finetune.split('/')[-2]
        weight_name = args.finetune.split('/')[-1].split('.')[0]

        args.log_dir = '/home/ubuntu/qujunlong/txyy/output/results/' + args.model + '/' + args.prefix + '/' + weight_name
        args.output_dir = '/home/ubuntu/qujunlong/txyy/output/results/' + args.model + '/' + args.prefix + '/' + weight_name
        # 如果'train_output/other/' + args.model + '_' + args.prefix这个文件夹不存在，则创建
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        # 将args以json格式保存到args.log_dir下的args.json文件中
        with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)


    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_test,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.model == 'resnet50':
        model = resnet50(num_classes=args.nb_classes)
    elif args.model == 'resnet101':
        model = resnet101(num_classes=args.nb_classes)
    elif args.model == 'resnet34':
        model = resnet34(num_classes=args.nb_classes)
    elif args.model == 'densenet121':
        model = densenet121(num_classes=args.nb_classes)
    elif args.model == 'swin_tiny_patch4_window7_224':
        model = swin_tiny_patch4_window7_224(num_classes=args.nb_classes)
    elif args.model == 'swin_small_patch4_window7_224':
        model = swin_small_patch4_window7_224(num_classes=args.nb_classes)
    elif args.model == 'efficientnet_b0':
        model = efficientnet_b0(num_classes=args.nb_classes)
    elif args.model == 'crossvit_tiny_224':
        model = crossvit_tiny_224(num_classes=args.nb_classes)
    elif args.model == 'crossvit_small_224':
        model = crossvit_small_224(num_classes=args.nb_classes)
    elif args.model == 'convnext_tiny_224':
        model = convnext_tiny(num_classes=args.nb_classes)
    elif args.model == 'convnext_small_224':
        model = convnext_small(num_classes=args.nb_classes)
    elif args.model == 'vit_small_patch16_224':
        model = timm.create_model('vit_small_patch16_224', num_classes=args.nb_classes)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
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
                del checkpoint_model[k]
        load_state_dict(model, checkpoint_model, prefix='')
    model.to(device)

    with torch.no_grad():
        model.eval()
        csv_file_path = "features_embed.csv"

        # 打开 CSV 文件并准备写入
        with open(os.path.join(args.output_dir, csv_file_path), mode='w', newline='') as file:
            writer = csv.writer(file)

            for data_iter_step, (samples, targets) in enumerate(data_loader_test):
                samples = samples.to(device, non_blocking=True)

                feature = model.forward_features(samples)

                if isinstance(feature, torch.Tensor):
                    feature = feature.cpu().detach().numpy()  # 移到 CPU 并转换为 NumPy 数组

                # 遍历 batch，将每个样本的特征逐行写入 CSV
                for single_feature in feature:
                    writer.writerow(single_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('All_models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)