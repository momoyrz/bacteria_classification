import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from datetime import timedelta
from inference.classification_metrics import all_metrics

import numpy as np
import pandas as pd
import scipy as sp
import timm
import torch
from sklearn.metrics import roc_curve
from torch.backends import cudnn
from torchvision import transforms

from models.convnext import convnext_tiny, convnext_small
from models.crossvit import crossvit_tiny_224, crossvit_small_224
from models.densenet import densenet121
from models.efficientnet import efficientnet_b0
from models.resnet import resnet50, resnet101, resnet34
from models.swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
from my_datasets.my_datasets import CsvDatasets, TryyDatasets
from utils.distributed_util import init_distributed_mode, get_rank, get_world_size
from utils.logger import create_logger
from utils.loss import FocalLoss
from utils.lr_factory import cosine_scheduler
from utils.optim_factory import create_optimizer
from utils.save_load_model import load_state_dict, save_model
from utils.amp_util import NativeScalerWithGradNormCount as NativeScaler
from utils.train_and_eval import train_one_epoch, evaluate
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
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--gpu_id', default='2', type=str)

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

    num_tasks = get_world_size()
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
        logger = create_logger(args.log_dir, 'visualize.log')
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
        logger.info(f"Loading model from {args.finetune}")
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


    start_time = time.time()


    with torch.no_grad():
        model.eval()
        y_true = []
        y_score = pd.DataFrame()

        for data_iter_step, (samples, targets) in enumerate(data_loader_test):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)


            outputs = model(samples)

            # softmax
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            y_true += targets.cpu().numpy().tolist()
            y_score = pd.concat([y_score, pd.DataFrame(outputs.detach().cpu().numpy())], axis=0)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    acc, pre, sen, f1, spec, kappa, my_auc, qwk = all_metrics(y_true, y_score)
    logger.info(f'acc={acc:.4f} pre={pre:.4f} sen={sen:.4f} f1={f1:.4f} spec={spec:.4f} kappa={kappa:.4f} '
                f'my_auc={my_auc:.4f} qwk={qwk:.4f}')

    for i in range(args.nb_classes):
        fpr, tpr, _ = roc_curve(y_true, y_score[:, i], pos_label=i)
        np.savetxt(os.path.join(args.output_dir, 'fpr_' + str(i) + '.csv'), fpr, delimiter=',')
        np.savetxt(os.path.join(args.output_dir, 'tpr_' + str(i) + '.csv'), tpr, delimiter=',')

    y_pred_test = np.argmax(y_score, axis=1)
    cm = sp.sparse.coo_matrix((np.ones(y_pred_test.shape[0]), (y_true, y_pred_test)),
                              shape=(args.nb_classes, args.nb_classes), dtype=np.int32).toarray()
    # 保存混淆矩阵
    np.savetxt(os.path.join(args.output_dir, 'confusion_matrix.csv'), cm, delimiter=',')

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='.20g', cmap='PuRd', annot_kws={'size': 20},
                xticklabels=lables, yticklabels=lables)
    plt.tick_params(axis='x', labelsize=20)  # 设置x轴标签的字体大小
    plt.tick_params(axis='y', labelsize=20)  # 设置y轴标签的字体大小
    plt.subplots_adjust(left=0.07, right=1.05, bottom=0.07, top=0.95)
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=480)
    plt.close()


    # 保存预测结果
    np.savetxt(os.path.join(args.log_dir, 'y_true_test.csv'), y_true, delimiter=',')
    np.savetxt(os.path.join(args.log_dir, 'y_pred_test.csv'), y_pred_test, delimiter=',')
    # 保存预测概率
    np.savetxt(os.path.join(args.log_dir, 'y_score_test.csv'), y_score, delimiter=',')

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info('Features extract time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('All_models training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)