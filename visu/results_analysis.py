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

def test_and_visualize(args, model, device, data_loader_test, logger, fold_dir):
    model.to(device)
    labels = json.load(open(args.category_to_idx_path))
    labels = [key for key in labels.keys()]
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
        np.savetxt(os.path.join(fold_dir, 'fpr_' + str(i) + '.csv'), fpr, delimiter=',')
        np.savetxt(os.path.join(fold_dir, 'tpr_' + str(i) + '.csv'), tpr, delimiter=',')

    y_pred_test = np.argmax(y_score, axis=1)
    cm = sp.sparse.coo_matrix((np.ones(y_pred_test.shape[0]), (y_true, y_pred_test)),
                              shape=(args.nb_classes, args.nb_classes), dtype=np.int32).toarray()
    # 保存混淆矩阵
    np.savetxt(os.path.join(fold_dir, 'confusion_matrix.csv'), cm, delimiter=',')

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='.20g', cmap='PuRd', annot_kws={'size': 20},
                xticklabels=labels, yticklabels=labels)
    plt.tick_params(axis='x', labelsize=20)  # 设置x轴标签的字体大小
    plt.tick_params(axis='y', labelsize=20)  # 设置y轴标签的字体大小
    plt.subplots_adjust(left=0.07, right=1.05, bottom=0.07, top=0.95)
    plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'), dpi=480)
    plt.close()


    # 保存预测结果
    np.savetxt(os.path.join(fold_dir, 'y_true_test.csv'), y_true, delimiter=',')
    np.savetxt(os.path.join(fold_dir, 'y_pred_test.csv'), y_pred_test, delimiter=',')
    # 保存预测概率
    np.savetxt(os.path.join(fold_dir, 'y_score_test.csv'), y_score, delimiter=',')
