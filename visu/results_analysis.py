import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from datetime import timedelta
from inference.classification_metrics import all_metrics
import matplotlib.pyplot as plt
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

    return acc, pre, sen, f1, spec, kappa, my_auc, qwk


def process_results(all_acc, all_pre, all_sen, all_f1, all_spec, all_kappa, all_my_auc, all_qwk, logger, args):
    logger.info(f"All acc: {all_acc}")
    logger.info(f"All pre: {all_pre}")
    logger.info(f"All sen: {all_sen}")
    logger.info(f"All f1: {all_f1}")
    logger.info(f"All spec: {all_spec}")
    logger.info(f"All kappa: {all_kappa}")
    logger.info(f"All my_auc: {all_my_auc}")
    logger.info(f"All qwk: {all_qwk}")

    # 平均
    all_acc = np.array(all_acc)
    all_pre = np.array(all_pre)
    all_sen = np.array(all_sen)
    all_f1 = np.array(all_f1)
    all_spec = np.array(all_spec)
    all_kappa = np.array(all_kappa) 
    all_my_auc = np.array(all_my_auc)
    all_qwk = np.array(all_qwk)
    logger.info(f"Mean acc: {all_acc.mean()} pre: {all_pre.mean()} sen: {all_sen.mean()} f1: {all_f1.mean()} spec: {all_spec.mean()} kappa: {all_kappa.mean()} my_auc: {all_my_auc.mean()} qwk: {all_qwk.mean()}")

    # 计算均值/标准差
    acc_mean = all_acc.mean()
    acc_std = all_acc.std()
    pre_mean = all_pre.mean()
    pre_std = all_pre.std()
    sen_mean = all_sen.mean()
    sen_std = all_sen.std()
    f1_mean = all_f1.mean()
    f1_std = all_f1.std()
    spec_mean = all_spec.mean()
    spec_std = all_spec.std()
    kappa_mean = all_kappa.mean()
    kappa_std = all_kappa.std()
    my_auc_mean = all_my_auc.mean()
    my_auc_std = all_my_auc.std()
    qwk_mean = all_qwk.mean()
    qwk_std = all_qwk.std()

    logger.info(f"Mean acc: {acc_mean} pre: {pre_mean} sen: {sen_mean} f1: {f1_mean} spec: {spec_mean} kappa: {kappa_mean} my_auc: {my_auc_mean} qwk: {qwk_mean}")
    logger.info(f"Std acc: {acc_std} pre: {pre_std} sen: {sen_std} f1: {f1_std} spec: {spec_std} kappa: {kappa_std} my_auc: {my_auc_std} qwk: {qwk_std}")

    # 保存结果
    with open(os.path.join(args.output_dir, 'results.csv'), 'w') as f:
        f.write(f"Mean acc: {acc_mean} pre: {pre_mean} sen: {sen_mean} f1: {f1_mean} spec: {spec_mean} kappa: {kappa_mean} my_auc: {my_auc_mean} qwk: {qwk_mean}\n")
        f.write(f"Std acc: {acc_std} pre: {pre_std} sen: {sen_std} f1: {f1_std} spec: {spec_std} kappa: {kappa_std} my_auc: {my_auc_std} qwk: {qwk_std}\n")


def plot_curve(train_csv_file, val_csv_file, fold_output_dir):
    # 读取csv文件
    train_df = pd.read_csv(train_csv_file)
    val_df = pd.read_csv(val_csv_file)

    # 绘制loss曲线
    plt.figure(figsize=(10, 8))
    plt.plot(train_df['loss'], label='train')
    plt.plot(val_df['loss'], label='val')
    plt.legend()
    plt.savefig(os.path.join(fold_output_dir, 'loss.png'))
    plt.close()
    
    # 绘制acc曲线
    plt.figure(figsize=(10, 8))
    plt.plot(train_df['acc'], label='train')
    plt.plot(val_df['acc'], label='val')
    plt.legend()
    plt.savefig(os.path.join(fold_output_dir, 'acc.png'))
    plt.close()

