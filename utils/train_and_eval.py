import sys
from typing import Iterable

import math
import numpy as np
import pandas as pd
import torch.nn
from sklearn.metrics import roc_auc_score

from inference.classification_metrics import all_metrics


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, start_epoch=0,
                    logger=None, lr_scheduler=None, num_training_steps_per_epoch=None, use_amp=False):
    model.train(True)
    optimizer.zero_grad()

    y_true = []
    y_score = pd.DataFrame()
    losses = []
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        if data_iter_step >= num_training_steps_per_epoch:
            continue

        it = start_epoch * num_training_steps_per_epoch + data_iter_step - 1

        if lr_scheduler is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_scheduler[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 如果criterion不是交叉熵
        if criterion.__class__.__name__ != 'CrossEntropyLoss':
            # 将targets转换为one-hot编码
            targets = torch.nn.functional.one_hot(targets, num_classes=len(data_loader.dataset.classes)).float()
        # logger.info(f'samples shape: {samples.size()}')
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)
        # softmax
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        losses += [loss.item()]
        y_true += targets.cpu().numpy().tolist()
        y_score = pd.concat([y_score, pd.DataFrame(outputs.detach().cpu().numpy())], axis=0)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.error("Loss is {}, stopping training".format(loss_value))
            logger.error(outputs)
            logger.error(targets)
            sys.exit(1)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % 1 == 0)
            if (data_iter_step + 1) % 1 == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            if (data_iter_step + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()

        class_acc = (outputs.argmax(dim=1) == targets).float().mean().item()

        if logger is not None:
            if (data_iter_step + 1) % 1 == 0:
                logger.info(f'Epoch {epoch} Iter {data_iter_step + 1}/{num_training_steps_per_epoch} '
                            f'lr={optimizer.param_groups[0]["lr"]:.6f} '
                            f'loss={loss_value:.4f} '
                            f'class_acc={class_acc:.4f} ')

    y_true = np.array(y_true)
    train_loss = np.mean(losses)
    y_score = np.array(y_score)
    acc, pre, sen, f1, spec, kappa, my_auc, qwk = all_metrics(y_true, y_score)
    logger.info(f'Epoch {epoch} train acc={acc:.4f} pre={pre:.4f} sen={sen:.4f} f1={f1:.4f} spec={spec:.4f} kappa={kappa:.4f} my_auc={my_auc:.4f} qwk={qwk:.4f}')
    return {'train_loss': train_loss, 'acc': acc, 'pre': pre, 'sen': sen, 'f1': f1, 'spec': spec, 'kappa': kappa, 'my_auc': my_auc, 'qwk': qwk}

@torch.no_grad()
def evaluate(model, criterion, device, data_loader, epoch, use_amp, logger=None, header=None):
    model.eval()
    y_true = []
    y_score = pd.DataFrame()
    losses = []

    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if criterion.__class__.__name__ != 'CrossEntropyLoss':
            # 将targets转换为one-hot编码
            targets = torch.nn.functional.one_hot(targets, num_classes=len(data_loader.dataset.classes)).float()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(outputs, targets)

        # softmax
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        losses += [loss.item()]
        y_true += targets.cpu().numpy().tolist()
        y_score = pd.concat([y_score, pd.DataFrame(outputs.cpu().numpy())], axis=0)

        if logger is not None:
            logger.info(f'Eval Iter {data_iter_step + 1}/{len(data_loader)} loss={loss.item():.4f}')

    y_true = np.array(y_true)
    eval_loss = np.mean(losses)
    y_score = np.array(y_score)
    acc, pre, sen, f1, spec, kappa, my_auc, qwk = all_metrics(y_true, y_score)
    logger.info(f'{header} acc={acc:.4f} pre={pre:.4f} sen={sen:.4f} f1={f1:.4f} spec={spec:.4f} kappa={kappa:.4f} '
                f'my_auc={my_auc:.4f} qwk={qwk:.4f}')
    return {'eval_loss': eval_loss, 'acc': acc, 'pre': pre, 'sen': sen, 'f1': f1, 'spec': spec, 'kappa': kappa,
            'my_auc': my_auc, 'qwk': qwk}
