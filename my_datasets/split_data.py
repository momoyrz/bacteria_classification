import json
import os
import random
import sys
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

def split_dataset(data_dir, data_jsonl_path, fold, val_size=0.1, random_state=42):
    """
    Split dataset using k-fold cross-validation, then further split training data into train and validation sets.
    
    Args:
        data_dir: Directory containing the images
        data_jsonl_path: Path to the jsonl file containing data annotations
        fold: Number of folds for cross-validation
        val_size: Proportion of the training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        A list of dictionaries, where each dictionary contains 'train', 'val', and 'test' data for each fold
    """
    # 读取jsonl文件
    all_samples = []
    
    with open(data_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            data['image_path'] = os.path.join(data_dir, data['image_path'])
            all_samples.append(data)
    
    # 提取所有样本的类别标签，用于分层抽样
    labels = [sample['category'] for sample in all_samples]
    
    # 创建分层K折交叉验证
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    
    # 将数据转换为数组以便进行索引
    samples_array = np.array(all_samples)
    
    # 存储每一折的训练集、验证集和测试集
    fold_data = []
    
    # 获取每一折的训练集和测试集索引，保证类别比例平衡
    for train_idx, test_idx in skf.split(samples_array, labels):
        # 使用索引获取当前fold的训练集和测试集
        train_samples = samples_array[train_idx].tolist()
        test_samples = samples_array[test_idx].tolist()
        
        # 从训练集中进一步划分出验证集，保持类别比例平衡
        train_labels = [sample['category'] for sample in train_samples]
        train_samples_array = np.array(train_samples)
        
        # 使用分层抽样从训练集中分离出验证集
        train_indices, val_indices = train_test_split(
            np.arange(len(train_samples_array)),
            test_size=val_size,
            stratify=train_labels,
            random_state=random_state
        )
        
        # 获取训练集和验证集
        final_train_samples = train_samples_array[train_indices].tolist()
        val_samples = train_samples_array[val_indices].tolist()
        
        # 随机打乱每个数据集
        random.shuffle(final_train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        # 将当前折的数据添加到结果列表中
        fold_data.append({
            'train': final_train_samples,
            'val': val_samples,
            'test': test_samples
        })
    
    return fold_data
