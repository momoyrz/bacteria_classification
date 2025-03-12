import json
import os
import random
import sys
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, data_jsonl_path, train_ratio=0.7, val_ratio=0.1):
    # 读取jsonl文件并按类别分组
    class_data = defaultdict(list)
    with open(data_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            data['image_path'] = os.path.join(data_dir, data['image_path'])
            label = data['category']
            class_data[label].append(data)

    train_sample = []
    val_sample = []
    test_sample = []

    # 对每个类别分别进行划分
    for label, samples in class_data.items():
        # 计算验证集和测试集的比例
        test_ratio = 1 - train_ratio - val_ratio
        
        # 随机打乱当前类别的数据
        random.shuffle(samples)
        
        # 计算当前类别各个数据集的大小
        total_size = len(samples)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # 划分当前类别的数据
        train_sample.extend(samples[:train_size])
        val_sample.extend(samples[train_size:train_size + val_size])
        test_sample.extend(samples[train_size + val_size:])

    # 随机打乱每个数据集
    random.shuffle(train_sample)
    random.shuffle(val_sample)
    random.shuffle(test_sample)



    return train_sample, val_sample, test_sample

if __name__ == '__main__':
    train_sample, val_sample, test_sample = split_dataset(
        '/home/ubuntu/qujunlong/data/bacteria',
        '/home/ubuntu/qujunlong/bacteria_classification/my_datasets/bacteria_dataset.jsonl')
    print(len(train_sample), len(val_sample), len(test_sample))
    print(train_sample)
    print(val_sample)
    print(test_sample)

    # 统计每个集合中每个类别的样本数量
    train_class_counts = Counter()
    val_class_counts = Counter()
    test_class_counts = Counter()

    for sample in train_sample:
        train_class_counts[sample['category']] += 1
    for sample in val_sample:
        val_class_counts[sample['category']] += 1
    for sample in test_sample:
        test_class_counts[sample['category']] += 1

    print(train_class_counts)
    print(val_class_counts)
    print(test_class_counts)

