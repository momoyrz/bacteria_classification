import json
import random
import sys
from collections import defaultdict
from sklearn.model_selection import train_test_split

def split_dataset(file_path, modality, train_ratio=0.7, val_ratio=0.1):
    with open(file_path, 'r', encoding='utf-8') as f:
        if modality == 'all':
            data = [json.loads(line) for line in f]
        elif modality == 'color':
            data = []
            for line in f:
                item = json.loads(line)
                try:
                    if item['modality'] == 'color':
                        data.append(item)
                except:
                    print("Error parsing line:", line)
        elif modality == 'FAF':
            data = []
            for line in f:
                item = json.loads(line)
                try:
                    if item['modality'] == 'FAF':
                        data.append(item)
                except:
                    print("Error parsing line:", line)
        elif modality == 'IR':
            data = []
            for line in f:
                item = json.loads(line)
                try:
                    if item['modality'] == 'IR':
                        data.append(item)
                except:
                    print("Error parsing line:", line)

    class_count = defaultdict(int)
    for item in data:
        class_count[item['class']] += 1

    train_target = {cls: int(count * train_ratio) for cls, count in class_count.items()}
    val_target = {cls: int(count * val_ratio) for cls, count in class_count.items()}
    test_target = {cls: count - train_target[cls] - val_target[cls] for cls, count in class_count.items()}

    train_sample, val_sample, test_sample = [], [], []

    train_current = defaultdict(int)
    val_current = defaultdict(int)
    test_current = defaultdict(int)

    for item in data:
        cls = item['class']
        if train_current[cls] < train_target[cls]:
            train_sample.append(item)
            train_current[cls] += 1
        elif val_current[cls] < val_target[cls]:
            val_sample.append(item)
            val_current[cls] += 1
        elif test_current[cls] < test_target[cls]:
            test_sample.append(item)
            test_current[cls] += 1

    def count_classes(samples):
        class_count = defaultdict(int)
        for sample in samples:
            class_count[sample['class']] += 1
        return class_count

    train_class_count = count_classes(train_sample)
    val_class_count = count_classes(val_sample)
    test_class_count = count_classes(test_sample)

    print("Train set class distribution:", train_class_count)
    print("Validation set class distribution:", val_class_count)
    print("Test set class distribution:", test_class_count)

    return train_sample, val_sample, test_sample

