from os.path import join
import os, torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import logging


class BacteriaDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item['image_path']
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        
        # 将类别名称映射为整数
        if not hasattr(self, 'category_to_idx'):
            # 获取所有唯一的类别名称并排序
            categories = sorted(list(set(item['category'] for item in self.data_list)))
            # 创建类别到索引的映射
            self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
            
        label = self.category_to_idx[item['category']]
        
        return image, label

