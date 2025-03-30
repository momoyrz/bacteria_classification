from os.path import join
import os, torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import logging
import json

from my_datasets.aug import DataAugmentationManager


class BacteriaDataset(Dataset):
    def __init__(self, data_list, category_to_idx=None, data_augmentation_name=None, **kwargs):
        self.data_list = data_list
        self.category_to_idx = category_to_idx
        self.data_augmentation_manager = DataAugmentationManager()
        self.transform = self.data_augmentation_manager.get_augmentation(data_augmentation_name, **kwargs)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item['image_path']
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        label = self.category_to_idx[item['category']]
        
        return image, label

