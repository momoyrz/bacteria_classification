from os.path import join
import os, torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import logging
import json


class BacteriaDataset(Dataset):
    def __init__(self, data_list, transform=None, category_to_idx=None):
        self.data_list = data_list
        self.transform = transform
        self.category_to_idx = category_to_idx

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

