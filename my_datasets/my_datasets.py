from os.path import join
import os, torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import logging

class_to_idx = {'BCD': 0, 'RP': 1, 'normal': 2}

class CsvDatasets(Dataset):
    def __init__(self, args, path, data_df, transform=None):
        self.args = args
        self.path = path
        self.data_df = data_df
        self.transform = transform
        self.classes = [i for i in range(args.nb_classes)]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img = self.data_df.iloc[idx]
        image = Image.open(join(self.path, img['图片']))
        i_np = np.array(image)[:, :, :3]
        pil = Image.fromarray(i_np)
        image = self.transform(pil)
        return image, img['标签']

class TryyDatasets(Dataset):
    def __init__(self, args, samples, transform=None):
        self.args = args
        self.samples = samples
        self.transform = transform


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pt_path = join(self.args.data_dir, sample['path'])
        try:
            image = torch.load(pt_path)[:3, :, :]
            image = self.transform(image)
        except Exception as e:
            logging.error(f"Error loading image from {pt_path}: {e}")
            image = torch.zeros(3, 256, 256)
        return image, class_to_idx[sample['class']]

