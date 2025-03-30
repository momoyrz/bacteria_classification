import re
import random
from dataclasses import dataclass
from typing import Any, Union, Optional, List, Dict, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import cv2
from skimage import exposure
from skimage.util import random_noise
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class DataAugmentationManager:
    def __init__(self):
        self.augmentations = {}

    def register_augmentation(self, name, augmentation_class):
        self.augmentations[name] = augmentation_class

    def get_augmentation(self, name, **kwargs):
        if name not in self.augmentations:
            raise ValueError(f"Augmentation '{name}' is not registered.")
        return self.augmentations[name](**kwargs)


class CutMixAugmentation:
    def __init__(self, alpha=2):
        self.alpha = alpha

    def __call__(self, batch):
        if isinstance(batch, list):
            batch = torch.utils.data.dataloader.default_collate(batch)
        data, targets = batch

        # 使用 torch 替代 numpy
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        # 使用 torch 生成 beta 分布的 lambda
        lam = torch.distributions.Beta(
            torch.tensor(self.alpha),
            torch.tensor(self.alpha)
        ).sample().item()

        image_h, image_w = data.shape[2:]

        # 使用 torch 的随机数生成
        cx = int(torch.rand(1).item() * image_w)
        cy = int(torch.rand(1).item() * image_h)

        w = int(image_w * torch.sqrt(torch.tensor(1. - lam)))
        h = int(image_h * torch.sqrt(torch.tensor(1. - lam)))

        x0 = max(cx - w // 2, 0)
        x1 = min(cx + w // 2, image_w)
        y0 = max(cy - h // 2, 0)
        y1 = min(cy + h // 2, image_h)

        # 按 lambda 系数混合
        data[:, :, y0:y1, x0:x1] = (
                lam * data[:, :, y0:y1, x0:x1] +
                (1 - lam) * shuffled_data[:, :, y0:y1, x0:x1]
        )

        targets = (targets, shuffled_targets, lam)
        return data, targets


class CutMixCriterion:
    def __init__(self, reduction='mean'):
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)


class CutoutAugmentation:
    def __init__(self, **kwargs):
        self.n_holes = kwargs['n_holes']
        self.length = kwargs['length']
        self.p = kwargs['p']

    def __call__(self, img):
        # 概率应用
        if random.uniform(0, 1) > self.p:
            return img

        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        # 创建掩码
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        # 应用掩码
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

    def generate_mask(self, img):
        """创建掩码但不应用它"""
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return mask


class HaSAugmentation:
    def __init__(self, **kwargs):
        """
        Initialize Hide-and-Seek augmentation

        Args:
            grid_sizes: 可选的网格大小列表
            hide_prob: 隐藏网格的概率
        """
        self.grid_sizes = kwargs['grid_sizes'] if kwargs['grid_sizes'] is not None else [0, 16, 32, 44, 56]
        self.hide_prob = kwargs['hide_prob']

    def __call__(self, img):
        """
        Apply Hide-and-Seek augmentation

        Args:
            img: Input image tensor (C, H, W) for PyTorch format

        Returns:
            Augmented image tensor
        """
        c, h, w = img.shape

        # 随机选择一个网格大小
        grid_size = self.grid_sizes[random.randint(0, len(self.grid_sizes) - 1)]

        if grid_size > 0:
            for y in range(0, h, grid_size):
                for x in range(0, w, grid_size):
                    y_end = min(h, y + grid_size)
                    x_end = min(w, x + grid_size)

                    # 按概率隐藏网格
                    if random.random() <= self.hide_prob:
                        img[:, y:y_end, x:x_end] = 0

        return img


class GridMaskAugmentation:
    def __init__(self, **kwargs):
        """
        GridMask Data Augmentation

        Args:
            d1: Minimum grid size
            d2: Maximum grid size
            rotate: Maximum rotation angle
            ratio: Ratio of the grid holes
            mode: 0 for masking pixels, 1 for keeping pixels in the grid
            prob: Probability of applying GridMask
        """
        self.d1 = kwargs['d1']  
        self.d2 = kwargs['d2']
        self.rotate = kwargs['rotate']
        self.ratio = kwargs['ratio']
        self.mode = kwargs['mode']
        self.st_prob = self.prob = kwargs['prob']

    def set_prob(self, epoch, max_epoch):
        """Adjust the probability based on current epoch"""
        self.prob = self.st_prob * epoch / max_epoch

    def _apply_grid_mask(self, img):
        """
        Apply grid mask to a single image

        Args:
            img: Input tensor image (C x H x W)

        Returns:
            Masked image
        """
        if random.random() > self.prob:
            return img

        h, w = img.size(1), img.size(2)

        # Randomly select grid size between d1 and d2
        d = random.randint(self.d1, self.d2)

        # Calculate mask parameters
        grid_ratio = self.ratio

        # Calculate the positions of the grid
        mask = torch.ones_like(img)
        st_h = random.randint(0, d)
        st_w = random.randint(0, d)

        # Apply rotation if needed
        if self.rotate:
            # Simple random horizontal/vertical flip
            horizontal_flip = random.random() < 0.5
            vertical_flip = random.random() < 0.5

            if horizontal_flip:
                img = torch.flip(img, [2])  # Flip horizontal
            if vertical_flip:
                img = torch.flip(img, [1])  # Flip vertical

        # Apply the grid mask
        for i in range(st_h, h, d):
            s_h = min(h, i + int(d * grid_ratio))
            for j in range(st_w, w, d):
                s_w = min(w, j + int(d * grid_ratio))
                if self.mode == 0:  # Masking pixels (set to 0)
                    mask[:, i:s_h, j:s_w] = 0
                else:  # Keep pixels (mode=1)
                    mask[:, :i, :] = 0
                    mask[:, s_h:, :] = 0
                    mask[:, :, :j] = 0
                    mask[:, :, s_w:] = 0

        img = img * mask

        # Revert flips if applied
        if self.rotate:
            if vertical_flip:
                img = torch.flip(img, [1])
            if horizontal_flip:
                img = torch.flip(img, [2])

        return img

    def __call__(self, x):
        """
        Apply GridMask to input data

        Args:
            x: Can be a single image or batch of images

        Returns:
            Augmented data
        """
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            # Batch of images
            return self.forward(x)
        else:
            # Single image
            return self._apply_grid_mask(x)

    def forward(self, x):
        """
        Forward pass applying GridMask to a batch

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Augmented tensor of same shape
        """
        if not self.training:
            return x

        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self._apply_grid_mask(x[i]))
        y = torch.cat(y).view(n, c, h, w)
        return y

    @property
    def training(self):
        # For compatibility with PyTorch modules
        return True


class RandomErasingAugmentation:
    def __init__(self, **kwargs):
        """
        Random Erasing Data Augmentation

        Args:
            probability: Probability of applying the augmentation
            sl: Minimum proportion of erased area
            sh: Maximum proportion of erased area
            r1: Minimum aspect ratio of erased area
            mean: Mean values for erased pixels
        """
        self.probability = kwargs['probability']
        self.mean = kwargs['mean']
        self.sl = kwargs['sl']
        self.sh = kwargs['sh']
        self.r1 = kwargs['r1']

    def __call__(self, img):
        """
        Apply random erasing to an image

        Args:
            img: Input tensor image

        Returns:
            Augmented image
        """
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

                return img

        return img


class AutoAugmentAugmentation:
    def __init__(self, **kwargs):
        """
        Initialize AutoAugment data augmentation

        Args:
            policies (list, optional): Custom policies list. If None, uses default policies.
            fixed_posterize (bool, optional): Whether to use fixed posterize method. Default is False.
        """
        # Define constants and helper functions
        self.PARAMETER_MAX = 30
        self.policies = kwargs['policies'] or self._get_default_policies()
        self.fixed_posterize = kwargs['fixed_posterize']

        # Setup transformations
        self._setup_transforms()

    def _setup_transforms(self):
        """Set up all transform functions and collections"""
        # Transform wrapper
        self.TransformFunction = type('TransformFunction', (object,), {
            '__init__': lambda self, func, name: setattr(self, 'f', func) or setattr(self, 'name', name),
            '__repr__': lambda self: '<' + self.name + '>',
            '__call__': lambda self, pil_img: self.f(pil_img)
        })

        # Transform base class
        self.TransformT = type('TransformT', (object,), {
            '__init__': lambda self, name, xform_fn: setattr(self, 'name', name) or setattr(self, 'xform', xform_fn),
            'pil_transformer': lambda self, probability, level: self._create_transform_function(self, probability,
                                                                                                level),
        })

        # Add method to TransformT
        def _create_transform_function(self, probability, level):
            def return_function(im):
                if random.random() < probability:
                    im = self.xform(im, level)
                return im

            name = self.name + '({:.1f},{})'.format(probability, level)
            return self.TransformFunction(return_function, name)

        self.TransformT._create_transform_function = _create_transform_function

        # Helper functions
        def pil_wrap(img):
            return img.convert('RGBA')

        def pil_unwrap(img):
            return img.convert('RGB')

        def float_parameter(level, maxval):
            return float(level) * maxval / self.PARAMETER_MAX

        def int_parameter(level, maxval):
            return int(level * maxval / self.PARAMETER_MAX)

        self.pil_wrap = pil_wrap
        self.pil_unwrap = pil_unwrap
        self.float_parameter = float_parameter
        self.int_parameter = int_parameter

        # Define transform implementations
        def _rotate_impl(pil_img, level):
            degrees = int_parameter(level, 30)
            if random.random() > 0.5:
                degrees = -degrees
            return pil_img.rotate(degrees)

        def _posterize_impl(pil_img, level):
            level = int_parameter(level, 4)
            return ImageOps.posterize(pil_img.convert('RGB'), 4 - level).convert('RGBA')

        def _enhancer_impl(enhancer):
            def impl(pil_img, level):
                v = float_parameter(level, 1.8) + .1
                return enhancer(pil_img).enhance(v)

            return impl

        def _shear_x_impl(pil_img, level):
            level = float_parameter(level, 0.3)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((32, 32), Image.AFFINE, (1, level, 0, 0, 1, 0))

        def _shear_y_impl(pil_img, level):
            level = float_parameter(level, 0.3)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0))

        def _translate_x_impl(pil_img, level):
            level = int_parameter(level, 10)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0))

        def _translate_y_impl(pil_img, level):
            level = int_parameter(level, 10)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level))

        # Create transform instances
        self.identity = self.TransformT('identity', lambda pil_img, level: pil_img)
        self.flip_lr = self.TransformT('FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
        self.flip_ud = self.TransformT('FlipUD', lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
        self.auto_contrast = self.TransformT('AutoContrast',
                                             lambda pil_img, level: ImageOps.autocontrast(
                                                 pil_img.convert('RGB')).convert('RGBA'))
        self.equalize = self.TransformT('Equalize',
                                        lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB')).convert(
                                            'RGBA'))
        self.invert = self.TransformT('Invert',
                                      lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB')).convert('RGBA'))
        self.blur = self.TransformT('Blur', lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
        self.smooth = self.TransformT('Smooth', lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
        self.rotate = self.TransformT('Rotate', _rotate_impl)
        self.posterize = self.TransformT('Posterize', _posterize_impl)
        self.color = self.TransformT('Color', _enhancer_impl(ImageEnhance.Color))
        self.contrast = self.TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
        self.brightness = self.TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
        self.sharpness = self.TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
        self.shear_x = self.TransformT('ShearX', _shear_x_impl)
        self.shear_y = self.TransformT('ShearY', _shear_y_impl)
        self.translate_x = self.TransformT('TranslateX', _translate_x_impl)
        self.translate_y = self.TransformT('TranslateY', _translate_y_impl)

        # Define transform lists
        self.ALL_TRANSFORMS = [
            self.identity, self.auto_contrast, self.equalize, self.rotate, self.posterize,
            self.color, self.contrast, self.brightness, self.sharpness,
            self.shear_x, self.shear_y, self.translate_x, self.translate_y
        ]

        self.AA_ALL_TRANSFORMS = [
            self.flip_lr, self.flip_ud, self.auto_contrast, self.equalize, self.invert,
            self.rotate, self.posterize, self.color, self.contrast, self.brightness,
            self.sharpness, self.shear_x, self.shear_y, self.translate_x, self.translate_y,
            self.blur, self.smooth
        ]

        # Create name-to-transform mappings
        self.AA_NAME_TO_TRANSFORM = {t.name: t for t in self.AA_ALL_TRANSFORMS}
        self.NAME_TO_TRANSFORM = {t.name: t for t in self.ALL_TRANSFORMS}

    def _apply_policy(self, policy, img, use_fixed_posterize=False):
        """Apply the policy to the image"""
        nametotransform = self.AA_NAME_TO_TRANSFORM
        pil_img = self.pil_wrap(img)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            xform_fn = nametotransform[name].pil_transformer(probability, level)
            pil_img = xform_fn(pil_img)
        return self.pil_unwrap(pil_img)

    def _get_default_policies(self):
        """Generate default augmentation policies"""
        return [
            [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
            [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
            [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
            [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
            [('Equalize', 0.8, 8), ('Invert', 0.1, 3)]
        ]

    def __call__(self, img):
        """
        Apply randomly selected data augmentation policy

        Args:
            img: Input image

        Returns:
            Augmented image
        """
        epoch_policy = self.policies[np.random.choice(len(self.policies))]
        final_img = self._apply_policy(epoch_policy, img, use_fixed_posterize=self.fixed_posterize)
        return final_img


class TrivialAugmentAugmentation:
    def __init__(self):
        """
        Initialize TrivialAugment augmentation
        """
        # Define necessary constants
        self.PARAMETER_MAX = 30

        # Set up transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Set up all transforms used by TrivialAugment"""
        # Define parameter classes
        self.MinMax = type('MinMax', (), {
            '__init__': lambda self, min_val, max_val: setattr(self, 'min', min_val) or setattr(self, 'max', max_val)
        })

        # Define parameter ranges
        self.min_max_vals = type('MinMaxVals', (), {
            'shear': self.MinMax(0.0, 0.3),
            'translate': self.MinMax(0, 10),
            'rotate': self.MinMax(0, 30),
            'solarize': self.MinMax(0, 256),
            'posterize': self.MinMax(0, 4),
            'enhancer': self.MinMax(0.1, 1.9),
            'cutout': self.MinMax(0.0, 0.2)
        })()

        # Define helper functions
        def float_parameter(level, maxval):
            return float(level) * maxval / self.PARAMETER_MAX

        def int_parameter(level, maxval):
            return int(level * maxval / self.PARAMETER_MAX)

        self.float_parameter = float_parameter
        self.int_parameter = int_parameter

        # Define transform wrapper classes
        self.TransformFunction = type('TransformFunction', (object,), {
            '__init__': lambda self, func, name: setattr(self, 'f', func) or setattr(self, 'name', name),
            '__repr__': lambda self: '<' + self.name + '>',
            '__call__': lambda self, pil_img: self.f(pil_img)
        })

        self.TransformT = type('TransformT', (object,), {
            '__init__': lambda self, name, xform_fn: setattr(self, 'name', name) or setattr(self, 'xform', xform_fn),
            '__repr__': lambda self: '<' + self.name + '>',
            'pil_transformer': lambda self, probability, level: self._create_transform_function(self, probability,
                                                                                                level)
        })

        # Add method to TransformT
        def _create_transform_function(self, probability, level):
            def return_function(im):
                if random.random() < probability:
                    im = self.xform(im, level)
                return im

            name = self.name + '({:.1f},{})'.format(probability, level)
            return self.TransformFunction(return_function, name)

        self.TransformT._create_transform_function = _create_transform_function

        # Define transform implementations
        def _rotate_impl(pil_img, level):
            degrees = int_parameter(level, self.min_max_vals.rotate.max)
            if random.random() > 0.5:
                degrees = -degrees
            return pil_img.rotate(degrees)

        def _posterize_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.posterize.max - self.min_max_vals.posterize.min)
            return ImageOps.posterize(pil_img, self.min_max_vals.posterize.max - level)

        def _solarize_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.solarize.max)
            return ImageOps.solarize(pil_img, 256 - level)

        def _shear_x_impl(pil_img, level):
            level = float_parameter(level, self.min_max_vals.shear.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

        def _shear_y_impl(pil_img, level):
            level = float_parameter(level, self.min_max_vals.shear.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

        def _translate_x_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.translate.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))

        def _translate_y_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.translate.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))

        def _enhancer_impl(enhancer):
            def impl(pil_img, level):
                v = float_parameter(level, 1.8) + .1
                return enhancer(pil_img).enhance(v)

            return impl

        # Create transform instances
        self.identity = self.TransformT('identity', lambda pil_img, level: pil_img)
        self.auto_contrast = self.TransformT('AutoContrast', lambda pil_img, level: ImageOps.autocontrast(pil_img))
        self.equalize = self.TransformT('Equalize', lambda pil_img, level: ImageOps.equalize(pil_img))
        self.rotate = self.TransformT('Rotate', _rotate_impl)
        self.solarize = self.TransformT('Solarize', _solarize_impl)
        self.color = self.TransformT('Color', _enhancer_impl(ImageEnhance.Color))
        self.posterize = self.TransformT('Posterize', _posterize_impl)
        self.contrast = self.TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
        self.brightness = self.TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
        self.sharpness = self.TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
        self.shear_x = self.TransformT('ShearX', _shear_x_impl)
        self.shear_y = self.TransformT('ShearY', _shear_y_impl)
        self.translate_x = self.TransformT('TranslateX', _translate_x_impl)
        self.translate_y = self.TransformT('TranslateY', _translate_y_impl)

        # Define transform list
        self.ALL_TRANSFORMS = [
            self.identity, self.auto_contrast, self.equalize, self.rotate,
            self.solarize, self.color, self.posterize, self.contrast,
            self.brightness, self.sharpness, self.shear_x, self.shear_y,
            self.translate_x, self.translate_y
        ]

    def __call__(self, img):
        """
        Apply TrivialAugment to an image

        Args:
            img: Input image

        Returns:
            Augmented image
        """
        # Randomly select an operation
        op = random.choices(self.ALL_TRANSFORMS, k=1)[0]

        # Apply with random level
        level = random.randint(0, self.PARAMETER_MAX)
        img = op.pil_transformer(1., level)(img)
        return img


class SoftAugAugmentation:
    """
    Simplified SoftAug that only uses SoftMixup for data augmentation.
    """

    def __init__(self,
                 n_class=10,
                 # SoftMixup parameters
                 alpha=1.0,
                 t_mixup=1.0,
                 pow_mixup=1):

        self.n_class = n_class

        # Initialize helper functions
        self._init_helper_functions()

        # Initialize SoftMixup augmentation
        self.softmixup = self._create_soft_mixup(
            alpha=alpha,
            t=t_mixup,
            pow=pow_mixup,
            n_classes=n_class
        )

        # Store parameters for convenience
        self.params = {
            'n_class': n_class,
            'alpha': alpha,
            't_mixup': t_mixup,
            'pow_mixup': pow_mixup
        }

    def _init_helper_functions(self):
        """Initialize helper functions used by SoftMixup"""

        # ComputeProb function
        def compute_prob(x, T=0.25, n_classes=10, max_prob=1.0, pow=2.0):
            max_prob = torch.clamp_min(torch.tensor(max_prob), 1 / n_classes)
            if T <= 0:
                T = 1e-10

            if x > T:
                return max_prob
            elif x > 0:
                a = (max_prob - 1 / float(n_classes)) / (T ** pow)
                return max_prob - a * (T - x) ** pow
            else:
                return np.ones_like(x) * 1 / n_classes

        # DecodeTargetProb function
        def decode_target_prob(targets):
            classes = targets.long()
            probs = 1 - (targets - classes)
            return classes, probs

        # EncodeTargetProb function
        def encode_target_prob(classes, probs=None):
            if probs is None:
                return classes.float()
            else:
                return classes.float() + 1 - probs

        # mixup_data function
        @torch.no_grad()
        def mixup_data(x, y, alpha=1.0, use_cuda=True, pow=1, t=1.0, n_classes=1e6):
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1 - 1e-8

            batch_size = x.size()[0]
            if use_cuda:
                index = torch.randperm(batch_size).cuda()
            else:
                index = torch.randperm(batch_size)

            mixed_x = lam * x + (1 - lam) * x[index, :]

            y_int = y.long()

            y_prob = (1 - y + y_int)
            prob_a = compute_prob(lam, max_prob=y_prob, pow=pow, T=t, n_classes=n_classes)
            prob_b = compute_prob((1 - lam), max_prob=y_prob[index], pow=pow, T=t, n_classes=n_classes)

            y_a = torch.tensor(y_int + 1 - prob_a)
            y_b = torch.tensor(y_int[index] + 1 - prob_b)

            return mixed_x, y_a, y_b, lam

        # mixup_criterion function
        def mixup_criterion(criterion, pred, y_a, y_b, lam):
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

        # Store helper functions as instance methods
        self.compute_prob = compute_prob
        self.decode_target_prob = decode_target_prob
        self.encode_target_prob = encode_target_prob
        self.mixup_data = mixup_data
        self.mixup_criterion = mixup_criterion

    def _create_soft_mixup(self, alpha, t, pow, n_classes):
        """Create SoftMixup augmentation"""

        class SoftMixup:
            def __init__(self, parent, alpha, t, pow, n_classes):
                self.parent = parent
                self.alpha = alpha
                self.t = t
                self.pow = pow
                self.n_classes = n_classes

            def __call__(self, batch):
                if isinstance(batch, tuple) and len(batch) == 2:
                    data, targets = batch
                else:
                    data, targets = batch

                device = data.device if hasattr(data, 'device') else torch.device('cpu')

                mixed_x, y_a, y_b, lam = self.parent.mixup_data(
                    data, targets,
                    alpha=self.alpha,
                    use_cuda=device.type == 'cuda',
                    t=self.t,
                    pow=self.pow,
                    n_classes=self.n_classes
                )

                return mixed_x, (y_a, y_b, lam)

        return SoftMixup(self, alpha, t, pow, n_classes)

    def __call__(self, *args):
        """
        Apply SoftMixup augmentation to batch data.

        Input formats supported:
        - (batch_data, batch_labels): Tensor, Tensor
        - ((batch_data, batch_labels)): Tuple of Tensors
        """
        # Case 1: Two tensors (batch_data, batch_labels)
        if len(args) == 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            return self.softmixup(args)

        # Case 2: Tuple format (batch_data, batch_labels)
        elif len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
            return self.softmixup(args[0])

        else:
            raise ValueError(f"Unsupported input format for SoftMixup: {args}. "
                             f"Expected (batch_data, batch_labels) or ((batch_data, batch_labels)).")

    # Expose utility methods
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return self.mixup_criterion(criterion, pred, y_a, y_b, lam)


class EntAugmentAugmentation:
    """
    EntAugment data augmentation method that randomly selects a transform
    operation and applies it at a specific intensity.
    """

    def __init__(self, **kwargs):
        """
        Initialize EntAugment with a specific intensity parameter

        Args:
            M (float): Intensity parameter in range [0,1]. Higher values produce stronger augmentation.
        """
        self.M = kwargs['entaugment_M']

        # Initialize all required components
        self._setup_transforms()

    def _setup_transforms(self):
        """Set up all transform functions and constants"""
        # Define constants
        self.PARAMETER_MAX = 30

        # Create MinMax dataclass
        self.MinMax = type('MinMax', (), {
            '__init__': lambda self, min_val, max_val: setattr(self, 'min', min_val) or setattr(self, 'max',
                                                                                                max_val)
        })

        # Create MinMaxVals
        self.min_max_vals = type('MinMaxVals', (), {
            'shear': self.MinMax(0.0, 0.3),
            'translate': self.MinMax(0, 10),
            'rotate': self.MinMax(0, 30),
            'solarize': self.MinMax(0, 256),
            'posterize': self.MinMax(0, 4),
            'enhancer': self.MinMax(0.1, 1.9),
            'cutout': self.MinMax(0.0, 0.2)
        })()

        # Define helper functions
        def float_parameter(level, maxval):
            return float(level) * maxval / self.PARAMETER_MAX

        def int_parameter(level, maxval):
            return int(level * maxval / self.PARAMETER_MAX)

        self.float_parameter = float_parameter
        self.int_parameter = int_parameter

        # Define transform classes
        self.TransformFunction = type('TransformFunction', (object,), {
            '__init__': lambda self, func, name: setattr(self, 'f', func) or setattr(self, 'name', name),
            '__repr__': lambda self: '<' + self.name + '>',
            '__call__': lambda self, pil_img: self.f(pil_img)
        })

        self.TransformT = type('TransformT', (object,), {
            '__init__': lambda self, name, xform_fn: setattr(self, 'name', name) or setattr(self, 'xform',
                                                                                            xform_fn),
            '__repr__': lambda self: '<' + self.name + '>',
            'pil_transformer': lambda self, probability, level: self._create_transform_function(self,
                                                                                                probability,
                                                                                                level)
        })

        # Add method to TransformT
        def _create_transform_function(self, probability, level):
            def return_function(im):
                if random.random() < probability:
                    im = self.xform(im, level)
                return im

            name = self.name + '({:.1f},{})'.format(probability, level)
            return self.TransformFunction(return_function, name)

        self.TransformT._create_transform_function = _create_transform_function

        # Define transform implementations
        def _rotate_impl(pil_img, level):
            degrees = int_parameter(level, self.min_max_vals.rotate.max)
            if random.random() > 0.5:
                degrees = -degrees
            return pil_img.rotate(degrees)

        def _posterize_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.posterize.max - self.min_max_vals.posterize.min)
            return ImageOps.posterize(pil_img, self.min_max_vals.posterize.max - level)

        def _solarize_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.solarize.max)
            return ImageOps.solarize(pil_img, 256 - level)

        def _shear_x_impl(pil_img, level):
            level = float_parameter(level, self.min_max_vals.shear.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

        def _shear_y_impl(pil_img, level):
            level = float_parameter(level, self.min_max_vals.shear.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

        def _translate_x_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.translate.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))

        def _translate_y_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.translate.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))

        def _enhancer_impl(enhancer):
            def impl(pil_img, level):
                v = float_parameter(level, 1.8) + .1
                return enhancer(pil_img).enhance(v)

            return impl

        # Create transform instances
        self.identity = self.TransformT('identity', lambda pil_img, level: pil_img)
        self.auto_contrast = self.TransformT('AutoContrast',
                                             lambda pil_img, level: ImageOps.autocontrast(pil_img))
        self.equalize = self.TransformT('Equalize', lambda pil_img, level: ImageOps.equalize(pil_img))
        self.rotate = self.TransformT('Rotate', _rotate_impl)
        self.solarize = self.TransformT('Solarize', _solarize_impl)
        self.color = self.TransformT('Color', _enhancer_impl(ImageEnhance.Color))
        self.posterize = self.TransformT('Posterize', _posterize_impl)
        self.contrast = self.TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
        self.brightness = self.TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
        self.sharpness = self.TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
        self.shear_x = self.TransformT('ShearX', _shear_x_impl)
        self.shear_y = self.TransformT('ShearY', _shear_y_impl)
        self.translate_x = self.TransformT('TranslateX', _translate_x_impl)
        self.translate_y = self.TransformT('TranslateY', _translate_y_impl)

        # Define transform list
        self.ALL_TRANSFORMS = [
            self.identity,
            self.auto_contrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.posterize,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y
        ]

    def __call__(self, img):
        """
        Apply randomly selected transform at specified intensity

        Args:
            img: Input image

        Returns:
            Augmented image
        """
        # Randomly select a transform operation
        op = random.choices(self.ALL_TRANSFORMS, k=1)[0]

        # Calculate level (intensity)
        level = min(int(self.PARAMETER_MAX * self.M) + 1, self.PARAMETER_MAX)

        # Apply transform
        img = op.pil_transformer(1., level)(img)
        return img


class MicrobialDataAugmentation:

    def __init__(self, **kwargs):
        """
        初始化细菌图像增强器

        Args:
            M (float): 增强强度参数，范围[0,1]，数值越大增强效果越强
            expansion_factor (int): 数据集扩展倍数
            online (bool): 是否在线增强（训练过程中动态生成）
        """
        self.M = kwargs['MicrobialDataAug_M']
        self.expansion_factor = kwargs['expansion_factor']
        self.online = kwargs['online']

        # 初始化所有必要组件
        self._setup_transforms()

        # 追踪已用于验证集的增强，防止数据泄露
        self.validation_augmentation_record = {}

    def _setup_transforms(self):
        """设置所有转换函数和常量"""
        # 定义常量
        self.PARAMETER_MAX = 30

        # 创建MinMax数据类
        self.MinMax = type('MinMax', (), {
            '__init__': lambda self, min_val, max_val: setattr(self, 'min', min_val) or setattr(self, 'max', max_val)
        })

        # 创建MinMaxVals - 调整后对细菌图像更合适的参数
        self.min_max_vals = type('MinMaxVals', (), {
            'shear': self.MinMax(0.0, 0.2),  # 降低剪切最大值，避免过度变形
            'translate': self.MinMax(0, 8),  # 调整平移限制
            'rotate': self.MinMax(0, 360),  # 细菌方向无关，允许完整旋转
            'solarize': self.MinMax(0, 256),
            'posterize': self.MinMax(2, 6),  # 调整posterize参数保留更多细节
            'enhancer': self.MinMax(0.5, 1.5),  # 降低增强器极值，避免过度处理
            'cutout': self.MinMax(0.0, 0.15),  # 减小cutout最大值，保留更多细节
            'zoom': self.MinMax(0.8, 1.2),  # 定义缩放范围
            'blur': self.MinMax(0.5, 1.5),  # 模糊参数
            'elastic': self.MinMax(0, 5),  # 弹性变形参数
            'clahe': self.MinMax(1.0, 4.0)  # CLAHE参数
        })()

        # 定义辅助函数
        def float_parameter(level, maxval):
            return float(level) * maxval / self.PARAMETER_MAX

        def int_parameter(level, maxval):
            return int(level * maxval / self.PARAMETER_MAX)

        self.float_parameter = float_parameter
        self.int_parameter = int_parameter

        # 定义转换类
        self.TransformFunction = type('TransformFunction', (object,), {
            '__init__': lambda self, func, name: setattr(self, 'f', func) or setattr(self, 'name', name),
            '__repr__': lambda self: '<' + self.name + '>',
            '__call__': lambda self, pil_img: self.f(pil_img)
        })

        self.TransformT = type('TransformT', (object,), {
            '__init__': lambda self, name, xform_fn: setattr(self, 'name', name) or setattr(self, 'xform', xform_fn),
            '__repr__': lambda self: '<' + self.name + '>',
            'pil_transformer': lambda self, probability, level: self._create_transform_function(self, probability,
                                                                                                level)
        })

        # 为TransformT添加方法
        def _create_transform_function(self, probability, level):
            def return_function(im):
                if random.random() < probability:
                    im = self.xform(im, level)
                return im

            name = self.name + '({:.1f},{})'.format(probability, level)
            return self.TransformFunction(return_function, name)

        self.TransformT._create_transform_function = _create_transform_function

        # 定义特定于细菌图像的转换实现

        # 1. 基础转换实现
        def _rotate_impl(pil_img, level):
            # 细菌图像可以任意角度旋转，没有"正确"的方向
            degrees = int_parameter(level, self.min_max_vals.rotate.max)
            return pil_img.rotate(degrees)

        def _posterize_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.posterize.max - self.min_max_vals.posterize.min)
            level = max(self.min_max_vals.posterize.min, self.min_max_vals.posterize.max - level)
            return ImageOps.posterize(pil_img, level)

        def _solarize_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.solarize.max)
            return ImageOps.solarize(pil_img, 256 - level)

        def _shear_x_impl(pil_img, level):
            level = float_parameter(level, self.min_max_vals.shear.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

        def _shear_y_impl(pil_img, level):
            level = float_parameter(level, self.min_max_vals.shear.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

        def _translate_x_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.translate.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))

        def _translate_y_impl(pil_img, level):
            level = int_parameter(level, self.min_max_vals.translate.max)
            if random.random() > 0.5:
                level = -level
            return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))

        def _enhancer_impl(enhancer):
            def impl(pil_img, level):
                v = float_parameter(level,
                                    self.min_max_vals.enhancer.max - self.min_max_vals.enhancer.min) + self.min_max_vals.enhancer.min
                return enhancer(pil_img).enhance(v)

            return impl

        # 2. 细菌图像特定增强方法

        # 随机缩放 - 细菌大小可能有变化
        def _zoom_impl(pil_img, level):
            zoom_factor = float_parameter(level,
                                          self.min_max_vals.zoom.max - self.min_max_vals.zoom.min) + self.min_max_vals.zoom.min

            # 获取原始尺寸
            width, height = pil_img.size

            # 计算新尺寸
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)

            # 调整大小
            resized_img = pil_img.resize((new_width, new_height), Image.BICUBIC)

            # 居中裁剪到原始尺寸
            left = max(0, (new_width - width) // 2)
            top = max(0, (new_height - height) // 2)
            right = min(new_width, left + width)
            bottom = min(new_height, top + height)

            if new_width < width or new_height < height:
                # 如果缩小了，我们需要创建一个新的大小为原始大小的图像
                new_img = Image.new(pil_img.mode, (width, height), (0, 0, 0))
                paste_left = max(0, (width - new_width) // 2)
                paste_top = max(0, (height - new_height) // 2)
                new_img.paste(resized_img, (paste_left, paste_top))
                return new_img
            else:
                # 如果放大了，我们裁剪到原始大小
                return resized_img.crop((left, top, right, bottom))

        # 添加高斯噪声 - 模拟显微镜成像噪声
        def _gaussian_noise_impl(pil_img, level):
            np_img = np.array(pil_img)
            noise_level = float_parameter(level, 0.1)  # 最大噪声水平为0.1

            # 添加高斯噪声
            noisy_img = random_noise(np_img, mode='gaussian', var=noise_level)

            # 将浮点数图像转换回uint8
            noisy_img = (noisy_img * 255).astype(np.uint8)

            return Image.fromarray(noisy_img)

        # 局部对比度增强 - 强调细菌细节
        def _clahe_impl(pil_img, level):
            np_img = np.array(pil_img)

            # 如果是RGB图像，转换为LAB并只增强L通道
            if len(np_img.shape) == 3 and np_img.shape[2] == 3:
                lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # 应用CLAHE
                clahe = cv2.createCLAHE(
                    clipLimit=float_parameter(level, self.min_max_vals.clahe.max),
                    tileGridSize=(8, 8)
                )
                cl = clahe.apply(l)

                # 合并通道
                merged = cv2.merge((cl, a, b))
                enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            else:
                # 灰度图像直接应用CLAHE
                clahe = cv2.createCLAHE(
                    clipLimit=float_parameter(level, self.min_max_vals.clahe.max),
                    tileGridSize=(8, 8)
                )
                enhanced_img = clahe.apply(np_img)

            return Image.fromarray(enhanced_img)

        # 弹性变形 - 模拟细菌形态略微变化
        def _elastic_transform_impl(pil_img, level):
            np_img = np.array(pil_img)

            # 参数
            alpha = float_parameter(level, self.min_max_vals.elastic.max)
            sigma = 4  # 控制变形平滑度

            # 创建随机位移场
            shape = np_img.shape[:2]
            dx = np.random.rand(*shape) * 2 - 1
            dy = np.random.rand(*shape) * 2 - 1

            # 高斯滤波使变形更平滑
            dx = ndimage.gaussian_filter(dx, sigma) * alpha
            dy = ndimage.gaussian_filter(dy, sigma) * alpha

            # 创建网格
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            # 执行变形
            if len(np_img.shape) == 3:
                # 彩色图像
                transformed = np.zeros_like(np_img)
                for i in range(np_img.shape[2]):
                    transformed[:, :, i] = ndimage.map_coordinates(np_img[:, :, i], indices, order=1).reshape(shape)
            else:
                # 灰度图像
                transformed = ndimage.map_coordinates(np_img, indices, order=1).reshape(shape)

            return Image.fromarray(transformed)

        # 添加模糊 - 模拟不同聚焦深度
        def _blur_impl(pil_img, level):
            radius = float_parameter(level, self.min_max_vals.blur.max)
            return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 随机裁剪 - 模拟部分视野
        def _cutout_impl(pil_img, level):
            width, height = pil_img.size
            cutout_size = int(float_parameter(level, self.min_max_vals.cutout.max) * min(width, height))

            # 创建裁剪区域
            x0 = random.randint(0, width - cutout_size)
            y0 = random.randint(0, height - cutout_size)
            x1 = x0 + cutout_size
            y1 = y0 + cutout_size

            # 创建裁剪掩码
            img_arr = np.array(pil_img)
            if len(img_arr.shape) == 3:
                # 彩色图像
                img_arr[y0:y1, x0:x1, :] = 0
            else:
                # 灰度图像
                img_arr[y0:y1, x0:x1] = 0

            return Image.fromarray(img_arr)

        # 随机照明变化 - 模拟不同光照条件
        def _lighting_impl(pil_img, level):
            np_img = np.array(pil_img)

            # 根据level决定是变亮还是变暗
            if random.random() > 0.5:
                # 增加曝光 - gamma < 1 使图像更亮
                gamma = 1.0 - float_parameter(level, 0.5)  # 范围 [0.5, 1.0]
            else:
                # 减少曝光 - gamma > 1 使图像更暗
                gamma = 1.0 + float_parameter(level, 1.0)  # 范围 [1.0, 2.0]

            # 应用伽马校正
            adjusted = exposure.adjust_gamma(np_img, gamma)
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

            return Image.fromarray(adjusted)

        # 创建转换实例
        self.identity = self.TransformT('identity', lambda pil_img, level: pil_img)
        self.auto_contrast = self.TransformT('AutoContrast', lambda pil_img, level: ImageOps.autocontrast(pil_img))
        self.equalize = self.TransformT('Equalize', lambda pil_img, level: ImageOps.equalize(pil_img))
        self.rotate = self.TransformT('Rotate', _rotate_impl)
        self.solarize = self.TransformT('Solarize', _solarize_impl)
        self.color = self.TransformT('Color', _enhancer_impl(ImageEnhance.Color))
        self.posterize = self.TransformT('Posterize', _posterize_impl)
        self.contrast = self.TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
        self.brightness = self.TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
        self.sharpness = self.TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
        self.shear_x = self.TransformT('ShearX', _shear_x_impl)
        self.shear_y = self.TransformT('ShearY', _shear_y_impl)
        self.translate_x = self.TransformT('TranslateX', _translate_x_impl)
        self.translate_y = self.TransformT('TranslateY', _translate_y_impl)

        # 为细菌特定增强添加转换
        self.zoom = self.TransformT('Zoom', _zoom_impl)
        self.gaussian_noise = self.TransformT('GaussianNoise', _gaussian_noise_impl)
        self.clahe = self.TransformT('CLAHE', _clahe_impl)
        self.elastic = self.TransformT('ElasticTransform', _elastic_transform_impl)
        self.blur = self.TransformT('Blur', _blur_impl)
        self.cutout = self.TransformT('Cutout', _cutout_impl)
        self.lighting = self.TransformT('Lighting', _lighting_impl)

        # 定义转换列表
        self.ALL_TRANSFORMS = [
            # 基础变换
            self.auto_contrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.posterize,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,

            # 细菌特定变换
            self.zoom,
            self.gaussian_noise,
            self.clahe,
            self.elastic,
            self.blur,
            self.cutout,
            self.lighting
        ]

        # 细菌形态变换 - 更适合细菌形态学特征的变换组合
        self.MORPHOLOGY_TRANSFORMS = [
            self.zoom,
            self.rotate,
            self.shear_x,
            self.shear_y,
            self.elastic,
            self.translate_x,
            self.translate_y
        ]

        # 细菌颜色和纹理变换 - 更适合细菌颜色和纹理特征的变换组合
        self.COLOR_TEXTURE_TRANSFORMS = [
            self.auto_contrast,
            self.equalize,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.solarize,
            self.posterize,
            self.clahe,
            self.lighting
        ]

        # 细菌成像变换 - 模拟显微镜成像变化的变换
        self.IMAGING_TRANSFORMS = [
            self.gaussian_noise,
            self.blur,
            self.cutout,
            self.lighting
        ]

    def _generate_transform_sequence(self):
        """
        生成一个变换序列，包含形态学、颜色/纹理和成像特性的组合
        """
        transforms_to_apply = []

        # 从每个类别中选择一个变换
        if random.random() < 0.9:  # 90%概率应用形态学变换
            transforms_to_apply.append(random.choice(self.MORPHOLOGY_TRANSFORMS))

        if random.random() < 0.8:  # 80%概率应用颜色/纹理变换
            transforms_to_apply.append(random.choice(self.COLOR_TEXTURE_TRANSFORMS))

        if random.random() < 0.7:  # 70%概率应用成像变换
            transforms_to_apply.append(random.choice(self.IMAGING_TRANSFORMS))

        # 确保至少应用一个变换
        if not transforms_to_apply:
            transforms_to_apply.append(random.choice(self.ALL_TRANSFORMS))

        return transforms_to_apply

    def __call__(self, img, fold_id=None, image_id=None):
        """
        应用随机选择的转换序列，强度由M参数控制

        Args:
            img: 输入图像
            fold_id: 当前交叉验证折编号，用于防止数据泄露
            image_id: 图像的唯一标识符，用于防止数据泄露

        Returns:
            增强的图像
        """
        if fold_id is not None and image_id is not None:
            # 在线模式下，我们需要确保验证集中使用的图像增强与训练集不同
            # 这样可以防止模型"记住"特定的增强，而不是学习真正的特征
            key = f"{fold_id}_{image_id}"

            if key in self.validation_augmentation_record:
                # 对于验证集，始终使用相同的随机种子，确保同一图像在不同epoch有相同的增强
                random.seed(self.validation_augmentation_record[key])
            else:
                # 如果是第一次遇到这个验证图像，生成一个新的随机种子
                seed = random.randint(0, 100000)
                self.validation_augmentation_record[key] = seed
                random.seed(seed)

        # 计算level（强度）
        level = min(int(self.PARAMETER_MAX * self.M) + 1, self.PARAMETER_MAX)

        # 应用变换序列
        transforms_to_apply = self._generate_transform_sequence()

        for transform in transforms_to_apply:
            img = transform.pil_transformer(1.0, level)(img)

        # 重置随机种子
        if fold_id is not None and image_id is not None:
            random.seed()

        return img
    
class DataAugmentationManager:
    def __init__(self):
        self.aug_manager = DataAugmentationManager()
        self.aug_manager.register_augmentation('cutmix', CutMixAugmentation)
        self.aug_manager.register_augmentation('cutout', CutoutAugmentation)
        self.aug_manager.register_augmentation('has', HaSAugmentation)
        self.aug_manager.register_augmentation('grid', GridMaskAugmentation)
        self.aug_manager.register_augmentation('random_erasing', RandomErasingAugmentation)
        self.aug_manager.register_augmentation('autoaugment', AutoAugmentAugmentation)
        self.aug_manager.register_augmentation('trivialaugment', TrivialAugmentAugmentation)
        self.aug_manager.register_augmentation('softaug', SoftAugAugmentation)
        self.aug_manager.register_augmentation('entaugment', EntAugmentAugmentation)
        self.aug_manager.register_augmentation('MicrobialDataAug', MicrobialDataAugmentation)

    def get_augmentation(self, name, **kwargs):
        return self.aug_manager.get_augmentation(name, **kwargs)

    def register_augmentation(self, name, augmentation_class):
        self.aug_manager.register_augmentation(name, augmentation_class)

