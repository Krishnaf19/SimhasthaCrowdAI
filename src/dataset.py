import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F

from .utils import list_image_files


class EnhancedAugmentationPipeline:
    """Enhanced augmentation pipeline for diverse cultural headgear detection.
    
    Includes augmentations specifically designed to handle:
    - Turbans and saffron headwear
    - Veils and cloth coverings
    - Various caps and traditional headgear
    - Occlusion and partial visibility
    """
    def __init__(self, jitter: float = 0.3, rotation: int = 15):
        self.jitter = jitter
        self.rotation = rotation
        self.color_jitter = transforms.ColorJitter(
            brightness=jitter,
            contrast=jitter,
            saturation=jitter,
            hue=jitter * 0.15
        )
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    
    def apply_augmentation(self, img: Image.Image, target: torch.Tensor, is_train: bool = True):
        """Apply augmentations to handle diverse headgear scenarios."""
        if not is_train:
            return img, target
        
        # Random color jitter (handles lighting variations in religious ceremonies)
        if random.random() > 0.3:
            img = self.color_jitter(img)
        
        # Random gaussian blur (simulates motion in crowded scenes)
        if random.random() > 0.6:
            img = self.gaussian_blur(img)
        
        # Random rotation (handles varying head orientations)
        if random.random() > 0.4:
            angle = random.uniform(-self.rotation, self.rotation)
            img = TF.rotate(img, angle, expand=False)
            target = torch.from_numpy(
                np.rot90(target.numpy(), k=int(angle // 90) if angle % 90 < 45 else 0)
            ).float()
        
        # Random horizontal flip (symmetry for head detection)
        if random.random() > 0.5:
            img = TF.hflip(img)
            target = torch.flip(target, dims=[-1])
        
        # Random vertical flip (for turbans and tall headgear)
        if random.random() > 0.6:
            img = TF.vflip(img)
            target = torch.flip(target, dims=[-2])
        
        # Random brightness adjustment (ceremonies at different times)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # Random contrast adjustment (fabric texture variation)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.3)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        return img, target


class SimhasthaDataset(Dataset):
    def __init__(
        self,
        root_dir: str = 'data',
        split: str = 'Train',
        crop_size: int = 512,
        downsample: int = 8,
        jitter: float = 0.3,
        rotation: int = 15,
        enable_advanced_augmentation: bool = True,
    ):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.downsample = downsample
        self.jitter = jitter
        self.rotation = rotation
        self.enable_advanced_augmentation = enable_advanced_augmentation

        self.samples = []

        self.splits = ['all'] if split == 'all' else ['Train'] if split == 'Train' else ['Test'] if split == 'Test' else [split]

        for subset in self.splits:
            if subset == 'all':
                images_dir = os.path.join(root_dir, 'images')
                heatmaps_dir = os.path.join(root_dir, 'heatmaps')
            else:
                images_dir = os.path.join(root_dir, subset, 'images')
                heatmaps_dir = os.path.join(root_dir, subset, 'heatmaps')

            if not os.path.exists(images_dir):
                raise FileNotFoundError(f"Images directory not found: {images_dir}")
            if not os.path.exists(heatmaps_dir):
                raise FileNotFoundError(f"Heatmaps directory not found: {heatmaps_dir}")

            image_files = list_image_files(images_dir)
            if not image_files:
                raise RuntimeError(f"No images found in: {images_dir}")

            for img_name in image_files:
                base_name = os.path.splitext(img_name)[0]
                heat_path = os.path.join(heatmaps_dir, base_name + '.npy')
                if os.path.exists(heat_path):
                    self.samples.append((img_name, os.path.join(images_dir, img_name), heat_path))
        
        # Initialize augmentation pipeline
        self.augmentation_pipeline = EnhancedAugmentationPipeline(jitter=jitter, rotation=rotation)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_heatmap(self, heat_path: str) -> torch.Tensor:
        target = np.load(heat_path).astype(np.float32)
        target = torch.from_numpy(target)
        if target.dim() == 2:
            target = target.unsqueeze(0)
        return target

    def _pad_if_needed(self, img: Image.Image, target: torch.Tensor):
        w, h = img.size
        pad_right = max(0, self.crop_size - w)
        pad_bottom = max(0, self.crop_size - h)
        if pad_right or pad_bottom:
            img = TF.pad(img, [0, 0, pad_right, pad_bottom], fill=0)
            target = TF.pad(target, [0, 0, pad_right, pad_bottom], fill=0)
        return img, target

    def _apply_train_augmentation(self, img: Image.Image, target: torch.Tensor):
        """Apply enhanced augmentation pipeline for diverse headgear."""
        if self.enable_advanced_augmentation:
            img, target = self.augmentation_pipeline.apply_augmentation(img, target, is_train=True)
        else:
            # Fallback to basic augmentation
            if random.random() > 0.5:
                img = self.color_jitter(img)
            angle = random.uniform(-self.rotation, self.rotation)
            if random.random() > 0.5:
                img = TF.rotate(img, angle, interpolation=Image.BILINEAR, fill=0)
                target = TF.rotate(target, angle, interpolation=Image.BILINEAR, fill=0)
        
        return img, target

    def _downsample_target(self, target: torch.Tensor, canvas_h: int, canvas_w: int) -> torch.Tensor:
        original_count = target.sum()
        out_h = max(canvas_h // self.downsample, 1)
        out_w = max(canvas_w // self.downsample, 1)

        target = F.interpolate(
            target.unsqueeze(0),
            size=(out_h, out_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        if target.sum() > 0 and original_count > 0:
            target *= original_count / target.sum()
        return target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name, img_path, heat_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        target = self._load_heatmap(heat_path)

        if self.split == 'Train':
            img, target = self._pad_if_needed(img, target)
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size)
            )
            img = TF.crop(img, i, j, h_crop, w_crop)
            target = TF.crop(target, i, j, h_crop, w_crop)
            if random.random() > 0.5:
                img = TF.hflip(img)
                target = TF.hflip(target)
            img, target = self._apply_train_augmentation(img, target)
            canvas_h, canvas_w = self.crop_size, self.crop_size
        else:
            img = TF.resize(img, [self.crop_size, self.crop_size])
            original_count = target.sum()
            target = F.interpolate(
                target.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            if target.sum() > 0 and original_count > 0:
                target *= original_count / target.sum()
            canvas_h, canvas_w = self.crop_size, self.crop_size

        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        target = self._downsample_target(target, canvas_h, canvas_w)
        return img_tensor, target
