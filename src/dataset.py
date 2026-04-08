import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F

from .utils import list_image_files


class SimhasthaDataset(Dataset):
    def __init__(
        self,
        root_dir: str = 'data',
        split: str = 'Train',
        crop_size: int = 512,
        downsample: int = 8,
        jitter: float = 0.2,
        rotation: int = 10,
    ):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.downsample = downsample
        self.jitter = jitter
        self.rotation = rotation

        self.images_dir = os.path.join(root_dir, split, 'images')
        self.heatmaps_dir = os.path.join(root_dir, split, 'heatmaps')

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.heatmaps_dir):
            raise FileNotFoundError(f"Heatmaps directory not found: {self.heatmaps_dir}")

        self.image_files: List[str] = list_image_files(self.images_dir)
        if not self.image_files:
            raise RuntimeError(f"No images found in: {self.images_dir}")

        self.color_jitter = transforms.ColorJitter(
            brightness=self.jitter,
            contrast=self.jitter,
            saturation=self.jitter,
            hue=min(self.jitter / 2, 0.1),
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_heatmap(self, img_name: str) -> torch.Tensor:
        base_name = os.path.splitext(img_name)[0]
        heat_path = os.path.join(self.heatmaps_dir, base_name + '.npy')
        if not os.path.exists(heat_path):
            raise FileNotFoundError(f"Heatmap not found for '{img_name}' at '{heat_path}'")

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
        if random.random() > 0.5:
            img = self.color_jitter(img)

        if random.random() > 0.5:
            angle = random.uniform(-self.rotation, self.rotation)
            img = TF.rotate(img, angle, resample=Image.BILINEAR, fill=0)
            target = TF.rotate(target, angle, resample=Image.BILINEAR, fill=0)

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
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        target = self._load_heatmap(img_name)

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
