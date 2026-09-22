import os, random
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from ..utils.common import CLASSES, list_image_files

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class SimhasthaDataset(Dataset):
    def __init__(self, root_dir='data', split='train', crop_size=512, downsample=8, single_channel=False):
        self.root_dir   = root_dir
        self.split      = split
        self.crop_size  = crop_size
        self.downsample = downsample
        self.single_channel = single_channel
        self.color_jitter  = T.ColorJitter(0.3, 0.3, 0.3, 0.05)
        self.gaussian_blur = T.GaussianBlur(3, sigma=(0.1, 2.0))
        self.samples = []
        subsets = ['processed'] if split == 'all' else [split]
        for subset in subsets:
            if subset == 'processed':
                img_dir  = os.path.join(root_dir, 'processed', 'images')
                heat_dir = os.path.join(root_dir, 'processed', 'heatmaps')
            else:
                img_dir  = os.path.join(root_dir, 'splits', subset, 'images')
                heat_dir = os.path.join(root_dir, 'splits', subset, 'heatmaps')
            if not os.path.exists(img_dir):
                continue
            for img_name in list_image_files(img_dir):
                stem = os.path.splitext(img_name)[0]
                if os.path.exists(os.path.join(heat_dir, stem + '.npy')):
                    self.samples.append((os.path.join(img_dir, img_name), heat_dir, stem))
        if not self.samples:
            raise RuntimeError('No samples found for split=' + split + ' in ' + root_dir)

    def __len__(self):
        return len(self.samples)

    def _load_target(self, heat_dir, stem):
        combined_path = os.path.join(heat_dir, stem + '.npy')
        combined = np.load(combined_path).astype(np.float32)
        if self.single_channel:
            return torch.from_numpy(combined).unsqueeze(0)
        maps = []
        for cls in CLASSES:
            p = os.path.join(heat_dir, stem + '_' + cls + '.npy')
            maps.append(torch.from_numpy(np.load(p).astype(np.float32)) if os.path.exists(p)
                        else torch.zeros_like(torch.from_numpy(combined)))
        try:
            return torch.stack(maps, dim=0)
        except RuntimeError:
            c = torch.from_numpy(combined).unsqueeze(0)
            z = torch.zeros_like(c)
            return torch.cat([c, z, z, z], dim=0)

    def _downsample_target(self, target, h, w):
        orig = target.sum()
        out_h, out_w = max(h // self.downsample, 1), max(w // self.downsample, 1)
        target = F.interpolate(target.unsqueeze(0), size=(out_h, out_w),
                               mode='bilinear', align_corners=False).squeeze(0)
        if target.sum() > 0 and orig > 0:
            target = target * (orig / target.sum())
        return target

    def __getitem__(self, idx):
        img_path, heat_dir, stem = self.samples[idx]
        img    = Image.open(img_path).convert('RGB')
        target = self._load_target(heat_dir, stem)
        is_train = self.split in ('train', 'all')
        if is_train:
            w, h = img.size
            pr, pb = max(0, self.crop_size - w), max(0, self.crop_size - h)
            if pr or pb:
                img = TF.pad(img, [0, 0, pr, pb])
                target = TF.pad(target, [0, 0, pr, pb])
            i, j, ch, cw = T.RandomCrop.get_params(img, (self.crop_size, self.crop_size))
            img, target = TF.crop(img, i, j, ch, cw), TF.crop(target, i, j, ch, cw)
            if random.random() > 0.5:
                img, target = TF.hflip(img), TF.hflip(target)
            if random.random() > 0.3:
                img = self.color_jitter(img)
            if random.random() > 0.6:
                img = self.gaussian_blur(img)
            canvas_h, canvas_w = self.crop_size, self.crop_size
        else:
            img = TF.resize(img, [self.crop_size, self.crop_size])
            orig = target.sum()
            target = F.interpolate(target.unsqueeze(0),
                                   size=(self.crop_size, self.crop_size),
                                   mode='bilinear', align_corners=False).squeeze(0)
            if target.sum() > 0 and orig > 0:
                target = target * (orig / target.sum())
            canvas_h, canvas_w = self.crop_size, self.crop_size
        img_tensor = TF.normalize(TF.to_tensor(img), MEAN, STD)
        target     = self._downsample_target(target, canvas_h, canvas_w)
        return img_tensor, target
