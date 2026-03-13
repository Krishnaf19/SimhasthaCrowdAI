# import os
# import torch
# import numpy as np
# import random
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# import torch.nn.functional as F

# class SimhasthaDataset(Dataset):
#     def __init__(self, root_dir, split='Train', crop_size=512, downsample=8):
#         self.root_dir = root_dir
#         self.split = split
#         self.crop_size = crop_size
#         self.downsample = downsample
        
#         self.images_dir = os.path.join(root_dir, split, 'images')
#         self.heatmaps_dir = os.path.join(root_dir, split, 'heatmaps')
        
#         if not os.path.exists(self.images_dir):
#             raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            
#         self.image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         img = Image.open(img_path).convert('RGB')
        
#         # 1. Locate Heatmap
#         base_name = os.path.splitext(img_name)[0]
#         gt_path = os.path.join(self.heatmaps_dir, base_name + ".npy")
        
#         # Fallback for name variations
#         if not os.path.exists(gt_path):
#             gt_path = os.path.join(self.heatmaps_dir, img_name + ".npy")

#         target = np.load(gt_path)
#         target = torch.from_numpy(target).float()
#         if len(target.shape) == 2:
#             target = target.unsqueeze(0)

#         # 2. Synchronized Augmentation
#         if self.split == 'Train':
#             w, h = img.size
#             # Padding if image is smaller than crop size
#             if w < self.crop_size or h < self.crop_size:
#                 pad_w = max(0, self.crop_size - w)
#                 pad_h = max(0, self.crop_size - h)
#                 img = TF.pad(img, (0, 0, pad_w, pad_h))
#                 target = TF.pad(target, (0, 0, pad_w, pad_h))

#             # CORRECT: Use transforms.RandomCrop to get the coordinates
#             i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
#                 img, output_size=(self.crop_size, self.crop_size)
#             )
            
#             img = TF.crop(img, i, j, h_crop, w_crop)
#             target = TF.crop(target, i, j, h_crop, w_crop)
            
#             if random.random() > 0.5:
#                 img = TF.hflip(img)
#                 target = TF.hflip(target)

#         # 3. Final Preprocessing
#         img_tensor = TF.to_tensor(img)
#         img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
#         # 4. Downsample target (e.g., 512 -> 64)
#         original_count = target.sum()
#         output_size = self.crop_size // self.downsample
        
#         target = F.interpolate(target.unsqueeze(0), size=(output_size, output_size), mode='bilinear', align_corners=False).squeeze(0)
        
#         # Re-normalize to keep the crowd count accurate
#         if target.sum() > 0:
#             target = target * (original_count / target.sum())
        
#         return img_tensor, target

import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.nn.functional as F


class SimhasthaDataset(Dataset):
    def __init__(
        self,
        root_dir:   str = 'data',
        split:      str = 'Train',
        crop_size:  int = 512,
        downsample: int = 8,
    ):
        self.root_dir  = root_dir
        self.split     = split
        self.crop_size = crop_size
        self.downsample = downsample

        self.images_dir   = os.path.join(root_dir, split, 'images')
        self.heatmaps_dir = os.path.join(root_dir, split, 'heatmaps')

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.heatmaps_dir):
            raise FileNotFoundError(f"Heatmaps directory not found: {self.heatmaps_dir}")

        # FIX 6: Sort for deterministic ordering across runs
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in: {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_heatmap(self, img_name: str) -> torch.Tensor:
        """FIX 1 & 5: Clean heatmap loading with proper fallback and clear error."""
        base_name = os.path.splitext(img_name)[0]
        gt_path   = os.path.join(self.heatmaps_dir, base_name + '.npy')

        if not os.path.exists(gt_path):
            raise FileNotFoundError(
                f"Heatmap not found for '{img_name}'.\n"
                f"  Expected: {gt_path}\n"
                f"  Run step2_generate_data.py to regenerate heatmaps."
            )

        target = np.load(gt_path).astype(np.float32)
        target = torch.from_numpy(target)               # shape: (H, W)

        if target.dim() == 2:
            target = target.unsqueeze(0)                # → (1, H, W)

        return target

    def _pad_if_needed(self, img: Image.Image, target: torch.Tensor):
        """
        FIX 2: Correct synchronized padding for PIL image + (1,H,W) tensor.
        TF.pad on a tensor takes (left, top, right, bottom) just like PIL,
        but we need to apply the same padding to both so they stay aligned.
        """
        w, h = img.size                                 # PIL: (width, height)
        pad_right  = max(0, self.crop_size - w)
        pad_bottom = max(0, self.crop_size - h)

        if pad_right > 0 or pad_bottom > 0:
            # PIL padding: (left, top, right, bottom)
            img    = TF.pad(img,    [0, 0, pad_right, pad_bottom], fill=0)
            # Tensor padding: same convention when target is (C, H, W)
            target = TF.pad(target, [0, 0, pad_right, pad_bottom], fill=0)

        return img, target

    def _downsample_target(self, target: torch.Tensor, canvas_h: int, canvas_w: int) -> torch.Tensor:
        """
        FIX 3: Compute output size from actual canvas dimensions, not
        the fixed crop_size — so Test images of any size work correctly.
        Count is preserved via rescaling after interpolation.
        """
        original_count = target.sum()

        out_h = canvas_h // self.downsample
        out_w = canvas_w // self.downsample

        # Ensure at least 1×1
        out_h = max(out_h, 1)
        out_w = max(out_w, 1)

        target = F.interpolate(
            target.unsqueeze(0),                        # (1, 1, H, W)
            size=(out_h, out_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)                                    # → (1, out_h, out_w)

        # Re-normalise to preserve count
        t_sum = target.sum()
        if t_sum > 0 and original_count > 0:
            target = target * (original_count / t_sum)

        return target

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        img    = Image.open(img_path).convert('RGB')
        target = self._load_heatmap(img_name)           # (1, H, W)

        # ── Training augmentations ────────────────────────────────────────────
        if self.split == 'Train':
            # FIX 2: Correct padding
            img, target = self._pad_if_needed(img, target)

            # Synchronized random crop
            i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size)
            )
            img    = TF.crop(img,    i, j, h_crop, w_crop)
            target = TF.crop(target, i, j, h_crop, w_crop)

            # Synchronized horizontal flip
            if random.random() > 0.5:
                img    = TF.hflip(img)
                target = TF.hflip(target)

            canvas_h, canvas_w = self.crop_size, self.crop_size

        else:
            # FIX 4: Resize test images to a fixed size so DataLoader
            # can batch them. Use crop_size as the canonical eval resolution.
            img    = TF.resize(img, [self.crop_size, self.crop_size])

            # Resize density map to match — count-preserving
            original_count = target.sum()
            target = F.interpolate(
                target.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            t_sum = target.sum()
            if t_sum > 0 and original_count > 0:
                target = target * (original_count / t_sum)

            canvas_h, canvas_w = self.crop_size, self.crop_size

        # ── Normalise image ───────────────────────────────────────────────────
        img_tensor = TF.to_tensor(img)                  # [0, 1], shape (3, H, W)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # ── Downsample density map to model output resolution ─────────────────
        # FIX 3: Use actual canvas dimensions
        target = self._downsample_target(target, canvas_h, canvas_w)

        return img_tensor, target