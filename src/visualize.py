import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .dataset import SimhasthaDataset
from .evaluate import load_checkpoint
from .model import CSRNet, get_device, clear_device_cache
from .utils import list_image_files, ensure_dir


def _tensor_to_display(img_tensor) -> np.ndarray:
    img = img_tensor.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std) + mean
    return np.clip(img, 0, 1)


def generate_previews(data_dir: str = 'data', output_dir: str = 'previews') -> None:
    ensure_dir(output_dir)
    print('Step 3: Generating previews')
    for split in ['Train', 'Test']:
        img_dir = os.path.join(data_dir, split, 'images')
        heat_dir = os.path.join(data_dir, split, 'heatmaps')
        for npy_file in sorted([f for f in os.listdir(heat_dir) if f.endswith('.npy')]):
            base = os.path.splitext(npy_file)[0]
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = os.path.join(img_dir, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            with Image.open(img_path) as pil_img:
                img = np.array(pil_img.convert('RGB'))
            density_map = np.load(os.path.join(heat_dir, npy_file))
            fig, axes = plt.subplots(1, 2, figsize=(15, 7))
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(img)
            axes[1].imshow(density_map, cmap='jet', alpha=0.55)
            axes[1].set_title(f'Density Overlay (count={density_map.sum():.1f})')
            axes[1].axis('off')
            sm = cm.ScalarMappable(cmap='jet')
            sm.set_array(density_map)
            fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
            filename = f'{split}_{base}_preview.png'
            fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f'  Saved {filename}')


def visualize_results(
    model_path: str = 'checkpoints/satark_best.pth',
    data_root: str = 'data',
    output_dir: str = 'previews',
    all_images: bool = True,
) -> None:
    ensure_dir(output_dir)
    device = get_device()
    clear_device_cache(device)
    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    if not load_checkpoint(model_path, model, device):
        return

    dataset = SimhasthaDataset(root_dir=data_root, split='Test')
    if not dataset:
        print('No test images found.')
        return

    indices = list(range(len(dataset))) if all_images else [max(range(len(dataset)), key=lambda i: float(dataset[i][1].sum()))]
    for idx in indices:
        img_tensor, target = dataset[idx]
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
        output_np = output.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()
        img_disp = _tensor_to_display(img_tensor)
        gt_count = float(target_np.sum())
        pred_count = float(output_np.sum())
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        axes[0].imshow(img_disp)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        im1 = axes[1].imshow(target_np, cmap='jet')
        axes[1].set_title(f'GT count={gt_count:.0f}')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(output_np, cmap='jet', vmin=0, vmax=max(target_np.max(), output_np.max(), 1.0))
        axes[2].set_title(f'Pred count={pred_count:.0f}')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        filename = f'satark_visual_{idx}.png'
        fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'  Saved {filename} (GT={gt_count:.0f}, Pred={pred_count:.0f})')
