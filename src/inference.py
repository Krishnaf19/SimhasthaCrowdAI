import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .evaluate import load_checkpoint
from .model import CSRNet, get_device, clear_device_cache
from .utils import IMAGE_EXTENSIONS, ensure_dir


MAX_SIDE_PX = 1000
DANGER_THRESHOLD = 250
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resize_for_inference(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_SIDE_PX:
        return img
    scale = MAX_SIDE_PX / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _preprocess(img: Image.Image, device: torch.device) -> torch.Tensor:
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return t.unsqueeze(0).to(device)


def _save_simple(img_raw: Image.Image, density_np: np.ndarray, img_name: str, count: float, output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"{img_name} | Estimated count: {int(round(count))}", fontsize=12, fontweight='bold')
    axes[0].imshow(np.array(img_raw))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    im = axes[1].imshow(density_np, cmap='jet', interpolation='bilinear')
    axes[1].set_title(f'Density Map | Count: {int(round(count))}')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Density')
    save_path = os.path.join(output_dir, f'result_{os.path.splitext(img_name)[0]}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return save_path


def _save_alert(img_raw: Image.Image, density_np: np.ndarray, img_name: str, count: float, output_dir: str) -> str:
    status = 'DANGER' if count > DANGER_THRESHOLD else 'NORMAL'
    status_color = '#FF2D2D' if status == 'DANGER' else '#2DFF6E'
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'SATARK ALERT: {status} | Count: {int(round(count))}',
        fontsize=14, fontweight='bold', color='white', backgroundcolor=status_color, y=1.01
    )
    axes[0].imshow(np.array(img_raw))
    axes[0].set_title(f'Original | {img_name}')
    axes[0].axis('off')
    axes[1].imshow(np.array(img_raw))
    im = axes[1].imshow(density_np, cmap='jet', alpha=0.55, interpolation='bilinear')
    axes[1].set_title(status, color='red' if status == 'DANGER' else 'green', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Density')
    save_path = os.path.join(output_dir, f'alert_{status.lower()}_{os.path.splitext(img_name)[0]}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return save_path


def run_batch_inference(
    model_path: str = 'checkpoints/satark_best.pth',
    inference_dir: str = 'data/Inference/images',
    output_dir: str = 'outputs/inference',
    simple: bool = False,
) -> Dict:
    ensure_dir(output_dir)
    device = get_device()
    clear_device_cache(device)
    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    if not load_checkpoint(model_path, model, device):
        return {}

    if not os.path.exists(inference_dir):
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    images = sorted([
        f for f in os.listdir(inference_dir)
        if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
    ])
    if not images:
        print(f"No images found in '{inference_dir}'.")
        return {}

    results = []
    danger_count = 0

    for img_name in images:
        img_path = os.path.join(inference_dir, img_name)
        with Image.open(img_path) as pil_img:
            img_rgb = pil_img.convert('RGB')
        resized = _resize_for_inference(img_rgb)
        tensor = _preprocess(resized, device)
        with torch.no_grad():
            output = model(tensor)
        count = float(output.sum().item())
        status = 'DANGER' if count > DANGER_THRESHOLD else 'NORMAL'
        if status == 'DANGER':
            danger_count += 1
        density_np = output.squeeze().cpu().numpy()
        save_path = _save_simple(resized, density_np, img_name, count, output_dir) if simple else _save_alert(resized, density_np, img_name, count, output_dir)
        results.append({'image': img_name, 'count': count, 'status': status, 'path': save_path})
        print(f"Processed {img_name}: count={count:.1f}, status={status}")
        clear_device_cache(device)

    print(f"Batch finished. {len(results)} images processed. Dangerous: {danger_count}")
    return {'results': results, 'danger_count': danger_count}
