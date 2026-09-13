import os
from typing import Dict

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask compatibility
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Import cv2 only if needed, with fallback
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .evaluate import load_checkpoint
from .model import CSRNet, get_device, clear_device_cache
from .utils import IMAGE_EXTENSIONS, ensure_dir


MAX_SIDE_PX = 1000
# Configurable thresholds - adjust these based on your crowd type
SAFE_THRESHOLD = 50  # More conservative threshold for diverse crowds
NORMAL_THRESHOLD = 150  # Middle ground between safe and critical
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing settings
ENABLE_CONTRAST_ENHANCEMENT = True
ENABLE_HISTOGRAM_EQUALIZATION = False
ADAPTIVE_THRESHOLD_SCALING = True


def _resize_for_inference(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_SIDE_PX:
        return img
    scale = MAX_SIDE_PX / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _enhance_contrast(img: Image.Image) -> Image.Image:
    """Enhance image contrast to improve crowd visibility in varying lighting conditions."""
    img_np = np.array(img).astype(np.float32)
    # Normalize to 0-1 range
    img_np = img_np / 255.0
    # Compute mean and std
    mean = np.mean(img_np)
    std = np.std(img_np)
    # Enhance contrast
    if std > 0.01:
        img_np = (img_np - mean) / (std + 1e-6) * 0.15 + 0.5
        img_np = np.clip(img_np, 0, 1)
    # Convert back to 0-255 range
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _adaptive_histogram_equalization(img: Image.Image) -> Image.Image:
    """Apply adaptive histogram equalization for lighting-invariant features."""
    if not HAS_CV2:
        # Fallback: just return the image if cv2 not available
        return img
    
    import cv2
    img_np = np.array(img)
    if len(img_np.shape) == 3:  # RGB image
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        img_yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_yuv[:, :, 0])
        img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)
    else:  # Grayscale
        img_np = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_np)
    return Image.fromarray(img_np)


def _preprocess(img: Image.Image, device: torch.device, enhance: bool = True) -> torch.Tensor:
    """Preprocess image with optional enhancement for better generalization."""
    # Apply enhancements if enabled
    if enhance:
        if ENABLE_CONTRAST_ENHANCEMENT:
            img = _enhance_contrast(img)
        if ENABLE_HISTOGRAM_EQUALIZATION:
            img = _adaptive_histogram_equalization(img)
    
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


def _save_zone_image(img_raw: Image.Image, density_np: np.ndarray, img_name: str, count: float, output_dir: str) -> str:
    zone = get_zone_label(count)
    zone_color = get_zone_color(zone)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f'SATARK ZONE: {zone} | Count: {int(round(count))}',
        fontsize=14, fontweight='bold', color='black' if zone != 'CRITICAL' else 'white', backgroundcolor=zone_color, y=1.01
    )
    axes[0].imshow(np.array(img_raw))
    axes[0].set_title(f'Original | {img_name}')
    axes[0].axis('off')
    axes[1].imshow(np.array(img_raw))
    im = axes[1].imshow(density_np, cmap='jet', alpha=0.55, interpolation='bilinear')
    axes[1].set_title(zone, color='red' if zone == 'CRITICAL' else 'black', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Density')
    save_path = os.path.join(output_dir, f'zone_{zone.lower()}_{os.path.splitext(img_name)[0]}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return save_path


def get_zone_label(count: float) -> str:
    if count <= SAFE_THRESHOLD:
        return 'SAFE'
    if count <= NORMAL_THRESHOLD:
        return 'NORMAL'
    return 'CRITICAL'


def get_zone_color(zone: str) -> str:
    return {
        'SAFE': '#2DFF6E',
        'NORMAL': '#FFD12D',
        'CRITICAL': '#FF2D2D',
    }.get(zone, '#FFFFFF')


def infer_image(
    img_path: str,
    model_path: str = 'checkpoints/satark_best.pth',
    output_dir: str = 'outputs/inference',
    model: CSRNet = None,
    device: torch.device = None,
    enhance: bool = True,
) -> Dict:
    """
    Infer crowd count on a single image with adaptive preprocessing.
    
    Args:
        img_path: Path to input image
        model_path: Path to model checkpoint
        output_dir: Output directory for results
        model: Pre-loaded model (optional)
        device: Torch device (optional)
        enhance: Whether to apply contrast enhancement for better generalization
    """
    ensure_dir(output_dir)
    if device is None:
        device = get_device()
    if model is None:
        model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
        if not load_checkpoint(model_path, model, device):
            return {}
    else:
        model = model.to(device)

    with Image.open(img_path) as pil_img:
        img_rgb = pil_img.convert('RGB')
    resized = _resize_for_inference(img_rgb)
    tensor = _preprocess(resized, device, enhance=enhance)
    with torch.no_grad():
        output = model(tensor)

    count = float(output.sum().item())
    zone = get_zone_label(count)
    density_np = output.squeeze().cpu().numpy()
    save_path = _save_zone_image(resized, density_np, os.path.basename(img_path), count, output_dir)
    return {'image': os.path.basename(img_path), 'count': count, 'zone': zone, 'path': save_path}


def infer_images(
    model_path: str = 'checkpoints/satark_best.pth',
    inference_dir: str = 'data/Inference/images',
    output_dir: str = 'outputs/inference',
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
    zone_counts = {'SAFE': 0, 'NORMAL': 0, 'CRITICAL': 0}
    for img_name in images:
        img_path = os.path.join(inference_dir, img_name)
        result = infer_image(img_path, model=model, device=device, output_dir=output_dir)
        results.append(result)
        zone = result.get('zone')
        if zone in zone_counts:
            zone_counts[zone] += 1

    print(
        f"Batch finished. {len(results)} images processed. "
        f"SAFE: {zone_counts['SAFE']}, NORMAL: {zone_counts['NORMAL']}, CRITICAL: {zone_counts['CRITICAL']}"
    )
    return {
        'results': results,
        'safe_count': zone_counts['SAFE'],
        'normal_count': zone_counts['NORMAL'],
        'critical_count': zone_counts['CRITICAL'],
    }


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
    critical_count = 0

    for img_name in images:
        img_path = os.path.join(inference_dir, img_name)
        with Image.open(img_path) as pil_img:
            img_rgb = pil_img.convert('RGB')
        resized = _resize_for_inference(img_rgb)
        tensor = _preprocess(resized, device)
        with torch.no_grad():
            output = model(tensor)
        count = float(output.sum().item())
        zone = get_zone_label(count)
        if zone == 'CRITICAL':
            critical_count += 1
        density_np = output.squeeze().cpu().numpy()
        save_path = _save_simple(resized, density_np, img_name, count, output_dir) if simple else _save_zone_image(resized, density_np, img_name, count, output_dir)
        results.append({'image': img_name, 'count': count, 'zone': zone, 'path': save_path})
        print(f"Processed {img_name}: count={count:.1f}, zone={zone}")
        clear_device_cache(device)

    print(f"Batch finished. {len(results)} images processed. Critical: {critical_count}")
    return {'results': results, 'critical_count': critical_count}
