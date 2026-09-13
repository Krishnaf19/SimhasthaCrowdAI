import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch
import torchvision.transforms.functional as TF
from PIL import Image
from ..engine.evaluator import load_checkpoint
from ..models.csrnet    import CSRNet, get_device, clear_device_cache
from ..utils.common     import IMAGE_EXTENSIONS, ensure_dir, CLASSES

MAX_SIDE_PX      = 1000
SAFE_THRESHOLD   = 50
NORMAL_THRESHOLD = 150
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_zone(count):
    if count <= SAFE_THRESHOLD:   return 'SAFE'
    if count <= NORMAL_THRESHOLD: return 'NORMAL'
    return 'CRITICAL'


def _resize(img):
    w, h   = img.size
    longest = max(w, h)
    if longest <= MAX_SIDE_PX: return img
    scale = MAX_SIDE_PX / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _preprocess(img, device):
    t = TF.normalize(TF.to_tensor(img), MEAN, STD)
    return t.unsqueeze(0).to(device)


def _checkpoint_output_channels(model_path, device):
    """Read the density-head width so legacy single-channel models can load."""
    try:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict', 'model', 'net'):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        return int(checkpoint['output_layer.weight'].shape[0])
    except (KeyError, TypeError, OSError, RuntimeError):
        return len(CLASSES)


def _save_result(img_raw, density_np, img_name, count, per_class_counts, output_dir):
    zone = get_zone(count)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    labels = CLASSES if len(per_class_counts) == len(CLASSES) else ['head count']
    info = '  |  '.join(cls + ': ' + str(int(round(c)))
                        for cls, c in zip(labels, per_class_counts))
    fig.suptitle('SATARK  Zone: ' + zone + '  |  Count: ' + str(int(round(count))) + '  |  ' + info,
                 fontsize=11, fontweight='bold')
    axes[0].imshow(np.array(img_raw)); axes[0].set_title(img_name); axes[0].axis('off')
    axes[1].imshow(np.array(img_raw))
    im = axes[1].imshow(density_np, cmap='jet', alpha=0.55, interpolation='bilinear')
    axes[1].set_title(zone, color='red' if zone == 'CRITICAL' else 'black', fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Density')
    save_path = os.path.join(output_dir, 'result_' + os.path.splitext(img_name)[0] + '.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150); plt.close(fig)
    return save_path


def infer_image(img_path, model_path='checkpoints/satark_best.pth',
                output_dir='outputs/inference', model=None, device=None):
    ensure_dir(output_dir)
    if device is None: device = get_device()
    if model is None:
        output_channels = _checkpoint_output_channels(model_path, device)
        model = CSRNet(load_weights=False, freeze_frontend=False,
                       output_channels=output_channels).to(device)
        if not load_checkpoint(model_path, model, device): return {}
    img_rgb = Image.open(img_path).convert('RGB')
    resized = _resize(img_rgb)
    with torch.no_grad():
        output = model(_preprocess(resized, device))
    per_class_counts = [output[0, c].sum().item() for c in range(output.shape[1])]
    count     = sum(per_class_counts)
    density_np = output[0].sum(dim=0).cpu().numpy()
    save_path  = _save_result(resized, density_np, os.path.basename(img_path),
                               count, per_class_counts, output_dir)
    return {'image': os.path.basename(img_path), 'count': count, 'zone': get_zone(count),
            'path': save_path,
            'per_class': dict(zip(CLASSES if len(per_class_counts) == len(CLASSES) else ['head count'], per_class_counts))}


def infer_images(model_path='checkpoints/satark_best.pth',
                 inference_dir='data/processed/images',
                 output_dir='outputs/inference'):
    ensure_dir(output_dir)
    device = get_device(); clear_device_cache(device)
    output_channels = _checkpoint_output_channels(model_path, device)
    model = CSRNet(load_weights=False, freeze_frontend=False,
                   output_channels=output_channels).to(device)
    if not load_checkpoint(model_path, model, device): return {}
    if not os.path.exists(inference_dir):
        raise FileNotFoundError('Inference dir not found: ' + inference_dir)
    images = sorted(f for f in os.listdir(inference_dir)
                    if os.path.splitext(f)[1] in IMAGE_EXTENSIONS)
    if not images: return {}
    results = []; zone_counts = {'SAFE': 0, 'NORMAL': 0, 'CRITICAL': 0}
    for img_name in images:
        r = infer_image(os.path.join(inference_dir, img_name),
                        model=model, device=device, output_dir=output_dir)
        results.append(r)
        z = r.get('zone')
        if z in zone_counts: zone_counts[z] += 1
    print('Batch done. {} images. SAFE={}, NORMAL={}, CRITICAL={}'.format(
        len(results), zone_counts['SAFE'], zone_counts['NORMAL'], zone_counts['CRITICAL']))
    return {'results': results, **zone_counts}
