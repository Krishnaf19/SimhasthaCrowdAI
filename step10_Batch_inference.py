import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF

from step5_model import CSRNet, get_device, clear_device_cache
from step6_baseline_eval import load_checkpoint


# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
MAX_SIDE_PX      = 1000
DANGER_THRESHOLD = 250
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]


def resize_for_inference(img: Image.Image) -> Image.Image:
    """Cap longest side to MAX_SIDE_PX to prevent MPS OOM on large images."""
    w, h    = img.size
    longest = max(w, h)
    if longest <= MAX_SIDE_PX:
        return img
    scale = MAX_SIDE_PX / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def preprocess(img: Image.Image, device: torch.device) -> torch.Tensor:
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return t.unsqueeze(0).to(device)


def save_simple(
    img_raw:     Image.Image,
    density_np:  np.ndarray,
    img_name:    str,
    count:       float,
    output_dir:  str,
) -> str:
    """Simple mode: side-by-side original + standalone density map, no alerts."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        f"{img_name}  |  Estimated Count: {int(round(count))} people",
        fontsize=12, fontweight='bold'
    )

    axes[0].imshow(np.array(img_raw))
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis('off')

    im = axes[1].imshow(density_np, cmap='jet', interpolation='bilinear')
    axes[1].set_title(f"Density Map  |  Count: {int(round(count))}", fontsize=11)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Density')

    plt.tight_layout()
    base      = os.path.splitext(img_name)[0]
    save_path = os.path.join(output_dir, f"result_{base}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return save_path


def save_alert(
    img_raw:      Image.Image,
    density_np:   np.ndarray,
    img_name:     str,
    count:        float,
    output_dir:   str,
) -> str:
    """Alert mode: overlay + DANGER/NORMAL banner burned into header."""
    status       = 'DANGER' if count > DANGER_THRESHOLD else 'NORMAL'
    status_color = '#FF2D2D' if status == 'DANGER' else '#2DFF6E'

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"  SATARK ALERT: {status}  |  Count: {int(round(count))} people  ",
        fontsize=14, fontweight='bold',
        color='white', backgroundcolor=status_color,
        y=1.01
    )

    img_np = np.array(img_raw)

    axes[0].imshow(img_np)
    axes[0].set_title(f"Original  |  {img_name}", fontsize=10)
    axes[0].axis('off')

    # Density overlaid on image — same scale for both panels
    axes[1].imshow(img_np)
    im = axes[1].imshow(density_np, cmap='jet', alpha=0.55, interpolation='bilinear')
    axes[1].set_title(
        f"Density Overlay  |  {status}",
        fontsize=10,
        color='red' if status == 'DANGER' else 'green',
        fontweight='bold'
    )
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Density')

    plt.tight_layout()
    base      = os.path.splitext(img_name)[0]
    save_path = os.path.join(output_dir, f"alert_{status.lower()}_{base}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return save_path


def run_batch_inference(
    model_path:    str  = 'checkpoints/satark_best.pth',
    inference_dir: str  = 'data/Inference/images',
    output_dir:    str  = 'outputs/inference',
    simple:        bool = False,    # True = no alerts, False = full alert mode
) -> None:
    mode = "Simple (no alerts)" if simple else "Alert mode (DANGER/NORMAL)"
    print("Step 11: SATARK Batch Inference")
    print("=" * 55)
    print(f"  Mode          : {mode}")
    print(f"  Model         : {model_path}")
    print(f"  Inference dir : {inference_dir}")
    print(f"  Output dir    : {output_dir}")
    if not simple:
        print(f"  Danger threshold: >{DANGER_THRESHOLD} people")
    print(f"  Max image side  : {MAX_SIDE_PX}px")

    os.makedirs(output_dir, exist_ok=True)

    # ── Device & model ────────────────────────────────────────────────────────
    device = get_device()
    clear_device_cache(device)
    print(f"  Device        : {device}\n")

    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    if not load_checkpoint(model_path, model, device):
        print("  Run step7_fine_tune.py first to generate the model.")
        return
    model.eval()

    # ── Find images ───────────────────────────────────────────────────────────
    if not os.path.exists(inference_dir):
        print(f"  Warning: '{inference_dir}' not found.")
        print("  Create the folder and drop your unlabeled .jpg images into it.")
        return

    all_files = sorted([
        f for f in os.listdir(inference_dir)
        if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
    ])

    if not all_files:
        print(f"  No images found in '{inference_dir}'")
        return

    print(f"  Images found: {len(all_files)}\n")

    header = f"{'#':<4} {'Image':<35} {'Count':>6}  {'Status'}"
    print(header)
    print("-" * len(header))

    results      = []
    danger_count = 0

    for i, img_name in enumerate(all_files):
        img_path = os.path.join(inference_dir, img_name)
        try:
            with Image.open(img_path) as pil_img:
                img_raw  = pil_img.convert('RGB')
                img_resized = resize_for_inference(img_raw)
                orig_size   = pil_img.size
                resized_size = img_resized.size

            tensor = preprocess(img_resized, device)

            with torch.no_grad():
                output = model(tensor)
                count  = float(output.sum().item())

            status = 'DANGER' if count > DANGER_THRESHOLD else 'NORMAL'
            if status == 'DANGER':
                danger_count += 1

            # Resize density map to match display image
            density_np = output.squeeze().cpu().numpy()
            from PIL import Image as PILImage
            dm_pil     = PILImage.fromarray(density_np).resize(
                resized_size, PILImage.BILINEAR
            )
            density_display = np.array(dm_pil)

            # Save in chosen mode
            if simple:
                save_path = save_simple(
                    img_resized, density_display, img_name, count, output_dir
                )
                status_col = '—'
            else:
                save_path = save_alert(
                    img_resized, density_display, img_name, count, output_dir
                )
                status_col = f"{status} {'⚠️' if status == 'DANGER' else '✓'}"

            print(f"{i+1:<4} {img_name:<35} {int(round(count)):>6}  {status_col}")
            if orig_size != resized_size:
                print(f"     (resized {orig_size[0]}×{orig_size[1]} → "
                      f"{resized_size[0]}×{resized_size[1]})")

            results.append({
                'image':  img_name,
                'count':  count,
                'status': status,
                'saved':  save_path,
            })

        except Exception as e:
            print(f"{i+1:<4} {img_name:<35} ERROR: {e}")
            continue

        clear_device_cache(device)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * len(header))
    print(f"\n  BATCH COMPLETE")
    print(f"  Processed : {len(results)} / {len(all_files)} images")
    if not simple:
        print(f"  DANGER    : {danger_count}")
        print(f"  NORMAL    : {len(results) - danger_count}")
        if danger_count > 0:
            print(f"\n  ⚠️  DANGER zones:")
            for r in results:
                if r['status'] == 'DANGER':
                    print(f"     → {r['image']} ({int(round(r['count']))} people)")
    print(f"\n  Outputs saved to: '{output_dir}/'")


if __name__ == '__main__':
    # Change simple=True for plain side-by-side without alert banners
    run_batch_inference(simple=False)