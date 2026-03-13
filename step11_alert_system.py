# import torch
# import os
# import matplotlib.pyplot as plt
# from PIL import Image
# import torchvision.transforms as transforms
# from step5_model import CSRNet

# def run_alert_inference():
#     print(" Step 11: Running SATARK Alert System...")
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
#     # 1. Load Model
#     model = CSRNet().to(device)
#     model.load_state_dict(torch.load('satark_tuned_weights.pth', map_location=device))
#     model.eval()

#     # 2. Setup Threshold (Adjust this based on bridge capacity)
#     DANGER_THRESHOLD = 250 

#     # 3. Process Inference Image
#     inference_dir = 'data/Inference/images'
#     img_name = [f for f in os.listdir(inference_dir) if f.endswith(('.jpg', '.png'))][0]
#     img_path = os.path.join(inference_dir, img_name)

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
    
#     img_raw = Image.open(img_path).convert('RGB')
#     img_tensor = transform(img_raw).unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = model(img_tensor)
#         count = output.sum().item()

#     # 4. Alert Logic
#     is_danger = count > DANGER_THRESHOLD
#     status_text = " DANGER: CAPACITY EXCEEDED" if is_danger else "STATUS: SAFE"
#     status_color = 'red' if is_danger else 'green'

#     # 5. Visualization with Alert Overlay
#     fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
#     axes[0].imshow(img_raw)
#     axes[0].set_title("Input Feed")
#     axes[0].axis('off')

#     im = axes[1].imshow(output.squeeze().cpu().numpy(), cmap='jet')
#     axes[1].set_title(f"Density Map\nCount: {count:.1f}")
#     axes[1].axis('off')

#     # Add the Alert Banner
#     plt.suptitle(status_text, color=status_color, fontsize=24, fontweight='bold', y=0.95)
    
#     save_path = 'satark_alert_output.png'
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

#     print(f"Final Count: {count:.1f}")
#     print(f"Status: {status_text}")
#     print(f"Alert report saved as '{save_path}'")

# if __name__ == '__main__':
#     run_alert_inference()


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import csv
from datetime import datetime

from step5_model import CSRNet, get_device, clear_device_cache
from step6_baseline_eval import load_checkpoint


# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
MAX_SIDE_PX      = 1000
DANGER_THRESHOLD = 250
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]


def resize_for_inference(img: Image.Image) -> Image.Image:
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


def save_alert_visualization(
    img_raw:     Image.Image,
    density_np:  np.ndarray,
    img_name:    str,
    count:       float,
    status:      str,
    output_dir:  str,
) -> str:
    """
    Save a 3-panel alert visualization:
    Left   — original image
    Middle — density map overlay
    Right  — alert status panel with count and verdict
    """
    status_color = '#FF2D2D' if status == 'DANGER' else '#2DFF6E'
    text_color   = 'white'

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Header banner burned into figure title
    fig.suptitle(
        f"  SATARK CROWD ALERT: {status}  |  "
        f"Estimated Count: {int(round(count))} people  |  "
        f"{img_name}  ",
        fontsize=13, fontweight='bold',
        color=text_color,
        backgroundcolor=status_color,
        y=1.01
    )

    img_np = np.array(img_raw)

    # Panel 1 — Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Feed", fontsize=11)
    axes[0].axis('off')

    # Panel 2 — Density overlay
    axes[1].imshow(img_np)
    im = axes[1].imshow(density_np, cmap='jet', alpha=0.55, interpolation='bilinear')
    axes[1].set_title("Crowd Density Map", fontsize=11)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Density')

    # Panel 3 — Alert status card
    axes[2].set_facecolor(status_color)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')

    # Status icon
    icon = '⚠️' if status == 'DANGER' else '✓'
    axes[2].text(
        0.5, 0.72, icon,
        fontsize=52, ha='center', va='center',
        transform=axes[2].transAxes
    )
    # Status label
    axes[2].text(
        0.5, 0.55, status,
        fontsize=28, fontweight='bold',
        ha='center', va='center',
        color='white',
        transform=axes[2].transAxes
    )
    # Count
    axes[2].text(
        0.5, 0.38,
        f"Count: {int(round(count))} people",
        fontsize=16, ha='center', va='center',
        color='white',
        transform=axes[2].transAxes
    )
    # Threshold note
    axes[2].text(
        0.5, 0.25,
        f"Threshold: {DANGER_THRESHOLD} people",
        fontsize=11, ha='center', va='center',
        color='white', alpha=0.85,
        transform=axes[2].transAxes
    )
    # Timestamp
    axes[2].text(
        0.5, 0.10,
        datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
        fontsize=9, ha='center', va='center',
        color='white', alpha=0.7,
        transform=axes[2].transAxes
    )

    plt.tight_layout()

    base      = os.path.splitext(img_name)[0]
    save_name = f"alert_{status.lower()}_{base}.png"
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return save_path


def save_alert_log(results: list, output_dir: str) -> str:
    """
    Save a CSV log of all alerts for audit trail.
    Useful when you scale to real-time monitoring.
    """
    log_path = os.path.join(output_dir, 'alert_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'image', 'count', 'status', 'threshold'])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            writer.writerow([
                ts, r['image'], int(round(r['count'])),
                r['status'], DANGER_THRESHOLD
            ])
    return log_path


def run_alert_system(
    model_path:    str = 'checkpoints/satark_best.pth',
    inference_dir: str = 'data/Inference/images',
    output_dir:    str = 'outputs/alerts',
) -> None:
    print("Step 11: SATARK Alert System — All Images")
    print("=" * 55)
    print(f"  Model           : {model_path}")
    print(f"  Inference dir   : {inference_dir}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Danger threshold: >{DANGER_THRESHOLD} people")
    print(f"  Max image side  : {MAX_SIDE_PX}px")

    os.makedirs(output_dir, exist_ok=True)

    # ── Device & model ────────────────────────────────────────────────────────
    device = get_device()
    clear_device_cache(device)
    print(f"  Device          : {device}\n")

    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    if not load_checkpoint(model_path, model, device):
        print("  Run step7_fine_tune.py first.")
        return
    model.eval()

    # ── Find images ───────────────────────────────────────────────────────────
    if not os.path.exists(inference_dir):
        print(f"  Warning: '{inference_dir}' not found.")
        print("  Create the folder and add .jpg images to it.")
        return

    all_files = sorted([
        f for f in os.listdir(inference_dir)
        if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
    ])

    if not all_files:
        print(f"  No images found in '{inference_dir}'")
        return

    print(f"  Images found: {len(all_files)}\n")

    header = f"{'#':<4} {'Image':<35} {'Count':>6}  {'Status':<8}  {'Action'}"
    print(header)
    print("-" * len(header))

    results      = []
    danger_count = 0

    for i, img_name in enumerate(all_files):
        img_path = os.path.join(inference_dir, img_name)
        try:
            with Image.open(img_path) as pil_img:
                img_raw      = pil_img.convert('RGB')
                img_resized  = resize_for_inference(img_raw)
                orig_size    = pil_img.size
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
            dm_pil     = Image.fromarray(density_np).resize(
                resized_size, Image.BILINEAR
            )
            density_display = np.array(dm_pil)

            save_path = save_alert_visualization(
                img_raw=img_resized,
                density_np=density_display,
                img_name=img_name,
                count=count,
                status=status,
                output_dir=output_dir,
            )

            action = "DISPATCH CONTROL TEAM" if status == 'DANGER' else "Continue monitoring"
            flag   = " ⚠️" if status == 'DANGER' else " ✓"
            print(f"{i+1:<4} {img_name:<35} {int(round(count)):>6}  "
                  f"{status:<8}  {action}{flag}")

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

    # ── Save audit log ────────────────────────────────────────────────────────
    if results:
        log_path = save_alert_log(results, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * len(header))
    print(f"\n  ALERT SYSTEM SUMMARY")
    print(f"  Processed : {len(results)} / {len(all_files)} images")
    print(f"  DANGER    : {danger_count}")
    print(f"  NORMAL    : {len(results) - danger_count}")
    print(f"  Alert log : '{log_path}'")
    print(f"  Visuals   : '{output_dir}/'")

    if danger_count > 0:
        print(f"\n  ⚠️  IMMEDIATE ACTION REQUIRED:")
        for r in results:
            if r['status'] == 'DANGER':
                print(f"     → {r['image']}  |  "
                      f"{int(round(r['count']))} people detected  |  "
                      f"DISPATCH CROWD CONTROL")
    else:
        print(f"\n  All zones NORMAL — no immediate action required.")


if __name__ == '__main__':
    run_alert_system()