# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from step5_model import CSRNet
# from step4_dataset import SimhasthaDataset
# import torch.nn.functional as F

# def visualize_results():
#     print(" Step 9: Generating Visual Comparison...")
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
#     # Load the SATARK model
#     model = CSRNet().to(device)
#     model.load_state_dict(torch.load('satark_tuned_weights.pth', map_location=device))
#     model.eval()

#     # Get the high-density image (Test_4)
#     dataset = SimhasthaDataset(root_dir='data', split='Test')
#     img, target = dataset[3] 
    
#     with torch.no_grad():
#         img_tensor = img.unsqueeze(0).to(device)
#         output = model(img_tensor)
        
#         # FIX: Ensure output is 2D (height, width) for imshow
#         output_np = output.squeeze().cpu().numpy() 
#         target_np = target.squeeze().cpu().numpy()

#     # Convert image back to displayable RGB format
#     img_display = img.permute(1, 2, 0).numpy()
#     # Normalize for display
#     img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

#     # Plotting
#     fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
#     # 1. Original Image
#     axes[0].imshow(img_display)
#     axes[0].set_title(f"Original Simhastha Image")
#     axes[0].axis('off')

#     # 2. Ground Truth (Heatmap from your dots)
#     axes[1].imshow(target_np, cmap='jet')
#     axes[1].set_title(f"Ground Truth Count: {target_np.sum():.1f}")
#     axes[1].axis('off')

#     # 3. SATARK Prediction (AI's Heatmap)
#     axes[2].imshow(output_np, cmap='jet')
#     axes[2].set_title(f"SATARK Prediction: {output_np.sum():.1f}")
#     axes[2].axis('off')

#     plt.tight_layout()
#     plt.savefig('satark_comparison_plot.png')
#     print(" Success! Visualization saved as 'satark_comparison_plot.png'!")

# if __name__ == '__main__':
#     visualize_results()

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from step5_model import CSRNet, get_device, clear_device_cache  # FIX 2 & 5
from step6_baseline_eval import load_checkpoint                  # FIX 4
from step4_dataset import SimhasthaDataset


def tensor_to_display(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised (ImageNet) image tensor back to displayable RGB.
    Reverses the normalisation applied in the Dataset so colours look correct.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = img_tensor.permute(1, 2, 0).numpy()   # (C,H,W) → (H,W,C)
    img = (img * std) + mean                     # undo ImageNet normalisation
    img = np.clip(img, 0, 1)                     # clamp to [0,1] for imshow
    return img


def find_hardest_index(dataset: SimhasthaDataset) -> int:
    """
    FIX 3: Dynamically find the test image with the highest GT count
    instead of hardcoding index 3.
    """
    max_count = -1
    max_idx   = 0
    for i in range(len(dataset)):
        _, target = dataset[i]
        count = float(target.sum())
        if count > max_count:
            max_count = count
            max_idx   = i
    return max_idx


def visualize_single(
    idx:       int,
    dataset:   SimhasthaDataset,
    model:     CSRNet,
    device:    torch.device,
    output_dir: str,
) -> None:
    """Render and save a 3-panel comparison for one test image."""
    img_name = dataset.image_files[idx]
    img, target = dataset[idx]

    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))

    output_np = output.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    img_disp  = tensor_to_display(img)           # FIX: proper denormalisation

    gt_count   = float(target_np.sum())
    pred_count = float(output_np.sum())
    err_pct    = abs(gt_count - pred_count) / gt_count * 100 if gt_count > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        f"{img_name}  |  GT={gt_count:.0f}  Pred={pred_count:.0f}  "
        f"Err={abs(gt_count-pred_count):.0f} ({err_pct:.1f}%)",
        fontsize=12, fontweight='bold'
    )

    # Panel 1 — Original image
    axes[0].imshow(img_disp)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis('off')

    # Panel 2 — Ground truth density map
    im1 = axes[1].imshow(target_np, cmap='jet', interpolation='bilinear')
    axes[1].set_title(f"Ground Truth  |  Count: {gt_count:.0f}", fontsize=11)
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Density')  # FIX 7

    # Panel 3 — SATARK prediction
    # Use same colour scale as GT so maps are directly comparable
    vmax = max(target_np.max(), output_np.max())
    im2 = axes[2].imshow(output_np, cmap='jet', interpolation='bilinear',
                         vmin=0, vmax=vmax)
    axes[2].set_title(f"SATARK Prediction  |  Count: {pred_count:.0f}", fontsize=11)
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Density')  # FIX 7

    plt.tight_layout()

    # FIX 6: Save to previews/ folder
    save_name = f"satark_viz_{os.path.splitext(img_name)[0]}.png"
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: '{save_path}'  (GT={gt_count:.0f}, Pred={pred_count:.0f})")


def visualize_results(
    model_path:  str = 'checkpoints/satark_best.pth',  # FIX 1
    data_root:   str = 'data',
    output_dir:  str = 'previews',
    all_images:  bool = True,    # FIX 8: loop all test images by default
) -> None:
    print("Step 9: Generating SATARK Visual Comparisons")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    # FIX 2: Shared device utility
    device = get_device()
    clear_device_cache(device)
    print(f"  Device     : {device}")
    print(f"  Model path : {model_path}")

    # FIX 5: freeze_frontend=False at inference
    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)

    # FIX 4: Robust checkpoint loader
    ok = load_checkpoint(model_path, model, device)
    if not ok:
        print("  Run step7_fine_tune.py first to generate the model.")
        return

    model.eval()

    dataset = SimhasthaDataset(root_dir=data_root, split='Test')
    if len(dataset) == 0:
        print("  No test images found.")
        return

    print(f"  Test images : {len(dataset)}\n")

    if all_images:
        # FIX 8: Visualise every test image
        indices = list(range(len(dataset)))
    else:
        # FIX 3: Dynamically find hardest image instead of hardcoding index 3
        hard_idx = find_hardest_index(dataset)
        print(f"  Hardest image: '{dataset.image_files[hard_idx]}'")
        indices = [hard_idx]

    for idx in indices:
        visualize_single(idx, dataset, model, device, output_dir)
        clear_device_cache(device)   # FIX 2: epoch-level cache clear

    print(f"\n  All visualisations saved to '{output_dir}/'")
    print(f"  Open them to verify the density map shapes look correct.")


if __name__ == '__main__':
    visualize_results()