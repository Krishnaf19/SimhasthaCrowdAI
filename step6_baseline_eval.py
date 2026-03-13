# import torch
# import numpy as np
# import os
# from step5_model import CSRNet
# from step4_dataset import SimhasthaDataset
# from torch.utils.data import DataLoader

# def run_baseline_comparison():
#     print(" Step 6: Running ShanghaiTech Baseline Evaluation...")
    
#     # 1. Setup Device (Optimized for Mac M1/M2/M3)
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     print(f"-> Using Device: {device}")

#     # 2. Initialize the Model
#     model = CSRNet().to(device)
    
#     # 3. Path to your downloaded weights
#     weights_path = 'baseline_weights.pth'
    
#     if not os.path.exists(weights_path):
#         print(f"Error: Could not find '{weights_path}'")
#         print("Action: Rename 'PartAmodel_best.pth.tar' to 'baseline_weights.pth' in this folder.")
#         return

#     # 4. Load Weights (Handling the .tar wrapper if present)
#     try:
#         checkpoint = torch.load(weights_path, map_location=device)
#         # If it's a dictionary (common in .tar files), grab the 'state_dict'
#         if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
#         print(" Weights loaded successfully!")
#     except Exception as e:
#         print(f" Error loading weights: {e}")
#         return

#     model.eval()

#     # 5. Load your Test Images (The 5 images you set aside)
#     try:
#         test_dataset = SimhasthaDataset(root_dir='data', split='Test')
#         test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#         if len(test_dataset) == 0:
#             print(" No images found in data/Test/images/")
#             return
#     except Exception as e:
#         print(f" Dataset Error: {e}")
#         return

#     total_mae = 0.0
#     print(f"\n{'Image Name':<25} | {'Manual Count':<12} | {'AI Guess':<10} | {'Error'}")
#     print("-" * 75)

#     with torch.no_grad():
#         for i, (img, target) in enumerate(test_loader):
#             img = img.to(device)
#             output = model(img)
            
#             # Sum the density map to get the count
#             ai_count = output.sum().item()
#             gt_count = target.sum().item()
            
#             error = abs(ai_count - gt_count)
#             total_mae += error
            
#             # Identify image using index since dataset names are handled internally
#             img_label = f"Simhastha_Test_{i+1}"
#             print(f"{img_label:<25} | {gt_count:<12.1f} | {ai_count:<10.1f} | {error:.1f}")

#     avg_mae = total_mae / len(test_dataset)
#     print("-" * 75)
#     print(f"📊 FINAL BASELINE MAE (ShanghaiTech Model): {avg_mae:.2f}")
#     print("\nNext Step: Record these errors. They prove why SATARK tuning is needed!")

# if __name__ == '__main__':
#     run_baseline_comparison()

import torch
import os
from torch.utils.data import DataLoader

# FIX 1: Reuse shared utilities from Step 5 instead of duplicating device logic
from step5_model import CSRNet, get_device
from step4_dataset import SimhasthaDataset


def load_checkpoint(weights_path: str, model: CSRNet, device: torch.device) -> bool:
    """
    FIX 2: Robust checkpoint loader that tries multiple common state_dict keys
    and gives clear feedback on exactly what went wrong.
    """
    if not os.path.exists(weights_path):
        print(f"  Error: Could not find '{weights_path}'")
        print("  Action: Rename 'PartAmodel_best.pth.tar' to 'baseline_weights.pth'")
        return False

    try:
        # FIX 7: weights_only=True suppresses PyTorch >=2.0 security warning
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for older checkpoints that contain non-tensor objects
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # FIX 2: Try all common key names used across CSRNet releases
    STATE_DICT_KEYS = ['state_dict', 'model_state_dict', 'model', 'net']

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in STATE_DICT_KEYS:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"  Found state dict under key: '{key}'")
                break
        if state_dict is None:
            # Assume the dict itself is the state dict (flat format)
            state_dict = checkpoint
            print("  Assuming checkpoint is a flat state dict.")
    else:
        print(f"  Error: Unexpected checkpoint type: {type(checkpoint)}")
        return False

    try:
        model.load_state_dict(state_dict, strict=True)
        print("  Weights loaded successfully (strict match).")
    except RuntimeError as e:
        print(f"  Strict load failed: {e}")
        print("  Retrying with strict=False (partial load)...")
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys  : {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            print("  Partial weights loaded.")
        except Exception as e2:
            print(f"  Fatal: Could not load weights at all. Error: {e2}")
            return False

    return True


def run_baseline_comparison(
    weights_path: str = 'baseline_weights.pth',
    data_root:    str = 'data',
    crop_size:    int = 512,
    downsample:   int = 8,
) -> None:
    print("Step 6: Running ShanghaiTech Baseline Evaluation on Simhastha Data")
    print("=" * 70)

    # FIX 1: Use shared device utility
    device = get_device()
    print(f"  Device : {device}")

    # FIX 6: freeze_frontend=False — no point freezing at eval time
    model = CSRNet(load_weights=True, freeze_frontend=False).to(device)

    if not load_checkpoint(weights_path, model, device):
        return

    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    try:
        test_dataset = SimhasthaDataset(
            root_dir=data_root,
            split='Test',
            crop_size=crop_size,
            downsample=downsample,
        )
    except Exception as e:
        print(f"  Dataset Error: {e}")
        return

    if len(test_dataset) == 0:
        print("  No images found in data/Test/images/")
        return

    # FIX 4: batch_size=1 so we can pair each result with its filename
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    print(f"\n  Test images : {len(test_dataset)}")
    print()

    # ── Evaluation loop ───────────────────────────────────────────────────────
    col = f"{'#':<4} {'Image':<28} {'GT Count':>9} {'AI Count':>9} {'MAE':>7} {'Err %':>7}"
    print(col)
    print("-" * len(col))

    results      = []
    total_mae    = 0.0
    # FIX 3: track actual batches processed, not dataset length
    batches_seen = 0

    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img    = img.to(device)
            output = model(img)

            ai_count = output.sum().item()
            gt_count = target.sum().item()
            mae      = abs(ai_count - gt_count)

            # FIX 5: Relative error percentage
            rel_err = (mae / gt_count * 100) if gt_count > 0 else float('nan')

            total_mae    += mae
            batches_seen += 1

            # FIX 4: Pull actual filename from dataset
            img_name = test_dataset.image_files[i]

            flag = "  ← HIGH ERR" if rel_err > 30 else ""
            print(f"{i+1:<4} {img_name:<28} {gt_count:>9.1f} {ai_count:>9.1f} "
                  f"{mae:>7.1f} {rel_err:>6.1f}%{flag}")

            results.append({
                'image':    img_name,
                'gt_count': gt_count,
                'ai_count': ai_count,
                'mae':      mae,
                'rel_err':  rel_err,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    # FIX 3: Divide by actual batches processed
    avg_mae = total_mae / batches_seen if batches_seen > 0 else 0.0

    all_gt      = [r['gt_count'] for r in results]
    all_rel_err = [r['rel_err']  for r in results if not (r['rel_err'] != r['rel_err'])]
    worst       = max(results, key=lambda r: r['mae'])

    print("-" * len(col))
    print(f"\n  BASELINE RESULTS (ShanghaiTech weights → Simhastha data)")
    print(f"    Average MAE      : {avg_mae:.2f} people")
    print(f"    Avg Relative Err : {sum(all_rel_err)/len(all_rel_err):.1f}%"
          if all_rel_err else "")
    print(f"    GT count range   : {min(all_gt):.0f} – {max(all_gt):.0f} people")
    print(f"    Worst prediction : '{worst['image']}' "
          f"(GT={worst['gt_count']:.0f}, AI={worst['ai_count']:.0f}, "
          f"err={worst['mae']:.0f})")
    print(f"\n  These numbers are your baseline. The SATARK fine-tuning in")
    print(f"  Step 7 should reduce MAE by at least 40% on this test set.")


if __name__ == '__main__':
    run_baseline_comparison()