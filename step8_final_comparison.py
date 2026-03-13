# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# from step5_model import CSRNet
# from step4_dataset import SimhasthaDataset
# from torch.utils.data import DataLoader

# def run_satark_metrics():
#     print("🚀 Project SATARK: Stable Performance Evaluation")
    
#     # 1. Device Setup with Memory Cleanup
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     if torch.backends.mps.is_available():
#         torch.mps.empty_cache()
    
#     # 2. Model Initialization
#     model = CSRNet(load_weights=False).to(device)
#     model_path = 'satark_final.pth'
    
#     if not os.path.exists(model_path):
#         print(f" Weights not found at {model_path}")
#         return

#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
#     model.eval()
#     print(f" Weights Loaded. Device: {device}")

#     # 3. Data Loader
#     test_dataset = SimhasthaDataset(root_dir='data', split='Test')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     print(f"\n{'Image':<15} | {'GT':<8} | {'Pred':<10} | {'Acc %'}")
#     print("-" * 55)

#     total_mae = 0
#     accuracy_list = []

#     with torch.no_grad():
#         for i, (img, target) in enumerate(test_loader):
#             try:
#                 # --- MEMORY SAFETY RESIZING ---
#                 # CSRNet struggles with 4K+ images on laptops. 
#                 # We cap the max dimension at 1280px for stability.
#                 h, w = img.shape[2], img.shape[3]
#                 if max(h, w) > 1280:
#                     scale = 1280 / max(h, w)
#                     img = F.interpolate(img, size=(int(h * scale), int(w * scale)), mode='bilinear', align_corners=False)
                
#                 img = img.to(device)
#                 output = model(img)
                
#                 gt = target.sum().item()
#                 pred = output.sum().item()
                
#                 abs_err = abs(gt - pred)
#                 total_mae += abs_err
                
#                 # Accuracy calc
#                 acc = max(0, (1 - (abs_err / gt)) * 100) if gt > 0 else 100.0
#                 accuracy_list.append(acc)
                
#                 img_name = test_dataset.image_files[i]
#                 print(f"{img_name[:15]:<15} | {gt:<8.1f} | {pred:<10.1f} | {acc:>8.1f}%")

#                 # Frequent Memory Flush
#                 if torch.backends.mps.is_available():
#                     torch.mps.empty_cache()
                    
#             except Exception as e:
#                 print(f" Error processing {test_dataset.image_files[i]}: {e}")
#                 continue

#     final_mae = total_mae / len(accuracy_list)
#     final_acc = np.mean(accuracy_list)

#     print("-" * 55)
#     print(f" SYSTEM SUMMARY:")
#     print(f"▶ Avg Error (MAE): {final_mae:.2f} people")
#     print(f"▶ System Accuracy: {final_acc:.2f}%")

# if __name__ == '__main__':
#     run_satark_metrics()

import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from step5_model import CSRNet, get_device, clear_device_cache  # FIX 1
from step6_baseline_eval import load_checkpoint                  # FIX 1: reuse loader
from step4_dataset import SimhasthaDataset


def run_satark_metrics(
    model_path: str = 'checkpoints/satark_best.pth',   # FIX 2 & 8
    data_root:  str = 'data',
    crop_size:  int = 512,
    downsample: int = 8,
) -> dict:
    print("Project SATARK: Final Performance Evaluation")
    print("=" * 60)

    # FIX 1: Shared device utility
    device = get_device()
    clear_device_cache(device)
    print(f"  Device     : {device}")
    print(f"  Model path : {model_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)

    # FIX 2: Use robust loader; aborts cleanly if file missing
    ok = load_checkpoint(model_path, model, device)
    if not ok:
        print(f"\n  Tip: Run step7_fine_tune.py first to generate "
              f"'checkpoints/satark_best.pth'")
        return {}

    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    try:
        test_dataset = SimhasthaDataset(
            root_dir=data_root, split='Test',
            crop_size=crop_size, downsample=downsample
        )
    except FileNotFoundError as e:
        print(f"  Dataset error: {e}")
        return {}

    if len(test_dataset) == 0:
        print("  No test images found in data/Test/images/")
        return {}

    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=0, pin_memory=False
    )
    print(f"  Test images: {len(test_dataset)}\n")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    header = f"{'#':<4} {'Image':<28} {'GT':>7} {'Pred':>7} {'MAE':>7} {'Err%':>7}"
    print(header)
    print("-" * len(header))

    results      = []
    # FIX 5: Track counts separately so a skipped batch doesn't corrupt MAE
    batches_ok   = 0

    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img_name = test_dataset.image_files[i]
            try:
                # FIX 3: img already resized by Dataset to crop_size × crop_size
                # No manual resizing needed here — Dataset handles it uniformly.
                # target.sum() is always the true GT count regardless of spatial size.
                img = img.to(device)

                output  = model(img)
                gt      = float(target.sum().item())
                pred    = float(output.sum().item())
                mae     = abs(gt - pred)
                # FIX 7: Collect squared error for RMSE
                sq_err  = (gt - pred) ** 2
                rel_err = (mae / gt * 100) if gt > 0 else 0.0

                results.append({
                    'image':   img_name,
                    'gt':      gt,
                    'pred':    pred,
                    'mae':     mae,
                    'sq_err':  sq_err,
                    'rel_err': rel_err,
                })
                batches_ok += 1

                flag = " ←" if rel_err > 30 else ""
                print(f"{i+1:<4} {img_name:<28} {gt:>7.1f} {pred:>7.1f} "
                      f"{mae:>7.1f} {rel_err:>6.1f}%{flag}")

            except Exception as e:
                print(f"{i+1:<4} {img_name:<28} ERROR: {e}")
                continue

            # FIX 4: Cache clear once per image (not per batch in a tight loop)
            clear_device_cache(device)

    if batches_ok == 0:
        print("  No images were processed successfully.")
        return {}

    # ── Metrics ───────────────────────────────────────────────────────────────
    # FIX 5: All metrics computed from results list — immune to skipped batches
    all_mae     = [r['mae']     for r in results]
    all_sq_err  = [r['sq_err']  for r in results]
    all_rel_err = [r['rel_err'] for r in results]
    all_gt      = [r['gt']      for r in results]

    avg_mae  = np.mean(all_mae)
    # FIX 7: RMSE — penalises large outlier errors more than MAE does
    rmse     = float(np.sqrt(np.mean(all_sq_err)))
    avg_rel  = float(np.mean(all_rel_err))
    worst    = max(results, key=lambda r: r['mae'])
    best     = min(results, key=lambda r: r['mae'])

    print("-" * len(header))
    print(f"\n  SATARK FINAL RESULTS")
    print(f"  {'MAE':<22}: {avg_mae:.2f} people")
    print(f"  {'RMSE':<22}: {rmse:.2f} people")          # FIX 7
    print(f"  {'Mean Relative Error':<22}: {avg_rel:.1f}%")
    # FIX 6: Removed non-standard "accuracy" metric
    print(f"  {'GT count range':<22}: {min(all_gt):.0f} – {max(all_gt):.0f} people")
    print(f"  {'Best prediction':<22}: '{best['image']}'  "
          f"(err={best['mae']:.1f})")
    print(f"  {'Worst prediction':<22}: '{worst['image']}'  "
          f"(GT={worst['gt']:.0f}, Pred={worst['pred']:.0f}, "
          f"err={worst['mae']:.1f})")

    # Contextual quality verdict
    print()
    if avg_rel < 10:
        verdict = "EXCELLENT — production ready"
    elif avg_rel < 20:
        verdict = "GOOD — acceptable for alerting system"
    elif avg_rel < 35:
        verdict = "FAIR — add more training images to improve"
    else:
        verdict = "NEEDS WORK — check annotations and re-train"
    print(f"  Verdict: {verdict}")
    print(f"\n  RMSE > MAE by {rmse - avg_mae:.1f} — "
          + ("a few large outliers are skewing results, investigate worst predictions."
             if rmse - avg_mae > avg_mae * 0.5
             else "errors are fairly consistent across images."))

    return {
        'mae':      avg_mae,
        'rmse':     rmse,
        'rel_err':  avg_rel,
        'results':  results,
    }


if __name__ == '__main__':
    run_satark_metrics()

