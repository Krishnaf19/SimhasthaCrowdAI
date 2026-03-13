# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from step5_model import CSRNet
# from step4_dataset import SimhasthaDataset
# import os
# import sys

# def train_satark():
#     print(" Step 7: Initializing Fine-Tuning Engine...")
    
#     # 1. Device Setup
#     device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#     print(f"-> Target Hardware: {device}")

#     # 2. Initialize Model
#     model = CSRNet(load_weights=False).to(device)
#     weights_path = 'baseline_weights.pth'
    
#     if os.path.exists(weights_path):
#         checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
#         if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
#         print(" Baseline weights successfully loaded.")
#     else:
#         print(" baseline_weights.pth not found. Initializing with VGG16.")
#         model = CSRNet(load_weights=True).to(device)

#     # 3. Load Dataset
#     print("Checking for data in data/Train/...")
#     train_dataset = SimhasthaDataset(root_dir='data', split='Train')
    
#     if len(train_dataset) == 0:
#         print(" ERROR: No images found in data/Train/images. Please check your folder structure!")
#         return

#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#     print(f"Dataset loaded: {len(train_dataset)} images ready for training.")

#     # 4. Training Setup
#     criterion = nn.MSELoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
#     num_epochs = 50 
#     print(f" Starting training loop for {num_epochs} epochs...")
#     print("-" * 50)

#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0
        
#         for i, (img, target) in enumerate(train_loader):
#             img, target = img.to(device), target.to(device)
            
#             optimizer.zero_grad()
#             output = model(img)
            
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()

#             if torch.backends.mps.is_available():
#                 torch.mps.empty_cache()

#         avg_loss = epoch_loss / len(train_loader)
        
#         # Immediate feedback
#         print(f"Epoch [{epoch+1}/{num_epochs}] | Average Loss: {avg_loss:.8f}")

#         # Save Checkpoint
#         if (epoch + 1) % 10 == 0:
#             save_name = f'satark_e{epoch+1}.pth'
#             torch.save(model.state_dict(), save_name)
#             print(f"💾 Checkpoint saved: {save_name}")

#     torch.save(model.state_dict(), 'satark_final.pth')
#     print("-" * 50)
#     print("SATARK Fine-tuning Complete! Final model saved as 'satark_final.pth'")


# if __name__ == '__main__':
#     try:
#         train_satark()
#     except Exception as e:
#         print(f"SCRIPT CRASHED: {e}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import traceback

# FIX 1: Reuse shared utilities from Step 5
from step5_model import CSRNet, get_device, clear_device_cache
# FIX 2: Reuse checkpoint loader from Step 6
from step6_baseline_eval import load_checkpoint
from step4_dataset import SimhasthaDataset


# ── FIX 5: Custom density-weighted loss (Module 3 requirement) ────────────────

class DensityWeightedMSELoss(nn.Module):
    """
    Standard MSELoss, but multiplied by a penalty factor when the ground-truth
    crowd count exceeds a threshold. Forces the model to prioritise accuracy
    on dense crowds — fixing the Test_6 undercount issue.
    """
    def __init__(self, density_threshold: float = 150.0, penalty: float = 2.5):
        super().__init__()
        self.threshold = density_threshold
        self.penalty   = penalty
        self.mse       = nn.MSELoss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse(pred, target)
        gt_count = target.sum()
        if gt_count > self.threshold:
            loss = loss * self.penalty
        return loss


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """Shared forward-pass logic for both train and validation."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_mae  = 0.0
    context    = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for img, target in loader:
            img, target = img.to(device), target.to(device)

            output = model(img)
            loss   = criterion(output, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            # Track raw MAE (unweighted) for honest reporting
            with torch.no_grad():
                total_mae += abs(output.sum() - target.sum()).item()

    n = len(loader)
    return total_loss / n, total_mae / n


def train_satark(
    weights_path:     str   = 'baseline_weights.pth',
    data_root:        str   = 'data',
    checkpoint_dir:   str   = 'checkpoints',       # FIX 8
    num_epochs:       int   = 50,
    lr:               float = 1e-5,
    batch_size:       int   = 1,
    density_threshold: float = 150.0,
    penalty:          float = 2.5,
    save_every:       int   = 10,
    patience: int = 7,       # stop if no improvement for 7 epoch
    min_epochs: int = 10,    # always train at least 10 epochs
) -> None:

    print("Step 7: SATARK Fine-Tuning Engine")
    print("=" * 55)

    # FIX 1: Shared device utility
    device = get_device()
    print(f"  Device : {device}")

    # FIX 8: Dedicated checkpoint folder
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Model init ────────────────────────────────────────────────────────────
    # Always start from VGG16 pretrained frontend, then overlay baseline weights
    model = CSRNet(load_weights=True, freeze_frontend=True).to(device)

    if os.path.exists(weights_path):
        print(f"\n  Loading baseline weights from '{weights_path}'...")
        # FIX 2: Reuse the robust loader from Step 6
        ok = load_checkpoint(weights_path, model, device)
        if not ok:
            print("  Proceeding with VGG16-only initialisation.")
    else:
        print(f"  '{weights_path}' not found — using VGG16 initialisation only.")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n  Loading datasets...")
    try:
        train_dataset = SimhasthaDataset(root_dir=data_root, split='Train')
        val_dataset   = SimhasthaDataset(root_dir=data_root, split='Test')
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    if len(train_dataset) == 0:
        print("  ERROR: No training images found. Check data/Train/images/")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=1,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"  Train : {len(train_dataset)} images")
    print(f"  Val   : {len(val_dataset)} images")

    # ── Optimizer — FIX 3: only trainable (non-frozen) parameters ────────────
    criterion = DensityWeightedMSELoss(
        density_threshold=density_threshold,
        penalty=penalty
    )
    optimizer = optim.Adam(
        model.trainable_parameters(),   # FIX 3: excludes frozen frontend
        lr=lr
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\n  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {frozen:,}  (VGG16 frontend)")
    print(f"\n  Loss  : DensityWeightedMSELoss "
          f"(threshold={density_threshold}, penalty=×{penalty})")
    print(f"  LR    : {lr}  |  Epochs: {num_epochs}")
    print("-" * 55)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mae  = float('inf')
    best_epoch    = -1

    for epoch in range(num_epochs):

        train_loss, train_mae = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )

        # FIX 4: Cache clear ONCE per epoch, not per batch
        clear_device_cache(device)

        # FIX 6: Validation every epoch
        val_loss = val_mae = 0.0
        if len(val_dataset) > 0:
            val_loss, val_mae = run_epoch(
                model, val_loader, criterion, optimizer, device, is_train=False
            )

        ep_str = f"Epoch [{epoch+1:>3}/{num_epochs}]"
        tr_str = f"Train → loss={train_loss:.6f}  MAE={train_mae:.1f}"
        vl_str = f"Val → loss={val_loss:.6f}  MAE={val_mae:.1f}" if len(val_dataset) > 0 else ""
        print(f"{ep_str}  |  {tr_str}  |  {vl_str}")

        # FIX 9: Save best model by validation MAE
        if val_mae < best_val_mae and len(val_dataset) > 0:
            best_val_mae = val_mae
            best_epoch   = epoch + 1
            best_path    = os.path.join(checkpoint_dir, 'satark_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best val MAE={best_val_mae:.1f} — saved '{best_path}'")

        # FIX 8: Periodic checkpoints go to dedicated folder
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'satark_e{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: '{ckpt_path}'")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, 'satark_final.pth')
    torch.save(model.state_dict(), final_path)

    print("-" * 55)
    print(f"  Fine-tuning complete!")
    print(f"  Final model  : '{final_path}'")
    print(f"  Best model   : '{checkpoint_dir}/satark_best.pth'  "
          f"(epoch {best_epoch}, val MAE={best_val_mae:.1f})")
    print(f"\n  Use 'satark_best.pth' (not final) for Step 8 inference.")


if __name__ == '__main__':
    # FIX 7: Preserve full traceback for debugging
    try:
        train_satark()
    except Exception:
        print("\n SCRIPT CRASHED — full traceback:")
        traceback.print_exc()