import os
import traceback
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import SimhasthaDataset
from .evaluate import load_checkpoint
from .model import CSRNet, get_device, clear_device_cache


class DensityWeightedMSELoss(nn.Module):
    def __init__(self, density_threshold: float = 150.0, penalty: float = 2.5):
        super().__init__()
        self.threshold = density_threshold
        self.penalty = penalty
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        
        # Apply penalty for dense crowds
        if target.sum() > self.threshold:
            mse_loss = mse_loss * self.penalty
        
        return mse_loss


def run_epoch(
    model: CSRNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """Run one epoch with improved gradient handling and metrics"""
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for img, target in loader:
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss = criterion(output, target)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Improved gradient clipping with adaptive scaling
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            total_loss += loss.item()
            
            # Compute MAE
            pred_count = output.sum().item()
            target_count = target.sum().item()
            mae = abs(pred_count - target_count)
            total_mae += mae
            
            # Compute RMSE
            rmse = (pred_count - target_count) ** 2
            total_rmse += rmse
            
            # Compute MAPE (avoid division by zero)
            if target_count > 0.01:
                mape = abs(pred_count - target_count) / target_count
                total_mape += mape

    size = len(loader)
    avg_loss = (total_loss / size) if size else 0.0
    avg_mae = (total_mae / size) if size else 0.0
    avg_rmse = (total_rmse / size) ** 0.5 if size else 0.0
    avg_mape = (total_mape / size) if size else 0.0
    
    return avg_loss, avg_mae, avg_rmse, avg_mape


def train_satark(
    weights_path: str = 'baseline_weights.pth',
    data_root: str = 'data',
    checkpoint_dir: str = 'checkpoints',
    num_epochs: int = 50,
    lr: float = 1e-5,
    batch_size: int = 1,
    density_threshold: float = 150.0,
    penalty: float = 2.5,
    save_every: int = 10,
    use_se: bool = True,
    unfreeze_after: int = 10,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-7,
    early_stopping_patience: int = 15,
    warmup_epochs: int = 2,
    grad_clip: float = 1.0,
) -> None:
    print('Step 7: SATARK Fine-Tuning Engine')
    print('=' * 55)
    device = get_device()
    print(f'  Device: {device}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = CSRNet(load_weights=True, freeze_frontend=True, use_se=use_se).to(device)
    if os.path.exists(weights_path):
        print(f'  Loading baseline weights from {weights_path}')
        load_checkpoint(weights_path, model, device)
    else:
        print(f'  Baseline weights not found at {weights_path}; using VGG16 init only.')

    try:
        train_dataset = SimhasthaDataset(root_dir=data_root, split='all')
        val_dataset = SimhasthaDataset(root_dir=data_root, split='all')  # Use all for validation too
    except Exception as exc:
        print(f'  Dataset initialization error: {exc}')
        return

    pin_memory = device.type != 'cpu'
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    criterion = DensityWeightedMSELoss(
        density_threshold=density_threshold, 
        penalty=penalty
    )
    optimizer = optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    print(f'  Train images    : {len(train_dataset)}')
    print(f'  Val images      : {len(val_dataset)}')
    print(f'  Epochs          : {num_epochs}')
    print(f'  LR              : {lr}')
    print(f'  Weight decay    : {weight_decay}')
    print(f'  Schedule factor : x{scheduler_factor} patience={scheduler_patience}')
    print(f'  Early stopping  : patience={early_stopping_patience}')
    print(f'  Penalty         : x{penalty} for dense crowds >{density_threshold}')
    print(f'  Use SE          : {use_se}')
    print(f'  Unfreeze epoch  : {unfreeze_after}')
    print(f'  Warmup epochs   : {warmup_epochs}')
    print(f'  Grad clip       : {grad_clip}')
    print('=' * 55)

    best_val_mae = float('inf')
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(1, num_epochs + 1):
        # Learning rate warmup
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * warmup_factor
        
        if epoch == unfreeze_after:
            model.unfreeze_frontend()
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.25, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=min_lr,
            )
            print(f'  >>> Frontend unfrozen at epoch {epoch} for full model fine-tuning.')

        train_loss, train_mae, train_rmse, train_mape = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            is_train=True,
            grad_clip=grad_clip,
        )
        clear_device_cache(device)
        val_loss, val_mae, val_rmse, val_mape = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            is_train=False,
            grad_clip=grad_clip,
        )

        scheduler.step(val_mae)

        print(
            f'Epoch {epoch}/{num_epochs} | '
            f'Train: loss={train_loss:.6f} MAE={train_mae:.2f} RMSE={train_rmse:.2f} MAPE={train_mape:.4f} | '
            f'Val: loss={val_loss:.6f} MAE={val_mae:.2f} RMSE={val_rmse:.2f} MAPE={val_mape:.4f}'
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            early_stop_counter = 0
            best_path = os.path.join(checkpoint_dir, 'satark_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f'  ★ New best validation MAE: {best_val_mae:.2f}; saved {best_path}')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                print(f'  ⚠ Early stopping triggered after {early_stop_counter} epochs without improvement.')
                break

        if epoch % save_every == 0:
            path = os.path.join(checkpoint_dir, f'satark_e{epoch}.pth')
            torch.save(model.state_dict(), path)
            print(f'  Checkpoint saved: {path}')

    final_path = os.path.join(checkpoint_dir, 'satark_final.pth')
    torch.save(model.state_dict(), final_path)
    print('=' * 55)
    print(f'Fine-tuning complete. Final model saved to {final_path}')
    print(f'Best model: {os.path.join(checkpoint_dir, "satark_best.pth")} (epoch {best_epoch})')


if __name__ == '__main__':
    try:
        train_satark()
    except Exception:
        traceback.print_exc()
