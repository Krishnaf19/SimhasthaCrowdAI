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
        loss = self.mse(pred, target)
        if target.sum() > self.threshold:
            loss = loss * self.penalty
        return loss


def run_epoch(
    model: CSRNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
) -> Tuple[float, float]:
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_mae = 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for img, target in loader:
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss = criterion(output, target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_mae += abs(output.sum() - target.sum()).item()

    size = len(loader)
    return (total_loss / size) if size else 0.0, (total_mae / size) if size else 0.0


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
        train_dataset = SimhasthaDataset(root_dir=data_root, split='Train')
        val_dataset = SimhasthaDataset(root_dir=data_root, split='Test')
    except Exception as exc:
        print(f'  Dataset initialization error: {exc}')
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    criterion = DensityWeightedMSELoss(density_threshold=density_threshold, penalty=penalty)
    optimizer = optim.Adam(model.trainable_parameters(), lr=lr)

    print(f'  Train images: {len(train_dataset)}')
    print(f'  Val images  : {len(val_dataset)}')
    print(f'  Epochs      : {num_epochs}')
    print(f'  LR          : {lr}')
    print(f'  Penalty     : x{penalty} for dense crowds >{density_threshold}')
    print(f'  Use SE      : {use_se}')
    print('=' * 55)

    best_val_mae = float('inf')
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        clear_device_cache(device)
        val_loss, val_mae = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)

        print(
            f'Epoch {epoch}/{num_epochs} | '
            f'Train loss={train_loss:.6f}, MAE={train_mae:.2f} | '
            f'Val loss={val_loss:.6f}, MAE={val_mae:.2f}'
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_path = os.path.join(checkpoint_dir, 'satark_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f'  ★ New best validation MAE: {best_val_mae:.2f}; saved {best_path}')

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
