import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from ..data.dataset   import SimhasthaDataset
from ..engine.evaluator import load_checkpoint
from ..models.csrnet  import CSRNet, get_device, clear_device_cache
from ..utils.inference import _checkpoint_output_channels


class DensityMSELoss(nn.Module):
    def __init__(self, density_threshold=150.0, penalty=2.5):
        super().__init__()
        self.threshold, self.penalty = density_threshold, penalty
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        loss = self.mse(pred, target)
        return loss * self.penalty if target.sum() > self.threshold else loss


def _run_epoch(model, loader, criterion, optimizer, device, is_train, grad_clip=1.0):
    model.train() if is_train else model.eval()
    total_loss = total_mae = total_se = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for img, target in loader:
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss   = criterion(output, target)
            if is_train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item()
            pred_n, gt_n = output.sum().item(), target.sum().item()
            total_mae += abs(pred_n - gt_n)
            total_se  += (pred_n - gt_n) ** 2
    n = max(len(loader), 1)
    return total_loss / n, total_mae / n, (total_se / n) ** 0.5


def train_satark(
    weights_path='checkpoints/baseline_weights.pth',
    data_root='data',
    checkpoint_dir='checkpoints',
    num_epochs=80,
    lr=5e-5,
    batch_size=1,
    density_threshold=150.0,
    penalty=3.0,
    save_every=10,
    use_se=True,
    unfreeze_after=15,
    weight_decay=1e-4,
    scheduler_patience=3,
    scheduler_factor=0.5,
    min_lr=1e-7,
    early_stopping_patience=15,
    grad_clip=1.0,
    num_workers=None,
    output_channels=None,
):
    print('SATARK Training — Headgear-Aware Crowd Counting')
    print('=' * 52)
    device = get_device()
    print('  Device:', device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if weights_path and os.path.exists(weights_path) and output_channels is None:
        output_channels = _checkpoint_output_channels(weights_path, device)
    if output_channels is None:
        output_channels = 1 if os.path.exists(os.path.join(checkpoint_dir, 'satark_best.pth')) else 4

    single_channel = (output_channels == 1)
    print(f'  Model output channels: {output_channels} (single_channel={single_channel})')

    model = CSRNet(load_weights=True, freeze_frontend=True, use_se=use_se, output_channels=output_channels).to(device)
    if weights_path and os.path.exists(weights_path):
        load_checkpoint(weights_path, model, device)

    try:
        train_ds = SimhasthaDataset(root_dir=data_root, split='train', single_channel=single_channel)
        val_ds   = SimhasthaDataset(root_dir=data_root, split='all', single_channel=single_channel)
    except Exception as e:
        print('Dataset error:', e); return

    pin = device.type != 'cpu'
    workers = num_workers if num_workers is not None else (0 if os.name == 'nt' else 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=workers, pin_memory=pin)

    criterion = DensityMSELoss(density_threshold, penalty)
    optimizer = optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                    factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr)

    print('  Train:', len(train_ds), '| Val:', len(val_ds), '| Epochs:', num_epochs)
    print('=' * 52)
    best_mae, no_improve = float('inf'), 0

    for epoch in range(1, num_epochs + 1):
        if epoch == unfreeze_after:
            model.unfreeze_frontend()
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.25, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                            factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr)
            print('  >>> Frontend unfrozen at epoch', epoch)

        tl, tm, tr = _run_epoch(model, train_loader, criterion, optimizer, device, True,  grad_clip)
        clear_device_cache(device)
        vl, vm, vr = _run_epoch(model, val_loader,   criterion, optimizer, device, False, grad_clip)
        scheduler.step(vm)

        print('Ep {}/{} | Train loss={:.4f} MAE={:.1f} RMSE={:.1f} | Val loss={:.4f} MAE={:.1f} RMSE={:.1f}'.format(
            epoch, num_epochs, tl, tm, tr, vl, vm, vr), flush=True)

        if vm < best_mae:
            best_mae, no_improve = vm, 0
            best_path = os.path.join(checkpoint_dir, 'satark_best.pth')
            torch.save(model.state_dict(), best_path)
            print('  * Best MAE {:.2f} -> {}'.format(best_mae, best_path), flush=True)
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print('  Early stop at epoch', epoch, flush=True); break

        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'satark_e{}.pth'.format(epoch)))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'satark_final.pth'))
    print('Training complete. Best MAE:', round(best_mae, 2), flush=True)
