import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SimhasthaDataset
from .model import CSRNet, get_device, clear_device_cache


STATE_DICT_KEYS = ['state_dict', 'model_state_dict', 'model', 'net']


def load_checkpoint(weights_path: str, model: CSRNet, device: torch.device) -> bool:
    if not os.path.exists(weights_path):
        print(f"Checkpoint not found: {weights_path}")
        return False

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    state_dict = checkpoint if not isinstance(checkpoint, dict) else None
    if isinstance(checkpoint, dict):
        for key in STATE_DICT_KEYS:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint

    if not isinstance(state_dict, dict):
        print(f"Invalid checkpoint format: {type(state_dict)}")
        return False

    try:
        model.load_state_dict(state_dict, strict=True)
        print('Weights loaded successfully (strict match).')
        return True
    except Exception as exc:
        print(f'Strict load failed: {exc}')
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f'  Missing keys: {missing}')
            if unexpected:
                print(f'  Unexpected keys: {unexpected}')
            print('Partial weights loaded.')
            return True
        except Exception as exc2:
            print(f'Failed to load checkpoint: {exc2}')
            return False


def _run_eval_loop(model, loader, device):
    model.eval()
    total_mae = 0.0
    entries = []
    with torch.no_grad():
        for idx, (img, target) in enumerate(loader):
            img, target = img.to(device), target.to(device)
            output = model(img)
            gt = float(target.sum().item())
            pred = float(output.sum().item())
            mae = abs(gt - pred)
            entries.append({'idx': idx, 'gt': gt, 'pred': pred, 'mae': mae})
            total_mae += mae
    return entries, total_mae


def run_baseline_comparison(
    weights_path: str = 'baseline_weights.pth',
    data_root: str = 'data',
    crop_size: int = 512,
    downsample: int = 8,
) -> Dict:
    device = get_device()
    print('Step 6: Baseline comparison')
    print(f'  Device: {device}')

    model = CSRNet(load_weights=True, freeze_frontend=False).to(device)
    if not load_checkpoint(weights_path, model, device):
        return {}

    try:
        dataset = SimhasthaDataset(root_dir=data_root, split='Test', crop_size=crop_size, downsample=downsample)
    except Exception as exc:
        print(f'  Could not create dataset: {exc}')
        return {}

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    entries, total_mae = _run_eval_loop(model, loader, device)
    if not entries:
        print('  No test images processed.')
        return {}

    avg_mae = total_mae / len(entries)
    print(f'  Baseline MAE: {avg_mae:.2f}')
    return {'mae': avg_mae, 'entries': entries}


def run_satark_metrics(
    model_path: str = 'checkpoints/satark_best.pth',
    data_root: str = 'data',
    crop_size: int = 512,
    downsample: int = 8,
) -> Dict:
    print('Step 8: SATARK final evaluation')
    device = get_device()
    clear_device_cache(device)
    print(f'  Device: {device}')
    print(f'  Model path: {model_path}')

    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    if not load_checkpoint(model_path, model, device):
        return {}

    try:
        dataset = SimhasthaDataset(root_dir=data_root, split='Test', crop_size=crop_size, downsample=downsample)
    except Exception as exc:
        print(f'  Dataset error: {exc}')
        return {}

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    entries, total_mae = _run_eval_loop(model, loader, device)
    if not entries:
        print('  No images were processed.')
        return {}

    gt_values = [entry['gt'] for entry in entries]
    mae_values = [entry['mae'] for entry in entries]
    sq_err = [(entry['gt'] - entry['pred']) ** 2 for entry in entries]
    rel_err = [(entry['mae'] / entry['gt'] * 100) if entry['gt'] > 0 else 0.0 for entry in entries]

    avg_mae = float(np.mean(mae_values))
    rmse = float(np.sqrt(np.mean(sq_err)))
    avg_rel = float(np.mean(rel_err))
    best = min(entries, key=lambda x: x['mae'])
    worst = max(entries, key=lambda x: x['mae'])

    print(f'  MAE           : {avg_mae:.2f}')
    print(f'  RMSE          : {rmse:.2f}')
    print(f'  Mean rel err  : {avg_rel:.1f}%')
    print(f'  GT range      : {min(gt_values):.0f}–{max(gt_values):.0f}')
    print(f'  Best result   : idx={best["idx"]}, err={best["mae"]:.1f}')
    print(f'  Worst result  : idx={worst["idx"]}, err={worst["mae"]:.1f}')

    return {
        'mae': avg_mae,
        'rmse': rmse,
        'rel_err': avg_rel,
        'best': best,
        'worst': worst,
        'entries': entries,
    }
