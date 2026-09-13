import os
import numpy as np, torch
from torch.utils.data import DataLoader
from ..data.dataset   import SimhasthaDataset
from ..models.csrnet  import CSRNet, get_device, clear_device_cache

STATE_DICT_KEYS = ['state_dict', 'model_state_dict', 'model', 'net']

def load_checkpoint(weights_path, model, device):
    if not os.path.exists(weights_path):
        print('Checkpoint not found:', weights_path); return False
    try:    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except: ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt if not isinstance(ckpt, dict) else None
    if isinstance(ckpt, dict):
        for k in STATE_DICT_KEYS:
            if k in ckpt: state = ckpt[k]; break
        if state is None: state = ckpt
    try:
        model.load_state_dict(state, strict=True)
        print('Weights loaded (strict).'); return True
    except:
        try:
            missing, _ = model.load_state_dict(state, strict=False)
            print('Partial weights loaded. Missing:', missing); return True
        except Exception as e:
            print('Load failed:', e); return False

def _eval_loop(model, loader, device):
    model.eval()
    entries = []
    with torch.no_grad():
        for idx, (img, target) in enumerate(loader):
            img, target = img.to(device), target.to(device)
            output = model(img)
            gt, pred = float(target.sum().item()), float(output.sum().item())
            entries.append({'idx': idx, 'gt': gt, 'pred': pred, 'mae': abs(gt - pred)})
    return entries

def run_satark_metrics(model_path='checkpoints/satark_best.pth',
                       data_root='data', split='test',
                       crop_size=512, downsample=8):
    print('Evaluating SATARK...')
    device = get_device()
    clear_device_cache(device)
    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    if not load_checkpoint(model_path, model, device): return {}
    try:
        ds = SimhasthaDataset(root_dir=data_root, split=split,
                              crop_size=crop_size, downsample=downsample)
    except Exception as e:
        print('Dataset error:', e); return {}
    entries = _eval_loop(model, DataLoader(ds, batch_size=1, shuffle=False, num_workers=0), device)
    if not entries:
        print('No images processed.'); return {}
    mae_vals = [e['mae'] for e in entries]
    sq_err   = [(e['gt'] - e['pred']) ** 2 for e in entries]
    rel_err  = [e['mae'] / e['gt'] * 100 if e['gt'] > 0 else 0.0 for e in entries]
    avg_mae  = float(np.mean(mae_vals))
    rmse     = float(np.sqrt(np.mean(sq_err)))
    within10 = sum(1 for e in entries
                   if (e['gt'] == 0 and e['pred'] == 0)
                   or (e['gt'] > 0 and abs(e['pred'] - e['gt']) / e['gt'] <= 0.10))
    print('  MAE      :', round(avg_mae, 2))
    print('  RMSE     :', round(rmse, 2))
    print('  Rel err  :', round(float(np.mean(rel_err)), 1), '%')
    print('  Within10%:', within10, '/', len(entries))
    return {'mae': avg_mae, 'rmse': rmse, 'within_10': within10 / len(entries) * 100, 'entries': entries}
