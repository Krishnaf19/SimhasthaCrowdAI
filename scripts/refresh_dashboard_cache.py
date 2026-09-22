import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from satark.utils.inference import infer_image, _checkpoint_output_channels, get_zone
from satark.models.csrnet import CSRNet, get_device
from satark.utils.common import list_image_files
from satark.engine.evaluator import load_checkpoint

def refresh_cache():
    device = get_device()
    model_path = 'checkpoints/satark_best.pth'
    output_channels = _checkpoint_output_channels(model_path, device)
    model = CSRNet(load_weights=False, freeze_frontend=False, output_channels=output_channels).to(device)
    load_checkpoint(model_path, model, device)
    model.eval()

    image_dir = 'data/processed/images'
    output_dir = 'outputs/inference'
    os.makedirs(output_dir, exist_ok=True)

    cache_file = os.path.join(output_dir, 'dashboard_results.json')
    existing_results = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for r in data.get('results', []):
                    existing_results[r['image']] = r
        except Exception:
            pass

    all_images = list_image_files(image_dir)
    print(f"Total images on disk in {image_dir}: {len(all_images)}")
    print(f"Previously cached: {len(existing_results)}")

    processed_new = 0
    for idx, img_name in enumerate(all_images):
        need_infer = (img_name not in existing_results) or (not os.path.exists(existing_results[img_name].get('path', '')))
        if need_infer:
            img_path = os.path.join(image_dir, img_name)
            r = infer_image(img_path, model=model, device=device, output_dir=output_dir)
            if r:
                r['display_name'] = img_name
                bn = os.path.basename(r.get('path', ''))
                r['view_url'] = f'/analysis/{bn}'
                r['image_url'] = f'/outputs/{bn}'
                r['url'] = r['image_url']
                existing_results[img_name] = r
                processed_new += 1
                if processed_new % 5 == 0 or processed_new == len(all_images):
                    print(f"  Processed {processed_new} new image inferences...")

    final_results = [existing_results[img] for img in all_images if img in existing_results]
    zone_counts = {'SAFE': 0, 'NORMAL': 0, 'CRITICAL': 0}
    for r in final_results:
        z = r.get('zone', 'SAFE')
        zone_counts[z] = zone_counts.get(z, 0) + 1

    counts = [r['count'] for r in final_results]
    avg_count = sum(counts) / max(len(counts), 1)

    payload = {
        'results': final_results,
        'total_images': len(final_results),
        'safe_count': zone_counts['SAFE'],
        'normal_count': zone_counts['NORMAL'],
        'critical_count': zone_counts['CRITICAL'],
        'avg_count': round(avg_count, 1)
    }

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print("=" * 50)
    print(f"Dashboard cache updated successfully!")
    print(f"Total images now in cache: {len(final_results)}")
    print(f"Breakdown: Safe={zone_counts['SAFE']}, Normal={zone_counts['NORMAL']}, Critical={zone_counts['CRITICAL']}")
    print(f"Average count: {round(avg_count, 1)}")
    print("=" * 50)

if __name__ == '__main__':
    refresh_cache()
