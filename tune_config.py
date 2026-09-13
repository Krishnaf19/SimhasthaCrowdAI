#!/usr/bin/env python3
"""
Configuration Tuning Script
Helps you find optimal thresholds and preprocessing settings for your crowd images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
import numpy as np

from src.model import CSRNet, get_device
from src.evaluate import load_checkpoint
from src.inference import (
    _resize_for_inference, _preprocess, 
    _enhance_contrast,
    get_zone_label
)


def infer_with_config(
    img_path: str,
    model: CSRNet,
    device: torch.device,
    enhance_contrast: bool = True,
    use_histogram_eq: bool = False,
) -> Tuple[float, np.ndarray]:
    """Run inference with specified preprocessing options."""
    with Image.open(img_path) as pil_img:
        img_rgb = pil_img.convert('RGB')
    
    resized = _resize_for_inference(img_rgb)
    
    # Apply preprocessing options
    if enhance_contrast:
        resized = _enhance_contrast(resized)
    # Note: histogram equalization requires cv2, skipped if not available
    
    tensor = _preprocess(resized, device, enhance=False)  # Don't enhance twice
    
    with torch.no_grad():
        output = model(tensor)
    
    count = float(output.sum().item())
    density_map = output.squeeze().cpu().numpy()
    
    return count, density_map


def tune_thresholds(
    image_dir: str,
    manual_counts_json: str,
    model_path: str = 'checkpoints/satark_best.pth',
    output_file: str = 'threshold_tuning_results.json',
) -> None:
    """
    Systematically test different threshold combinations.
    
    Args:
        image_dir: Directory with test images
        manual_counts_json: JSON file with manual crowd counts
            Format: {"image_name.jpg": 150, "image_name2.jpg": 45, ...}
        model_path: Path to model checkpoint
        output_file: Output JSON file with results
    """
    # Load manual counts
    with open(manual_counts_json, 'r') as f:
        manual_counts = json.load(f)
    
    device = get_device()
    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    load_checkpoint(model_path, model, device)
    
    # Get predictions for all images
    predictions = {}
    for img_name, manual_count in manual_counts.items():
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        pred_count, _ = infer_with_config(img_path, model, device)
        predictions[img_name] = {
            'predicted': pred_count,
            'manual': manual_count,
            'error': abs(pred_count - manual_count),
            'pct_error': abs(pred_count - manual_count) / manual_count * 100 if manual_count > 0 else 0,
        }
        print(f"  {img_name}: Predicted={pred_count:.1f}, Manual={manual_count}, Error={predictions[img_name]['error']:.1f}")
    
    # Test different threshold combinations
    results = []
    safe_thresholds = range(20, 150, 10)
    normal_thresholds = range(100, 300, 20)
    
    print(f"\n🔍 Testing {len(safe_thresholds) * len(normal_thresholds)} threshold combinations...")
    
    for safe_th in safe_thresholds:
        for normal_th in normal_thresholds:
            if safe_th >= normal_th:
                continue
            
            correct_zones = 0
            total = 0
            
            for img_name, pred_data in predictions.items():
                manual_count = pred_data['manual']
                pred_count = pred_data['predicted']
                
                # Determine ground truth zone
                if manual_count <= safe_th:
                    true_zone = 'SAFE'
                elif manual_count <= normal_th:
                    true_zone = 'NORMAL'
                else:
                    true_zone = 'CRITICAL'
                
                # Determine predicted zone
                if pred_count <= safe_th:
                    pred_zone = 'SAFE'
                elif pred_count <= normal_th:
                    pred_zone = 'NORMAL'
                else:
                    pred_zone = 'CRITICAL'
                
                if true_zone == pred_zone:
                    correct_zones += 1
                total += 1
            
            zone_accuracy = correct_zones / total * 100 if total > 0 else 0
            
            results.append({
                'safe_threshold': safe_th,
                'normal_threshold': normal_th,
                'zone_accuracy': zone_accuracy,
                'correct_zones': correct_zones,
                'total': total,
            })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['zone_accuracy'], reverse=True)
    
    # Display top 10 results
    print(f"\n✅ Top 10 Threshold Combinations:")
    print("┌─────────────────────────────────────────────┐")
    print("│ SAFE │ NORMAL │ Zone Accuracy │ Correct/Total │")
    print("├─────────────────────────────────────────────┤")
    for i, result in enumerate(results[:10]):
        print(f"│ {result['safe_threshold']:>4} │ {result['normal_threshold']:>6} │ {result['zone_accuracy']:>12.1f}% │ {result['correct_zones']:>4}/{result['total']:<4} │")
    print("└─────────────────────────────────────────────┘")
    
    # Save all results
    with open(output_file, 'w') as f:
        json.dump({
            'predictions': predictions,
            'threshold_combinations': results,
            'best': results[0],
            'summary': {
                'total_images': len(predictions),
                'best_zone_accuracy': results[0]['zone_accuracy'],
                'recommended_safe_threshold': results[0]['safe_threshold'],
                'recommended_normal_threshold': results[0]['normal_threshold'],
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"\n🎯 RECOMMENDED SETTINGS:")
    print(f"   SAFE_THRESHOLD = {results[0]['safe_threshold']}")
    print(f"   NORMAL_THRESHOLD = {results[0]['normal_threshold']}")
    print(f"   Expected Zone Accuracy: {results[0]['zone_accuracy']:.1f}%")


def test_preprocessing_options(
    img_path: str,
    model_path: str = 'checkpoints/satark_best.pth',
) -> None:
    """Test different preprocessing combinations on a single image."""
    device = get_device()
    model = CSRNet(load_weights=False, freeze_frontend=False).to(device)
    load_checkpoint(model_path, model, device)
    
    print(f"\n🧪 Testing preprocessing options on: {os.path.basename(img_path)}")
    print("┌────────────────────────┬──────────────┐")
    print("│ Preprocessing Option   │ Crowd Count  │")
    print("├────────────────────────┼──────────────┤")
    
    configs = [
        ('No enhancement', False, False),
        ('Contrast only', True, False),
    ]
    
    for name, enhance_contrast, use_histogram_eq in configs:
        count, _ = infer_with_config(
            img_path, model, device,
            enhance_contrast=enhance_contrast,
            use_histogram_eq=use_histogram_eq
        )
        print(f"│ {name:<22} │ {count:>12.1f} │")
    
    print("└────────────────────────┴──────────────┘")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune model configuration for your crowd type.')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Threshold tuning command
    tune_parser = subparsers.add_parser('tune-thresholds', help='Tune thresholds using manual annotations')
    tune_parser.add_argument('--image-dir', required=True, help='Directory with test images')
    tune_parser.add_argument('--counts-json', required=True, help='JSON file with manual crowd counts')
    tune_parser.add_argument('--model-path', default='checkpoints/satark_best.pth', help='Model checkpoint path')
    tune_parser.add_argument('--output', default='threshold_tuning_results.json', help='Output JSON file')
    
    # Preprocessing test command
    prep_parser = subparsers.add_parser('test-preprocessing', help='Test preprocessing options on an image')
    prep_parser.add_argument('--image', required=True, help='Test image path')
    prep_parser.add_argument('--model-path', default='checkpoints/satark_best.pth', help='Model checkpoint path')
    
    args = parser.parse_args()
    
    if args.command == 'tune-thresholds':
        tune_thresholds(args.image_dir, args.counts_json, args.model_path, args.output)
    elif args.command == 'test-preprocessing':
        test_preprocessing_options(args.image, args.model_path)
    else:
        parser.print_help()
