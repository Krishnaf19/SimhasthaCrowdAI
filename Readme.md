### Simhastha Crowd Counting using CSRNet on Apple Silicon (MPS)

## Overview
AI-powered crowd density estimation system for Simhastha Kumbh Mela
using fine-tuned CSRNet with VGG16 backbone on Apple M-series hardware.

## Results (15 training images)
- MAE: 28.27 people
- RMSE: 40.69 people  
- Mean Relative Error: 32%
- Best prediction: 5.4% error

## Execution Sequence
```
01_build_master_index.py
02_stratified_train_test_split.py
03_generate_heatmaps.py
04_visualize_heatmaps.py
05_evaluate_baseline.py
06_fine_tune.py
07_evaluate_finetuned_model.py
08_visualize_predictions.py
09_batch_inference.py
10_alert_inference.py
```

## Requirements
```
torch torchvision scipy numpy matplotlib Pillow opencv-python pandas
```

## Hardware
Optimized for Apple Silicon (MPS backend). Falls back to CUDA or CPU.

## Code structure
The project now uses a clean `src/` package for dataset, model, training, evaluation, visualization, and inference code.

## Improvements added
- Train-time augmentation with color jitter, random horizontal flip, and randomized rotation to improve robustness for turbans, saffron hats, and varied headgear.
- Optional squeeze-and-excitation block in CSRNet for better feature calibration.
- CLI entrypoints for all major steps, with `argparse` support to run custom paths and hyperparameters.

## Dataset
19 labeled Simhastha crowd images. Expanding to 100 images.
Annotations in CVAT XML format, converted to Gaussian density maps.