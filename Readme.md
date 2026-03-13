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
step1_build_csv.py
fix_stratified_split.py
step2_generate_data.py
step3_visualize_data.py
step5_model.py
step6_baseline_eval.py
step7_fine_tune.py
step8_final_comparison.py
step9_visualize.py
step10_batch_inference.py
step11_alert_systems.py
```

## Requirements
```
torch torchvision scipy numpy matplotlib Pillow opencv-python pandas
```

## Hardware
Optimized for Apple Silicon (MPS backend). Falls back to CUDA or CPU.

## Dataset
19 labeled Simhastha crowd images. Expanding to 100 images.
Annotations in CVAT XML format, converted to Gaussian density maps.