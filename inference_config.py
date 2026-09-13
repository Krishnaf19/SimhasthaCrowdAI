"""
Configuration for model inference on diverse crowd types.
Edit these values to improve performance on your specific crowd images.
"""

# ============================================================================
# INFERENCE THRESHOLDS - Calibrate these for your crowd type
# ============================================================================
# Set these based on your crowd density expectations
# Original Simhastha values: SAFE=100, NORMAL=250

SAFE_THRESHOLD = 50  
"""
Crowd count <= this is classified as SAFE.
Lower values = stricter safety classification.
For dense crowds: increase to 75-100
For sparse crowds: decrease to 25-50
"""

NORMAL_THRESHOLD = 150  
"""
Crowd count between SAFE and this is NORMAL.
Anything above NORMAL is CRITICAL.
For dense crowds: increase to 200-250
For sparse crowds: decrease to 100-150
"""

# ============================================================================
# PREPROCESSING OPTIONS - Enable/disable enhancements
# ============================================================================

ENABLE_CONTRAST_ENHANCEMENT = True
"""
Enhance image contrast to handle varying lighting conditions.
Helps with: dark images, low-contrast crowds, mixed lighting
Disable if: images are already high-contrast or artificial
"""

ENABLE_HISTOGRAM_EQUALIZATION = False
"""
Apply Adaptive Histogram Equalization (CLAHE) for lighting-invariant features.
Requires OpenCV (cv2) to be installed.
Helps with: extreme lighting variations, backlighting
Disable if: standard preprocessing is sufficient, or cv2 not available
Note: Can sometimes over-enhance and create artifacts
"""

# ============================================================================
# MODEL INFERENCE SETTINGS
# ============================================================================

MAX_SIDE_PX = 1000
"""
Maximum image size for inference (longer side).
Larger = slower but potentially more accurate
Smaller = faster but may lose fine details
Typical range: 800-1200 pixels
"""

IMAGE_NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],  # ImageNet normalization
    'std': [0.229, 0.224, 0.225],
}
"""
ImageNet normalization values used for model training.
Only change if you fine-trained with different normalization.
"""

# ============================================================================
# DEBUGGING & ANALYSIS
# ============================================================================

SAVE_DENSITY_MAP = True
"""Whether to save density map visualization."""

SAVE_INTERMEDIATE_IMAGES = False
"""Whether to save preprocessing intermediate outputs for debugging."""

VERBOSE_INFERENCE = False
"""Whether to print detailed inference information."""

# ============================================================================
# FINE-TUNING RECOMMENDATIONS
# ============================================================================
"""
If thresholds alone don't work:

1. QUICK FIXES:
   - Try different threshold values
   - Enable/disable contrast enhancement
   - Use images of similar size/type for testing

2. MEDIUM FIXES (1-2 hours):
   - Collect 5-10 new crowd images with manual counts
   - Fine-tune with just these images (5-10 epochs)
   - Use different learning rates: 1e-5 to 1e-4

3. LONG-TERM (1-2 weeks):
   - Collect 50+ diverse crowd images
   - Create proper density map annotations
   - Re-train model with mixed Simhastha + new data
   - Track metrics: MAE, MAPE, zone accuracy

See MODEL_GENERALIZATION_ANALYSIS.md for detailed guidance.
"""
