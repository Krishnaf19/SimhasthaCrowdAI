# SATARK - Headgear Aware Crowd Counting System
#### SATARK is an AI crowd-counter that accurately counts people at large cultural gatherings by recognizing turbans, veils, caps, and bare heads as separate classes instead of missing them like standard models do.
---

##  Executive Summary

Modern deep learning crowd counting models (e.g., standard CSRNet, MCNN, DM-Count) are trained on Western or urban benchmarks like **ShanghaiTech**, **UCF-QNRF**, and **WorldExpo**. These datasets predominantly depict crowds with bare heads or standard caps.

During massive cultural and religious gatherings such as **Simhastha Kumbh Mela**, over **60% to 70%** of attendees wear traditional headgear:
- **Turbans (Pagri, Dastar)**
- **Veils (Dupatta, Ghoonghat, Hijab)**
- **Saffron caps / Religious cloths**
- **Upward-folded hair buns (Jataa)**

###  Why Today's Models Fail
1. **Geometric Mismatch:** Traditional models look for circular/oval skin/hair geometries. Turbans and drapery distort standard contours into irregular, wide shapes.
2. **Texture Absence:** Fabric textures lack characteristic hair gradient signatures, causing standard models to drop heads as background artifacts.
3. **Severe Undercounting:** Modern models underestimate crowd numbers by **40% to 65%** in religious events, directly blinding public safety alert systems and triggering stampede risks.

###  What SATARK Solves
SATARK introduces **4-Class Simultaneous Density Estimation**:
- **`head`**: Bare or short-hair heads
- **`turban`**: Turbans, saffron hats, coiled top-knot hair (Jataa)
- **`veil`**: Dupattas, shawls, hijabs, and head coverings
- **`cap`**: Modern caps and brimmed hats

Instead of a single density map, SATARK outputs **4 specialized density channels**, capturing unique spatial textures for each group. The sum across all 4 channels delivers **robust, reliable total headcount accuracy**.

---

##  End-to-End Model Working Flow

```mermaid
flowchart TD
    A[Raw Input Image\nCultural Crowd Scene] --> B[Preprocessing & Standardization\nResize max 1000px, ImageNet Normalization]
    B --> C[VGG-16 Backbone\nFirst 13 Conv Layers - Shallow & Mid Features]
    C --> D[Dilated Conv Backend\n6 Dilated Layers - Expanded Receptive Field]
    D --> E[Squeeze-and-Excitation SE Block\nChannel Recalibration & Feature Attention]
    E --> F[Output 1x1 Convolution\nConv2d: 64 channels to 4 channels]
    
    F --> G1[Channel 0: Head Density Map]
    F --> G2[Channel 1: Turban Density Map]
    F --> G3[Channel 2: Veil Density Map]
    F --> G4[Channel 3: Cap Density Map]
    
    G1 & G2 & G3 & G4 --> H[Integral Summation\nsum over spatial grid & channels]
    
    H --> I[Detailed Breakdown Output\nHead: N1 | Turban: N2 | Veil: N3 | Cap: N4]
    H --> J[Total Crowd Count\nTotal = N1 + N2 + N3 + N4]
    
    J --> K{Safety Zone Evaluator}
    K -->|Count <= 50| L1[ SAFE Zone]
    K -->|51 to 150| L2[ NORMAL Zone]
    K -->|Count > 150| L3[ CRITICAL Zone Alert]
```

---

##  Step-by-Step Architecture Breakdown

```text
Input Image (H × W × 3)
      │
      ▼
[ VGG-16 Feature Extractor ]
├── First 13 Conv layers (pretrained ImageNet)
└── Frozen during initial warmup, unfrozen at epoch 15
      │
      ▼
[ Dilated Conv Backend ]
├── 6 dilated convolutional layers (dilation rate = 2)
└── Quadruples receptive field without reducing spatial resolution
      │
      ▼
[ Squeeze-and-Excitation (SE) Attention ]
├── Adaptive channel-wise feature recalibration
└── Enhances fabric/turban patterns, suppresses noisy backgrounds
      │
      ▼
[ Multi-Channel Output Head ]
└── Conv2d(64 → 4, kernel_size=1)
      │
      ▼
4 Density Maps (H/8 × W/8) ───► Total Count = Head + Turban + Veil + Cap
```

### 1. Feature Extraction (VGG-16 Frontend)
- Takes input image tensor `(B, 3, H, W)`.
- Pretrained weights extract fine-grained edge and textural cues.
- Frozen during initial epochs to protect pretrained representations, then fine-tuned.

### 2. Context Aggregation (Dilated Backend)
- Uses dilation rate $d=2$ without downsampling pooling.
- Enlarges receptive field exponentially without sacrificing spatial resolution (critical for dense, overlapping crowds).

### 3. Cultural Channel Attention (SE Block)
- Squeeze-and-Excitation layer adaptively weights feature channels.
- Enhances channels capturing textile folds, fabrics, and turban contours while suppressing background clutter.

### 4. 4-Channel Density Generation
- Final $1 \times 1$ convolution projects 64 feature maps into 4 separate density channels: `(B, 4, H/8, W/8)`.
- Supervised using **Density-Weighted MSE Loss**, placing a 3x higher penalty on ultra-dense crowd regions.

$$\text{Total Crowd Count} = \sum_{c=0}^{3} \sum_{h=1}^{H/8} \sum_{w=1}^{W/8} D_{c}(h, w)$$

---

##  Project Directory Structure

```text
SimhasthaCrowdAI/
├── satark/                      # Core Python Package
│   ├── data/
│   │   ├── builder.py          # CVAT XML annotation parser & indexer
│   │   ├── dataset.py          # PyTorch multi-channel Dataset loader
│   │   └── heatmap.py          # KDTree adaptive-sigma density map generator
│   ├── models/
│   │   └── csrnet.py           # 4-Channel CSRNet with SE Attention
│   ├── engine/
│   │   ├── trainer.py          # Training loop with Density-Weighted MSE
│   │   └── evaluator.py        # Validation, MAE, and RMSE metrics
│   └── utils/
│       ├── common.py           # Constants, classes, and path utilities
│       ├── inference.py        # Single & batch image inference pipeline
│       └── video.py            # Continuous video crowd engine & HUD overlay
├── scripts/                    # Command-Line Entry Points
│   ├── build_dataset.py        # Dataset preparation & heatmap builder
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Evaluation on test splits
│   └── process_video.py        # Continuous video crowd counter with HUD
├── app/                        # Production Web Dashboard
│   ├── main.py                 # Flask server & inference endpoints
│   ├── templates/              # Web UI templates
│   └── static/                 # Stylesheets & assets
├── configs/                    # Declarative YAML Configurations
│   ├── train.yaml              # Training hyper-parameters
│   └── inference.yaml          # Inference thresholds & visual settings
├── data/                       # Data Directory (gitignored)
│   ├── raw/                    # Original images & XML annotations
│   ├── processed/              # Formatted images, JSONs, and .npy heatmaps
│   └── splits/                 # Train / Test splits
├── checkpoints/                # Model weights (.pth)
├── outputs/                    # Visual predictions & heatmaps
├── requirements.txt            # Dependency list
├── setup.py                    # Package installer
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
```bash
# Clone repository
git clone https://github.com/your-org/SimhasthaCrowdAI.git
cd SimhasthaCrowdAI

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # On Windows
# source .venv/bin/activate     # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Data & Generate Heatmaps
Place raw images in `data/raw/images/` and CVAT XML annotations in `data/raw/annotations/`. Then run:
```bash
python scripts/build_dataset.py --gen-heatmaps --split-data
```

### 3. Train the Model
```bash
python scripts/train.py --epochs 80 --lr 5e-5 --batch-size 1
```

### 4. Evaluate Performance
```bash
python scripts/evaluate.py --split test
```

### 5. Continuous Video Crowd Counting (CLI)
```bash
python scripts/process_video.py --video-path path/to/crowd.mp4 --stride 1
```
Generates:
- `outputs/inference/result_<name>.mp4`: Full annotated video with live counter HUD, safety zone badges, and heatmap overlays.
- `outputs/inference/result_<name>_analytics.json`: Complete time-series telemetry with peak counts and timestamps.

### 6. Launch Web Dashboard (Image & Video Uploads)
```bash
python app/main.py
```
Open **`http://localhost:5000`** in your browser to analyze images or videos with real-time video playback and continuous count graphs.


---

##  Evaluation Metrics

| Metric | Purpose |
|---|---|
| **MAE (Mean Absolute Error)** | Measures average headcount discrepancy per frame. |
| **RMSE (Root Mean Square Error)** | Penalizes large estimation errors and crowd burst anomalies. |
| **Within 10% Accuracy** | % of test frames where headcount error is within $\pm 10\%$. |

---

##  License & Intended Use
Developed for research and public-safety operations during large-scale mass gatherings. Please ensure compliance with local privacy and surveillance guidelines prior to production deployment.
