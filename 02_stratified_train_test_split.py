"""
Run this ANY TIME you add new images to re-split with stratified density bands.
Replaces the random split in simhastha_master_index.csv with a stratified one
so every density range is proportionally represented in both Train and Test.
"""
import pandas as pd
import random

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.8

# Density bands — adjust upper bound if your max count changes
BANDS = [
    (0,    50,   '0-50'),
    (50,   150,  '50-150'),
    (150,  300,  '150-300'),
    (300,  600,  '300-600'),
    (600,  9999, '600+'),
]

def get_band(count):
    for lo, hi, label in BANDS:
        if lo <= count < hi:
            return label
    return '600+'

random.seed(RANDOM_SEED)

df = pd.read_csv('simhastha_master_index.csv')

# Only stratify labeled images
labeled   = df[df['status'] == 'Labeled'].copy()
unlabeled = df[df['status'] != 'Labeled'].copy()

labeled['density_band'] = labeled['head_count'].apply(get_band)

train_names = set()
test_names  = set()

print("Stratified split breakdown:")
print(f"{'Band':<12} {'Total':>6} {'Train':>6} {'Test':>6}")
print("-" * 35)

for _, _, label in BANDS:
    band_imgs = labeled[labeled['density_band'] == label]['image_name'].tolist()
    if not band_imgs:
        continue
    random.shuffle(band_imgs)
    split_idx = max(1, int(len(band_imgs) * TRAIN_RATIO))
    # Ensure at least 1 in test if band has >1 image
    if len(band_imgs) > 1:
        split_idx = min(split_idx, len(band_imgs) - 1)
    t  = set(band_imgs[:split_idx])
    te = set(band_imgs[split_idx:])
    train_names.update(t)
    test_names.update(te)
    print(f"{label:<12} {len(band_imgs):>6} {len(t):>6} {len(te):>6}")

print("-" * 35)
print(f"{'TOTAL':<12} {len(labeled):>6} {len(train_names):>6} {len(test_names):>6}")

# Apply new splits back to dataframe
def assign_split(row):
    if row['status'] != 'Labeled':
        return 'Inference'
    if row['image_name'] in train_names:
        return 'Train'
    return 'Test'

df['split_assignment'] = df.apply(assign_split, axis=1)
df.to_csv('simhastha_master_index.csv', index=False)

print("\nCSV updated with stratified splits.")
print("Now run: python 03_generate_heatmaps.py && python 06_fine_tune.py")
