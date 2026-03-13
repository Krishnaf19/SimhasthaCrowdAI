# import os
# import cv2
# import numpy as np
# import json
# import shutil
# import random
# from scipy.ndimage import gaussian_filter

# def generate_heatmaps():
#     image_dir = 'data/images'
#     anno_dir = 'data/annotations'
#     output_dir = 'data/heatmaps'
#     os.makedirs(output_dir, exist_ok=True)

#     print("Step 2: Generating Density Maps...")
    
#     image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for img_name in image_files:
#         img = cv2.imread(os.path.join(image_dir, img_name))
#         if img is None: continue
#         h, w = img.shape[:2]
        
#         density_map = np.zeros((h, w), dtype=np.float32)
        
#         json_path = os.path.join(anno_dir, os.path.splitext(img_name)[0] + '.json')
#         if not os.path.exists(json_path):
#             print(f"Skipping {img_name}, no JSON found.")
#             continue
            
#         with open(json_path, 'r') as f:
#             points = json.load(f)
            
#         for pt in points:
#             x, y = int(pt['x']), int(pt['y'])
#             if 0 <= x < w and 0 <= y < h:
#                 density_map[y, x] = 1.0
        
#         # Apply Gaussian Blur
#         density_map = gaussian_filter(density_map, sigma=4)
        
#         # --- CRITICAL CORRECTION: Count Preservation ---
#         if len(points) > 0:
#             current_sum = np.sum(density_map)
#             if current_sum > 0:
#                 density_map = density_map * (len(points) / current_sum)
        
#         np.save(os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy'), density_map)
#         print(f"Generated Heatmap for: {img_name} ({len(points)} people)")

# def split_data():
#     print(" Splitting into Train/Test folders...")
#     # Get base names that have BOTH an image and a heatmap
#     image_dir = 'data/images'
#     heat_dir = 'data/heatmaps'
    
#     valid_files = []
#     for f in os.listdir(image_dir):
#         base = os.path.splitext(f)[0]
#         if os.path.exists(os.path.join(heat_dir, base + '.npy')):
#             valid_files.append(f)

#     random.shuffle(valid_files)
#     split_idx = int(0.8 * len(valid_files))
#     train_files = valid_files[:split_idx]
    
#     for folder in ['Train', 'Test']:
#         # Use shutil.rmtree with ignore_errors to avoid 'Folder Busy' issues on Mac
#         if os.path.exists(f'data/{folder}'):
#             shutil.rmtree(f'data/{folder}', ignore_errors=True)
#         os.makedirs(f'data/{folder}/images', exist_ok=True)
#         os.makedirs(f'data/{folder}/heatmaps', exist_ok=True)
        
#     for f in valid_files:
#         dest = 'Train' if f in train_files else 'Test'
#         base = os.path.splitext(f)[0]
        
#         shutil.copy(os.path.join(image_dir, f), f'data/{dest}/images/{f}')
#         shutil.copy(os.path.join(heat_dir, base + '.npy'), f'data/{dest}/heatmaps/{base}.npy')
    
#     print(f" Split Complete! {len(train_files)} Train, {len(valid_files)-split_idx} Test.")

# if __name__ == "__main__":
#     generate_heatmaps()
#     split_data()


import os
import csv
import cv2
import numpy as np
import json
import shutil
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

RANDOM_SEED   = 42
MIN_SIGMA     = 4
MAX_SIGMA     = 15
K_NEIGHBOURS  = 3
SIGMA_BUCKETS = 5   # group sigmas into N bands → N blur passes instead of N_people passes


def compute_sigmas(coords: np.ndarray) -> np.ndarray:
    """
    Vectorised: compute adaptive sigma for every point at once using KDTree.
    Returns a 1-D float32 array of sigma values, one per point.
    """
    n = len(coords)
    if n == 1:
        return np.array([float(MIN_SIGMA)], dtype=np.float32)

    tree   = KDTree(coords)
    k      = min(K_NEIGHBOURS + 1, n)          # +1 because self is always nearest
    dists, _ = tree.query(coords, k=k)         # shape (n, k)
    # drop self-column (index 0, distance 0) and average the rest
    avg_dists = dists[:, 1:].mean(axis=1)      # shape (n,)
    sigmas    = np.clip(avg_dists * 0.25, MIN_SIGMA, MAX_SIGMA).astype(np.float32)
    return sigmas


def generate_density_map(points: list, h: int, w: int) -> np.ndarray:
    """
    Vectorised bucket-blur approach.
    Instead of one gaussian_filter per person (extremely slow for 300+ people),
    group points by similar sigma → one blur pass per bucket.
    Result: SIGMA_BUCKETS blur passes instead of N_people blur passes.
    """
    if len(points) == 0:
        return np.zeros((h, w), dtype=np.float32)

    coords = np.array([[pt['x'], pt['y']] for pt in points], dtype=np.float64)
    sigmas = compute_sigmas(coords)

    density_map = np.zeros((h, w), dtype=np.float32)

    sigma_min_val = sigmas.min()
    sigma_max_val = sigmas.max()

    if sigma_min_val == sigma_max_val:
        buckets = [(float(sigma_min_val), np.arange(len(points)))]
    else:
        bucket_edges = np.linspace(sigma_min_val, sigma_max_val, SIGMA_BUCKETS + 1)
        buckets = []
        for i in range(SIGMA_BUCKETS):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (sigmas >= lo) & (sigmas <= hi if i == SIGMA_BUCKETS - 1 else sigmas < hi)
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                buckets.append((float(sigmas[idxs].mean()), idxs))

    for sigma, idxs in buckets:
        bucket_map = np.zeros((h, w), dtype=np.float32)
        for i in idxs:
            x = int(np.clip(coords[i, 0], 0, w - 1))
            y = int(np.clip(coords[i, 1], 0, h - 1))
            bucket_map[y, x] += 1.0
        density_map += gaussian_filter(bucket_map, sigma=sigma)

    # Count preservation
    total_sum = float(np.sum(density_map))
    expected  = float(len(points))
    if total_sum > 0 and abs(total_sum - expected) > 0.5:
        density_map *= expected / total_sum

    return density_map


def generate_heatmaps(
    image_dir:  str = 'data/images',
    anno_dir:   str = 'data/annotations',
    output_dir: str = 'data/heatmaps'
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("Step 2a: Generating Density Maps...")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    skipped = 0
    for img_name in image_files:
        img = cv2.imread(os.path.join(image_dir, img_name))
        if img is None:
            print(f"  Warning: Could not read '{img_name}'. Skipping.")
            skipped += 1
            continue

        h, w = img.shape[:2]

        json_path = os.path.join(anno_dir, os.path.splitext(img_name)[0] + '.json')
        if not os.path.exists(json_path):
            print(f"  Skipping '{img_name}' — no JSON annotation found.")
            skipped += 1
            continue

        with open(json_path, 'r') as f:
            points = json.load(f)

        print(f"  Processing '{img_name}' ({len(points)} people) ...", end=' ', flush=True)
        density_map = generate_density_map(points, h, w)

        np.save(os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy'), density_map)
        print(f"done. map_sum={np.sum(density_map):.2f}")

    print(f"\n  Heatmap generation done. Skipped {skipped} file(s).")


def split_data(
    master_csv: str = 'simhastha_master_index.csv',
    image_dir:  str = 'data/images',
    heat_dir:   str = 'data/heatmaps'
) -> None:
    print("\nStep 2b: Splitting into Train/Test folders (from master CSV)...")

    if not os.path.exists(master_csv):
        raise FileNotFoundError(
            f"Master CSV '{master_csv}' not found. Run step1_build_csv.py first."
        )

    train_set: set = set()
    test_set:  set = set()

    with open(master_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['split_assignment'] == 'Train':
                train_set.add(row['image_name'])
            elif row['split_assignment'] == 'Test':
                test_set.add(row['image_name'])

    for folder in ['Train', 'Test']:
        target = f'data/{folder}'
        if os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)
        os.makedirs(f'{target}/images',   exist_ok=True)
        os.makedirs(f'{target}/heatmaps', exist_ok=True)

    copied       = {'Train': 0, 'Test': 0}
    missing_heat = []

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        base      = os.path.splitext(img_name)[0]
        heat_path = os.path.join(heat_dir, base + '.npy')

        if img_name in train_set:
            dest = 'Train'
        elif img_name in test_set:
            dest = 'Test'
        else:
            continue

        if not os.path.exists(heat_path):
            missing_heat.append(img_name)
            print(f"  Warning: No heatmap for '{img_name}'. Skipping.")
            continue

        shutil.copy(os.path.join(image_dir, img_name), f'data/{dest}/images/{img_name}')
        shutil.copy(heat_path, f'data/{dest}/heatmaps/{base}.npy')
        copied[dest] += 1

    print(f"\n  Split Complete!")
    print(f"    Train : {copied['Train']} images")
    print(f"    Test  : {copied['Test']} images")
    if missing_heat:
        print(f"    Skipped (no heatmap): {missing_heat}")


if __name__ == "__main__":
    generate_heatmaps()
    split_data()