import csv
import json
import os
import shutil
from typing import List

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

from .utils import ensure_dir, list_image_files

RANDOM_SEED = 42
MIN_SIGMA = 4.0
MAX_SIGMA = 15.0
K_NEIGHBOURS = 3
SIGMA_BUCKETS = 5


def compute_sigmas(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 1:
        return np.array([MIN_SIGMA], dtype=np.float32)
    tree = KDTree(coords)
    k = min(K_NEIGHBOURS + 1, len(coords))
    dists, _ = tree.query(coords, k=k)
    avg_dists = dists[:, 1:].mean(axis=1)
    sigmas = np.clip(avg_dists * 0.25, MIN_SIGMA, MAX_SIGMA).astype(np.float32)
    return sigmas


def generate_density_map(points: List[dict], h: int, w: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((h, w), dtype=np.float32)

    coords = np.array([[pt['x'], pt['y']] for pt in points], dtype=np.float64)
    sigmas = compute_sigmas(coords)
    density_map = np.zeros((h, w), dtype=np.float32)

    if np.all(sigmas == sigmas[0]):
        buckets = [(float(sigmas[0]), np.arange(len(points)))]
    else:
        edges = np.linspace(sigmas.min(), sigmas.max(), SIGMA_BUCKETS + 1)
        buckets = []
        for i in range(SIGMA_BUCKETS):
            low, high = edges[i], edges[i + 1]
            if i == SIGMA_BUCKETS - 1:
                mask = (sigmas >= low) & (sigmas <= high)
            else:
                mask = (sigmas >= low) & (sigmas < high)
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                buckets.append((float(sigmas[idxs].mean()), idxs))

    for sigma, idxs in buckets:
        bucket_map = np.zeros((h, w), dtype=np.float32)
        for idx in idxs:
            x = int(np.clip(coords[idx, 0], 0, w - 1))
            y = int(np.clip(coords[idx, 1], 0, h - 1))
            bucket_map[y, x] += 1.0
        density_map += gaussian_filter(bucket_map, sigma=sigma)

    total = density_map.sum()
    expected = float(len(points))
    if total > 0 and abs(total - expected) > 0.5:
        density_map *= expected / total

    return density_map


def generate_heatmaps(
    image_dir: str = 'data/images',
    anno_dir: str = 'data/annotations',
    output_dir: str = 'data/heatmaps',
) -> None:
    ensure_dir(output_dir)
    image_files = list_image_files(image_dir)

    if not image_files:
        print(f"No images found in '{image_dir}'.")
        return

    print('Step 2a: Generating Density Maps...')
    skipped = 0
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        json_path = os.path.join(anno_dir, os.path.splitext(img_name)[0] + '.json')

        if not os.path.exists(json_path):
            print(f"  Skipping '{img_name}': annotation missing.")
            skipped += 1
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                points = json.load(f)
        except Exception as exc:
            print(f"  Skipping '{img_name}': invalid annotation file ({exc}).")
            skipped += 1
            continue

        if not isinstance(points, list):
            print(f"  Skipping '{img_name}': expected a list of point dictionaries.")
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping '{img_name}': cannot load image.")
            skipped += 1
            continue

        h, w = img.shape[:2]
        density_map = generate_density_map(points, h, w)
        out_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        np.save(out_path, density_map)
        print(f"  Saved '{out_path}' (count={density_map.sum():.2f})")

    print(f"Finished heatmap generation. Skipped {skipped} image(s).")


def split_data(
    master_csv: str = 'simhastha_master_index.csv',
    image_dir: str = 'data/images',
    heat_dir: str = 'data/heatmaps',
) -> None:
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    train_set = set()
    test_set = set()
    with open(master_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split_assignment'] == 'Train':
                train_set.add(row['image_name'])
            elif row['split_assignment'] == 'Test':
                test_set.add(row['image_name'])

    for folder in ['Train', 'Test']:
        target_dir = os.path.join('data', folder)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        ensure_dir(os.path.join(target_dir, 'images'))
        ensure_dir(os.path.join(target_dir, 'heatmaps'))

    copied = 0
    missing = 0
    for img_name in list_image_files(image_dir):
        base_name = os.path.splitext(img_name)[0]
        source_heat = os.path.join(heat_dir, base_name + '.npy')
        if img_name in train_set:
            dest = 'Train'
        elif img_name in test_set:
            dest = 'Test'
        else:
            continue

        if not os.path.exists(source_heat):
            print(f"  Missing heatmap for '{img_name}'.")
            missing += 1
            continue

        shutil.copy(os.path.join(image_dir, img_name), os.path.join('data', dest, 'images', img_name))
        shutil.copy(source_heat, os.path.join('data', dest, 'heatmaps', base_name + '.npy'))
        copied += 1

    print(f"Step 2b: Split complete. Copied {copied} images.")
    if missing:
        print(f"  Missing heatmaps: {missing}")
