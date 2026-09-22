import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import shutil
import json
import csv
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

import argparse
from satark.utils.common import CLASSES, ensure_dir

RAW_IMG_DIR = "data/raw/images"
RAW_ANNO_DIR = "data/raw/annotations"
PROC_IMG_DIR = "data/processed/images"
PROC_ANNO_DIR = "data/processed/annotations"
PROC_HEAT_DIR = "data/processed/heatmaps"
TRAIN_IMG_DIR = "data/splits/train/images"
TRAIN_HEAT_DIR = "data/splits/train/heatmaps"
METADATA_CSV = "satark_dataset_metadata.csv"
BACKUP_WEIGHTS = "checkpoints/satark_best_backup.pth"
ORIGINAL_WEIGHTS = "checkpoints/satark_best.pth"
BACKUP_WEIGHTS = "checkpoints/satark_best_backup.pth"
ORIGINAL_WEIGHTS = "checkpoints/satark_best.pth"

MIN_SIGMA, MAX_SIGMA, K_NEIGHBOURS = 4.0, 15.0, 3

def compute_sigmas(coords):
    if len(coords) == 1:
        return np.array([MIN_SIGMA], dtype=np.float32)
    k = min(K_NEIGHBOURS + 1, len(coords))
    dists, _ = KDTree(coords).query(coords, k=k)
    return np.clip(dists[:, 1:].mean(axis=1) * 0.25, MIN_SIGMA, MAX_SIGMA).astype(np.float32)

def make_density_map(points, h, w):
    if not points:
        return np.zeros((h, w), dtype=np.float32)
    coords = np.array([[p['x'], p['y']] for p in points], dtype=np.float64)
    sigmas = compute_sigmas(coords)
    density = np.zeros((h, w), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        tmp = np.zeros((h, w), dtype=np.float32)
        tmp[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] = 1.0
        density += gaussian_filter(tmp, sigma=float(sigmas[i]))
    total = density.sum()
    if total > 0:
        density *= len(points) / total
    return density

def run_integration(source_img_dir, source_xml_path, xml_tag_name="batch"):
    print("=" * 60)
    print("STEP 1: Backing up current weights...")
    ensure_dir("checkpoints")
    if os.path.exists(ORIGINAL_WEIGHTS) and not os.path.exists(BACKUP_WEIGHTS):
        shutil.copy2(ORIGINAL_WEIGHTS, BACKUP_WEIGHTS)
        print(f"  Backed up {ORIGINAL_WEIGHTS} -> {BACKUP_WEIGHTS}")
    elif os.path.exists(BACKUP_WEIGHTS):
        print(f"  Backup already exists at {BACKUP_WEIGHTS}")

    print("\nSTEP 2: Ensuring directories exist...")
    for d in [RAW_IMG_DIR, RAW_ANNO_DIR, PROC_IMG_DIR, PROC_ANNO_DIR, PROC_HEAT_DIR, TRAIN_IMG_DIR, TRAIN_HEAT_DIR]:
        ensure_dir(d)

    print("\nSTEP 3: Copying raw annotations XML...")
    target_xml = os.path.join(RAW_ANNO_DIR, f"annotations_{xml_tag_name}.xml")
    shutil.copy2(source_xml_path, target_xml)
    print(f"  Copied {source_xml_path} -> {target_xml}")

    print("\nSTEP 4: Parsing XML annotations and processing images...")
    tree = ET.parse(source_xml_path)
    root = tree.getroot()
    image_tags = root.findall("image")
    print(f"  Found {len(image_tags)} images in XML.")

    new_metadata_rows = []

    for img_tag in image_tags:
        img_name = img_tag.get("name")
        stem = os.path.splitext(img_name)[0]
        src_img_path = os.path.join(source_img_dir, img_name)
        if not os.path.exists(src_img_path):
            raise FileNotFoundError(f"Missing source image: {src_img_path}")

        # Copy raw image to data/raw/images, data/processed/images, data/splits/train/images
        shutil.copy2(src_img_path, os.path.join(RAW_IMG_DIR, img_name))
        shutil.copy2(src_img_path, os.path.join(PROC_IMG_DIR, img_name))
        shutil.copy2(src_img_path, os.path.join(TRAIN_IMG_DIR, img_name))

        # Read actual image dimensions
        img_mat = cv2.imread(src_img_path)
        h, w = img_mat.shape[:2]

        # Parse points per class
        per_class = {cls: [] for cls in CLASSES}
        for p_tag in img_tag.findall("points"):
            label = p_tag.get("label", "head").strip().lower()
            if label not in per_class:
                label = "head"
            for raw in p_tag.get("points", "").split(";"):
                parts = raw.strip().split(",")
                if len(parts) == 2:
                    try:
                        per_class[label].append({"x": float(parts[0]), "y": float(parts[1])})
                    except ValueError:
                        pass

        # Save JSON annotations
        json_path = os.path.join(PROC_ANNO_DIR, stem + ".json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(per_class, jf)

        # Generate density maps
        combined = np.zeros((h, w), dtype=np.float32)
        for cls in CLASSES:
            pts = per_class.get(cls, [])
            dm = make_density_map(pts, h, w)
            # save per-class map
            proc_cls_path = os.path.join(PROC_HEAT_DIR, f"{stem}_{cls}.npy")
            np.save(proc_cls_path, dm)
            shutil.copy2(proc_cls_path, os.path.join(TRAIN_HEAT_DIR, f"{stem}_{cls}.npy"))
            combined += dm

        proc_comb_path = os.path.join(PROC_HEAT_DIR, f"{stem}.npy")
        np.save(proc_comb_path, combined)
        shutil.copy2(proc_comb_path, os.path.join(TRAIN_HEAT_DIR, f"{stem}.npy"))

        total_heads = sum(len(per_class[c]) for c in CLASSES)
        # Compute lighting from grayscale image
        gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
        lighting = "Low-light" if gray.mean() < 75 else "Day"

        # Crowd density classification
        if total_heads <= 50:
            density_str = "Low"
        elif total_heads <= 150:
            density_str = "Medium"
        else:
            density_str = "High"

        new_metadata_rows.append({
            "image_id": img_name,
            "image_path": f"data/images/{img_name}",
            "image_width": w,
            "image_height": h,
            "head_count": total_heads,
            "crowd_density": density_str,
            "headwear_type": "Mixed",
            "dataset_split": "Train",
            "annotation_confidence": 0.85,
            "lighting": lighting
        })
        print(f"  [OK] {img_name}: {w}x{h}, heads={total_heads} ({density_str}), lighting={lighting}, heat_sum={combined.sum():.2f}")

    print("\nSTEP 5: Updating CSV metadata file...")
    # Check if entries are already present in satark_dataset_metadata.csv
    existing_ids = set()
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["image_id"])

    added_count = 0
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "image_path", "image_width", "image_height", "head_count",
            "crowd_density", "headwear_type", "dataset_split", "annotation_confidence", "lighting"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in new_metadata_rows:
            if row["image_id"] not in existing_ids:
                writer.writerow(row)
                existing_ids.add(row["image_id"])
                added_count += 1

    print(f"  Added {added_count} new rows to {METADATA_CSV}.")

    # Also build/update simhastha_master_index.csv
    from satark.data.builder import build_master_index
    build_master_index(
        img_dir=RAW_IMG_DIR,
        xml_dir=RAW_ANNO_DIR,
        output_csv="simhastha_master_index.csv",
        train_ratio=0.8
    )

    print("\nData integration complete!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate new batch of crowd images and annotations into dataset.")
    parser.add_argument("--img-dir", default=r"C:\Users\Krishna\Downloads\simhasthasadhuimages", help="Path to images directory")
    parser.add_argument("--xml-path", default=r"C:\Users\Krishna\Downloads\label000\annotations.xml", help="Path to CVAT annotations XML")
    parser.add_argument("--tag", default="label000", help="Unique identifier/tag for this batch")
    args = parser.parse_args()

    run_integration(
        source_img_dir=args.img_dir,
        source_xml_path=args.xml_path,
        xml_tag_name=args.tag
    )
