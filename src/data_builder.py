import csv
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from typing import Dict

from .utils import ensure_dir

RANDOM_SEED = 42


def _parse_xml_points(p_tag):
    raw = p_tag.get('points', '')
    points = []
    for raw_point in raw.split(';'):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        parts = raw_point.split(',')
        if len(parts) != 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            points.append({'x': x, 'y': y})
        except ValueError:
            continue
    return points


def build_master_index(
    img_dir: str = 'Images',
    xml_dir: str = 'Annotations',
    output_csv: str = 'simhastha_master_index.csv',
    train_ratio: float = 0.8,
) -> None:
    print('Step 1: Indexing & Preparing Simhastha Dataset...')
    ensure_dir('data/images')
    ensure_dir('data/annotations')

    if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
        raise FileNotFoundError(f"Could not find '{img_dir}' or '{xml_dir}' folder.")

    all_images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    xml_data: Dict[str, Dict] = {}

    for xml_file in sorted(xml_files):
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as exc:
            print(f"  Warning: failed to parse '{xml_file}': {exc}")
            continue

        for img_tag in root.findall('image'):
            name = img_tag.get('name')
            if not name:
                continue
            if name in xml_data:
                print(f"  Warning: duplicate annotation for '{name}' in '{xml_file}'. Skipping.")
                continue

            width = img_tag.get('width')
            height = img_tag.get('height')
            try:
                width = int(width) if width is not None else 0
                height = int(height) if height is not None else 0
            except ValueError:
                width, height = 0, 0

            points = []
            for p_tag in img_tag.findall('points'):
                points.extend(_parse_xml_points(p_tag))

            json_name = os.path.splitext(name)[0] + '.json'
            json_path = os.path.join('data/annotations', json_name)
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(points, jf, indent=2)

            src_img = os.path.join(img_dir, name)
            dst_img = os.path.join('data/images', name)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"  Warning: image file '{src_img}' not found.")

            xml_data[name] = {
                'count': len(points),
                'xml_file': xml_file,
                'width': width,
                'height': height,
            }

    random.seed(RANDOM_SEED)
    labeled_images = [img for img in all_images if img in xml_data]
    random.shuffle(labeled_images)
    split_idx = int(len(labeled_images) * train_ratio)
    train_images = set(labeled_images[:split_idx])
    test_images = set(labeled_images[split_idx:])

    with open(output_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'image_name', 'status', 'head_count',
            'source_xml', 'width', 'height', 'split_assignment'
        ])
        for img_name in all_images:
            if img_name in xml_data:
                record = xml_data[img_name]
                split = 'Train' if img_name in train_images else 'Test'
                writer.writerow([
                    img_name,
                    'Labeled',
                    record['count'],
                    record['xml_file'],
                    record['width'],
                    record['height'],
                    split,
                ])
            else:
                writer.writerow([
                    img_name,
                    'Unlabeled',
                    0,
                    'None',
                    'Unknown',
                    'Unknown',
                    'Inference',
                ])

    print(f"Done! Master index saved to '{output_csv}'")
    print(f"  Total images : {len(all_images)}")
    print(f"  Labeled      : {len(labeled_images)}")
    print(f"    Train      : {len(train_images)}")
    print(f"    Test       : {len(test_images)}")
    print(f"  Unlabeled    : {len(all_images) - len(labeled_images)}")
    print('  Centralized data ready in the data/ folder.')
