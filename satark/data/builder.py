import csv
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

from ..utils.common import CLASSES, ensure_dir

RANDOM_SEED = 42


def _parse_xml_annotations(xml_dir: str) -> dict:
    xml_data = {}
    for xml_file in sorted(f for f in os.listdir(xml_dir) if f.endswith('.xml')):
        try:
            root = ET.parse(os.path.join(xml_dir, xml_file)).getroot()
        except ET.ParseError as e:
            print('  Warning: failed to parse', xml_file, ':', e)
            continue
        for img_tag in root.findall('image'):
            name = img_tag.get('name')
            if not name or name in xml_data:
                continue
            try:
                w, h = int(img_tag.get('width', 0)), int(img_tag.get('height', 0))
            except ValueError:
                w, h = 0, 0
            per_class = {cls: [] for cls in CLASSES}
            for p_tag in img_tag.findall('points'):
                label = p_tag.get('label', 'head').strip().lower()
                if label not in per_class:
                    label = 'head'
                for raw in p_tag.get('points', '').split(';'):
                    parts = raw.strip().split(',')
                    if len(parts) == 2:
                        try:
                            per_class[label].append({'x': float(parts[0]), 'y': float(parts[1])})
                        except ValueError:
                            pass
            xml_data[name] = {'classes': per_class, 'width': w, 'height': h, 'xml': xml_file}
    return xml_data


def build_master_index(
    img_dir: str = 'data/raw/images',
    xml_dir: str = 'data/raw/annotations',
    output_csv: str = 'simhastha_master_index.csv',
    train_ratio: float = 0.8,
) -> None:
    print('Step 1: Indexing Simhastha Dataset...')
    ensure_dir('data/processed/images')
    ensure_dir('data/processed/annotations')

    if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
        raise FileNotFoundError('Cannot find ' + repr(img_dir) + ' or ' + repr(xml_dir))

    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    xml_data = _parse_xml_annotations(xml_dir)

    for name, data in xml_data.items():
        json_path = os.path.join('data/processed/annotations', os.path.splitext(name)[0] + '.json')
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(data['classes'], jf)
        src = os.path.join(img_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join('data/processed/images', name))
        else:
            print('  Warning: image not found:', name)

    random.seed(RANDOM_SEED)
    labeled = [img for img in all_images if img in xml_data]
    random.shuffle(labeled)
    split_idx = int(len(labeled) * train_ratio)
    train_set, test_set = set(labeled[:split_idx]), set(labeled[split_idx:])

    with open(output_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['image_name', 'status', 'head_count', 'turban_count', 'veil_count',
                         'cap_count', 'source_xml', 'width', 'height', 'split_assignment'])
        for img in all_images:
            if img in xml_data:
                d = xml_data[img]
                split = 'train' if img in train_set else 'test'
                writer.writerow([img, 'labeled',
                                 len(d['classes']['head']), len(d['classes']['turban']),
                                 len(d['classes']['veil']), len(d['classes']['cap']),
                                 d['xml'], d['width'], d['height'], split])
            else:
                writer.writerow([img, 'unlabeled', 0, 0, 0, 0, 'none', 0, 0, 'inference'])

    print('  Master index ->', output_csv)
    print('  Total:', len(all_images), '| Labeled:', len(labeled),
          '(train=' + str(len(train_set)) + ', test=' + str(len(test_set)) + ')')
