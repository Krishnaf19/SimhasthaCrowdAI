import json, os, shutil, csv
import cv2, numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from ..utils.common import CLASSES, ensure_dir, list_image_files

MIN_SIGMA, MAX_SIGMA, K_NEIGHBOURS = 4.0, 15.0, 3

def _compute_sigmas(coords):
    if len(coords) == 1:
        return np.array([MIN_SIGMA], dtype=np.float32)
    k = min(K_NEIGHBOURS + 1, len(coords))
    dists, _ = KDTree(coords).query(coords, k=k)
    return np.clip(dists[:, 1:].mean(axis=1) * 0.25, MIN_SIGMA, MAX_SIGMA).astype(np.float32)

def _make_density_map(points, h, w):
    if not points:
        return np.zeros((h, w), dtype=np.float32)
    coords = np.array([[p['x'], p['y']] for p in points], dtype=np.float64)
    sigmas = _compute_sigmas(coords)
    density = np.zeros((h, w), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        tmp = np.zeros((h, w), dtype=np.float32)
        tmp[int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))] = 1.0
        density += gaussian_filter(tmp, sigma=float(sigmas[i]))
    total = density.sum()
    if total > 0:
        density *= len(points) / total
    return density

def generate_heatmaps(image_dir='data/processed/images',
                      anno_dir='data/processed/annotations',
                      output_dir='data/processed/heatmaps'):
    ensure_dir(output_dir)
    image_files = list_image_files(image_dir)
    if not image_files:
        print('No images found in', image_dir); return
    print('Generating density maps...')
    skipped = 0
    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        jp = os.path.join(anno_dir, stem + '.json')
        if not os.path.exists(jp):
            skipped += 1; continue
        try:
            data = json.load(open(jp, 'r', encoding='utf-8'))
        except Exception:
            skipped += 1; continue
        if isinstance(data, list):
            data = {'head': data, 'turban': [], 'veil': [], 'cap': []}
        img = cv2.imread(os.path.join(image_dir, img_name))
        if img is None:
            skipped += 1; continue
        h, w = img.shape[:2]
        combined = np.zeros((h, w), dtype=np.float32)
        for cls in CLASSES:
            pts = data.get(cls, [])
            dm = _make_density_map(pts, h, w)
            np.save(os.path.join(output_dir, stem + '_' + cls + '.npy'), dm)
            combined += dm
        np.save(os.path.join(output_dir, stem + '.npy'), combined)
        counts = ', '.join(c + '=' + str(len(data.get(c, []))) for c in CLASSES)
        print('  ' + img_name + ': total=' + str(int(round(combined.sum()))) + ' (' + counts + ')')
    print('Done. Skipped', skipped)

def split_data(master_csv='simhastha_master_index.csv',
               image_dir='data/processed/images',
               heat_dir='data/processed/heatmaps'):
    if not os.path.exists(master_csv):
        raise FileNotFoundError('Master CSV not found: ' + master_csv)
    train_set, test_set = set(), set()
    with open(master_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            s = row['split_assignment']
            if s == 'train': train_set.add(row['image_name'])
            elif s == 'test': test_set.add(row['image_name'])
    for folder in ('train', 'test'):
        t = os.path.join('data/splits', folder)
        shutil.rmtree(t, ignore_errors=True)
        ensure_dir(os.path.join(t, 'images'))
        ensure_dir(os.path.join(t, 'heatmaps'))
    copied = missing = 0
    for img_name in list_image_files(image_dir):
        stem = os.path.splitext(img_name)[0]
        dest = 'train' if img_name in train_set else ('test' if img_name in test_set else None)
        if dest is None: continue
        if not os.path.exists(os.path.join(heat_dir, stem + '.npy')):
            missing += 1; continue
        shutil.copy(os.path.join(image_dir, img_name),
                    os.path.join('data/splits', dest, 'images', img_name))
        for fname in [stem + '.npy'] + [stem + '_' + c + '.npy' for c in CLASSES]:
            src = os.path.join(heat_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join('data/splits', dest, 'heatmaps', fname))
        copied += 1
    print('Split complete. Copied', copied, 'images. Missing:', missing)
