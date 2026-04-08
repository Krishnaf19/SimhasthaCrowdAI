import json
import os
from typing import Iterable, List

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_image_files(folder: str) -> List[str]:
    if not os.path.exists(folder):
        return []
    return sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
    ])


def safe_load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def int_or_zero(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
