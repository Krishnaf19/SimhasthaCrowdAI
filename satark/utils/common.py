import json
import os
from typing import List

# 4 headgear classes. Index: 0=head, 1=turban, 2=veil, 3=cap
CLASSES = ['head', 'turban', 'veil', 'cap']

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.avif',
                    '.JPG', '.JPEG', '.PNG', '.WEBP', '.AVIF'}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm',
                    '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM'}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_image_files(folder: str) -> List[str]:
    if not os.path.exists(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[1] in IMAGE_EXTENSIONS)


def list_video_files(folder: str) -> List[str]:
    if not os.path.exists(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[1] in VIDEO_EXTENSIONS)


def safe_load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

