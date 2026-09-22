__all__ = [
    "CLASSES",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "ensure_dir",
    "list_image_files",
    "list_video_files",
    "infer_image",
    "infer_images",
    "infer_video",
]


def __getattr__(name):
    if name in {"CLASSES", "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "ensure_dir", "list_image_files", "list_video_files"}:
        from .common import CLASSES, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, ensure_dir, list_image_files, list_video_files
        return {
            "CLASSES": CLASSES,
            "IMAGE_EXTENSIONS": IMAGE_EXTENSIONS,
            "VIDEO_EXTENSIONS": VIDEO_EXTENSIONS,
            "ensure_dir": ensure_dir,
            "list_image_files": list_image_files,
            "list_video_files": list_video_files,
        }[name]
    if name in {"infer_image", "infer_images"}:
        from .inference import infer_image, infer_images
        return infer_image if name == "infer_image" else infer_images
    if name == "infer_video":
        from .video import infer_video
        return infer_video
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

