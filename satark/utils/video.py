import json
import os
import time
from collections import deque
import cv2
import numpy as np
import torch

from ..engine.evaluator import load_checkpoint
from ..models.csrnet import CSRNet, get_device, clear_device_cache
from ..utils.common import ensure_dir, CLASSES
from .inference import _checkpoint_output_channels, get_zone, MEAN, STD, MAX_SIDE_PX

_MEAN_TENSOR = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_STD_TENSOR = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)


def _preprocess_frame(frame_bgr, device, max_dim=640):
    """Resize and normalize a BGR OpenCV frame for CSRNet inference with direct tensor conversion."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        frame_resized = frame_bgr

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    mean = _MEAN_TENSOR.to(device)
    std = _STD_TENSOR.to(device)
    tensor = tensor.to(device).sub_(mean).div_(std)
    return tensor, frame_resized.shape[:2]


def _temporal_median(samples, classes):
    """Return a robust count for a near-static crowd scene.

    Crowd density should not jump sharply between adjacent video samples.  A
    median rejects blur, compression, and partial-occlusion outliers without
    inventing a count from frames that were not actually analysed.
    """
    if not samples:
        return 0.0, {name: 0.0 for name in classes}
    totals = np.asarray([sample['total'] for sample in samples], dtype=np.float32)
    per_class = {
        name: float(np.median([sample['classes'].get(name, 0.0) for sample in samples]))
        for name in classes
    }
    return float(np.median(totals)), per_class


def _draw_hud(frame_bgr, density_map_raw, count, per_class_dict, zone, frame_idx, total_frames, fps, cached_heatmap=None):
    """Draw semi-transparent heatmap overlay and real-time Heads-Up Display (HUD)."""
    h, w = frame_bgr.shape[:2]

    # 1. Overlay Heatmap (reuse cached_heatmap to avoid re-rendering every frame)
    if cached_heatmap is not None:
        annotated = cv2.addWeighted(frame_bgr, 0.75, cached_heatmap, 0.25, 0)
        heatmap_color = cached_heatmap
    elif density_map_raw is not None and density_map_raw.max() > 0:
        norm_density = np.clip(density_map_raw / (density_map_raw.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        heatmap_resized = cv2.resize(norm_density, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        annotated = cv2.addWeighted(frame_bgr, 0.75, heatmap_color, 0.25, 0)
    else:
        annotated = frame_bgr.copy()
        heatmap_color = None

    # 2. Semi-transparent top HUD banner
    hud_height = 80
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (18, 20, 24), -1)
    # Bottom status strip
    cv2.rectangle(overlay, (0, h - 28), (w, h), (18, 20, 24), -1)
    annotated = cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0)

    # 3. Zone color definitions (BGR)
    zone_colors = {
        'SAFE': (60, 210, 110),      # Bright green
        'NORMAL': (30, 210, 255),    # Amber/Yellow
        'CRITICAL': (50, 50, 255),   # Intense Red
    }
    accent_color = zone_colors.get(zone, (200, 200, 200))

    # Title & Brand
    cv2.putText(annotated, "SATARK AI", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, "CROWD INTELLIGENCE", (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 165, 175), 1, cv2.LINE_AA)

    # Live Total Count (Large Font)
    count_str = f"LIVE COUNT: {int(round(count))}"
    cv2.putText(annotated, count_str, (210, 34), cv2.FONT_HERSHEY_DUPLEX, 0.95, accent_color, 2, cv2.LINE_AA)

    # Zone Badge
    badge_text = f"[{zone} ZONE]"
    cv2.putText(annotated, badge_text, (210, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent_color, 1, cv2.LINE_AA)

    # Cultural Headgear Breakdown
    if per_class_dict:
        breakdown_parts = []
        for cls_name in CLASSES:
            if cls_name in per_class_dict:
                val = int(round(per_class_dict[cls_name]))
                breakdown_parts.append(f"{cls_name.capitalize()}: {val}")
        breakdown_str = " | ".join(breakdown_parts)
        cv2.putText(annotated, breakdown_str, (460, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 235, 245), 1, cv2.LINE_AA)

    # Bottom Bar: Timeline & Frame Counter
    cur_sec = int(frame_idx / fps) if fps > 0 else 0
    tot_sec = int(total_frames / fps) if fps > 0 else 0
    time_str = f"Time: {cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}  |  Frame: {frame_idx}/{total_frames}  |  {fps:.1f} FPS"
    cv2.putText(annotated, time_str, (16, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 185, 195), 1, cv2.LINE_AA)

    return annotated, heatmap_color


def infer_video(
    video_path: str,
    model_path: str = 'checkpoints/satark_best.pth',
    output_dir: str = 'outputs/inference',
    frame_stride = 'auto',
    max_dim: int = MAX_SIDE_PX,
    temporal_window: int = 5,
    model=None,
    device=None,
    progress_callback=None,
) -> dict:
    """Process video, estimating continuous multi-class crowd density with high throughput.

    Args:
        video_path: Path to source video.
        model_path: Path to trained PyTorch weights.
        output_dir: Destination folder for output video and JSON metrics.
        frame_stride: 'auto' (e.g. ~2 FPS sampling for fast real-time counting), or integer stride.
        max_dim: Maximum resolution boundary for inference (default 1000, matching images).
        temporal_window: Odd-numbered history length used to stabilize sampled counts.
        model: Optional pre-loaded CSRNet instance.
        device: Optional torch.device.
        progress_callback: Optional callable receiving progress dictionary.

    Returns:
        Dictionary containing paths, time-series telemetry, and summary statistics.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ensure_dir(output_dir)
    if device is None:
        device = get_device()

    if device.type == 'cpu':
        cores = os.cpu_count() or 4
        torch.set_num_threads(min(8, cores))

    if model is None:
        out_channels = _checkpoint_output_channels(model_path, device)
        model = CSRNet(load_weights=False, freeze_frontend=False, output_channels=out_channels).to(device)
        if not load_checkpoint(model_path, model, device):
            raise RuntimeError(f"Could not load checkpoint from {model_path}")
    else:
        model = model.to(device)

    model.eval()
    headgear_supported = getattr(getattr(model, 'output_layer', None), 'out_channels', 0) == len(CLASSES)
    temporal_window = max(1, int(temporal_window))
    if temporal_window % 2 == 0:
        temporal_window += 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video source: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = orig_fps if (orig_fps and orig_fps > 0) else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine numeric frame stride
    if frame_stride == 'auto' or frame_stride is None:
        # Sample at ~2 FPS by default for fast real-time counting
        stride = max(1, int(round(fps / 2.0)))
    else:
        try:
            stride = max(1, int(frame_stride))
        except (ValueError, TypeError):
            stride = max(1, int(round(fps / 2.0)))

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video_name = f"result_{stem}.mp4"
    out_video_path = os.path.join(output_dir, out_video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    telemetry = []
    zone_counts = {'SAFE': 0, 'NORMAL': 0, 'CRITICAL': 0}
    peak_count = 0.0
    peak_time_sec = 0.0
    cumulative_count = 0.0

    last_raw_density = None
    last_count = 0.0
    last_per_class = {c: 0.0 for c in CLASSES}
    last_zone = 'SAFE'
    cached_heatmap = None
    count_history = deque(maxlen=temporal_window)

    frame_idx = 0
    t_start = time.time()

    try:
        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                should_infer = (frame_idx % stride == 0) or (frame_idx == 1)

                if should_infer:
                    tensor, (rh, rw) = _preprocess_frame(frame, device, max_dim=max_dim)
                    output = model(tensor)  # (1, C, H_feat, W_feat)

                    out_c = output.shape[1]
                    class_counts = [float(output[0, c].sum().item()) for c in range(out_c)]
                    raw_total_count = sum(class_counts)

                    if out_c == len(CLASSES):
                        raw_per_class = dict(zip(CLASSES, class_counts))
                    else:
                        raw_per_class = {'head count': raw_total_count}

                    count_history.append({'total': raw_total_count, 'classes': raw_per_class})
                    total_count, per_class = _temporal_median(count_history, raw_per_class.keys())

                    zone = get_zone(total_count)
                    raw_density = output[0].sum(dim=0).cpu().numpy()

                    last_raw_density = raw_density
                    last_count = total_count
                    last_per_class = per_class
                    last_zone = zone
                    cached_heatmap = None  # Re-render on new density
                else:
                    total_count = last_count
                    per_class = last_per_class
                    zone = last_zone
                    raw_density = last_raw_density

                # Update statistics
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
                cumulative_count += total_count
                if total_count > peak_count:
                    peak_count = total_count
                    peak_time_sec = round(frame_idx / fps, 2)

                # Draw Heads-Up Display
                annotated_frame, cached_heatmap = _draw_hud(
                    frame, raw_density, total_count, per_class, zone,
                    frame_idx, total_frames, fps, cached_heatmap=cached_heatmap
                )
                writer.write(annotated_frame)

                # Record telemetry periodically
                if frame_idx % max(1, stride) == 0 or frame_idx == 1 or frame_idx == total_frames:
                    telemetry.append({
                        'frame': frame_idx,
                        'time_sec': round(frame_idx / fps, 2),
                        'total_count': round(total_count, 1),
                        'zone': zone,
                        'classes': {k: round(v, 1) for k, v in per_class.items()}
                    })

                # Stream continuous progress callback
                if progress_callback and (should_infer or frame_idx % 5 == 0 or frame_idx == total_frames):
                    now = time.time()
                    elapsed = now - t_start
                    proc_fps = frame_idx / max(elapsed, 0.001)
                    eta = (total_frames - frame_idx) / max(proc_fps, 0.001) if total_frames > frame_idx else 0.0

                    cb_payload = {
                        'current_frame': frame_idx,
                        'total_frames': total_frames,
                        'progress_pct': round((frame_idx / max(total_frames, 1)) * 100, 1),
                        'current_count': round(total_count, 1),
                        'per_class': {k: round(v, 1) for k, v in per_class.items()},
                        'zone': zone,
                        'time_sec': round(frame_idx / fps, 2),
                        'fps_processed': round(proc_fps, 1),
                        'elapsed_sec': round(elapsed, 1),
                        'eta_sec': round(eta, 1),
                        'is_inferred': should_infer,
                        'headgear_supported': headgear_supported,
                    }
                    try:
                        progress_callback(cb_payload)
                    except TypeError:
                        progress_callback(frame_idx, total_frames, total_count)

    finally:
        cap.release()
        writer.release()

    duration_sec = round(frame_idx / fps, 2)
    avg_count = round(cumulative_count / max(frame_idx, 1), 1)

    # Save telemetry JSON
    analytics_file = os.path.join(output_dir, f"result_{stem}_analytics.json")
    summary_payload = {
        'video': os.path.basename(video_path),
        'output_video': out_video_name,
        'total_frames': frame_idx,
        'fps': round(fps, 1),
        'duration_seconds': duration_sec,
        'peak_count': round(peak_count, 1),
        'peak_timestamp_seconds': peak_time_sec,
        'average_count': avg_count,
        'temporal_window': temporal_window,
        'inference_max_dimension': max_dim,
        'zone_distribution': {
            'safe_frames': zone_counts['SAFE'],
            'normal_frames': zone_counts['NORMAL'],
            'critical_frames': zone_counts['CRITICAL'],
            'safe_pct': round(zone_counts['SAFE'] / max(frame_idx, 1) * 100, 1),
            'normal_pct': round(zone_counts['NORMAL'] / max(frame_idx, 1) * 100, 1),
            'critical_pct': round(zone_counts['CRITICAL'] / max(frame_idx, 1) * 100, 1),
        },
        'telemetry': telemetry
    }

    with open(analytics_file, 'w', encoding='utf-8') as jf:
        json.dump(summary_payload, jf, indent=2)

    return {
        'video': os.path.basename(video_path),
        'output_video': out_video_name,
        'output_path': out_video_path,
        'analytics_file': analytics_file,
        'total_frames': frame_idx,
        'duration_sec': duration_sec,
        'peak_count': round(peak_count, 1),
        'peak_time_sec': peak_time_sec,
        'avg_count': avg_count,
        'zone': get_zone(peak_count),
        'zone_stats': summary_payload['zone_distribution'],
        'telemetry': telemetry,
    }
