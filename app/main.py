import json
import os
import uuid
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import torch
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, send_from_directory, url_for, jsonify, Response, redirect

from satark.utils.inference import infer_image, infer_images, _checkpoint_output_channels
from satark.utils.video import infer_video
from satark.utils.common import list_image_files, ensure_dir
from satark.models.csrnet import CSRNet, get_device
from satark.engine.evaluator import load_checkpoint


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'checkpoints' / 'satark_best.pth'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'inference'
UPLOAD_DIR = BASE_DIR / 'uploads'
RESULTS_CACHE = OUTPUT_DIR / 'dashboard_results.json'
NAME_MAP_PATH = BASE_DIR / 'configs' / 'dataset_name_map.json'

# Global Model Singleton
_MODEL_LOCK = threading.Lock()
_CACHED_MODEL = None
_CACHED_DEVICE = None


def get_shared_model():
    global _CACHED_MODEL, _CACHED_DEVICE
    with _MODEL_LOCK:
        if _CACHED_MODEL is None:
            _CACHED_DEVICE = get_device()
            if _CACHED_DEVICE.type == 'cpu':
                cores = os.cpu_count() or 4
                torch.set_num_threads(min(8, cores))
            out_channels = _checkpoint_output_channels(str(MODEL_PATH), _CACHED_DEVICE)
            _CACHED_MODEL = CSRNet(load_weights=False, freeze_frontend=False, output_channels=out_channels).to(_CACHED_DEVICE)
            if not load_checkpoint(str(MODEL_PATH), _CACHED_MODEL, _CACHED_DEVICE):
                print(f"[WARN] Failed to load checkpoint {MODEL_PATH}")
            _CACHED_MODEL.eval()
        return _CACHED_MODEL, _CACHED_DEVICE


# Asynchronous Video Task Registry
VIDEO_TASKS = {}
VIDEO_TASKS_LOCK = threading.Lock()

# Search candidates for dataset images
CANDIDATE_IMAGE_DIRS = [
    BASE_DIR / 'data' / 'processed' / 'images',
    BASE_DIR / 'data' / 'splits' / 'train' / 'images',
    BASE_DIR / 'data' / 'images',
    BASE_DIR / 'data' / 'Train' / 'images',
    BASE_DIR / 'data' / 'Test' / 'images',
]

IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'avif'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
MAX_FILE_SIZE = 150 * 1024 * 1024  # 150MB

ensure_dir(str(OUTPUT_DIR))
ensure_dir(str(UPLOAD_DIR))

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.secret_key = 'satark_secret_key_2025'


def load_dataset_name_map():
    """Load names used to render legacy cached inference results cleanly."""
    try:
        with open(NAME_MAP_PATH, 'r', encoding='utf-8') as f:
            values = json.load(f)
        return values if isinstance(values, dict) else {}
    except (OSError, ValueError):
        return {}


DATASET_NAME_MAP = load_dataset_name_map()


def display_dataset_name(value: str) -> str:
    filename = os.path.basename(value or '')
    return DATASET_NAME_MAP.get(filename, filename or 'Unknown image')


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS


def find_image_dir():
    for path in CANDIDATE_IMAGE_DIRS:
        if path.exists() and list_image_files(str(path)):
            return path
    return None


def _enrich_dashboard_row(row):
    # The cache may predate a dataset rename.  Always derive the label from the
    # canonical name map instead of preserving an old, random source filename.
    row['display_name'] = display_dataset_name(row.get('image') or row.get('display_name') or row.get('path', ''))
    row['view_url'] = row.get('view_url') or url_for('analysis_view', filename=os.path.basename(row.get('path', '')))
    row['image_url'] = row.get('image_url') or url_for('output_image', filename=os.path.basename(row.get('path', '')))
    return row


def load_cached_results():
    if RESULTS_CACHE.exists():
        try:
            with open(RESULTS_CACHE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, dict) and isinstance(cached.get('results'), list):
                cached['results'] = [_enrich_dashboard_row(r) for r in cached['results'] if isinstance(r, dict)]
                return cached
        except Exception:
            return None
    return None


def save_cached_results(data):
    try:
        with open(RESULTS_CACHE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def compute_dashboard_results(force_refresh=False):
    image_dir = find_image_dir()
    total_on_disk = len(list_image_files(str(image_dir))) if image_dir else 0

    if not force_refresh:
        cached = load_cached_results()
        if cached and cached.get('total_images') == total_on_disk and len(cached.get('results', [])) == total_on_disk:
            return cached

    if image_dir is None:
        return {
            'results': [],
            'total_images': 0,
            'safe_count': 0,
            'normal_count': 0,
            'critical_count': 0,
            'message': 'No dataset image directory found in data/.',
        }

    batch_results = infer_images(
        model_path=str(MODEL_PATH),
        inference_dir=str(image_dir),
        output_dir=str(OUTPUT_DIR),
    )
    if not batch_results or 'results' not in batch_results:
        return {
            'results': [],
            'total_images': 0,
            'safe_count': 0,
            'normal_count': 0,
            'critical_count': 0,
            'message': 'Failed to process images with model.',
        }

    for row in batch_results['results']:
        row['display_name'] = display_dataset_name(row.get('image') or row.get('display_name') or row.get('path', ''))
        row['view_url'] = row.get('view_url') or url_for('analysis_view', filename=os.path.basename(row.get('path', '')))
        row['image_url'] = row.get('image_url') or url_for('output_image', filename=os.path.basename(row.get('path', '')))
        row['url'] = row.get('url') or row['image_url']

    counts = [r['count'] for r in batch_results['results']]
    avg_count = sum(counts) / max(len(counts), 1)

    payload = {
        'results': batch_results['results'],
        'total_images': len(batch_results['results']),
        'safe_count': batch_results.get('SAFE', 0),
        'normal_count': batch_results.get('NORMAL', 0),
        'critical_count': batch_results.get('CRITICAL', 0),
        'avg_count': round(avg_count, 1),
    }
    save_cached_results(payload)
    return payload


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('index', reloaded='1'), code=303)

    is_reloaded = (request.args.get('reloaded') == '1')
    message = 'Dashboard reloaded.' if is_reloaded else None
    result = None
    dashboard_results = compute_dashboard_results(force_refresh=is_reloaded)
    return render_template(
        'index.html',
        dashboard_results=dashboard_results,
        message=message,
        result=result,
    )


def start_video_processing(file_path: str, filename: str, unique_name: str, speed_mode: str = 'balanced', extract_fps: float = 1.0) -> str:
    """Launch background video inference with frame extraction and continuous live telemetry."""
    task_id = uuid.uuid4().hex

    # Read quick video properties
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = orig_fps if orig_fps > 0 else 25.0
    cap.release()

    # Preset configurations - always preserve 1000px max_dim to match image model accuracy
    max_dim = 1000
    if speed_mode == 'fast':
        # High speed: ~1-2 FPS sampling while preserving full image model resolution
        stride = max(1, int(round(fps / 1.5)))
    elif speed_mode == 'balanced':
        # Balanced: ~3-4 FPS sampling for responsive counting
        stride = max(1, int(round(fps / 3.0)))
    else:  # precision
        stride = max(1, int(round(fps / 6.0)))

    orig_url = f"/uploads/{unique_name}"

    task_payload = {
        'task_id': task_id,
        'status': 'processing',
        'speed_mode': speed_mode,
        'filename': filename,
        'unique_name': unique_name,
        'uploaded_media_url': orig_url,
        'progress_pct': 0.0,
        'current_frame': 0,
        'total_frames': total_frames,
        'current_count': 0.0,
        'per_class': {},
        'headgear_supported': None,
        'zone': 'SAFE',
        'fps_processed': 0.0,
        'elapsed_sec': 0.0,
        'eta_sec': 0.0,
        'recent_counts': [],
        'extracted_count': 0,
        'latest_extracted': None,
        'result': None,
        'error': None,
        'created_at': time.time(),
    }

    with VIDEO_TASKS_LOCK:
        VIDEO_TASKS[task_id] = task_payload

    def _worker():
        with app.app_context():
            try:
                model, device = get_shared_model()

                def _on_progress(p):
                    with VIDEO_TASKS_LOCK:
                        t = VIDEO_TASKS.get(task_id)
                        if t:
                            t['current_frame'] = p['current_frame']
                            t['total_frames'] = p['total_frames']
                            t['progress_pct'] = p['progress_pct']
                            t['current_count'] = p['current_count']
                            t['per_class'] = p['per_class']
                            t['headgear_supported'] = p.get('headgear_supported')
                            t['zone'] = p['zone']
                            t['fps_processed'] = p['fps_processed']
                            t['elapsed_sec'] = p['elapsed_sec']
                            t['eta_sec'] = p['eta_sec']
                            t['time_sec'] = p['time_sec']
                            t['extracted_count'] = p.get('extracted_count', 0)
                            t['latest_extracted'] = p.get('latest_extracted')
                            t['recent_counts'].append({
                                'time_sec': p['time_sec'],
                                'frame': p['current_frame'],
                                'count': p['current_count'],
                                'zone': p['zone']
                            })
                            if len(t['recent_counts']) > 30:
                                t['recent_counts'] = t['recent_counts'][-30:]

                v_res = infer_video(
                    file_path,
                    model_path=str(MODEL_PATH),
                    output_dir=str(OUTPUT_DIR),
                    frame_stride=stride,
                    max_dim=max_dim,
                    temporal_window=1,
                    extract_frames=True,
                    extract_fps=extract_fps,
                    model=model,
                    device=device,
                    progress_callback=_on_progress,
                )

                with VIDEO_TASKS_LOCK:
                    t = VIDEO_TASKS.get(task_id)
                    if t:
                        t['status'] = 'completed'
                        t['progress_pct'] = 100.0
                        t['result'] = {
                            'filename': filename,
                            'original_video_url': orig_url,
                            'output_video_url': f"/outputs/{v_res['output_video']}",
                            'total_frames': v_res['total_frames'],
                            'duration_sec': v_res['duration_sec'],
                            'peak_count': v_res['peak_count'],
                            'peak_time_sec': v_res['peak_time_sec'],
                            'avg_count': v_res['avg_count'],
                            'zone': v_res['zone'],
                            'zone_stats': v_res['zone_stats'],
                            'extracted_frames': v_res.get('extracted_frames', []),
                            'telemetry': v_res['telemetry'],
                        }
            except Exception as ex:
                with VIDEO_TASKS_LOCK:
                    t = VIDEO_TASKS.get(task_id)
                    if t:
                        t['status'] = 'error'
                        t['error'] = str(ex)

    threading.Thread(target=_worker, daemon=True).start()
    return task_id


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    result = None
    video_result = None
    message = None
    uploaded_media_url = None
    active_task_id = request.args.get('task_id')

    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file part in the request'
        else:
            file = request.files['file']
            if file.filename == '':
                message = 'No file selected for uploading'
            elif not allowed_file(file.filename):
                message = f'File type not allowed. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'
            else:
                try:
                    filename = secure_filename(file.filename)
                    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                    file.save(file_path)

                    speed_mode = request.form.get('speed_mode', 'balanced')
                    try:
                        extract_fps = float(request.form.get('extract_fps', 1.0) or 1.0)
                    except (ValueError, TypeError):
                        extract_fps = 1.0

                    if is_video_file(filename):
                        # Start continuous live counting and frame extraction task in background
                        task_id = start_video_processing(
                            file_path=file_path,
                            filename=filename,
                            unique_name=unique_name,
                            speed_mode=speed_mode,
                            extract_fps=extract_fps,
                        )

                        # Return JSON if called from modern JS fetch
                        if (
                            request.headers.get('X-Requested-With') == 'XMLHttpRequest'
                            or 'application/json' in request.headers.get('Accept', '')
                        ):
                            return jsonify({
                                'status': 'started',
                                'task_id': task_id,
                                'status_url': url_for('video_task_status', task_id=task_id),
                                'stream_url': url_for('video_task_stream', task_id=task_id),
                            })

                        # Standard form POST: render with active_task_id for live updates
                        active_task_id = task_id
                        uploaded_media_url = url_for('uploaded_file', filename=unique_name)
                    else:
                        # Image inference with cached model
                        model, device = get_shared_model()
                        img_res = infer_image(
                            file_path,
                            model_path=str(MODEL_PATH),
                            output_dir=str(OUTPUT_DIR),
                            model=model,
                            device=device,
                        )
                        if img_res:
                            img_res['url'] = url_for('output_image', filename=os.path.basename(img_res['path']))
                            uploaded_media_url = url_for('uploaded_file', filename=unique_name)
                            pc = img_res.get('per_class', {})
                            breakdown = ', '.join(f"{k.capitalize()}: {int(round(v))}" for k, v in pc.items()) if pc else ''
                            message = f"Image analyzed! Total count: {int(round(img_res.get('count', 0)))} | {breakdown}"
                            result = img_res
                        else:
                            message = 'Inference failed. Please try with another image.'
                            if os.path.exists(file_path):
                                os.remove(file_path)

                except Exception as e:
                    message = f'Error processing file: {str(e)}'

    # Check active task state if present
    active_task = None
    if active_task_id:
        with VIDEO_TASKS_LOCK:
            active_task = VIDEO_TASKS.get(active_task_id)
        if active_task and active_task.get('status') == 'completed':
            video_result = active_task.get('result')
            uploaded_media_url = active_task.get('uploaded_media_url')

    return render_template(
        'upload.html',
        result=result,
        video_result=video_result,
        active_task_id=active_task_id,
        active_task=active_task,
        message=message,
        uploaded_media_url=uploaded_media_url,
        max_file_size=MAX_FILE_SIZE / (1024 * 1024),
        allowed_extensions=", ".join(ALLOWED_EXTENSIONS),
    )


@app.route('/api/video/upload', methods=['POST'])
def api_video_upload():
    """JSON API for uploading a video and starting continuous counting."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not is_video_file(file.filename):
        return jsonify({'error': 'File is not a supported video format'}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(file_path)

    speed_mode = request.form.get('speed_mode', 'balanced')
    try:
        extract_fps = float(request.form.get('extract_fps', 1.0) or 1.0)
    except (ValueError, TypeError):
        extract_fps = 1.0

    task_id = start_video_processing(
        file_path,
        filename,
        unique_name,
        speed_mode=speed_mode,
        extract_fps=extract_fps,
    )

    return jsonify({
        'status': 'started',
        'task_id': task_id,
        'filename': filename,
        'uploaded_media_url': url_for('uploaded_file', filename=unique_name),
        'status_url': url_for('video_task_status', task_id=task_id),
        'stream_url': url_for('video_task_stream', task_id=task_id),
    })


@app.route('/api/video/status/<task_id>')
def video_task_status(task_id):
    """Poll current status, real-time count, and telemetry for a video task."""
    with VIDEO_TASKS_LOCK:
        task = VIDEO_TASKS.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


@app.route('/api/video/stream/<task_id>')
def video_task_stream(task_id):
    """Server-Sent Events (SSE) stream delivering live count continuously."""
    def event_generator():
        while True:
            with VIDEO_TASKS_LOCK:
                task = VIDEO_TASKS.get(task_id)
            if not task:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            status = task.get('status', 'processing')
            yield f"data: {json.dumps(task)}\n\n"

            if status in ('completed', 'error'):
                break

            time.sleep(0.18)

    return Response(event_generator(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    })


@app.route('/model')
def model_info():
    """Explain the crowd-counting model and analysis workflow."""
    return render_template('model.html')


@app.route('/analysis/<path:filename>')
def analysis_view(filename):
    """Show an inference density map within the SATARK interface."""
    target_path = OUTPUT_DIR / filename
    if not target_path.is_file():
        target_path = OUTPUT_DIR / os.path.basename(filename)
    if not target_path.is_file():
        return 'Analysis image not found.', 404
    rel_path = target_path.relative_to(OUTPUT_DIR).as_posix()
    scene = request.args.get('scene', type=int)
    scene_name = f'Simhastha Crowd Scene {scene:02d}' if scene else 'Crowd Density Analysis'
    return render_template(
        'analysis.html',
        scene_name=scene_name,
        image_url=url_for('output_image', filename=rel_path),
    )


@app.route('/outputs/<path:filename>')
def output_image(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route('/api/info')
def api_info():
    return jsonify({
        'model': 'CSRNet with Squeeze-and-Excitation attention',
        'features': [
            'Multi-class headgear density estimation',
            'Continuous real-time video crowd counting',
            'Live HUD overlays',
            'Adaptive temporal sampling engine'
        ],
        'classes': ['head', 'turban', 'veil', 'cap'],
        'framework': 'PyTorch',
    })


if __name__ == '__main__':
    # Pre-warm model in background
    threading.Thread(target=get_shared_model, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=5000)

