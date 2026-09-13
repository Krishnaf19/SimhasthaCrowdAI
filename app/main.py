import json
import os
from collections import Counter
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid

from flask import Flask, render_template, request, send_from_directory, url_for

from satark.utils.inference import infer_image, infer_images
from satark.utils.common import list_image_files, ensure_dir

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'checkpoints' / 'satark_best.pth'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'inference'
UPLOAD_DIR = BASE_DIR / 'uploads'
RESULTS_CACHE = OUTPUT_DIR / 'dashboard_results.json'

# Prefer the combined dataset image folder if available.
CANDIDATE_IMAGE_DIRS = [
    BASE_DIR / 'data' / 'images',
    BASE_DIR / 'data' / 'Train' / 'images',
    BASE_DIR / 'data' / 'Test' / 'images',
]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'avif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

ensure_dir(str(OUTPUT_DIR))
ensure_dir(str(UPLOAD_DIR))

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.secret_key = 'satark_secret_key_2025'


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_image_dir():
    """Find the primary image directory."""
    for path in CANDIDATE_IMAGE_DIRS:
        if path.exists() and list_image_files(str(path)):
            return path
    return None


def build_image_list():
    """Build list of available images from dataset."""
    image_dir = find_image_dir()
    return list_image_files(str(image_dir)) if image_dir else []


def load_cached_results():
    """Load cached dashboard results."""
    if RESULTS_CACHE.exists():
        try:
            with open(RESULTS_CACHE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cached_results(data):
    """Save dashboard results to cache."""
    try:
        with open(RESULTS_CACHE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def compute_dashboard_results(force_refresh=False):
    """Compute or retrieve cached dashboard results."""
    if not force_refresh:
        cached = load_cached_results()
        if cached:
            return cached

    image_dir = find_image_dir()
    if image_dir is None:
        return {
            'results': [],
            'total_images': 0,
            'safe_count': 0,
            'normal_count': 0,
            'critical_count': 0,
            'message': 'No image directory found. Please place images in data/images or data/Train/images.',
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
            'message': 'Inference failed or no images were processed.',
        }

    zone_counts = Counter(row.get('zone', 'UNKNOWN') for row in batch_results['results'])
    batch_results['total_images'] = len(batch_results['results'])
    batch_results['safe_count'] = zone_counts.get('SAFE', 0)
    batch_results['normal_count'] = zone_counts.get('NORMAL', 0)
    batch_results['critical_count'] = zone_counts.get('CRITICAL', 0)
    save_cached_results(batch_results)
    return batch_results


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main dashboard page with dataset inference."""
    images = build_image_list()
    result = None
    dashboard_results = None
    message = None
    refresh = False

    if request.method == 'POST':
        action = request.form.get('action')
        selected_image = request.form.get('image_name')
        if action == 'infer' and selected_image:
            image_dir = find_image_dir()
            img_path = image_dir / selected_image if image_dir else None
            if img_path is None or not img_path.exists():
                message = f"Image not found: {selected_image}"
            else:
                result = infer_image(
                    str(img_path),
                    model_path=str(MODEL_PATH),
                    output_dir=str(OUTPUT_DIR),
                )
                if result:
                    result['url'] = url_for('output_image', filename=os.path.basename(result['path']))
        elif action == 'refresh':
            message = 'Dashboard refreshed. Your existing dataset results have not been changed.'
        else:
            message = 'Dashboard updated automatically. Use refresh to recompute results.'

    dashboard_results = compute_dashboard_results(force_refresh=refresh)
    if dashboard_results and 'results' in dashboard_results:
        for index, row in enumerate(dashboard_results['results'], start=1):
            output_filename = os.path.basename(row['path'])
            row['url'] = url_for('output_image', filename=output_filename)
            row['display_name'] = f'Simhastha Crowd Scene {index:02d}'
            row['view_url'] = url_for('analysis_view', filename=output_filename, scene=index)

    return render_template(
        'index.html',
        images=images,
        result=result,
        dashboard_results=dashboard_results,
        message=message,
        inference_dir=str(find_image_dir().relative_to(BASE_DIR)) if find_image_dir() else 'data/images',
        output_dir=str(OUTPUT_DIR.relative_to(BASE_DIR)),
    )


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload and test page for custom images."""
    result = None
    message = None
    uploaded_image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file part in the request'
        else:
            file = request.files['file']
            if file.filename == '':
                message = 'No file selected for uploading'
            elif not allowed_file(file.filename):
                message = f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            elif file.content_length and file.content_length > MAX_FILE_SIZE:
                message = f'File size exceeds maximum limit of {MAX_FILE_SIZE / (1024*1024):.1f}MB'
            else:
                try:
                    # Save uploaded file
                    filename = secure_filename(file.filename)
                    # Add timestamp to avoid overwrites
                    filename = f"{uuid.uuid4()}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)

                    # Run inference
                    result = infer_image(
                        file_path,
                        model_path=str(MODEL_PATH),
                        output_dir=str(OUTPUT_DIR),
                    )

                    if result:
                        result['url'] = url_for('output_image', filename=os.path.basename(result['path']))
                        uploaded_image_url = url_for('uploaded_file', filename=filename)
                        pc = result.get('per_class', {})
                        breakdown = ', '.join('{}: {}'.format(k, int(round(v))) for k, v in pc.items()) if pc else ''
                        message = 'Analyzed! Total count: {} | {}'.format(int(round(result.get('count', 0))), breakdown)
                    else:
                        message = 'Inference failed. Please try with a different image.'
                        # Clean up failed upload
                        if os.path.exists(file_path):
                            os.remove(file_path)

                except Exception as e:
                    message = f'Error processing file: {str(e)}'

    return render_template(
        'upload.html',
        result=result,
        message=message,
        uploaded_image_url=uploaded_image_url,
        max_file_size=MAX_FILE_SIZE / (1024*1024),
        allowed_extensions=", ".join(ALLOWED_EXTENSIONS),
    )


@app.route('/model')
def model_info():
    """Explain the crowd-counting model and analysis workflow."""
    return render_template('model.html')


@app.route('/outputs/<path:filename>')
def output_image(filename):
    """Serve inference output images."""
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route('/analysis/<path:filename>')
def analysis_view(filename):
    """Show an inference density map within the SATARK interface."""
    safe_filename = os.path.basename(filename)
    if not (OUTPUT_DIR / safe_filename).is_file():
        return 'Analysis image not found.', 404
    scene = request.args.get('scene', type=int)
    scene_name = f'Simhastha Crowd Scene {scene:02d}' if scene else 'Crowd Density Analysis'
    return render_template(
        'analysis.html',
        scene_name=scene_name,
        image_url=url_for('output_image', filename=safe_filename),
    )


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded user images."""
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route('/api/info')
def api_info():
    """API endpoint for model and system information."""
    return json.dumps({
        'model': 'CSRNet with Squeeze-and-Excitation blocks',
        'backend': 'VGG16 (ImageNet pre-trained)',
        'framework': 'PyTorch',
        'focus': 'Crowd counting with cultural headgear diversity',
        'training_data': 'Simhastha Kumbh Mela religious gathering',
        'headgear_types': [
            'Saffron turbans',
            'Religious veils',
            'Traditional caps',
            'Various cloth headwear',
            'Bare heads'
        ],
        'metrics': {
            'MAE': '9.86 people',
            'RMSE': '~15 people',
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
