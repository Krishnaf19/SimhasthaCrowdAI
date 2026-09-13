# SATARK Crowd Monitoring

A lightweight crowd density estimation and alert system built for large public gatherings such as the Simhastha festival. The project combines a CSRNet-style density estimation model with a Flask dashboard for inference, alerting, and image analysis.

## Overview

This application is designed to:

- estimate crowd density from images
- classify scenes as Safe, Normal, or Critical
- visualize density heatmaps and predictions
- support batch processing and single-image inference
- provide a simple web UI for uploading and reviewing results

## Key Features

- Crowd counting using a deep learning density map model
- Zone-based alert logic for crowd intensity
- Heatmap and prediction visualization
- Batch inference for multiple images
- Flask-based dashboard and upload interface
- Training, evaluation, and fine-tuning pipeline

## Project Structure

```text
.
├── app.py                        # Flask dashboard and upload app
├── 01_build_master_index.py      # Build dataset index from images and annotations
├── 02_stratified_train_test_split.py
├── 03_generate_heatmaps.py
├── 04_visualize_heatmaps.py
├── 05_evaluate_baseline.py
├── 06_fine_tune.py
├── 07_evaluate_finetuned_model.py
├── 08_visualize_predictions.py
├── 09_batch_inference.py
├── 10_alert_inference.py
├── generate_dataset_csv.py
├── inference_config.py
├── tune_config.py
├── requirements.txt
├── checkpoints/                 # trained model weights
├── data/                        # dataset and annotation files
├── Images/                      # source image dataset
├── outputs/                     # generated inference and alert outputs
├── templates/                   # Flask HTML templates
├── static/                      # CSS and static assets
├── uploads/                     # uploaded user images
├── src/                         # model, dataset, training, inference code
└── README.md
```

## Tech Stack

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy / Pandas
- Matplotlib
- Flask

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web App

Start the dashboard locally:

```bash
python app.py
```

Then open the app in your browser:

```text
http://localhost:5000
```

You can use the dashboard to:

- browse available images
- run inference on selected images
- refresh dataset results
- upload a new image for analysis

## Training Pipeline

The workflow is organized as a sequence of scripts:

1. Build the dataset index
   ```bash
   python 01_build_master_index.py
   ```

2. Split data into train/test sets
   ```bash
   python 02_stratified_train_test_split.py
   ```

3. Generate density heatmaps
   ```bash
   python 03_generate_heatmaps.py
   ```

4. Train or fine-tune the model
   ```bash
   python 06_fine_tune.py
   ```

5. Evaluate the trained model
   ```bash
   python 05_evaluate_baseline.py
   python 07_evaluate_finetuned_model.py
   ```

6. Run batch or alert inference
   ```bash
   python 09_batch_inference.py
   python 10_alert_inference.py
   ```

## Model Output

The project stores trained checkpoints in the `checkpoints/` directory and outputs visual results in the `outputs/` directory, including:

- inference images
- density heatmaps
- alert summaries
- processed prediction results

## Notes

- The project is tuned for crowd-heavy scenes and public event monitoring.
- You can adjust thresholds and configurations in the config files such as `inference_config.py` and `tune_config.py`.
- Model performance depends on image quality, dataset consistency, and annotation accuracy.

## License

This project is intended for academic and research use. Please check your local usage requirements before deploying in production or public systems.
