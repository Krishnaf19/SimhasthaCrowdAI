import argparse
from src.inference import run_batch_inference


def parse_args():
    parser = argparse.ArgumentParser(description='Run alert-style batch inference for SATARK.')
    parser.add_argument('--model-path', default='checkpoints/satark_best.pth', help='SATARK model path')
    parser.add_argument('--inference-dir', default='data/Inference/images', help='Input images folder')
    parser.add_argument('--output-dir', default='outputs/inference', help='Output folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_batch_inference(
        model_path=args.model_path,
        inference_dir=args.inference_dir,
        output_dir=args.output_dir,
        simple=False,
    )
