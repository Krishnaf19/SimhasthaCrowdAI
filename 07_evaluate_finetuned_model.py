import argparse
from src.evaluate import run_satark_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SATARK model on a data split.')
    parser.add_argument('--model-path', default='checkpoints/satark_best.pth', help='SATARK model path')
    parser.add_argument('--data-root', default='data', help='Data root folder')
    parser.add_argument('--split', choices=['Train', 'Test', 'All'], default='Test', help='Split to evaluate')
    parser.add_argument('--crop-size', type=int, default=512, help='Crop size for model inputs')
    parser.add_argument('--downsample', type=int, default=8, help='Density map downsample factor')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_satark_metrics(
        model_path=args.model_path,
        data_root=args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        downsample=args.downsample,
    )
