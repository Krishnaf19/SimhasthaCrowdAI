import argparse
from src.evaluate import run_baseline_comparison


def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline evaluation with pre-trained ShanghaiTech weights.')
    parser.add_argument('--weights-path', default='baseline_weights.pth', help='Baseline weight file')
    parser.add_argument('--data-root', default='data', help='Root directory for data folders')
    parser.add_argument('--crop-size', type=int, default=512, help='Input crop size for evaluation')
    parser.add_argument('--downsample', type=int, default=8, help='Density map downsample factor')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_baseline_comparison(
        weights_path=args.weights_path,
        data_root=args.data_root,
        crop_size=args.crop_size,
        downsample=args.downsample,
    )
