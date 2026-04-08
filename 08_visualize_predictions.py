import argparse
from src.visualize import visualize_results


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize SATARK predictions.')
    parser.add_argument('--model-path', default='checkpoints/satark_best.pth', help='SATARK model path')
    parser.add_argument('--data-root', default='data', help='Data root folder')
    parser.add_argument('--output-dir', default='previews', help='Visualization output folder')
    parser.add_argument('--all-images', action='store_true', help='Visualize every test image')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_results(
        model_path=args.model_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        all_images=args.all_images,
    )
