import argparse
from src.visualize import generate_previews


def parse_args():
    parser = argparse.ArgumentParser(description='Generate preview visualizations for heatmaps.')
    parser.add_argument('--data-dir', default='data', help='Base data folder')
    parser.add_argument('--output-dir', default='previews', help='Preview output folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_previews(data_dir=args.data_dir, output_dir=args.output_dir)
