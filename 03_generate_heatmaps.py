import argparse
from src.heatmap import generate_heatmaps, split_data


def parse_args():
    parser = argparse.ArgumentParser(description='Generate heatmaps and split Train/Test folders.')
    parser.add_argument('--image-dir', default='data/images', help='Centralized image folder')
    parser.add_argument('--anno-dir', default='data/annotations', help='Annotation JSON folder')
    parser.add_argument('--heat-dir', default='data/heatmaps', help='Output density map folder')
    parser.add_argument('--master-csv', default='simhastha_master_index.csv', help='Master CSV file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_heatmaps(image_dir=args.image_dir, anno_dir=args.anno_dir, output_dir=args.heat_dir)
    split_data(master_csv=args.master_csv, image_dir=args.image_dir, heat_dir=args.heat_dir)
