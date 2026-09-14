import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from satark.data.builder import build_master_index
from satark.data.heatmap  import generate_heatmaps, split_data


def main():
    p = argparse.ArgumentParser(description='Build Simhastha dataset index and density maps.')
    p.add_argument('--img-dir',      default='data/raw/images')
    p.add_argument('--xml-dir',      default='data/raw/annotations')
    p.add_argument('--output-csv',   default='simhastha_master_index.csv')
    p.add_argument('--train-ratio',  type=float, default=0.8)
    p.add_argument('--gen-heatmaps', action='store_true')
    p.add_argument('--split-data',   action='store_true')
    args = p.parse_args()

    build_master_index(args.img_dir, args.xml_dir, args.output_csv, args.train_ratio)
    if args.gen_heatmaps:
        generate_heatmaps()
    if args.split_data:
        split_data(master_csv=args.output_csv)


if __name__ == '__main__':
    main()
