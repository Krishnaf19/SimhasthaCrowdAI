import argparse
from src.data_builder import build_master_index


def parse_args():
    parser = argparse.ArgumentParser(description='Build the Simhastha master CSV index.')
    parser.add_argument('--img-dir', default='Images', help='Source image folder')
    parser.add_argument('--xml-dir', default='Annotations', help='Source annotation folder')
    parser.add_argument('--output-csv', default='simhastha_master_index.csv', help='Output CSV file')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_master_index(
        img_dir=args.img_dir,
        xml_dir=args.xml_dir,
        output_csv=args.output_csv,
        train_ratio=args.train_ratio,
    )
