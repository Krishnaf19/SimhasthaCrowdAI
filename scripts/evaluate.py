import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from satark.engine.evaluator import run_satark_metrics


def main():
    p = argparse.ArgumentParser(description='Evaluate the fine-tuned SATARK model.')
    p.add_argument('--model-path', default='checkpoints/satark_best.pth')
    p.add_argument('--data-root',  default='data')
    p.add_argument('--split',      default='test')
    args = p.parse_args()
    run_satark_metrics(model_path=args.model_path, data_root=args.data_root, split=args.split)


if __name__ == '__main__':
    main()
