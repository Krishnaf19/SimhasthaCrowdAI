import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from satark.engine.trainer import train_satark


def main():
    p = argparse.ArgumentParser(description='Train SATARK headgear-aware crowd counter.')
    p.add_argument('--weights-path',    default='checkpoints/baseline_weights.pth')
    p.add_argument('--data-root',       default='data')
    p.add_argument('--checkpoint-dir',  default='checkpoints')
    p.add_argument('--epochs',          type=int,   default=80)
    p.add_argument('--batch-size',      type=int,   default=1)
    p.add_argument('--lr',              type=float, default=5e-5)
    p.add_argument('--unfreeze-after',  type=int,   default=15)
    p.add_argument('--early-stop',      type=int,   default=15, dest='early_stopping_patience')
    args = p.parse_args()
    train_satark(
        weights_path=args.weights_path,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        unfreeze_after=args.unfreeze_after,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == '__main__':
    main()
