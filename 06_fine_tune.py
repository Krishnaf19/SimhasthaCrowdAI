import argparse
from src.train import train_satark


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune SATARK on Simhastha training data.')
    parser.add_argument('--weights-path', default='baseline_weights.pth', help='Initial baseline weights')
    parser.add_argument('--data-root', default='data', help='Root data folder')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--density-threshold', type=float, default=150.0, help='Density threshold for weighted loss')
    parser.add_argument('--penalty', type=float, default=2.5, help='Penalty multiplier for dense crowds')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--use-se', action='store_true', help='Use squeeze-and-excitation block in CSRNet')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_satark(
        weights_path=args.weights_path,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        density_threshold=args.density_threshold,
        penalty=args.penalty,
        save_every=args.save_every,
        use_se=args.use_se,
    )
