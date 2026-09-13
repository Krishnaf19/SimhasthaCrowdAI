import argparse
from src.train import train_satark


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune SATARK on Simhastha training data.')
    parser.add_argument('--weights-path', default='baseline_weights.pth', help='Initial baseline weights')
    parser.add_argument('--data-root', default='data', help='Root data folder')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')
    parser.add_argument('--density-threshold', type=float, default=150.0, help='Density threshold for weighted loss')
    parser.add_argument('--penalty', type=float, default=3.0, help='Penalty multiplier for dense crowds')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--unfreeze-after', type=int, default=15, help='Epoch after which to unfreeze the frontend')
    parser.add_argument('--scheduler-patience', type=int, default=3, help='ReduceLROnPlateau patience')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help='LR reduction factor on plateau')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate for scheduler')
    parser.add_argument('--use-se', action='store_true', help='Use squeeze-and-excitation block in CSRNet')
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--warmup-epochs', type=int, default=2, help='Number of warmup epochs')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')
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
        unfreeze_after=args.unfreeze_after,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        early_stopping_patience=args.early_stopping_patience,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
    )
