__all__ = ["train_satark", "run_satark_metrics", "load_checkpoint"]


def __getattr__(name):
    if name == "train_satark":
        from .trainer import train_satark
        return train_satark
    if name in {"run_satark_metrics", "load_checkpoint"}:
        from .evaluator import run_satark_metrics, load_checkpoint
        return run_satark_metrics if name == "run_satark_metrics" else load_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
