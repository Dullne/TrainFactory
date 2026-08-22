"""Base trainer module."""

__all__ = ["BaseTrainer", "TrainingResult"]


def __getattr__(name: str):
    if name == "BaseTrainer":
        from .base_trainer import BaseTrainer

        return BaseTrainer
    if name == "TrainingResult":
        from .training_result import TrainingResult

        return TrainingResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
