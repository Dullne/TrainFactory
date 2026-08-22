"""Data loading and processing for TrainFactory."""

from .data_loader import DataLoader

# Import submodules for easy access
from . import formats
from . import preprocessors

# Backward-compatible imports (embedding_recipe moved to utils/)
from ..utils import (
    EmbeddingRecipeSpec,
    EmbeddingRecipeError,
    detect_data_format,
    prepare_dataset_for_recipe,
    build_embedding_loss,
    normalize_loss_name,
)

__all__ = [
    "DataLoader",
    # submodules
    "formats",
    "preprocessors",
    # Backward-compatible (use train_factory.utils instead)
    "EmbeddingRecipeSpec",
    "EmbeddingRecipeError",
    "detect_data_format",
    "prepare_dataset_for_recipe",
    "build_embedding_loss",
    "normalize_loss_name",
]
