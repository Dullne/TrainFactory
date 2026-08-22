"""
TrainFactory - A standalone training framework for Embedding and Reranker models.

This package provides a clean, decoupled training framework that supports:
- Embedding model training (SentenceTransformer)
- Reranker model training (CrossEncoder)
- Multi-GPU training
- HuggingFace and ModelScope model/dataset support
- MySQL-based task persistence
- Simple REST API for training management
"""

__version__ = "0.1.0"

__all__ = ["settings", "__version__"]


def __getattr__(name: str):
    """Lazily load settings to avoid side effects at package import time."""
    if name == "settings":
        from .config import settings
        return settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
