"""Similarity-based loss functions."""

try:
    from sentence_transformers.sentence_transformer.losses import (
        AnglELoss,
        CoSENTLoss,
        CosineSimilarityLoss,
        MSELoss,
    )
except ImportError:  # pragma: no cover - compatibility with sentence-transformers 3.x
    from sentence_transformers.losses import (
        AnglELoss,
        CoSENTLoss,
        CosineSimilarityLoss,
        MSELoss,
    )

__all__ = [
    "CosineSimilarityLoss",
    "CoSENTLoss",
    "AnglELoss",
    "MSELoss",
]
