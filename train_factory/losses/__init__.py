"""Loss functions for TrainFactory."""

from .contrastive import DynamicExplicitNegativesRankingLoss
from .registry import LossRegistry
from .similarity import AnglELoss, CoSENTLoss, CosineSimilarityLoss, MSELoss

__all__ = [
    "LossRegistry",
    "DynamicExplicitNegativesRankingLoss",
    "CosineSimilarityLoss",
    "CoSENTLoss",
    "AnglELoss",
    "MSELoss",
]
