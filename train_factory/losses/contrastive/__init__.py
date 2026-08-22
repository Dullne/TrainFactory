"""Contrastive learning loss functions."""

from .dynamic_negatives_loss import DynamicExplicitNegativesRankingLoss

# TODO: Add more contrastive losses
# from .infonce import InfoNCELoss
# from .multiple_negatives import MultipleNegativesRankingLoss
# from .triplet import TripletLoss

__all__ = [
    "DynamicExplicitNegativesRankingLoss",
]
