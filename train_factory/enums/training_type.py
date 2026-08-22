"""Training type and related enumerations."""

from enum import Enum


class TrainingType(str, Enum):
    """Supported training types."""

    EMBEDDING = "embedding"       # Sentence embedding model (SentenceTransformer)
    RERANKER = "reranker"         # Reranker model (CrossEncoder)


class DatasetFormat(str, Enum):
    """Dataset format types for embedding training."""

    AUTO = "auto"                             # Auto-detect format
    PAIR_SCORE = "pair-score"                 # (text1, text2, score)
    PAIR_CLASS = "pair-class"                 # (text1, text2, label)
    TRIPLET = "triplet"                       # (anchor, positive, negative)
    PAIR = "pair"                             # (text1, text2) - positive pairs
    QUERY_POSITIVE_NEGATIVES = "qpn"          # (query, positive, [negatives])


class LossType(str, Enum):
    """Loss function types."""

    # === Contrastive Learning Losses ===
    MULTIPLE_NEGATIVES_RANKING = "MultipleNegativesRankingLoss"
    CACHED_MULTIPLE_NEGATIVES_RANKING = "CachedMultipleNegativesRankingLoss"
    CONTRASTIVE = "ContrastiveLoss"
    ONLINE_CONTRASTIVE = "OnlineContrastiveLoss"
    MEGA_BATCH_MARGIN = "MegaBatchMarginLoss"

    # === Similarity Losses ===
    COSINE_SIMILARITY = "CosineSimilarityLoss"
    COSENT = "CoSENTLoss"
    ANGLE = "AnglELoss"

    # === Triplet Losses ===
    TRIPLET = "TripletLoss"
    BATCH_HARD_TRIPLET = "BatchHardTripletLoss"
    BATCH_SEMI_HARD_TRIPLET = "BatchSemiHardTripletLoss"
    BATCH_ALL_TRIPLET = "BatchAllTripletLoss"

    # === Classification Losses ===
    SOFTMAX = "SoftmaxLoss"

    # === Knowledge Distillation Losses ===
    MSE = "MSELoss"
    DISTILL_KL_DIV = "DistillKLDivLoss"

    # === GIST Embedding Loss ===
    GIST_EMBED = "GISTEmbedLoss"

    # === Matryoshka Losses (nested dimension training) ===
    MATRYOSHKA = "MatryoshkaLoss"
    MATRYOSHKA_2D = "Matryoshka2dLoss"

    # === Custom Losses ===
    DYNAMIC_NEGATIVES = "DynamicExplicitNegativesRankingLoss"

    # === Reranker Classification Losses ===
    CROSS_ENTROPY = "CrossEntropyLoss"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"
    BCE = "BCELoss"

    # === Reranker Regression Losses ===
    RERANKER_MSE = "MSELoss"
    L1 = "L1Loss"
    SMOOTH_L1 = "SmoothL1Loss"
    HUBER = "HuberLoss"

    # === Reranker Ranking Losses ===
    MARGIN_RANKING = "MarginRankingLoss"
    COSINE_EMBEDDING = "CosineEmbeddingLoss"
    PAIRWISE_HINGE = "PairwiseHingeLoss"
    LISTWISE = "ListwiseLoss"
    LAMBDA = "LambdaLoss"
