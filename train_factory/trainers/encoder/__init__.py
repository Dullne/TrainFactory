"""Encoder-based trainers (SentenceTransformer, CrossEncoder)."""

__all__ = ["EmbeddingTrainer", "RerankerTrainer"]


def __getattr__(name: str):
    if name == "EmbeddingTrainer":
        from .embedding_trainer import EmbeddingTrainer

        return EmbeddingTrainer
    if name == "RerankerTrainer":
        from .reranker_trainer import RerankerTrainer

        return RerankerTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
