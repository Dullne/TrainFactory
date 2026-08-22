"""
Trainers for TrainFactory.

Directory Structure:
    trainers/
    ├── base/           # Base trainer classes
    ├── encoder/        # Encoder-based (Embedding, CrossEncoder Reranker) - SFT only
    ├── decoder/        # Decoder-based (LLM, Decoder Reranker) - SFT + RL (DPO, GRPO, etc.)
    └── factory.py      # Trainer factory
"""

__all__ = [
    "BaseTrainer",
    "TrainingResult",
    "EmbeddingTrainer",
    "RerankerTrainer",
    "DecoderRerankerTrainer",
    "LLMTrainer",
    "TrainerFactory",
]


def __getattr__(name: str):
    if name == "BaseTrainer":
        from .base.base_trainer import BaseTrainer

        return BaseTrainer
    if name == "TrainingResult":
        from .base.training_result import TrainingResult

        return TrainingResult
    if name == "EmbeddingTrainer":
        from .encoder.embedding_trainer import EmbeddingTrainer

        return EmbeddingTrainer
    if name == "RerankerTrainer":
        from .encoder.reranker_trainer import RerankerTrainer

        return RerankerTrainer
    if name == "DecoderRerankerTrainer":
        from .decoder.decoder_reranker_trainer import DecoderRerankerTrainer

        return DecoderRerankerTrainer
    if name == "LLMTrainer":
        from .decoder.llm_trainer import LLMTrainer

        return LLMTrainer
    if name == "TrainerFactory":
        from .factory import TrainerFactory

        return TrainerFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
