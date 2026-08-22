"""Trainer factory for creating trainers based on (model_type, training_method)."""

import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)


class TrainerFactory:
    """
    Factory for creating trainers based on (model_type, training_method) combination.

    Usage:
        trainer = TrainerFactory.create("embedding", "sft", **config)
        trainer = TrainerFactory.create("decoder_reranker", "grpo", **config)
    """

    _registry: Dict[tuple, str] = {
        # Encoder-based (only SFT)
        ("embedding", "sft"): "encoder.EmbeddingTrainer",
        ("reranker", "sft"): "encoder.RerankerTrainer",

        # Decoder-based Reranker (SFT and RL)
        ("decoder_reranker", "sft"): "decoder.DecoderRerankerTrainer",
        ("decoder_reranker", "grpo"): "decoder.DecoderRerankerTrainer",
        ("decoder_reranker", "dapo"): "decoder.DecoderRerankerTrainer",
        ("decoder_reranker", "dr_grpo"): "decoder.DecoderRerankerTrainer",
        ("decoder_reranker", "dpo"): "decoder.DecoderRerankerTrainer",

        # LLM (SFT, DPO, ORPO)
        ("llm", "sft"): "decoder.LLMTrainer",
        ("llm", "dpo"): "decoder.LLMTrainer",
        ("llm", "orpo"): "decoder.LLMTrainer",
    }

    @classmethod
    def create(cls, model_type: str, training_method: str = "sft", **config) -> Any:
        """
        Create a trainer based on model type and training method.

        Args:
            model_type: Type of model (embedding, reranker, llm, decoder_reranker)
            training_method: Training method (sft, dpo, grpo, dapo, dr_grpo)
            **config: Training configuration

        Returns:
            Trainer instance
        """
        key = (model_type, training_method)

        if key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported combination: {key}. "
                f"Available combinations: {available}"
            )

        trainer_path = cls._registry[key]
        trainer_cls = cls._import_trainer(trainer_path)

        logger.info(f"Creating trainer: {trainer_cls.__name__} for {key}")
        return trainer_cls(config)

    @classmethod
    def _import_trainer(cls, trainer_path: str) -> Type:
        """Import trainer class from path string."""
        module_name, class_name = trainer_path.rsplit(".", 1)

        if module_name == "encoder":
            from .encoder import EmbeddingTrainer, RerankerTrainer
            return {"EmbeddingTrainer": EmbeddingTrainer, "RerankerTrainer": RerankerTrainer}[class_name]
        elif module_name == "decoder":
            from .decoder import DecoderRerankerTrainer, LLMTrainer
            return {"DecoderRerankerTrainer": DecoderRerankerTrainer,
                    "LLMTrainer": LLMTrainer}[class_name]

        raise ValueError(f"Unknown trainer module: {module_name}")

    @classmethod
    def register(cls, model_type: str, training_method: str, trainer_cls: Type):
        """
        Register a custom trainer.

        Args:
            model_type: Type of model
            training_method: Training method
            trainer_cls: Trainer class
        """
        key = (model_type, training_method)
        cls._registry[key] = f"custom.{trainer_cls.__name__}"
        logger.info(f"Registered custom trainer: {trainer_cls.__name__} for {key}")

    @classmethod
    def list_available(cls) -> Dict[tuple, str]:
        """List all available trainer combinations."""
        return cls._registry.copy()
