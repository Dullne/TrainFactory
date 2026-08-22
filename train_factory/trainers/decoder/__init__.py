"""Decoder-based trainers (LLM SFT/DPO/ORPO, Decoder Reranker)."""

__all__ = ["DecoderRerankerTrainer", "LLMTrainer"]


def __getattr__(name: str):
    if name == "DecoderRerankerTrainer":
        from .decoder_reranker_trainer import DecoderRerankerTrainer

        return DecoderRerankerTrainer
    if name == "LLMTrainer":
        from .llm_trainer import LLMTrainer

        return LLMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
