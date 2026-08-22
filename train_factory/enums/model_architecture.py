"""Model architecture enumerations."""

from enum import Enum


class ModelArchitecture(str, Enum):
    """Model architecture type."""

    ENCODER = "encoder"    # Encoder-only (BERT-like)
    DECODER = "decoder"    # Decoder-only (GPT-like)


class ModelType(str, Enum):
    """Model type for training."""

    # Encoder-based
    EMBEDDING = "embedding"            # SentenceTransformer
    RERANKER = "reranker"              # CrossEncoder

    # Decoder-based
    DECODER_RERANKER = "decoder_reranker"  # Qwen3-Reranker
    LLM = "llm"                        # General LLM


class TrainingMethod(str, Enum):
    """Training method."""

    # Supervised Training
    SFT = "sft"                # Supervised Fine-Tuning
    CPT = "cpt"                # Continual Pre-Training

    # Preference Optimization
    DPO = "dpo"                # Direct Preference Optimization
    GRPO = "grpo"              # Group Relative Policy Optimization
    KTO = "kto"                # Kahneman-Tversky Optimization
    ORPO = "orpo"              # Odds Ratio Preference Optimization
    SIMPO = "simpo"            # Simple Preference Optimization

    # Reinforcement Learning
    PPO = "ppo"                # Proximal Policy Optimization

    # Multi-stage
    TWO_STAGE = "two_stage"    # SFT + RL two-stage training


class TunerType(str, Enum):
    """Parameter-efficient fine-tuning method type."""

    LORA = "lora"              # Low-Rank Adaptation
    QLORA = "qlora"            # Quantized LoRA (4-bit/8-bit)
    FULL = "full"              # Full parameter fine-tuning
    FREEZE = "freeze"          # Freeze layers tuning
    ADAPTER = "adapter"        # Adapter layers
    PREFIX = "prefix"          # Prefix tuning
    PROMPT = "prompt"          # Prompt tuning
