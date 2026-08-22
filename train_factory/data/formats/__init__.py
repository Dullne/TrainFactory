"""
Data format definitions for TrainFactory.

This module provides data format classes for different training types:
- Embedding formats (pair, triplet, query-positive-negatives)
- SFT formats (instruction, conversation, alpaca, sharegpt)
- DPO formats (preference, ranked preference)
"""

from .base import (
    DataFormatType,
    DataSample,
    BaseDataFormat,
    EmbeddingPairSample,
    EmbeddingTripletSample,
    EmbeddingQPNSample,
    SFTSample,
    ConversationSample,
    ConversationMessage,
    PreferenceSample,
    RerankerSample,
)

from .embedding import (
    PairFormat,
    PairScoreFormat,
    PairClassFormat,
    TripletFormat,
    QPNFormat,
    get_embedding_format,
    detect_embedding_format,
    EMBEDDING_FORMATS,
)

from .sft import (
    InstructionFormat,
    AlpacaFormat,
    ShareGPTFormat,
    ConversationFormat,
    get_sft_format,
    SFT_FORMATS,
)

from .dpo import (
    PreferenceFormat,
    PreferenceRankedFormat,
    UltraFeedbackFormat,
    get_dpo_format,
    DPO_FORMATS,
)

__all__ = [
    # Base
    "DataFormatType",
    "DataSample",
    "BaseDataFormat",
    # Data samples
    "EmbeddingPairSample",
    "EmbeddingTripletSample",
    "EmbeddingQPNSample",
    "SFTSample",
    "ConversationSample",
    "ConversationMessage",
    "PreferenceSample",
    "RerankerSample",
    # Embedding formats
    "PairFormat",
    "PairScoreFormat",
    "PairClassFormat",
    "TripletFormat",
    "QPNFormat",
    "get_embedding_format",
    "detect_embedding_format",
    "EMBEDDING_FORMATS",
    # SFT formats
    "InstructionFormat",
    "AlpacaFormat",
    "ShareGPTFormat",
    "ConversationFormat",
    "get_sft_format",
    "SFT_FORMATS",
    # DPO formats
    "PreferenceFormat",
    "PreferenceRankedFormat",
    "UltraFeedbackFormat",
    "get_dpo_format",
    "DPO_FORMATS",
]
