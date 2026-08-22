"""Base data format definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class DataFormatType(str, Enum):
    """Data format type enumeration."""

    # Embedding formats
    PAIR = "pair"                              # (text1, text2)
    PAIR_SCORE = "pair_score"                  # (text1, text2, score)
    PAIR_CLASS = "pair_class"                  # (text1, text2, label)
    TRIPLET = "triplet"                        # (anchor, positive, negative)
    QUERY_POSITIVE_NEGATIVES = "qpn"           # (query, positive, [negatives])

    # SFT formats
    INSTRUCTION = "instruction"               # (instruction, input, output)
    CONVERSATION = "conversation"             # Multi-turn conversation
    ALPACA = "alpaca"                         # Alpaca format
    SHAREGPT = "sharegpt"                     # ShareGPT format

    # DPO formats
    PREFERENCE = "preference"                 # (prompt, chosen, rejected)
    PREFERENCE_RANKED = "preference_ranked"   # (prompt, responses, rankings)

    # Reranker formats
    RERANKER_PAIR = "reranker_pair"          # (query, passage, label)
    RERANKER_LIST = "reranker_list"          # (query, passages, labels)


@dataclass
class DataSample:
    """Base class for data samples."""
    pass


@dataclass
class EmbeddingPairSample(DataSample):
    """Embedding pair sample."""
    text1: str
    text2: str
    score: Optional[float] = None
    label: Optional[int] = None


@dataclass
class EmbeddingTripletSample(DataSample):
    """Embedding triplet sample."""
    anchor: str
    positive: str
    negative: str


@dataclass
class EmbeddingQPNSample(DataSample):
    """Embedding query-positive-negatives sample."""
    query: str
    positive: str
    negatives: List[str]


@dataclass
class SFTSample(DataSample):
    """SFT (Supervised Fine-Tuning) sample."""
    instruction: str
    input: str = ""
    output: str = ""
    system: Optional[str] = None


@dataclass
class ConversationMessage:
    """Single message in a conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ConversationSample(DataSample):
    """Multi-turn conversation sample."""
    messages: List[ConversationMessage]


@dataclass
class PreferenceSample(DataSample):
    """DPO preference sample."""
    prompt: str
    chosen: str
    rejected: str
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None


@dataclass
class RerankerSample(DataSample):
    """Reranker sample."""
    query: str
    passage: str
    label: Union[int, float]  # 0/1 for classification, float for regression


class BaseDataFormat(ABC):
    """Abstract base class for data formats."""

    @property
    @abstractmethod
    def format_type(self) -> DataFormatType:
        """Return the data format type."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Return the required column names."""
        pass

    @property
    def optional_columns(self) -> List[str]:
        """Return optional column names."""
        return []

    @abstractmethod
    def validate(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a data sample.

        Args:
            sample: Data sample dictionary

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def parse(self, sample: Dict[str, Any]) -> DataSample:
        """
        Parse a data sample into structured format.

        Args:
            sample: Data sample dictionary

        Returns:
            Parsed DataSample object
        """
        pass

    def validate_columns(self, sample: Dict[str, Any]) -> bool:
        """Check if all required columns are present."""
        return all(col in sample for col in self.required_columns)
