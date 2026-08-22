"""Embedding training data formats."""

from typing import Any, Dict, List

from .base import (
    BaseDataFormat,
    DataFormatType,
    EmbeddingPairSample,
    EmbeddingTripletSample,
    EmbeddingQPNSample,
)


def normalize_embedding_negatives(value: Any) -> List[str] | None:
    """Normalize a scalar/list negative field without iterating strings."""
    if isinstance(value, str):
        return [value] if value.strip() else None
    if not isinstance(value, list) or not value:
        return None
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return None
    return list(value)


def _get_qpn_negatives(sample: Dict[str, Any]) -> Any:
    for field in ("negatives", "neg", "negative"):
        if field in sample:
            return sample[field]
    return None


class PairFormat(BaseDataFormat):
    """
    Simple pair format for embedding training.

    Expected columns:
        - text1/sentence1/anchor: First text
        - text2/sentence2/positive: Second text
    """

    # Column name aliases
    TEXT1_ALIASES = ["text1", "sentence1", "anchor", "query", "question"]
    TEXT2_ALIASES = ["text2", "sentence2", "positive", "passage", "answer"]

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.PAIR

    @property
    def required_columns(self) -> List[str]:
        return ["text1", "text2"]

    def _get_column(self, sample: Dict[str, Any], aliases: List[str]) -> str:
        """Get column value from sample using aliases."""
        for alias in aliases:
            if alias in sample:
                return sample[alias]
        return ""

    def validate(self, sample: Dict[str, Any]) -> bool:
        text1 = self._get_column(sample, self.TEXT1_ALIASES)
        text2 = self._get_column(sample, self.TEXT2_ALIASES)
        return bool(text1) and bool(text2)

    def parse(self, sample: Dict[str, Any]) -> EmbeddingPairSample:
        return EmbeddingPairSample(
            text1=self._get_column(sample, self.TEXT1_ALIASES),
            text2=self._get_column(sample, self.TEXT2_ALIASES),
        )


class PairScoreFormat(BaseDataFormat):
    """
    Pair with score format for embedding training.

    Expected columns:
        - text1/sentence1: First text
        - text2/sentence2: Second text
        - score/similarity: Similarity score (float)
    """

    TEXT1_ALIASES = ["text1", "sentence1", "anchor", "query"]
    TEXT2_ALIASES = ["text2", "sentence2", "positive", "passage"]
    SCORE_ALIASES = ["score", "similarity", "label"]

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.PAIR_SCORE

    @property
    def required_columns(self) -> List[str]:
        return ["text1", "text2", "score"]

    def _get_column(self, sample: Dict[str, Any], aliases: List[str], default=None):
        for alias in aliases:
            if alias in sample:
                return sample[alias]
        return default

    def validate(self, sample: Dict[str, Any]) -> bool:
        text1 = self._get_column(sample, self.TEXT1_ALIASES)
        text2 = self._get_column(sample, self.TEXT2_ALIASES)
        score = self._get_column(sample, self.SCORE_ALIASES)
        return bool(text1) and bool(text2) and score is not None

    def parse(self, sample: Dict[str, Any]) -> EmbeddingPairSample:
        return EmbeddingPairSample(
            text1=self._get_column(sample, self.TEXT1_ALIASES),
            text2=self._get_column(sample, self.TEXT2_ALIASES),
            score=float(self._get_column(sample, self.SCORE_ALIASES)),
        )


class PairClassFormat(BaseDataFormat):
    """
    Pair with class label format for embedding training.

    Expected columns:
        - text1/sentence1: First text
        - text2/sentence2: Second text
        - label: Class label (int)
    """

    TEXT1_ALIASES = ["text1", "sentence1", "premise", "anchor"]
    TEXT2_ALIASES = ["text2", "sentence2", "hypothesis", "positive"]
    LABEL_ALIASES = ["label", "class", "category"]

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.PAIR_CLASS

    @property
    def required_columns(self) -> List[str]:
        return ["text1", "text2", "label"]

    def _get_column(self, sample: Dict[str, Any], aliases: List[str], default=None):
        for alias in aliases:
            if alias in sample:
                return sample[alias]
        return default

    def validate(self, sample: Dict[str, Any]) -> bool:
        text1 = self._get_column(sample, self.TEXT1_ALIASES)
        text2 = self._get_column(sample, self.TEXT2_ALIASES)
        label = self._get_column(sample, self.LABEL_ALIASES)
        return bool(text1) and bool(text2) and label is not None

    def parse(self, sample: Dict[str, Any]) -> EmbeddingPairSample:
        return EmbeddingPairSample(
            text1=self._get_column(sample, self.TEXT1_ALIASES),
            text2=self._get_column(sample, self.TEXT2_ALIASES),
            label=int(self._get_column(sample, self.LABEL_ALIASES)),
        )


class TripletFormat(BaseDataFormat):
    """
    Triplet format for embedding training.

    Expected columns:
        - anchor: Anchor text
        - positive: Positive text
        - negative: Negative text
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.TRIPLET

    @property
    def required_columns(self) -> List[str]:
        return ["anchor", "positive", "negative"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False
        return (
            bool(sample.get("anchor"))
            and bool(sample.get("positive"))
            and bool(sample.get("negative"))
        )

    def parse(self, sample: Dict[str, Any]) -> EmbeddingTripletSample:
        return EmbeddingTripletSample(
            anchor=sample["anchor"],
            positive=sample["positive"],
            negative=sample["negative"],
        )


class QPNFormat(BaseDataFormat):
    """
    Query-Positive-Negatives format for embedding training.

    Expected columns:
        - query: Query text
        - positive/pos: Positive text
        - negatives/neg/negative: List of negative texts
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.QUERY_POSITIVE_NEGATIVES

    @property
    def required_columns(self) -> List[str]:
        return ["query", "positive", "negatives"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        query = sample.get("query")
        positive = sample.get("positive") or sample.get("pos")
        negatives = normalize_embedding_negatives(_get_qpn_negatives(sample))

        if not query or not positive:
            return False

        return negatives is not None

    def parse(self, sample: Dict[str, Any]) -> EmbeddingQPNSample:
        positive = sample.get("positive") or sample.get("pos")
        negatives = normalize_embedding_negatives(_get_qpn_negatives(sample))
        if negatives is None:
            raise ValueError("negative texts must be a non-empty string or list")

        return EmbeddingQPNSample(
            query=sample["query"],
            positive=positive,
            negatives=negatives,
        )


# Format registry
EMBEDDING_FORMATS = {
    DataFormatType.PAIR: PairFormat,
    DataFormatType.PAIR_SCORE: PairScoreFormat,
    DataFormatType.PAIR_CLASS: PairClassFormat,
    DataFormatType.TRIPLET: TripletFormat,
    DataFormatType.QUERY_POSITIVE_NEGATIVES: QPNFormat,
}


def get_embedding_format(format_type: DataFormatType) -> BaseDataFormat:
    """Get embedding format by type."""
    if format_type not in EMBEDDING_FORMATS:
        raise ValueError(f"Unknown embedding format: {format_type}")
    return EMBEDDING_FORMATS[format_type]()


def detect_embedding_format(sample: Dict[str, Any]) -> DataFormatType:
    """
    Auto-detect embedding data format from sample.

    Args:
        sample: A data sample dictionary

    Returns:
        Detected DataFormatType
    """
    # Check for QPN format
    if "query" in sample and any(
        field in sample for field in ("negatives", "neg", "negative")
    ):
        return DataFormatType.QUERY_POSITIVE_NEGATIVES

    # Check for triplet format
    if "anchor" in sample and "positive" in sample and "negative" in sample:
        return DataFormatType.TRIPLET

    # Check for pair with score
    if any(col in sample for col in PairScoreFormat.SCORE_ALIASES):
        return DataFormatType.PAIR_SCORE

    # Check for pair with label
    if any(col in sample for col in PairClassFormat.LABEL_ALIASES):
        return DataFormatType.PAIR_CLASS

    # Default to simple pair
    return DataFormatType.PAIR
