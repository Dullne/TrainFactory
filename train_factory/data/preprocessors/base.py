"""Base preprocessor interface."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for data preprocessors.

    Preprocessors are responsible for transforming raw data into
    the format expected by the training pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessor configuration
        """
        self.config = config or {}

    @abstractmethod
    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset.

        Args:
            dataset: Input dataset

        Returns:
            Preprocessed dataset
        """
        pass

    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single sample.

        Args:
            sample: Input sample

        Returns:
            Preprocessed sample
        """
        return sample

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate a single sample.

        Args:
            sample: Sample to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def filter_invalid(self, dataset: Dataset) -> Dataset:
        """
        Filter out invalid samples from dataset.

        Args:
            dataset: Input dataset

        Returns:
            Filtered dataset
        """
        original_size = len(dataset)
        dataset = dataset.filter(self.validate_sample)
        filtered_size = len(dataset)

        if filtered_size < original_size:
            logger.info(
                f"Filtered {original_size - filtered_size} invalid samples "
                f"({filtered_size}/{original_size} remaining)"
            )

        return dataset

    @property
    def preprocessor_type(self) -> str:
        """Return the preprocessor type name."""
        return self.__class__.__name__


class TokenizingPreprocessor(BasePreprocessor):
    """
    Base class for preprocessors that require tokenization.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tokenizing preprocessor.

        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            config: Additional configuration
        """
        super().__init__(config)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = True,
        padding: Union[bool, str] = False,
    ) -> Dict[str, List[int]]:
        """
        Tokenize text.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate
            padding: Padding strategy

        Returns:
            Tokenized output with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=self.max_length,
            padding=padding,
            return_tensors=None,
        )

    def get_token_length(self, text: str) -> int:
        """Get the token length of text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
