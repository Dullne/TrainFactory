"""Base tuner interface for parameter-efficient fine-tuning methods."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseTuner(ABC):
    """
    Abstract base class for all tuners.

    Tuners are responsible for applying parameter-efficient fine-tuning
    methods to models, such as LoRA, QLoRA, Adapter, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tuner.

        Args:
            config: Tuner configuration dictionary
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate tuner configuration. Override in subclasses for specific validation."""
        pass

    @abstractmethod
    def prepare_model(self, model: Any) -> Any:
        """
        Apply the tuner to the model.

        Args:
            model: The model to apply tuning to

        Returns:
            The model with tuner applied
        """
        pass

    @abstractmethod
    def get_trainable_parameters(self, model: Any) -> int:
        """
        Get the number of trainable parameters after tuning.

        Args:
            model: The tuned model

        Returns:
            Number of trainable parameters
        """
        pass

    def get_total_parameters(self, model: Any) -> int:
        """
        Get the total number of parameters in the model.

        Args:
            model: The model

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in model.parameters())

    def get_trainable_percentage(self, model: Any) -> float:
        """
        Get the percentage of trainable parameters.

        Args:
            model: The tuned model

        Returns:
            Percentage of trainable parameters
        """
        total = self.get_total_parameters(model)
        trainable = self.get_trainable_parameters(model)
        return (trainable / total * 100) if total > 0 else 0.0

    def print_trainable_parameters(self, model: Any) -> None:
        """Print trainable parameters information."""
        trainable = self.get_trainable_parameters(model)
        total = self.get_total_parameters(model)
        percentage = self.get_trainable_percentage(model)
        logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} ({percentage:.2f}%)"
        )

    @property
    @abstractmethod
    def tuner_type(self) -> str:
        """Return the tuner type name."""
        pass
