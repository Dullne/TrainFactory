"""Full parameter fine-tuning (no tuner applied)."""

import logging
from typing import Any, Dict, List

from .base import BaseTuner
from .policy import ensure_tuner_supported

logger = logging.getLogger(__name__)


class FullParameterTuner(BaseTuner):
    """
    Full parameter fine-tuning tuner.

    This tuner makes all parameters trainable, which is the traditional
    fine-tuning approach. Optionally supports freezing specific layers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize full parameter tuner.

        Args:
            config: Configuration dictionary with the following keys:
                - freeze_layers: List of layer name patterns to freeze (optional)
                - freeze_embeddings: Whether to freeze embedding layers (default: False)
                - freeze_lm_head: Whether to freeze the LM head (default: False)
        """
        ensure_tuner_supported("full")
        super().__init__(config)

    def prepare_model(self, model: Any) -> Any:
        """
        Prepare model for full parameter fine-tuning.

        Args:
            model: The model to prepare

        Returns:
            The model with appropriate parameters set to trainable
        """
        # First, make all parameters trainable
        for param in model.parameters():
            param.requires_grad = True

        # Handle frozen layers
        freeze_layers = self.config.get("freeze_layers", [])
        freeze_embeddings = self.config.get("freeze_embeddings", False)
        freeze_lm_head = self.config.get("freeze_lm_head", False)

        frozen_count = 0

        for name, param in model.named_parameters():
            should_freeze = False

            # Check freeze_layers patterns
            for pattern in freeze_layers:
                if pattern in name:
                    should_freeze = True
                    break

            # Check embeddings
            if freeze_embeddings and "embed" in name.lower():
                should_freeze = True

            # Check LM head
            if freeze_lm_head and "lm_head" in name.lower():
                should_freeze = True

            if should_freeze:
                param.requires_grad = False
                frozen_count += 1

        if frozen_count > 0:
            logger.info(f"Frozen {frozen_count} parameters")

        self.print_trainable_parameters(model)
        return model

    def get_trainable_parameters(self, model: Any) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @property
    def tuner_type(self) -> str:
        """Return the tuner type name."""
        return "full"


class FreezeLayersTuner(BaseTuner):
    """
    Freeze specific layers tuner.

    This tuner freezes all layers except the specified ones,
    which is useful for transfer learning scenarios.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize freeze layers tuner.

        Args:
            config: Configuration dictionary with the following keys:
                - trainable_layers: List of layer name patterns to keep trainable
                - num_unfrozen_layers: Number of top layers to keep trainable (optional)
        """
        ensure_tuner_supported("freeze")
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate configuration."""
        trainable_layers = self.config.get("trainable_layers", [])
        num_unfrozen_layers = self.config.get("num_unfrozen_layers")

        if not trainable_layers and num_unfrozen_layers is None:
            raise ValueError(
                "Either 'trainable_layers' or 'num_unfrozen_layers' must be specified"
            )

    def prepare_model(self, model: Any) -> Any:
        """
        Prepare model by freezing most layers.

        Args:
            model: The model to prepare

        Returns:
            The model with most parameters frozen
        """
        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        trainable_layers = self.config.get("trainable_layers", [])
        num_unfrozen_layers = self.config.get("num_unfrozen_layers")

        unfrozen_count = 0

        # Unfreeze by layer patterns
        for name, param in model.named_parameters():
            for pattern in trainable_layers:
                if pattern in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    break

        # Unfreeze top N layers if specified
        if num_unfrozen_layers is not None:
            layer_names = self._get_layer_names(model)
            top_layers = layer_names[-num_unfrozen_layers:]

            for name, param in model.named_parameters():
                for layer_name in top_layers:
                    if layer_name in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                        break

        logger.info(f"Unfrozen {unfrozen_count} parameters")
        self.print_trainable_parameters(model)
        return model

    def _get_layer_names(self, model: Any) -> List[str]:
        """Get unique layer names from model."""
        layer_names = set()
        for name, _ in model.named_parameters():
            parts = name.split(".")
            # Look for numbered layers (e.g., layers.0, layers.1)
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:
                    layer_name = ".".join(parts[:i+1])
                    layer_names.add(layer_name)
                    break
        return sorted(layer_names)

    def get_trainable_parameters(self, model: Any) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @property
    def tuner_type(self) -> str:
        """Return the tuner type name."""
        return "freeze"
