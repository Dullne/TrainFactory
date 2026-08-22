"""Loss function registry with registration mechanism."""

import logging
from typing import Any, Dict, Type, Callable

logger = logging.getLogger(__name__)


class LossRegistry:
    """
    Registry for loss functions with decorator-based registration.

    Usage:
        @LossRegistry.register(
            name="lambda_loss",
            category="ranking",
            model_architecture="all"
        )
        class LambdaLoss:
            ...

        # Get loss function
        loss = LossRegistry.get("lambda_loss", metric="ndcg", sigma=1.0)
    """

    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: str,
        model_architecture: str = "all",
        description: str = ""
    ) -> Callable:
        """
        Decorator for registering a loss function.

        Args:
            name: Unique name for the loss function
            category: Category (contrastive, ranking, rl, similarity)
            model_architecture: Compatible architecture (encoder, decoder, all)
            description: Description of the loss function
        """
        def decorator(loss_cls: Type) -> Type:
            cls._registry[name] = {
                "class": loss_cls,
                "category": category,
                "model_architecture": model_architecture,
                "description": description,
            }
            logger.debug(f"Registered loss function: {name}")
            return loss_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """
        Get a loss function instance by name.

        Args:
            name: Name of the loss function
            **kwargs: Arguments to pass to the loss function constructor

        Returns:
            Loss function instance
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown loss function: {name}. Available: {available}")

        loss_info = cls._registry[name]
        return loss_info["class"](**kwargs)

    @classmethod
    def list_all(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered loss functions."""
        return {
            name: {k: v for k, v in info.items() if k != "class"}
            for name, info in cls._registry.items()
        }

    @classmethod
    def list_by_category(cls, category: str) -> Dict[str, Dict[str, Any]]:
        """List loss functions by category."""
        return {
            name: {k: v for k, v in info.items() if k != "class"}
            for name, info in cls._registry.items()
            if info["category"] == category
        }

    @classmethod
    def list_for_architecture(cls, arch: str) -> Dict[str, Dict[str, Any]]:
        """List loss functions compatible with an architecture."""
        return {
            name: {k: v for k, v in info.items() if k != "class"}
            for name, info in cls._registry.items()
            if info["model_architecture"] in [arch, "all"]
        }
