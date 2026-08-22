"""Tuner registry with registration mechanism."""

import logging
from typing import Any, Dict, Type, Callable, Optional

from .base import BaseTuner
from .policy import KNOWN_TUNER_TYPES, ensure_tuner_supported

logger = logging.getLogger(__name__)


class TunerRegistry:
    """
    Registry for tuners with factory pattern support.

    Usage:
        # Register a custom tuner
        @TunerRegistry.register("my_tuner")
        class MyTuner(BaseTuner):
            ...

        # Get a tuner instance
        tuner = TunerRegistry.create("lora", config={"r": 16})

        # Apply tuner to model
        model = tuner.prepare_model(model)
    """

    _registry: Dict[str, Type[BaseTuner]] = {}

    @staticmethod
    def _canonical_registry_name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("Tuner name must be a string")
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Tuner name must not be empty")
        if normalized in KNOWN_TUNER_TYPES:
            supported = ensure_tuner_supported(normalized)
            if supported is None:
                raise ValueError("Tuner name must not be empty")
            return supported
        return normalized

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator for registering a tuner.

        Args:
            name: Unique name for the tuner

        Returns:
            Decorator function
        """
        canonical_name = cls._canonical_registry_name(name)

        def decorator(tuner_cls: Type[BaseTuner]) -> Type[BaseTuner]:
            if not issubclass(tuner_cls, BaseTuner):
                raise TypeError(f"{tuner_cls} must be a subclass of BaseTuner")

            cls._registry[canonical_name] = tuner_cls
            logger.debug(f"Registered tuner: {canonical_name}")
            return tuner_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseTuner:
        """
        Create a tuner instance by name.

        Args:
            name: Name of the tuner
            config: Tuner configuration dictionary

        Returns:
            Tuner instance
        """
        name = cls._canonical_registry_name(name)
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown tuner: {name}. Available: {available}")

        tuner_cls = cls._registry[name]
        return tuner_cls(config or {})

    @classmethod
    def get(cls, name: str) -> Type[BaseTuner]:
        """
        Get a tuner class by name.

        Args:
            name: Name of the tuner

        Returns:
            Tuner class
        """
        name = cls._canonical_registry_name(name)
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown tuner: {name}. Available: {available}")

        return cls._registry[name]

    @classmethod
    def list_all(cls) -> Dict[str, Type[BaseTuner]]:
        """List all registered tuners."""
        return cls._registry.copy()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a tuner is registered."""
        try:
            name = cls._canonical_registry_name(name)
        except ValueError:
            return False
        return name in cls._registry


# Register built-in tuners
def _register_builtin_tuners():
    """Register built-in tuners."""
    from .lora import LoRATuner
    from .full_param import FullParameterTuner

    TunerRegistry._registry["lora"] = LoRATuner
    TunerRegistry._registry["full"] = FullParameterTuner


# Auto-register on import
_register_builtin_tuners()
