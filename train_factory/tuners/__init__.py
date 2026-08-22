"""
Tuners for TrainFactory.

This module provides the currently supported fine-tuning methods including:
- LoRA (Low-Rank Adaptation)
- Full parameter fine-tuning

Usage:
    from train_factory.tuners import TunerRegistry, LoRATuner

    # Create tuner via registry
    tuner = TunerRegistry.create("lora", config={"r": 16, "lora_alpha": 32})
    model = tuner.prepare_model(model)

    # Or use tuner class directly
    tuner = LoRATuner({"r": 16, "lora_alpha": 32})
    model = tuner.prepare_model(model)
"""

from .base import BaseTuner
from .lora import LoRATuner
from .full_param import FullParameterTuner
from .registry import TunerRegistry

__all__ = [
    # Base
    "BaseTuner",
    # Implementations
    "LoRATuner",
    "FullParameterTuner",
    # Registry
    "TunerRegistry",
]
