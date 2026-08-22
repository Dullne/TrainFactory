"""LoRA (Low-Rank Adaptation) tuner implementation."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseTuner

logger = logging.getLogger(__name__)


class LoRATuner(BaseTuner):
    """
    LoRA tuner for parameter-efficient fine-tuning.

    LoRA (Low-Rank Adaptation) freezes the pre-trained model weights and
    injects trainable rank decomposition matrices into each layer of the
    Transformer architecture.

    Reference: https://arxiv.org/abs/2106.09685
    """

    # Default target modules for common model architectures
    DEFAULT_TARGET_MODULES = {
        "default": ["q_proj", "v_proj"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bert": ["query", "value"],
        "roberta": ["query", "value"],
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LoRA tuner.

        Args:
            config: LoRA configuration dictionary with the following keys:
                - r: LoRA rank (default: 16)
                - lora_alpha: LoRA alpha scaling factor (default: r * 2)
                - lora_dropout: Dropout probability for LoRA layers (default: 0.0)
                - target_modules: List of module names to apply LoRA to
                - bias: Bias type: "none", "all", or "lora_only" (default: "none")
                - task_type: Task type for PEFT (optional)
        """
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate LoRA configuration."""
        r = self.config.get("r", 16)
        if not isinstance(r, int) or r < 1:
            raise ValueError(f"LoRA rank 'r' must be a positive integer, got {r}")

        lora_dropout = self.config.get("lora_dropout", 0.0)
        if not 0.0 <= lora_dropout <= 1.0:
            raise ValueError(f"lora_dropout must be between 0 and 1, got {lora_dropout}")

    def prepare_model(self, model: Any) -> Any:
        """
        Apply LoRA to the model.

        Args:
            model: The model to apply LoRA to

        Returns:
            The model with LoRA adapters
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA. "
                "Install it with: pip install peft"
            )

        # Get configuration values
        r = self.config.get("r", 16)
        lora_alpha = self.config.get("lora_alpha", r * 2)
        lora_dropout = self.config.get("lora_dropout", 0.0)
        target_modules = self.config.get("target_modules", self.DEFAULT_TARGET_MODULES["default"])
        bias = self.config.get("bias", "none")
        task_type_str = self.config.get("task_type")

        # Handle task type
        task_type = None
        if task_type_str:
            task_type = getattr(TaskType, task_type_str, None)

        # Create LoRA config
        lora_config_params = {
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "bias": bias,
        }
        if task_type:
            lora_config_params["task_type"] = task_type

        peft_config = LoraConfig(**lora_config_params)

        # Apply LoRA to model
        logger.info(f"Applying LoRA with r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Target modules: {target_modules}")

        model = get_peft_model(model, peft_config)
        self.print_trainable_parameters(model)

        return model

    def get_trainable_parameters(self, model: Any) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @property
    def tuner_type(self) -> str:
        """Return the tuner type name."""
        return "lora"

    @classmethod
    def get_target_modules_for_model(cls, model_type: str) -> List[str]:
        """
        Get default target modules for a specific model type.

        Args:
            model_type: Model type (e.g., "llama", "qwen", "bert")

        Returns:
            List of target module names
        """
        return cls.DEFAULT_TARGET_MODULES.get(
            model_type.lower(),
            cls.DEFAULT_TARGET_MODULES["default"]
        )

    @classmethod
    def find_all_linear_modules(cls, model: Any, exclude: Optional[List[str]] = None) -> List[str]:
        """
        Find all linear module names in the model.

        Args:
            model: The model to search
            exclude: Module names to exclude

        Returns:
            List of linear module names
        """
        import torch.nn as nn

        exclude = exclude or ["lm_head"]
        linear_module_names = set()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get the last part of the module name
                names = name.split(".")
                if names[-1] not in exclude:
                    linear_module_names.add(names[-1])

        return list(linear_module_names)
