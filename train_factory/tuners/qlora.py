"""QLoRA (Quantized LoRA) tuner implementation."""

import logging
from typing import Any, Dict, Optional

from .lora import LoRATuner
from .policy import ensure_tuner_supported

logger = logging.getLogger(__name__)


class QLoRATuner(LoRATuner):
    """
    QLoRA tuner for quantized parameter-efficient fine-tuning.

    QLoRA combines 4-bit quantization with LoRA to enable fine-tuning
    of large language models on consumer hardware.

    Reference: https://arxiv.org/abs/2305.14314
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize QLoRA tuner.

        Args:
            config: QLoRA configuration dictionary with the following keys:
                - All LoRA config keys (r, lora_alpha, etc.)
                - load_in_4bit: Whether to load model in 4-bit (default: True)
                - load_in_8bit: Whether to load model in 8-bit (default: False)
                - bnb_4bit_compute_dtype: Compute dtype for 4-bit (default: "float16")
                - bnb_4bit_quant_type: Quantization type (default: "nf4")
                - bnb_4bit_use_double_quant: Use double quantization (default: True)
        """
        ensure_tuner_supported("qlora")
        super().__init__(config)

    def _validate_config(self) -> None:
        """Validate QLoRA configuration."""
        super()._validate_config()

        load_in_4bit = self.config.get("load_in_4bit", True)
        load_in_8bit = self.config.get("load_in_8bit", False)

        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")

    def get_quantization_config(self) -> Optional[Any]:
        """
        Get BitsAndBytes quantization configuration.

        Returns:
            BitsAndBytesConfig for quantization, or None if not using quantization
        """
        try:
            from transformers import BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for QLoRA. "
                "Install them with: pip install transformers torch"
            )

        load_in_4bit = self.config.get("load_in_4bit", True)
        load_in_8bit = self.config.get("load_in_8bit", False)

        if not load_in_4bit and not load_in_8bit:
            return None

        # Get compute dtype
        compute_dtype_str = self.config.get("bnb_4bit_compute_dtype", "float16")
        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = compute_dtype_map.get(compute_dtype_str, torch.float16)

        if load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.config.get("bnb_4bit_use_double_quant", True),
            )
        else:
            return BitsAndBytesConfig(load_in_8bit=True)

    def prepare_model(self, model: Any) -> Any:
        """
        Apply QLoRA to the model.

        Note: For QLoRA, the model should already be loaded with quantization.
        This method applies LoRA on top of the quantized model.

        Args:
            model: The quantized model to apply LoRA to

        Returns:
            The model with LoRA adapters
        """
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError:
            raise ImportError(
                "PEFT library is required for QLoRA. "
                "Install it with: pip install peft"
            )

        # Prepare model for k-bit training
        logger.info("Preparing model for quantized training")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.config.get("use_gradient_checkpointing", True)
        )

        # Apply LoRA on top
        return super().prepare_model(model)

    @property
    def tuner_type(self) -> str:
        """Return the tuner type name."""
        return "qlora"

    @classmethod
    def load_model_with_quantization(
        cls,
        base_model_path: str,
        config: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Load a model with quantization configuration.

        Args:
            base_model_path: Base model name or path
            config: QLoRA configuration
            **kwargs: Additional arguments for model loading

        Returns:
            Quantized model
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers is required. Install it with: pip install transformers"
            )

        tuner = cls(config)
        quantization_config = tuner.get_quantization_config()

        logger.info(f"Loading model with quantization: {base_model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            **kwargs
        )

        return model
