"""
Configuration builder for training.

Handles creation and validation of training configurations using Pydantic.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from ..schemas.training_config import TrainingParametersManager
from ..config import settings

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """Builds and validates training configurations using Pydantic."""

    def __init__(self, training_config: Dict[str, Any]):
        """
        Initialize config builder.

        Args:
            training_config: Raw training configuration dictionary
        """
        self.raw_config = training_config

    def build_training_config(self, model_type: str) -> Dict[str, Any]:
        """
        Build validated training configuration using Pydantic.

        Args:
            model_type: Model type ('embedding', 'reranker', 'decoder_reranker', 'llm')

        Returns:
            Validated training configuration dictionary
        """
        logger.info(f"Building {model_type} training configuration")

        config_data = self.raw_config.copy()

        # Ensure output directory exists
        if not config_data.get("output_dir"):
            config_data["output_dir"] = self._generate_default_output_dir(model_type)

        # Handle SwanLab configuration
        config_data = self._handle_swanlab_config(config_data)

        # Use TrainingParametersManager for validation and conversion
        try:
            param_manager = TrainingParametersManager()
            param_manager.load_from_config(config_data)

            # Get official training parameters
            training_args_dict = param_manager.get_training_args_dict()

            logger.info(f"Training configuration built: {len(training_args_dict)} official parameters")
            return training_args_dict

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            if hasattr(e, 'errors'):
                for error in e.errors():
                    logger.error(f"Config error: {error.get('loc', [])} - {error.get('msg', '')} (input: {error.get('input', 'N/A')})")
            raise ValueError(f"Training configuration validation failed: {e}") from e

    def _generate_default_output_dir(self, model_type: str) -> str:
        """Generate default output directory."""
        root = str(settings.output_dir)

        model_name = self.raw_config.get("base_model_path", "unknown-model")
        model_name = model_name.replace("/", "-")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return f"{root.rstrip('/')}/training_{model_type}_{model_name}_{timestamp}"

    def _handle_swanlab_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SwanLab configuration to prevent initialization errors."""
        report_to = config.get('report_to', 'none')
        logger.debug(f"SwanLab config check - report_to: {report_to}")

        if isinstance(report_to, str) and 'swanlab' in report_to.lower():
            swanlab_api_key = self.raw_config.get('swanlab_api_key', '')
            if not swanlab_api_key:
                logger.warning("SwanLab configured but missing API key, disabling SwanLab")
                if report_to == 'swanlab':
                    config['report_to'] = 'none'
                else:
                    report_tools = [tool.strip() for tool in report_to.split(',')]
                    filtered_tools = [tool for tool in report_tools if 'swanlab' not in tool.lower()]
                    config['report_to'] = ','.join(filtered_tools) if filtered_tools else 'none'
                logger.info(f"SwanLab removed from report_to, current: {config['report_to']}")

        return config

    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return {
            'train_dataset_path': self.raw_config.get('train_dataset_path'),
            'dataset_configs': self.raw_config.get('dataset_configs'),
            'HF_subset': self.raw_config.get('HF_subset'),
            'train_sample_size': self.raw_config.get('train_sample_size', -1),
            'eval_sample_size': self.raw_config.get('eval_sample_size', -1),
            'test_sample_size': self.raw_config.get('test_sample_size', -1)
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model-related configuration."""
        return {
            'base_model_path': self.raw_config.get('base_model_path'),
            'model_type': self.raw_config.get('model_type', 'embedding').lower()
        }
