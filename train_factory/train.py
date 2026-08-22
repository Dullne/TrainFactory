"""
Main training entry point for TrainFactory.

Provides a unified interface for training embedding and reranker models.
"""

import logging
import multiprocessing
from typing import Dict, Any, Optional, Callable

from dotenv import load_dotenv

from .config import settings
from .utils.common_utils import init_swanlab, setup_logging
from .trainers.base.training_result import TrainingResult
from .tuners.policy import canonicalize_tuner_config

logger = logging.getLogger(__name__)

# Set multiprocessing start method to 'spawn' to avoid CUDA issues in forked processes
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn')
        logger.info("Set multiprocessing start_method='spawn'")
    except RuntimeError as e:
        logger.debug(f"multiprocessing start_method already set: {e}")


def train(
    model_type: str = "embedding",
    base_model_path: str = None,
    train_dataset_path: str = None,
    output_dir: Optional[str] = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    learning_rate: float = 2e-5,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> TrainingResult:
    """
    Train a model with the specified configuration.

    Args:
        model_type: Model type ('embedding', 'reranker', 'decoder_reranker', 'llm')
        base_model_path: Base model name or path (HuggingFace/ModelScope or local)
        train_dataset_path: Train dataset path
        output_dir: Output directory for training results
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate
        progress_callback: Optional progress callback function
        **kwargs: Additional training parameters

    Returns:
        TrainingResult containing trained model and save directory

    Example:
        >>> from train_factory import train
        >>> result = train(
        ...     model_type="embedding",
        ...     base_model_path="BAAI/bge-base-zh-v1.5",
        ...     train_dataset_path="./data/train.jsonl",
        ...     num_train_epochs=3
        ... )
        >>> print(f"Model saved to: {result.save_dir}")
    """
    # Build training config
    training_config = {
        'model_type': model_type,
        'base_model_path': base_model_path,
        'train_dataset_path': train_dataset_path,
        'dataset_configs': [{'path': train_dataset_path, 'max_samples': None, 'split': 'train'}] if train_dataset_path else None,
        'output_dir': output_dir or str(settings.output_dir),
        'num_train_epochs': num_train_epochs,
        'per_device_train_batch_size': per_device_train_batch_size,
        'learning_rate': learning_rate,
        **kwargs
    }

    return train_with_config(training_config, progress_callback)


def train_with_config(
    training_config: Dict[str, Any],
    progress_callback: Optional[Callable] = None
) -> TrainingResult:
    """
    Train with a complete configuration dictionary.

    Args:
        training_config: Complete training configuration
        progress_callback: Optional progress callback function

    Returns:
        TrainingResult containing trained model and save directory
    """
    logger.info("Starting unified training - TrainFactory")

    # Load environment variables
    load_dotenv()

    training_config = canonicalize_tuner_config(
        training_config,
        allow_registered_custom=True,
    )

    # Validate required parameters
    if not training_config.get('base_model_path'):
        raise ValueError("base_model_path is required")
    if not training_config.get('train_dataset_path') and not training_config.get('dataset_configs'):
        raise ValueError("train_dataset_path or dataset_configs is required")

    model_type = training_config.get('model_type', 'embedding').lower()
    supported_model_types = ['embedding', 'reranker', 'decoder_reranker', 'llm']
    if model_type not in supported_model_types:
        raise ValueError(f"Unsupported model_type: {model_type}")

    task_id = training_config.get('task_id')

    try:
        # Initialize SwanLab if configured
        if training_config.get('swanlab_api_key'):
            logger.info("Initializing SwanLab")
            try:
                init_swanlab(training_config)
            except Exception as e:
                logger.warning(f"SwanLab initialization failed, continuing: {e}")

        # Import appropriate trainer
        if model_type == 'decoder_reranker':
            from .trainers.decoder import DecoderRerankerTrainer
            trainer = DecoderRerankerTrainer(training_config)
            result = trainer.train(progress_callback)
        elif model_type == 'llm':
            from .trainers.decoder import LLMTrainer
            trainer = LLMTrainer(training_config)
            result = trainer.train(progress_callback)
        elif model_type == 'embedding':
            from .trainers.encoder.embedding_trainer import EmbeddingTrainer
            trainer = EmbeddingTrainer(training_config)
            result = trainer.train(progress_callback)
        else:
            from .trainers.encoder.reranker_trainer import RerankerTrainer
            trainer = RerankerTrainer(training_config)
            result = trainer.train(progress_callback)

        logger.info(f"Training completed! Model saved to: {result.save_dir}")

        return TrainingResult(
            model=result.model,
            save_dir=result.save_dir,
            final_metrics=result.final_metrics,
            task_id=task_id
        )

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg, exc_info=True)

        # Update task status if task_id is available
        if task_id:
            try:
                from .storage.services.training_task_service import training_task_service
                training_task_service.update_task_status(
                    task_id,
                    "failed",
                    error_msg,
                    run_token=training_config.get("_run_token"),
                )
            except Exception as update_error:
                logger.error(f"Failed to update task status: {update_error}")

        raise RuntimeError(error_msg) from e


def get_supported_model_types() -> list:
    """Get list of supported model types."""
    return ['embedding', 'reranker', 'decoder_reranker', 'llm']


def create_default_config(
    model_type: str,
    base_model_path: str,
    train_dataset_path: str
) -> Dict[str, Any]:
    """
    Create default training configuration.

    Args:
        model_type: Model type ('embedding', 'reranker', 'decoder_reranker', 'llm')
        base_model_path: Model name or path
        train_dataset_path: Dataset path

    Returns:
        Default configuration dictionary
    """
    if model_type not in get_supported_model_types():
        raise ValueError(f"Unsupported model type: {model_type}")

    return {
        'model_type': model_type,
        'base_model_path': base_model_path,
        'train_dataset_path': train_dataset_path,
        'dataset_configs': [{'path': train_dataset_path, 'max_samples': None, 'split': 'train'}],
        'num_train_epochs': 3,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'logging_steps': 10,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'bf16': False,
        'fp16': False,
    }


if __name__ == "__main__":
    # Example usage
    setup_logging("INFO")

    example_config = {
        'model_type': 'embedding',
        'base_model_path': 'BAAI/bge-small-zh-v1.5',
        'train_dataset_path': 'sentence-transformers/all-nli',
        'dataset_configs': [{'path': 'sentence-transformers/all-nli', 'max_samples': None, 'split': 'train'}],
        'output_dir': './output/test',
        'num_train_epochs': 1,
        'per_device_train_batch_size': 8,
        'learning_rate': 2e-5,
    }

    try:
        result = train_with_config(example_config)
        print(f"Training completed! Model saved to: {result.save_dir}")
    except Exception as e:
        print(f"Training failed: {e}")
