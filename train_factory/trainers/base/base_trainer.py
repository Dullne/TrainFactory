"""
Abstract base trainer for all training types.

Defines the common interface and shared functionality for all training types.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

from .training_result import TrainingResult
from train_factory.core.time_utils import now_naive
from ...core.device_manager import DeviceManager
from ...core.config_builder import ConfigBuilder
from ...data.data_loader import DataLoader
from ...schemas.training_config import TrainingParametersManager
from ...tuners.policy import canonicalize_tuner_config
from ...utils.training_metrics import extract_final_loss_metrics

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Defines the common training pipeline and abstract methods that
    must be implemented by specific trainer types.
    """

    def __init__(self, training_config: Dict[str, Any]):
        """
        Initialize base trainer.

        Args:
            training_config: Complete training configuration dictionary
        """
        self.raw_config = canonicalize_tuner_config(
            training_config,
            allow_registered_custom=True,
        )
        self.config_builder = ConfigBuilder(self.raw_config)
        self.device_manager = DeviceManager()
        self.data_loader = None

        # Extract common configurations
        self.model_config = self.config_builder.get_model_config()
        self.data_config = self.config_builder.get_data_config()
        self.model_type = self.model_config['model_type']

        logger.info(f"Initializing {self.model_type} trainer")

    def train(self, progress_callback=None) -> TrainingResult:
        """
        Execute the complete training pipeline.

        Args:
            progress_callback: Optional progress callback function

        Returns:
            TrainingResult containing model and metadata
        """
        logger.info(f"Starting {self.model_type} training")

        try:
            self._update_task_stage("preparing")

            # Step 1: Initialize components
            self._initialize_components()

            # Step 2: Load and prepare data
            datasets = self._load_datasets()
            datasets = self._prepare_datasets(datasets)
            self._record_dataset_stats(datasets)

            # Step 3: Initialize model and loss
            model, loss = self._initialize_model_and_loss(datasets['train'])

            # Optional: evaluate test dataset before training
            output_dir = self.raw_config.get('output_dir', './output')
            os.makedirs(output_dir, exist_ok=True)
            test_dataset = datasets.get('test')
            test_metrics_before = self._evaluate_test(
                model=model,
                test_dataset=test_dataset,
                output_dir=output_dir,
                stage="before"
            )

            # Step 4: Create training configuration
            training_args = self._create_training_args()

            # Step 5: Create evaluator
            evaluator = self.create_evaluator(datasets.get('eval'))

            # Step 6: Create trainer instance
            trainer = self.create_trainer_instance(
                model=model,
                args=training_args,
                train_dataset=datasets['train'],
                eval_dataset=datasets.get('eval'),
                loss=loss,
                evaluator=evaluator
            )

            # Step 7: Execute training
            result = self._execute_training(trainer, model, progress_callback)

            # Optional: evaluate test dataset after training
            if test_dataset is not None:
                self._update_task_stage("evaluating")
            test_metrics_after = self._evaluate_test(
                model=model,
                test_dataset=test_dataset,
                output_dir=output_dir,
                stage="after"
            )

            # Merge test metrics into final_metrics
            result.final_metrics = self._merge_test_metrics(
                result.final_metrics,
                test_metrics_before,
                test_metrics_after
            )

            logger.info(f"{self.model_type} training completed")
            return result

        except Exception as e:
            logger.error(f"{self.model_type} training failed: {e}")
            raise

    def _update_task_stage(self, status: str, error_message: Optional[str] = None) -> None:
        """Best-effort task stage update for DB-backed training tasks."""
        task_id = self.raw_config.get("task_id")
        if not task_id:
            return

        try:
            from ...storage.services.training_task_service import training_task_service

            training_task_service.update_task_status(
                task_id,
                status,
                error_message,
                run_token=self.raw_config.get("_run_token"),
            )
        except Exception as e:
            logger.warning(f"Failed to update task {task_id} stage to {status}: {e}")

    def _initialize_components(self):
        """Initialize data loader."""
        self.data_loader = DataLoader(
            hf_subset=self.data_config.get('HF_subset'),
            train_sample_size=self.data_config.get('train_sample_size', -1),
            eval_sample_size=self.data_config.get('eval_sample_size', -1),
            test_sample_size=self.data_config.get('test_sample_size', -1)
        )

    def _prepare_datasets(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclasses to transform loaded datasets before training.

        Default implementation returns datasets unchanged.
        """
        return datasets

    def _record_dataset_stats(self, datasets: Dict[str, Any]) -> None:
        """Record actual sample counts per split back into training_params.dataset_configs."""
        task_id = self.raw_config.get('task_id')
        if not task_id:
            return
        run_token = self.raw_config.get("_run_token")

        # Count samples per split
        split_counts: Dict[str, int] = {}
        for split_name in ('train', 'eval', 'test'):
            ds = datasets.get(split_name)
            if ds is not None:
                try:
                    split_counts[split_name] = len(ds)
                except Exception:
                    pass

        if not split_counts:
            return

        try:
            from ...storage.database import get_session
            from ...storage.entities.training_task_entity import TrainingTaskDB
            from sqlalchemy import update
            from sqlmodel import select

            with get_session() as session:
                conditions = [TrainingTaskDB.task_id == task_id]
                if run_token is not None:
                    conditions.extend(
                        [
                            TrainingTaskDB.run_token == run_token,
                            TrainingTaskDB.status.in_(
                                ("preparing", "running", "evaluating")
                            ),
                        ]
                    )
                task = session.exec(select(TrainingTaskDB).where(*conditions)).first()
                if not task:
                    return

                params = dict(task.training_params or {})
                configs = list(params.get('dataset_configs') or [])

                # Enrich each config with num_rows
                for cfg in configs:
                    split = cfg.get('split', 'train')
                    if split in split_counts:
                        cfg['num_rows'] = split_counts[split]

                params['dataset_configs'] = configs
                result = session.exec(
                    update(TrainingTaskDB)
                    .where(*conditions)
                    .values(training_params=params, updated_at=now_naive())
                )
                session.commit()
                if result.rowcount != 1:
                    return

            logger.info(f"Recorded dataset stats: {split_counts}")
        except Exception as e:
            logger.warning(f"Failed to record dataset stats: {e}")

    def _load_datasets(self) -> Dict[str, Any]:
        """Load and return datasets."""
        dataset_configs = self.data_config.get('dataset_configs')
        if dataset_configs:
            datasets = self._load_datasets_from_configs(dataset_configs)
            if datasets.get('train') is None:
                raise ValueError("Failed to load training dataset from dataset_configs (split='train')")
            return datasets

        dataset_path = self.data_config['train_dataset_path']
        logger.info(f"Loading dataset: {dataset_path}")

        train_dataset, eval_dataset, test_dataset = self.data_loader.load_all_splits(dataset_path)

        if train_dataset is None:
            raise ValueError(f"Failed to load training dataset: {dataset_path}")

        return {
            'train': train_dataset,
            'eval': eval_dataset,
            'test': test_dataset
        }

    def _load_datasets_from_configs(self, dataset_configs: list) -> Dict[str, Any]:
        """Load datasets from dataset_configs with split awareness."""
        def normalize_split(name: Optional[str]) -> str:
            split = (name or "train").lower()
            if split in ("val", "validation", "dev"):
                return "eval"
            if split in ("train", "training"):
                return "train"
            return split

        def apply_max_samples(dataset: Any, max_samples: Optional[int]) -> Any:
            if not max_samples or max_samples <= 0 or dataset is None:
                return dataset
            try:
                if isinstance(dataset, dict):
                    return {k: apply_max_samples(v, max_samples) for k, v in dataset.items()}
                if hasattr(dataset, "select"):
                    limit = min(max_samples, len(dataset))
                    return dataset.select(range(limit))
            except Exception:
                logger.warning("Failed to apply max_samples, using full dataset")
            return dataset

        def merge_datasets(items: list) -> Optional[Any]:
            if not items:
                return None
            flattened = []
            for item in items:
                if isinstance(item, dict):
                    flattened.extend(item.values())
                else:
                    flattened.append(item)
            if len(flattened) == 1:
                return flattened[0]
            try:
                from datasets import concatenate_datasets
                return concatenate_datasets(flattened)
            except Exception:
                try:
                    from torch.utils.data import ConcatDataset
                    return ConcatDataset(flattened)
                except Exception as e:
                    logger.warning(f"Failed to merge datasets: {e}")
                    return flattened[0]

        split_map = {"train": [], "eval": [], "test": []}

        for cfg in dataset_configs:
            path = cfg.get("path")
            if not path:
                continue
            split = normalize_split(cfg.get("split"))
            if split not in split_map:
                logger.warning(f"Unknown dataset split '{split}', skipping")
                continue
            dataset = self.data_loader.load_data(split, path)
            dataset = apply_max_samples(dataset, cfg.get("max_samples"))
            if dataset is None:
                logger.warning(f"Failed to load {split} dataset from {path}")
                continue
            split_map[split].append(dataset)

        return {
            "train": merge_datasets(split_map["train"]),
            "eval": merge_datasets(split_map["eval"]),
            "test": merge_datasets(split_map["test"]),
        }

    def _evaluate_test(self, model: Any, test_dataset: Any, output_dir: str, stage: str) -> Optional[Dict[str, Any]]:
        """Evaluate test dataset before/after training (encoder/reranker)."""
        if test_dataset is None:
            return None
        try:
            evaluator = self.create_evaluator(test_dataset)
            if evaluator is None:
                return None
            was_training = getattr(model, "training", False)
            model.eval()
            metrics = evaluator(
                model,
                output_path=os.path.join(output_dir, f"test_{stage}"),
                epoch=0,
                steps=0,
            )
            if was_training:
                model.train()
            if isinstance(metrics, dict):
                return metrics
            return {"score": metrics}
        except Exception as e:
            logger.warning(f"Test evaluation ({stage}) failed: {e}")
            return None

    def _merge_test_metrics(
        self,
        final_metrics: Optional[Dict[str, Any]],
        test_before: Optional[Dict[str, Any]],
        test_after: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Merge test metrics into final_metrics and compute deltas."""
        if not test_before and not test_after:
            return final_metrics

        merged: Dict[str, Any] = dict(final_metrics or {})
        if test_before is not None:
            merged["test_before"] = test_before
        if test_after is not None:
            merged["test_after"] = test_after

        if test_before and test_after:
            delta = {}
            for key, before_val in test_before.items():
                after_val = test_after.get(key)
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    delta[key] = after_val - before_val
            if delta:
                merged["test_delta"] = delta

        return merged

    def _initialize_model_and_loss(self, train_dataset) -> Tuple[Any, Any]:
        """Initialize model and loss function."""
        model_name = self.model_config['base_model_path']

        # Initialize model
        model = self.initialize_model(model_name)

        # Handle LoRA if configured
        # SentenceTransformer handles LoRA via add_adapter(); applying
        # get_peft_model() there wraps it into PeftModel and breaks
        # SentenceTransformerTrainer. CrossEncoder does not share this limit.
        try:
            from sentence_transformers import SentenceTransformer as _ST
            is_sentence_transformer = isinstance(model, _ST)
        except ImportError:
            is_sentence_transformer = False

        tuner_type = self.raw_config.get("tuner_type")
        if tuner_type not in {"lora", "full"}:
            tuner_config = dict(self.raw_config.get("tuner_config") or {})
            tuner_config["tuner_type"] = tuner_type
            model = self._apply_tuner(model, tuner_config)
        elif not is_sentence_transformer:
            param_manager = TrainingParametersManager()
            param_manager.load_from_config(self.raw_config)
            custom_params = param_manager.get_custom_params_dict()
            lora_config = custom_params.get('lora_config', {})

            if lora_config.get('use_lora', False):
                logger.info("LoRA enabled, adding LoRA adapter")
                model = self._add_lora_adapter(model, lora_config)

        # Prepare model for training
        user_device = self.raw_config.get('device')
        device = self.device_manager.get_training_device(user_device)
        model = self.device_manager.prepare_model_for_training(model, device)

        # Create loss function
        loss = self.create_loss_function(model, train_dataset)

        return model, loss

    def _apply_tuner(self, model, tuner_config: Dict[str, Any]):
        """
        Apply tuner (LoRA, QLoRA, etc.) to model using TunerRegistry.

        Args:
            model: Model to apply tuner to
            tuner_config: Tuner configuration dict

        Returns:
            Model with tuner applied
        """
        from ...tuners import TunerRegistry

        tuner_type = tuner_config.get('tuner_type', 'lora')
        tuner = TunerRegistry.create(tuner_type, tuner_config)
        model = tuner.prepare_model(model)
        logger.info(f"{tuner_type.upper()} tuner applied successfully")
        return model

    def _add_lora_adapter(self, model, lora_config: Dict[str, Any]):
        """Add LoRA adapter to model (uses new tuner system)."""
        return self._apply_tuner(model, lora_config)

    def _add_lora_adapter_legacy(self, model, lora_config: Dict[str, Any]):
        """Legacy LoRA adapter method (fallback)."""
        try:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=lora_config.get('r', 16),
                lora_alpha=lora_config.get('lora_alpha', 32),
                lora_dropout=lora_config.get('lora_dropout', 0.0),
                target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
                bias=lora_config.get('bias', 'none'),
            )

            model = get_peft_model(model, peft_config)
            logger.info("LoRA adapter added successfully (legacy)")
            return model

        except ImportError as exc:
            raise RuntimeError("PEFT is required for LoRA training") from exc

    def _check_gpu_bf16_compatibility(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check GPU bf16 compatibility and adjust config if needed."""
        config_copy = config.copy()

        if config_copy.get('bf16', False):
            if not self.device_manager.check_gpu_bf16_support():
                logger.warning("GPU doesn't support bf16, falling back to fp16")
                config_copy['bf16'] = False
                config_copy['fp16'] = True

        return config_copy

    def _create_training_args(self) -> Any:
        """Create training arguments."""
        config = self.config_builder.build_training_config(self.model_type)

        # Check GPU compatibility
        config = self._check_gpu_bf16_compatibility(config)

        return self.create_training_args(config)

    def _execute_training(self, trainer, model, progress_callback) -> TrainingResult:
        """Execute training and return result."""
        logger.info("Executing training loop")

        # Get task_id and output_dir from config
        task_id = self.raw_config.get('task_id')
        run_token = self.raw_config.get("_run_token")
        output_dir = self.raw_config.get('output_dir', './output')

        # Add MetricsCallback if task_id is available
        if task_id:
            try:
                from ...monitoring.metrics_callback import MetricsCallback
                from ...storage.services.training_task_service import training_task_service

                def db_update_callback(metrics: dict):
                    """Update database with current training metrics."""
                    try:
                        training_task_service.update_task_metrics(
                            task_id,
                            metrics,
                            run_token=run_token,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update task metrics in DB: {e}")

                metrics_callback = MetricsCallback(
                    output_dir=output_dir,
                    task_id=task_id,
                    progress_callback=progress_callback,
                    db_update_callback=db_update_callback,
                )
                trainer.add_callback(metrics_callback)
                logger.info(f"Added MetricsCallback for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to add MetricsCallback: {e}")

        # Run training (支持断点续传)
        self._update_task_stage("running")
        resume_from = self.raw_config.get('resume_from_checkpoint')
        if resume_from:
            logger.info(f"Resuming training from checkpoint: {resume_from}")
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            trainer.train()

        # Get final metrics from trainer state if available. The last log_history
        # entry is often a train_runtime summary without loss, so scan backwards
        # for the most recent entry that actually carries loss/eval_loss.
        log_history = getattr(getattr(trainer, 'state', None), 'log_history', None) or []
        final_metrics = extract_final_loss_metrics(log_history)

        # Save model
        save_dir = f"{output_dir}/final_model"
        self._save_model(model, save_dir)
        logger.info(f"Model saved to: {save_dir}")

        return TrainingResult(
            model=model,
            save_dir=save_dir,
            final_metrics=final_metrics
        )

    def _save_model(self, model, save_dir: str):
        """Save model to directory. Can be overridden by subclasses."""
        model.save(save_dir)

    @abstractmethod
    def initialize_model(self, model_name: str) -> Any:
        """Initialize the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_loss_function(self, model: Any, train_dataset: Any) -> Any:
        """Create loss function. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_training_args(self, config: Dict[str, Any]) -> Any:
        """Create training arguments. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_trainer_instance(self, model, args, train_dataset, eval_dataset, loss, evaluator) -> Any:
        """Create trainer instance. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def create_evaluator(self, eval_dataset: Any) -> Optional[Any]:
        """Create evaluator. Must be implemented by subclasses."""
        pass
