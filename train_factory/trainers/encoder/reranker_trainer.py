"""
Reranker model trainer.

Specialized trainer for reranker models using CrossEncoder.
"""

import logging
import os
from typing import Any, Dict, Optional

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments

from ..base.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

_MNR_LOSS_NAMES = {
    "MultipleNegativesRankingLoss",
    "CachedMultipleNegativesRankingLoss",
}


class RerankerTrainer(BaseTrainer):
    """
    Specialized trainer for reranker models.

    Handles CrossEncoder model training with appropriate loss functions
    and training arguments.
    """

    def __init__(self, training_config: Dict[str, Any]):
        super().__init__(training_config)

    def _prepare_datasets(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten universal triplet data when CrossEncoder MNR losses are used.

        CrossEncoder MultipleNegativesRankingLoss expects scalar text columns in
        the order ``query, positive, negative_0, ...``. Our reranker datasets
        commonly arrive as ``query, positives: [...], negatives: [...]``.
        """
        if not self._uses_cross_encoder_mnr():
            return datasets

        prepared = dict(datasets)
        for split_name in ("train", "eval", "test"):
            dataset = prepared.get(split_name)
            if dataset is None:
                continue
            prepared[split_name] = self._prepare_mnr_dataset(dataset, split_name)

        return prepared

    def initialize_model(self, model_name: str) -> CrossEncoder:
        """
        Initialize CrossEncoder model.

        Args:
            model_name: Name or path of the model

        Returns:
            Initialized CrossEncoder model
        """
        logger.info(f"Initializing CrossEncoder model: {model_name}")

        try:
            # Try ModelScope download first
            try:
                from modelscope import snapshot_download
                from ...config import settings

                logger.info(f"Trying ModelScope download: {model_name}")
                model_dir = snapshot_download(model_name, cache_dir=str(settings.training_cache))
                model = CrossEncoder(model_dir)
                logger.info(f"ModelScope download successful: {model_dir}")
                return model

            except ImportError:
                logger.info("ModelScope not installed, using HuggingFace")
                model = CrossEncoder(model_name)
                logger.info(f"HuggingFace download successful: {model_name}")
                return model

            except Exception as e:
                if "CUDA" in str(e) or "cuda" in str(e):
                    error_msg = f"CUDA initialization failed: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                logger.warning(f"ModelScope download failed: {e}, falling back to HuggingFace")
                model = CrossEncoder(model_name)
                logger.info(f"HuggingFace download successful: {model_name}")
                return model

        except Exception as e:
            if "couldn't connect" in str(e).lower() or "connection" in str(e).lower():
                logger.warning(f"Network connection failed, trying local cache: {model_name}")
                try:
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    model = CrossEncoder(model_name)
                    logger.info(f"Local cache load successful: {model_name}")
                    return model
                except Exception:
                    error_msg = f"Reranker model initialization failed: {model_name}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"Reranker model initialization failed: {model_name}, error: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

    def create_loss_function(self, model: CrossEncoder, train_dataset: Any) -> Any:
        """
        Create loss function for reranker training.

        Args:
            model: CrossEncoder model
            train_dataset: Training dataset

        Returns:
            Loss function or None (uses default if not specified)
        """
        # Get loss config from raw_config
        loss_name = self.raw_config.get('reranker_loss_name', 'auto')
        loss_config = self.raw_config.get('loss_config') or {}

        if loss_name == 'auto' or loss_name is None:
            logger.info("Reranker uses default internal loss function")
            return None

        logger.info(f"Creating reranker loss function: {loss_name}")

        if (
            loss_name in _MNR_LOSS_NAMES
            and train_dataset is not None
            and not self._is_mnr_dataset_compatible(train_dataset)
        ):
            logger.warning(
                "Reranker loss %s requires scalar text columns. The current dataset is still "
                "incompatible after preprocessing, so TrainFactory will fall back to the "
                "default CrossEncoder loss.",
                loss_name,
            )
            return None

        try:
            from sentence_transformers.cross_encoder import losses as ce_losses
            builder_map = {
                "CrossEntropyLoss": lambda: ce_losses.CrossEntropyLoss(model),
                # Keep the old config value for compatibility with existing tasks/UI.
                "BCEWithLogitsLoss": lambda: ce_losses.BinaryCrossEntropyLoss(
                    model,
                    pos_weight=loss_config.get("pos_weight"),
                ),
                "BinaryCrossEntropyLoss": lambda: ce_losses.BinaryCrossEntropyLoss(
                    model,
                    pos_weight=loss_config.get("pos_weight"),
                ),
                "MSELoss": lambda: ce_losses.MSELoss(model),
                "MarginMSELoss": lambda: ce_losses.MarginMSELoss(model),
                "MultipleNegativesRankingLoss": lambda: ce_losses.MultipleNegativesRankingLoss(
                    model,
                    num_negatives=loss_config.get("num_negatives", 4),
                    scale=loss_config.get("scale", 10.0),
                ),
                "CachedMultipleNegativesRankingLoss": lambda: ce_losses.CachedMultipleNegativesRankingLoss(
                    model,
                    num_negatives=loss_config.get("num_negatives", 4),
                    scale=loss_config.get("scale", 10.0),
                    mini_batch_size=loss_config.get("mini_batch_size", 32),
                    show_progress_bar=loss_config.get("show_progress_bar", False),
                ),
                "RankNetLoss": lambda: ce_losses.RankNetLoss(
                    model,
                    k=loss_config.get("k"),
                    sigma=loss_config.get("sigma", 1.0),
                    mini_batch_size=loss_config.get("mini_batch_size"),
                ),
                "LambdaLoss": lambda: ce_losses.LambdaLoss(
                    model,
                    k=loss_config.get("k"),
                    sigma=loss_config.get("sigma", 1.0),
                    mini_batch_size=loss_config.get("mini_batch_size"),
                ),
                "ListMLELoss": lambda: ce_losses.ListMLELoss(
                    model,
                    mini_batch_size=loss_config.get("mini_batch_size"),
                    respect_input_order=loss_config.get("respect_input_order", True),
                ),
                "ListNetLoss": lambda: ce_losses.ListNetLoss(
                    model,
                    mini_batch_size=loss_config.get("mini_batch_size"),
                ),
                "PListMLELoss": lambda: ce_losses.PListMLELoss(
                    model,
                    mini_batch_size=loss_config.get("mini_batch_size"),
                    respect_input_order=loss_config.get("respect_input_order", True),
                ),
            }

            if loss_name in builder_map:
                return builder_map[loss_name]()

            unsupported_losses = {
                "BCELoss",
                "L1Loss",
                "SmoothL1Loss",
                "HuberLoss",
                "MarginRankingLoss",
                "CosineEmbeddingLoss",
                "PairwiseHingeLoss",
                "ListwiseLoss",
            }
            if loss_name in unsupported_losses:
                logger.warning(
                    "Reranker loss %s is not supported by CrossEncoderTrainer in the current "
                    "sentence-transformers integration; using default loss instead.",
                    loss_name,
                )
                return None

            logger.warning("Unknown reranker loss: %s, using default", loss_name)
            return None

        except Exception as e:
            logger.error(f"Failed to create reranker loss function: {e}")
            return None

    def _uses_cross_encoder_mnr(self) -> bool:
        return self.raw_config.get("reranker_loss_name") in _MNR_LOSS_NAMES

    def _prepare_mnr_dataset(self, dataset: Any, split_name: str) -> Any:
        if isinstance(dataset, dict):
            dataset = next(iter(dataset.values()))

        columns = dataset.column_names if hasattr(dataset, "column_names") else []
        if not columns:
            return dataset

        if self._is_mnr_dataset_compatible(dataset):
            logger.info(
                "Reranker %s split already compatible with CrossEncoder MNR: columns=%s",
                split_name,
                columns,
            )
            return dataset

        if "positives" not in columns or "negatives" not in columns:
            logger.warning(
                "Reranker %s split is not in universal triplet format, cannot auto-flatten "
                "for CrossEncoder MNR: columns=%s",
                split_name,
                columns,
            )
            return dataset

        anchor_col = "query" if "query" in columns else columns[0]
        target_negatives = int((self.raw_config.get("loss_config") or {}).get("num_negatives", 0) or 0)

        anchors: list[str] = []
        positives: list[str] = []
        normalized_negatives: list[list[str]] = []

        for row in dataset:
            anchor = self._normalize_text(row.get(anchor_col))
            positive = self._normalize_text(self._pick_first_text(row.get("positives")))
            negatives = self._normalize_text_list(row.get("negatives"))

            if not anchor or not positive:
                continue

            anchors.append(anchor)
            positives.append(positive)
            normalized_negatives.append(negatives)

        if not anchors:
            logger.warning(
                "Reranker %s split has no valid rows after MNR flattening, keeping original dataset",
                split_name,
            )
            return dataset

        negative_counts = [len(items) for items in normalized_negatives]
        min_neg = min(negative_counts) if negative_counts else 0
        max_neg = max(negative_counts) if negative_counts else 0
        shared_neg = min_neg
        if target_negatives > 0:
            shared_neg = min(shared_neg, target_negatives) if shared_neg > 0 else 0

        flat_data: Dict[str, list[str]] = {
            "query": anchors,
            "positive": positives,
        }
        for idx in range(shared_neg):
            flat_data[f"negative_{idx}"] = [negatives[idx] for negatives in normalized_negatives]

        from datasets import Dataset as HFDataset

        flat_dataset = HFDataset.from_dict(flat_data)
        logger.info(
            "Flattened reranker %s split for CrossEncoder MNR: rows=%s, min_neg=%s, "
            "max_neg=%s, hard_negatives=%s, columns=%s",
            split_name,
            len(flat_dataset),
            min_neg,
            max_neg,
            shared_neg,
            flat_dataset.column_names,
        )
        return flat_dataset

    def _is_mnr_dataset_compatible(self, dataset: Any) -> bool:
        if dataset is None:
            return False
        if isinstance(dataset, dict):
            dataset = next(iter(dataset.values()))

        columns = dataset.column_names if hasattr(dataset, "column_names") else []
        if not columns or len(dataset) == 0:
            return False

        sample = dataset[0]
        feature_columns = [
            column
            for column in columns
            if column not in {"label", "labels", "score", "scores", "dataset_name"}
            and not column.endswith("_prompt_length")
        ]
        if len(feature_columns) < 2:
            return False

        for column in feature_columns:
            value = sample[column]
            if isinstance(value, (list, tuple, dict, set)):
                return False
            if value is None:
                return False

        return True

    @staticmethod
    def _pick_first_text(value: Any) -> Any:
        if isinstance(value, list):
            for item in value:
                if item:
                    return item
            return None
        return value

    def _normalize_text_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            normalized = self._normalize_text(value)
            return [normalized] if normalized else []

        result = []
        for item in value:
            normalized = self._normalize_text(item)
            if normalized:
                result.append(normalized)
        return result

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            for key in ("content", "response", "text", "value", "output", "answer"):
                if value.get(key):
                    return RerankerTrainer._normalize_text(value[key])
            return ""
        return str(value).strip()

    def create_training_args(self, config: Dict[str, Any]) -> CrossEncoderTrainingArguments:
        """
        Create CrossEncoderTrainingArguments.

        Args:
            config: Training configuration dictionary

        Returns:
            CrossEncoderTrainingArguments instance
        """
        logger.info("Creating CrossEncoderTrainingArguments")

        try:
            args = CrossEncoderTrainingArguments(**config)
            logger.info("CrossEncoderTrainingArguments created successfully")
            return args
        except Exception as e:
            logger.error(f"CrossEncoderTrainingArguments creation failed: {e}")
            raise

    def create_trainer_instance(self, model, args, train_dataset, eval_dataset, loss, evaluator) -> CrossEncoderTrainer:
        """
        Create CrossEncoderTrainer instance.

        Args:
            model: CrossEncoder model
            args: CrossEncoderTrainingArguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            loss: Loss function (CrossEncoder-compatible losses only)
            evaluator: Evaluator (optional)

        Returns:
            CrossEncoderTrainer instance
        """
        logger.info("Creating CrossEncoderTrainer")

        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
        }

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        if evaluator is not None:
            trainer_kwargs["evaluator"] = evaluator

        if loss is not None:
            trainer_kwargs["loss"] = loss

        try:
            trainer = CrossEncoderTrainer(**trainer_kwargs)
            logger.info("CrossEncoderTrainer created successfully")
            return trainer
        except Exception as e:
            logger.error(f"CrossEncoderTrainer creation failed: {e}")
            raise

    def create_evaluator(self, eval_dataset: Any) -> Optional[Any]:
        """
        Create evaluator for reranker model evaluation.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluator instance or None
        """
        if eval_dataset is None:
            logger.info("No eval dataset, skipping evaluator creation")
            return None

        logger.info("Creating reranker evaluator")

        try:
            from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

            # Format data for evaluator
            if isinstance(eval_dataset, dict):
                eval_dataset = next(iter(eval_dataset.values()))

            columns = eval_dataset.column_names
            if len(columns) >= 3:
                # Convert to expected format
                samples = []
                for item in eval_dataset:
                    samples.append({
                        'query': item[columns[0]],
                        'positive': [item[columns[1]]],
                        'negative': []
                    })

                evaluator = CERerankingEvaluator(
                    samples=samples,
                    name="reranker_eval"
                )

                logger.info("Reranker evaluator created successfully")
                return evaluator

            logger.warning("Eval dataset format not suitable for evaluator")
            return None

        except Exception as e:
            logger.warning(f"Evaluator creation failed: {e}, skipping evaluation")
            return None
