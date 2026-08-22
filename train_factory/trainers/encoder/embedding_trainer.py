"""
Embedding model trainer.

Specialized trainer for sentence embedding models using SentenceTransformers.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
try:
    from sentence_transformers.sentence_transformer import losses, modules as models
except ImportError:  # pragma: no cover - compatibility with sentence-transformers 3.x
    from sentence_transformers import losses, models
from transformers import AutoConfig

from ...config import settings
from ...data.formats.embedding import normalize_embedding_negatives
from ..base.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class EmbeddingTrainer(BaseTrainer):
    """
    Specialized trainer for embedding models.

    Handles SentenceTransformer model training with appropriate loss functions
    and training arguments.
    """

    def __init__(self, training_config: Dict[str, Any]):
        super().__init__(training_config)

    def initialize_model(self, model_name: str) -> SentenceTransformer:
        """
        Initialize SentenceTransformer model.

        Args:
            model_name: Name or path of the model

        Returns:
            Initialized SentenceTransformer model
        """
        # Model dtype is selected during loading, so resolve precision support
        # before constructing the SentenceTransformer rather than only when
        # training arguments are created later in the pipeline.
        self.raw_config.update(
            self._check_gpu_bf16_compatibility(self.raw_config)
        )

        logger.info(f"Initializing SentenceTransformer model: {model_name}")

        try:
            if os.path.isdir(model_name):
                logger.info(f"Loading SentenceTransformer from local directory: {model_name}")
                model = self._load_sentence_transformer(model_name)
                logger.info(f"Local model load successful: {model_name}")
            else:
                # Try ModelScope download first (if available)
                try:
                    from modelscope import snapshot_download

                    logger.info(f"Trying ModelScope download: {model_name}")
                    model_dir = snapshot_download(model_name, cache_dir=str(settings.training_cache))
                    model = self._load_sentence_transformer(model_dir)
                    logger.info(f"ModelScope download successful: {model_dir}")

                except ImportError:
                    logger.info("ModelScope not installed, using HuggingFace")
                    model = self._load_sentence_transformer(model_name)
                    logger.info(f"HuggingFace download successful: {model_name}")

                except Exception as e:
                    # Check if CUDA error
                    if "CUDA" in str(e) or "cuda" in str(e):
                        error_msg = f"CUDA initialization failed: {e}"
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)

                    # Fall back to HuggingFace
                    logger.warning(f"ModelScope download failed: {e}, falling back to HuggingFace")
                    model = self._load_sentence_transformer(model_name)
                    logger.info(f"HuggingFace download successful: {model_name}")

        except Exception as e:
            # Try offline mode as last resort
            if "couldn't connect" in str(e).lower() or "connection" in str(e).lower():
                logger.warning(f"Network connection failed, trying local cache: {model_name}")
                try:
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    model = self._load_sentence_transformer(model_name, local_files_only=True)
                    logger.info(f"Local cache load successful: {model_name}")
                except Exception:
                    error_msg = f"Embedding model initialization failed: {model_name}, network unavailable and local cache not found"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"Embedding model initialization failed: {model_name}, error: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        configured_max_seq_length = self.raw_config.get('max_seq_length')
        if configured_max_seq_length is None:
            # Backward compatibility for direct trainer callers that still use
            # the public API field name without route-level normalization.
            configured_max_seq_length = self.raw_config.get('max_length')
        if configured_max_seq_length is not None:
            model.max_seq_length = int(configured_max_seq_length)
            logger.info("Set SentenceTransformer max_seq_length=%s", configured_max_seq_length)

        # Apply LoRA via SentenceTransformer's native add_adapter() API.
        # This keeps the model as a SentenceTransformer (iterable) instead of
        # wrapping it in PeftModel which breaks SentenceTransformerTrainer.
        model = self._apply_lora_if_configured(model)
        return model

    def _load_sentence_transformer(self, model_name_or_path: str, **kwargs) -> SentenceTransformer:
        """Load a SentenceTransformer, with fallback for configs that nest hidden_size under text_config."""
        kwargs = self._prepare_sentence_transformer_load_kwargs(kwargs)
        if self._requires_text_config_hidden_size_fallback(model_name_or_path, **kwargs):
            logger.warning(
                "Model config stores hidden_size under text_config; using Transformer + mean Pooling fallback for %s",
                model_name_or_path,
            )
            return self._load_sentence_transformer_with_text_config_hidden_size(model_name_or_path, **kwargs)

        try:
            return SentenceTransformer(model_name_or_path, **kwargs)
        except AttributeError as e:
            if "hidden_size" not in str(e):
                raise
            logger.warning(
                "SentenceTransformer auto-load failed because config.hidden_size is missing; "
                "trying Transformer + mean Pooling fallback for %s",
                model_name_or_path,
            )
            return self._load_sentence_transformer_with_text_config_hidden_size(model_name_or_path, **kwargs)

    def _prepare_sentence_transformer_load_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge load options while enforcing the operator remote-code policy."""
        load_kwargs = dict(kwargs)
        configured_model_kwargs = self.raw_config.get("model_kwargs") or {}
        explicit_model_kwargs = load_kwargs.get("model_kwargs") or {}
        model_kwargs = {**configured_model_kwargs, **explicit_model_kwargs}

        # SentenceTransformers merges these dictionaries after its shared
        # arguments, so a nested request value could otherwise override the
        # trusted top-level policy.
        for nested_key in ("model_kwargs", "tokenizer_kwargs", "config_kwargs"):
            nested_kwargs = dict(load_kwargs.get(nested_key) or {})
            nested_kwargs.pop("trust_remote_code", None)
            if nested_key == "model_kwargs":
                model_kwargs.pop("trust_remote_code", None)
                nested_kwargs = model_kwargs
            if nested_kwargs:
                load_kwargs[nested_key] = nested_kwargs
            else:
                load_kwargs.pop(nested_key, None)
        load_kwargs["trust_remote_code"] = settings.allow_model_remote_code

        has_explicit_dtype = "dtype" in model_kwargs or "torch_dtype" in model_kwargs
        configured_device = str(self.raw_config.get("device") or "").lower()
        if not has_explicit_dtype and configured_device != "cpu":
            if self.raw_config.get("bf16", False):
                model_kwargs["dtype"] = torch.bfloat16
            elif self.raw_config.get("fp16", False):
                model_kwargs["dtype"] = torch.float16

        if model_kwargs:
            load_kwargs["model_kwargs"] = model_kwargs
        return load_kwargs

    def _requires_text_config_hidden_size_fallback(self, model_name_or_path: str, **kwargs) -> bool:
        """Return whether a plain HF model needs manual pooling because hidden_size is nested."""
        if os.path.isdir(model_name_or_path) and os.path.exists(os.path.join(model_name_or_path, "modules.json")):
            return False

        config_kwargs = {
            key: kwargs[key]
            for key in ("trust_remote_code", "local_files_only", "revision", "token")
            if key in kwargs and kwargs[key] is not None
        }
        try:
            config = AutoConfig.from_pretrained(model_name_or_path, **config_kwargs)
        except Exception as e:
            logger.debug("Could not inspect model config before SentenceTransformer load: %s", e)
            return False

        return (
            getattr(config, "hidden_size", None) is None
            and self._extract_hidden_size(config) is not None
        )

    def _load_sentence_transformer_with_text_config_hidden_size(
        self,
        model_name_or_path: str,
        **kwargs,
    ) -> SentenceTransformer:
        """Build a SentenceTransformer for configs whose hidden size is stored in text_config."""
        transformer_kwargs = {}
        for key in ("cache_folder", "trust_remote_code", "local_files_only", "revision", "token"):
            if key in kwargs and kwargs[key] is not None:
                transformer_kwargs[key] = kwargs[key]

        shared_config_args = {}
        for key in ("trust_remote_code", "local_files_only", "revision", "token"):
            if key in transformer_kwargs:
                shared_config_args[key] = transformer_kwargs.pop(key)

        model_args = {
            **shared_config_args,
            **(kwargs.get("model_kwargs") or {}),
        }
        transformer_model = models.Transformer(
            model_name_or_path,
            cache_dir=transformer_kwargs.get("cache_folder"),
            model_args=model_args,
            tokenizer_args=shared_config_args,
            config_args=shared_config_args,
        )

        hidden_size = self._extract_hidden_size(transformer_model.auto_model.config)
        if hidden_size is None:
            raise AttributeError(
                f"Cannot infer hidden_size from model config for {model_name_or_path}"
            )

        if getattr(transformer_model.auto_model.config, "hidden_size", None) is None:
            transformer_model.auto_model.config.hidden_size = hidden_size
            logger.info("Patched transformer config.hidden_size from nested text_config: %s", hidden_size)

        pooling_model = models.Pooling(hidden_size, pooling_mode="mean")
        return SentenceTransformer(modules=[transformer_model, pooling_model])

    @staticmethod
    def _extract_hidden_size(config: Any) -> Optional[int]:
        """Return hidden_size from common top-level or nested transformer config layouts."""
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)

        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            nested_hidden_size = getattr(text_config, "hidden_size", None)
            if nested_hidden_size is not None:
                return int(nested_hidden_size)

        if isinstance(config, dict):
            hidden_size = config.get("hidden_size")
            if hidden_size is not None:
                return int(hidden_size)
            text_config = config.get("text_config") or {}
            nested_hidden_size = text_config.get("hidden_size")
            if nested_hidden_size is not None:
                return int(nested_hidden_size)

        return None

    def _apply_lora_if_configured(self, model: SentenceTransformer) -> SentenceTransformer:
        """Apply LoRA adapter using SentenceTransformer's native PEFT integration."""
        lora_config = self.raw_config.get('lora_config') or {}
        if not lora_config.get('use_lora', False):
            # Also check top-level use_lora flag
            if not self.raw_config.get('use_lora', False):
                return model

        try:
            from peft import LoraConfig, TaskType

            peft_config = LoraConfig(
                r=lora_config.get('r', 16),
                lora_alpha=lora_config.get('lora_alpha', 32),
                lora_dropout=lora_config.get('lora_dropout', 0.0),
                target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
                bias=lora_config.get('bias', 'none'),
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model.add_adapter(peft_config)
            logger.info(
                f"LoRA adapter added via SentenceTransformer.add_adapter() "
                f"(r={peft_config.r}, alpha={peft_config.lora_alpha})"
            )
        except ImportError as exc:
            raise RuntimeError(
                "PEFT is required for LoRA embedding training"
            ) from exc
        except Exception as e:
            logger.error(f"Failed to add LoRA adapter to SentenceTransformer: {e}")
            raise

        return model

    def create_loss_function(self, model: SentenceTransformer, train_dataset: Any) -> Any:
        """
        Create loss function for embedding training.

        Args:
            model: SentenceTransformer model
            train_dataset: Training dataset

        Returns:
            Loss function
        """
        logger.info("Creating embedding loss function")

        # Get loss config from raw_config
        loss_name = self.raw_config.get('embedding_loss_name', 'auto')
        loss_config = self.raw_config.get('loss_config') or {}

        # Auto-detect loss based on dataset format
        if loss_name == 'auto':
            loss_name = self._auto_detect_loss(train_dataset)

        # Create loss function
        try:
            loss = self._create_loss_by_name(model, loss_name, loss_config)

            # Check if Matryoshka wrapper is enabled
            if self.raw_config.get('use_matryoshka', False):
                matryoshka_dims = self.raw_config.get('matryoshka_dims', [768, 512, 256, 128, 64])
                matryoshka_weights = self.raw_config.get('matryoshka_weights', None)
                loss = losses.MatryoshkaLoss(model, loss, matryoshka_dims, matryoshka_weights)
                logger.info(f"Wrapped with MatryoshkaLoss, dims: {matryoshka_dims}")

            logger.info(f"Created loss function: {loss_name}")
            return loss

        except Exception as e:
            logger.error(f"Failed to create loss function: {e}")
            raise

    def _create_loss_by_name(self, model: SentenceTransformer, loss_name: str, loss_config: dict) -> Any:
        """
        Create a specific loss function by name.

        Args:
            model: SentenceTransformer model
            loss_name: Name of the loss function
            loss_config: Additional configuration for the loss function

        Returns:
            Loss function instance
        """
        # === Contrastive Learning Losses ===
        if loss_name == 'MultipleNegativesRankingLoss':
            scale = loss_config.get('scale', 20.0)
            return losses.MultipleNegativesRankingLoss(model, scale=scale)

        elif loss_name == 'CachedMultipleNegativesRankingLoss':
            scale = loss_config.get('scale', 20.0)
            mini_batch_size = loss_config.get('mini_batch_size', 32)
            return losses.CachedMultipleNegativesRankingLoss(
                model, scale=scale, mini_batch_size=mini_batch_size
            )

        elif loss_name == 'ContrastiveLoss':
            margin = loss_config.get('margin', 0.5)
            return losses.ContrastiveLoss(model, margin=margin)

        elif loss_name == 'OnlineContrastiveLoss':
            margin = loss_config.get('margin', 0.5)
            return losses.OnlineContrastiveLoss(model, margin=margin)

        elif loss_name == 'MegaBatchMarginLoss':
            positive_margin = loss_config.get('positive_margin', 0.8)
            negative_margin = loss_config.get('negative_margin', 0.3)
            return losses.MegaBatchMarginLoss(
                model, positive_margin=positive_margin, negative_margin=negative_margin
            )

        # === Similarity Losses ===
        elif loss_name == 'CosineSimilarityLoss':
            return losses.CosineSimilarityLoss(model)

        elif loss_name == 'CoSENTLoss':
            scale = loss_config.get('scale', 20.0)
            return losses.CoSENTLoss(model, scale=scale)

        elif loss_name == 'AnglELoss':
            scale = loss_config.get('scale', 20.0)
            return losses.AnglELoss(model, scale=scale)

        # === Triplet Losses ===
        elif loss_name == 'TripletLoss':
            distance_metric = loss_config.get('distance_metric', 'cosine')
            triplet_margin = loss_config.get('triplet_margin', 5.0)

            # Map string to distance metric function
            if distance_metric == 'cosine':
                metric = losses.TripletDistanceMetric.COSINE
            elif distance_metric == 'euclidean':
                metric = losses.TripletDistanceMetric.EUCLIDEAN
            else:
                metric = losses.TripletDistanceMetric.COSINE

            return losses.TripletLoss(model, distance_metric=metric, triplet_margin=triplet_margin)

        elif loss_name == 'BatchHardTripletLoss':
            margin = loss_config.get('margin', 5.0)
            return losses.BatchHardTripletLoss(model, margin=margin)

        elif loss_name == 'BatchSemiHardTripletLoss':
            margin = loss_config.get('margin', 5.0)
            return losses.BatchSemiHardTripletLoss(model, margin=margin)

        elif loss_name == 'BatchAllTripletLoss':
            margin = loss_config.get('margin', 5.0)
            return losses.BatchAllTripletLoss(model, margin=margin)

        # === Classification Losses ===
        elif loss_name == 'SoftmaxLoss':
            num_labels = loss_config.get('num_labels', 2)
            sentence_embedding_dimension = model.get_sentence_embedding_dimension()
            return losses.SoftmaxLoss(
                model=model,
                sentence_embedding_dimension=sentence_embedding_dimension,
                num_labels=num_labels
            )

        # === Knowledge Distillation Losses ===
        elif loss_name == 'DistillKLDivLoss':
            # Note: This requires teacher model embeddings to be pre-computed
            return losses.MSELoss(model)  # Fallback to MSE for now

        elif loss_name == 'MSELoss':
            return losses.MSELoss(model)

        # === GIST Embedding Loss ===
        elif loss_name == 'GISTEmbedLoss':
            guide_model_name = loss_config.get('guide_model', None)
            if guide_model_name:
                guide_model = SentenceTransformer(guide_model_name)
                return losses.GISTEmbedLoss(model, guide_model)
            else:
                logger.warning("GISTEmbedLoss requires guide_model, falling back to MNR")
                return losses.MultipleNegativesRankingLoss(model)

        # === Custom: Dynamic Negatives Loss ===
        # This stays in TrainFactory on purpose:
        # - It is an embedding/SentenceTransformer-side loss, not a qwen3 reranker loss.
        # - It supports row-wise explicit negatives with variable counts after our
        #   universal-format dataset is flattened and padded with empty negatives.
        # - qwen3-rerank-trainer remains the source of truth only for decoder reranker
        #   ranking / RL losses and should not absorb this embedding-specific adapter.
        elif loss_name == 'DynamicExplicitNegativesRankingLoss':
            from ...losses.contrastive.dynamic_negatives_loss import DynamicExplicitNegativesRankingLoss
            scale = loss_config.get('scale', 20.0)
            normalize_by_logc = loss_config.get('normalize_by_logc', False)
            ignore_empty = loss_config.get('ignore_empty', True)
            return DynamicExplicitNegativesRankingLoss(
                model, scale=scale, normalize_by_logc=normalize_by_logc, ignore_empty=ignore_empty
            )

        # === MatryoshkaLoss (standalone, wraps MNR by default) ===
        elif loss_name == 'MatryoshkaLoss':
            matryoshka_dims = loss_config.get('matryoshka_dims', [768, 512, 256, 128, 64])
            inner_loss = losses.MultipleNegativesRankingLoss(model)
            return losses.MatryoshkaLoss(model, inner_loss, matryoshka_dims)

        # === Adaptive Layer Loss (for layer-wise training) ===
        elif loss_name == 'Matryoshka2dLoss':
            matryoshka_dims = loss_config.get('matryoshka_dims', [768, 512, 256, 128, 64])
            inner_loss = losses.MultipleNegativesRankingLoss(model)
            return losses.Matryoshka2dLoss(model, inner_loss, matryoshka_dims)

        # === Default ===
        else:
            logger.warning(f"Unknown loss: {loss_name}, using MultipleNegativesRankingLoss")
            return losses.MultipleNegativesRankingLoss(model)

    def _prepare_datasets(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten universal format datasets for SentenceTransformerTrainer.

        Universal format: {query, positives: [...], negatives: [...], ...}
        Target format: {anchor, positive, negative_0, negative_1, ..., negative_N}

        positive = positives[0], negatives padded to global max count with
        empty strings.  DynamicExplicitNegativesRankingLoss filters empty
        negatives by embedding norm, so no data is lost.
        """
        train = datasets.get('train')
        if train is None:
            return datasets

        ds = train
        if isinstance(ds, dict):
            ds = next(iter(ds.values()))

        columns = ds.column_names if hasattr(ds, 'column_names') else []
        if not ('positives' in columns and 'negatives' in columns):
            return datasets  # not universal format

        logger.info("Detected universal format dataset, flattening for SentenceTransformer")

        # Determine anchor column
        anchor_col = 'query' if 'query' in columns else columns[0]

        # Pad-to-max strategy: keep all negatives, pad short rows with ""
        normalized_negatives: List[List[str]] = []
        for value in ds['negatives']:
            negatives = normalize_embedding_negatives(value)
            if negatives is None:
                raise ValueError(
                    "Universal embedding rows require non-empty string/list negatives"
                )
            normalized_negatives.append(negatives)
        neg_counts = [len(negatives) for negatives in normalized_negatives]
        max_neg = max(neg_counts) if neg_counts else 0
        min_neg = min(neg_counts) if neg_counts else 0
        logger.info(
            f"Universal format: {len(ds)} rows, anchor='{anchor_col}', "
            f"min_neg={min_neg}, max_neg={max_neg} (pad-to-max)"
        )

        # Build flattened dict
        anchors = []
        positives = []
        neg_columns: Dict[str, list] = {f"negative_{i}": [] for i in range(max_neg)}

        for row, negs in zip(ds, normalized_negatives):
            pos_list = row['positives']
            if isinstance(pos_list, str):
                # 字符串当作单元素正例，而不是按字符切成单字母垃圾数据
                # （对照 reranker 的 _pick_first_text/_normalize_text_list 语义）
                pos_list = [pos_list]
            if not pos_list:
                continue  # skip rows without positives
            anchors.append(row[anchor_col])
            positives.append(pos_list[0])
            for i in range(max_neg):
                neg_columns[f"negative_{i}"].append(negs[i] if i < len(negs) else "")

        from datasets import Dataset as HFDataset

        flat_data = {"anchor": anchors, "positive": positives, **neg_columns}
        flat_ds = HFDataset.from_dict(flat_data)
        logger.info(
            f"Flattened dataset: {len(flat_ds)} rows, "
            f"columns={flat_ds.column_names}"
        )

        datasets['train'] = flat_ds
        return datasets

    def _auto_detect_loss(self, dataset) -> str:
        """Auto-detect appropriate loss function based on dataset format."""
        try:
            if isinstance(dataset, dict):
                dataset = next(iter(dataset.values()))

            columns = dataset.column_names
            sample = dataset[0] if len(dataset) > 0 else None

            if sample is None:
                return 'MultipleNegativesRankingLoss'

            # Flattened universal format: (anchor, positive, negative_0, ...)
            neg_cols = [c for c in columns if c.startswith('negative_')]
            if 'anchor' in columns and 'positive' in columns and neg_cols:
                logger.info(
                    f"Auto-detect: flattened universal format with {len(neg_cols)} "
                    f"explicit negatives -> DynamicExplicitNegativesRankingLoss"
                )
                return 'DynamicExplicitNegativesRankingLoss'

            # Check for score/label column
            if len(columns) == 3:
                third_col = columns[2]
                value = sample[third_col]
                if isinstance(value, float) and 0 <= value <= 1:
                    return 'CosineSimilarityLoss'
                elif isinstance(value, int):
                    return 'MultipleNegativesRankingLoss'

            return 'MultipleNegativesRankingLoss'

        except Exception as e:
            logger.warning(f"Auto-detect loss failed: {e}, using default")
            return 'MultipleNegativesRankingLoss'

    def create_training_args(self, config: Dict[str, Any]) -> SentenceTransformerTrainingArguments:
        """
        Create SentenceTransformerTrainingArguments.

        Args:
            config: Training configuration dictionary

        Returns:
            SentenceTransformerTrainingArguments instance
        """
        logger.info("Creating SentenceTransformerTrainingArguments")

        try:
            args = SentenceTransformerTrainingArguments(**config)
            logger.info("SentenceTransformerTrainingArguments created successfully")
            return args
        except Exception as e:
            logger.error(f"SentenceTransformerTrainingArguments creation failed: {e}")
            raise

    def create_trainer_instance(self, model, args, train_dataset, eval_dataset, loss, evaluator) -> SentenceTransformerTrainer:
        """
        Create SentenceTransformerTrainer instance.

        Args:
            model: SentenceTransformer model
            args: SentenceTransformerTrainingArguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            loss: Loss function
            evaluator: Evaluator (optional)

        Returns:
            SentenceTransformerTrainer instance
        """
        logger.info("Creating SentenceTransformerTrainer")

        trainer_kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "loss": loss,
        }

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        if evaluator is not None:
            trainer_kwargs["evaluator"] = evaluator

        try:
            trainer = SentenceTransformerTrainer(**trainer_kwargs)
            logger.info("SentenceTransformerTrainer created successfully")
            return trainer
        except Exception as e:
            logger.error(f"SentenceTransformerTrainer creation failed: {e}")
            raise

    def create_evaluator(self, eval_dataset: Any) -> Optional[Any]:
        """
        Create evaluator for embedding model evaluation.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluator instance or None
        """
        if eval_dataset is None:
            logger.info("No eval dataset, skipping evaluator creation")
            return None

        logger.info("Creating embedding evaluator")

        try:
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

            # Get sample to determine format
            if isinstance(eval_dataset, dict):
                eval_dataset = next(iter(eval_dataset.values()))

            columns = eval_dataset.column_names
            if len(columns) >= 3:
                sentences1 = eval_dataset[columns[0]]
                sentences2 = eval_dataset[columns[1]]
                scores = eval_dataset[columns[2]]

                evaluator = EmbeddingSimilarityEvaluator(
                    sentences1=sentences1,
                    sentences2=sentences2,
                    scores=scores,
                    name="embedding_eval"
                )

                logger.info("Embedding evaluator created successfully")
                return evaluator

            logger.warning("Eval dataset format not suitable for evaluator")
            return None

        except Exception as e:
            logger.warning(f"Evaluator creation failed: {e}, skipping evaluation")
            return None
