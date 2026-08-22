"""
Decoder-based Reranker Trainer.

Integrates qwen3-rerank-trainer package for training Qwen3-Reranker models.
Supports SFT and RL (GRPO) training methods.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from ...config import settings
from ...tuners.policy import canonicalize_tuner_config
from ..base.training_result import TrainingResult
from ...utils.training_metrics import extract_final_loss_metrics

logger = logging.getLogger(__name__)


def _resolve_data_path(data_path: str, split: Optional[str] = None) -> str:
    """Resolve dataset path: handle directories by finding the first data file.

    qwen3-rerank-trainer now supports multiple formats (jsonl, json, parquet, csv, arrow)
    natively via HuggingFace datasets, so no format conversion is needed here.
    """
    p = Path(data_path)

    # If directory, find the first data file
    if p.is_dir():
        def _is_arrow_dataset_dir(path: Path) -> bool:
            return path.is_dir() and (
                (path / "dataset_info.json").exists() or any(path.glob("*.arrow"))
            )

        def _normalize_split(name: Optional[str]) -> str:
            split_name = (name or "train").lower()
            if split_name in ("val", "validation", "dev"):
                return "eval"
            if split_name in ("train", "training"):
                return "train"
            return split_name

        split_key = _normalize_split(split)
        if split_key in ("train", "eval", "test"):
            split_dirs = {
                "train": ["train", "training"],
                "eval": ["val", "validation", "dev", "eval"],
                "test": ["test"],
            }[split_key]

            for split_dir_name in split_dirs:
                split_dir = p / split_dir_name
                if _is_arrow_dataset_dir(split_dir):
                    return str(split_dir)

            split_patterns = []
            for split_dir_name in split_dirs:
                split_patterns.extend([
                    f"{split_dir_name}*.jsonl",
                    f"{split_dir_name}*.json",
                    f"{split_dir_name}*.parquet",
                    f"{split_dir_name}*.csv",
                    f"{split_dir_name}*.arrow",
                ])

            for pattern in split_patterns:
                files = sorted(p.glob(pattern))
                if files:
                    return str(files[0])

        if _is_arrow_dataset_dir(p):
            return str(p)

        arrow_fallback = None
        for split_name in ["train", "training", "val", "validation", "dev", "test"]:
            split_dir = p / split_name
            if _is_arrow_dataset_dir(split_dir):
                if split_name in ("train", "training"):
                    return str(split_dir)
                if arrow_fallback is None:
                    arrow_fallback = split_dir
        if arrow_fallback is not None:
            return str(arrow_fallback)

        train_patterns = ['train*.jsonl', 'train*.json', 'train*.parquet']
        for pattern in train_patterns:
            files = sorted(p.glob(pattern))
            if files:
                return str(files[0])

        for ext in ['*.jsonl', '*.json', '*.parquet', '*.csv', '*.arrow']:
            files = sorted(p.glob(ext))
            if files:
                return str(files[0])

        raise FileNotFoundError(f"No data files found in directory: {data_path}")

    return str(p)


class DecoderRerankerTrainer:
    """
    Decoder-based reranker trainer using qwen3-rerank-trainer.

    Supports:
    - SFT: Contrastive learning with BCE/InfoNCE/ListMLE/LambdaLoss/RankNet
    - RL: GRPO/DAPO/DR-GRPO reinforcement learning

    Args:
        training_config: TrainFactory training configuration dict
    """

    def __init__(self, training_config: Dict[str, Any]):
        self.config = canonicalize_tuner_config(training_config)
        self.training_method = self.config.get('training_method', 'sft')

    def _update_task_stage(self, status: str, error_message: Optional[str] = None) -> None:
        """Best-effort task stage update for DB-backed training tasks."""
        task_id = self.config.get("task_id")
        if not task_id:
            return

        try:
            from ...storage.services.training_task_service import training_task_service

            training_task_service.update_task_status(
                task_id,
                status,
                error_message,
                run_token=self.config.get("_run_token"),
            )
        except Exception as exc:
            logger.warning(f"Failed to update task {task_id} stage to {status}: {exc}")

    def _run_trainer_train(self, trainer, resume_from: Optional[str], label: str) -> None:
        """Run the underlying trainer after marking the task as actively training."""
        self._update_task_stage("running")
        if resume_from:
            logger.info(f"Resuming {label} training from checkpoint: {resume_from}")
            trainer.train(resume_from_checkpoint=resume_from)
        else:
            logger.info(f"Starting {label} training...")
            trainer.train()

    @staticmethod
    def _normalize_split(name: Optional[str]) -> str:
        split = (name or "train").lower()
        if split in ("val", "validation", "dev"):
            return "eval"
        if split in ("train", "training"):
            return "train"
        return split

    def train(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """Execute training pipeline."""
        self._update_task_stage("preparing")
        if self.training_method == 'sft':
            return self._train_sft(progress_callback)
        elif self.training_method in ('grpo', 'dapo', 'dr_grpo', 'dpo'):
            return self._train_rl(progress_callback)
        else:
            raise ValueError(f"Unsupported training method: {self.training_method}")

    def _load_datasets(self, dataset_configs: list, tokenizer, n_docs: int, n_pos: int,
                       max_length: int, seed: int, split_filter: str = None) -> Any:
        """Load multiple datasets and combine them.

        Args:
            dataset_configs: List of {'path': str, 'max_samples': int|None, 'split': str}
            tokenizer: Tokenizer instance
            n_docs, n_pos, max_length, seed: RerankDataset parameters
            split_filter: If provided, only load datasets with matching split ('train', 'eval', or 'test')

        Returns:
            Combined dataset (single RerankDataset or ConcatDataset), or None if no matching datasets
        """
        from qwen3_rerank_trainer import RerankDataset

        # Filter by split if specified
        if split_filter:
            target_split = self._normalize_split(split_filter)
            filtered_configs = [
                cfg for cfg in dataset_configs
                if self._normalize_split(cfg.get('split', 'train')) == target_split
            ]
        else:
            filtered_configs = dataset_configs

        if not filtered_configs:
            return None

        datasets = []
        total_samples = 0

        for cfg in filtered_configs:
            split_label = self._normalize_split(cfg.get('split', 'train'))
            data_path = _resolve_data_path(cfg['path'], split_label)
            max_samples = cfg.get('max_samples') or 0  # 0 means no limit

            ds = RerankDataset(
                data_path, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, max_samples=max_samples, seed=seed,
            )
            datasets.append(ds)
            total_samples += len(ds)
            logger.info(f"Loaded {len(ds)} {split_label} samples from {data_path}" +
                       (f" (max_samples={max_samples})" if max_samples else ""))

        if len(datasets) == 1:
            return datasets[0]

        # Combine multiple datasets
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset(datasets)
        logger.info(f"Combined {len(datasets)} datasets, total {total_samples} samples")
        return combined

    def _evaluate_test_metrics(
        self,
        model,
        tokenizer,
        test_dataset,
        max_length: int,
        batch_size: int,
        ks: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compute ranking metrics (MRR/NDCG/etc.) for decoder reranker test set."""
        if test_dataset is None:
            return None

        from qwen3_rerank_trainer.data import tokenize_for_training, forward_and_get_logits
        from qwen3_rerank_trainer.evaluation import compute_all_metrics, aggregate_metrics

        def iter_samples():
            for item in test_dataset:
                if not isinstance(item, dict):
                    continue
                query = item.get("query", "")
                if not query:
                    continue

                if "documents" in item and "labels" in item:
                    docs = item.get("documents") or []
                    labels = item.get("labels") or []
                else:
                    positives = item.get("positives") or item.get("positive") or []
                    negatives = item.get("negatives") or item.get("negative") or []
                    if isinstance(positives, str):
                        positives = [positives]
                    if isinstance(negatives, str):
                        negatives = [negatives]
                    docs = list(positives) + list(negatives)
                    labels = [1] * len(positives) + [0] * len(negatives)

                if not docs:
                    continue
                if hasattr(labels, "tolist"):
                    labels = labels.tolist()
                yield query, docs, labels

        def batch_iter(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i:i + size]

        device = next(model.parameters()).device
        was_training = getattr(model, "training", False)
        model.eval()

        all_results = []
        with torch.no_grad():
            for query, docs, labels in iter_samples():
                scores = []
                for chunk in batch_iter(docs, batch_size):
                    inputs = tokenize_for_training(
                        tokenizer,
                        query,
                        chunk,
                        max_length=max_length,
                    )
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    yes_id = inputs["yes_token_id"]
                    no_id = inputs["no_token_id"]
                    _, _, score_tensor = forward_and_get_logits(
                        model, input_ids, attention_mask, yes_id, no_id
                    )
                    scores.extend(score_tensor.detach().cpu().tolist())

                if not scores or len(scores) != len(labels):
                    continue

                ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                positive_indices = {i for i, label in enumerate(labels) if label}
                if not positive_indices:
                    continue
                metrics = compute_all_metrics(ranking, positive_indices, ks=ks)
                all_results.append(metrics)

        if was_training:
            model.train()

        if not all_results:
            return None

        aggregated = aggregate_metrics(all_results)
        aggregated["num_samples"] = len(all_results)
        return aggregated

    @staticmethod
    def _merge_test_metrics(
        final_metrics: Optional[Dict[str, Any]],
        test_before: Optional[Dict[str, Any]],
        test_after: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
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

    def _train_sft(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """Run SFT training using qwen3-rerank-trainer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from qwen3_rerank_trainer import ContrastiveSFTTrainer, RerankDataset, RerankCollator
        from qwen3_rerank_trainer.training.sft_trainer import get_yes_no_token_ids

        config = self.config
        model_path = config['base_model_path']
        output_dir = Path(config.get('output_dir') or './output')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Loss config
        loss_config = config.get('loss_config') or {}
        loss_type = loss_config.get('name', 'bce').lower()
        n_docs = loss_config.get('n_docs', 8)
        n_pos = loss_config.get('n_pos', 1)
        max_length = loss_config.get('max_length', 4096)
        chunk_size = loss_config.get('chunk_size', 0)
        temperature = loss_config.get('temperature', 0.05)
        lambda_metric = loss_config.get('metric', 'ndcg')
        infonce_mode = loss_config.get('infonce_mode', 'single')
        ranknet_max_pairs_per_batch = loss_config.get('ranknet_max_pairs_per_batch', 2000000)

        # Precision（默认关闭 bf16，与 llm_trainer 一致：旧卡直连不会默认 bf16 失败）
        use_bf16 = config.get('bf16', False)
        use_fp16 = config.get('fp16', False)
        dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=settings.allow_model_remote_code,
            padding_side='left',
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if progress_callback:
            progress_callback(5.0)

        # DeepSpeed manages device placement; avoid device_map="auto" when
        # using it (mirrors llm_trainer) to prevent placement conflicts / OOM.
        use_deepspeed = bool(config.get('deepspeed'))
        device_map = None if use_deepspeed else "auto"
        logger.info(f"Loading model from {model_path} (device_map={device_map})")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=settings.allow_model_remote_code,
            torch_dtype=dtype,
            **({"device_map": device_map} if device_map else {}),
        )

        # LoRA
        if config.get('use_lora', False):
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=config.get('lora_r', 16),
                lora_alpha=config.get('lora_alpha', 32),
                lora_dropout=config.get('lora_dropout', 0.0),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        if progress_callback:
            progress_callback(10.0)

        # Get yes/no token IDs
        yes_id, no_id = get_yes_no_token_ids(tokenizer)
        logger.info(f"yes_token_id: {yes_id}, no_token_id: {no_id}")

        # Dataset - support multiple datasets with individual max_samples and split
        seed = config.get('seed', 42)
        dataset_configs = config.get('dataset_configs')

        if dataset_configs:
            # Load training datasets (split='train')
            logger.info(f"Loading datasets from {len(dataset_configs)} config(s)")
            train_dataset = self._load_datasets(
                dataset_configs, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, seed=seed,
                split_filter='train',
            )
            if train_dataset is None:
                raise ValueError("No training datasets found (split='train')")

            # Load eval datasets (split='eval')
            val_dataset = self._load_datasets(
                dataset_configs, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, seed=seed,
                split_filter='eval',
            )
            if val_dataset:
                logger.info(f"Eval samples: {len(val_dataset)}")

            test_dataset = self._load_datasets(
                dataset_configs, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, seed=seed,
                split_filter='test',
            )
            if test_dataset:
                logger.info(f"Test samples: {len(test_dataset)}")
        else:
            # Backward compatibility: single dataset
            data_path = _resolve_data_path(config['train_dataset_path'], "train")
            max_samples = config.get('max_samples', 0)
            logger.info(f"Loading dataset from {data_path}")
            train_dataset = RerankDataset(
                data_path, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, max_samples=max_samples, seed=seed,
            )
            val_dataset = None
            test_dataset = None

        logger.info(f"Training samples: {len(train_dataset)}")

        # Backward compatibility: val_dataset_path override
        val_data_path = config.get('val_dataset_path')
        if val_data_path and os.path.exists(val_data_path):
            val_dataset = RerankDataset(
                val_data_path, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, seed=seed,
            )
            logger.info(f"Loaded eval dataset from {val_data_path}: {len(val_dataset)} samples")

        test_data_path = config.get('test_dataset_path')
        if test_data_path and os.path.exists(test_data_path):
            test_dataset = RerankDataset(
                test_data_path, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, seed=seed,
            )
            logger.info(f"Loaded test dataset from {test_data_path}: {len(test_dataset)} samples")

        collator = RerankCollator(tokenizer, max_length=max_length)

        if progress_callback:
            progress_callback(15.0)

        # Test metrics before training
        eval_batch_size = config.get('per_device_eval_batch_size') or config.get('per_device_train_batch_size') or 4
        test_metrics_before = self._evaluate_test_metrics(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            max_length=max_length,
            batch_size=eval_batch_size,
        )

        # Training arguments
        eval_strategy = config.get('eval_strategy', 'no')
        # Disable eval if no eval dataset available
        if val_dataset is None:
            eval_strategy = 'no'

        training_args_kwargs = dict(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.get('per_device_train_batch_size') or 4,
            gradient_accumulation_steps=config.get('gradient_accumulation_steps') or 1,
            learning_rate=config.get('learning_rate') or 1e-5,
            num_train_epochs=config.get('num_train_epochs') or 3,
            warmup_ratio=config.get('warmup_ratio') or 0.1,
            logging_steps=config.get('logging_steps') or 1,  # Log every step for loss curve
            save_steps=config.get('save_steps') or 500,
            save_total_limit=3,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to=config.get('report_to') or 'none',
            remove_unused_columns=False,
            dataloader_num_workers=4,
            seed=seed,
            eval_strategy=eval_strategy,
        )

        # DeepSpeed configuration
        ds_preset = config.get('deepspeed')
        if ds_preset:
            from .llm_trainer import DEEPSPEED_PRESETS
            if ds_preset in DEEPSPEED_PRESETS:
                training_args_kwargs['deepspeed'] = DEEPSPEED_PRESETS[ds_preset]
                logger.info(f"Using DeepSpeed preset: {ds_preset}")
            elif isinstance(ds_preset, dict):
                training_args_kwargs['deepspeed'] = ds_preset
                logger.info("Using custom DeepSpeed config")
        if eval_strategy != 'no':
            training_args_kwargs['per_device_eval_batch_size'] = config.get('per_device_eval_batch_size') or config.get('per_device_train_batch_size') or 4
            if eval_strategy == 'steps':
                training_args_kwargs['eval_steps'] = config.get('eval_steps') or config.get('save_steps') or 500

        training_args = TrainingArguments(**training_args_kwargs)

        # Create trainer
        trainer = ContrastiveSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            yes_token_id=yes_id,
            no_token_id=no_id,
            chunk_size=chunk_size,
            loss_type=loss_type,
            temperature=temperature,
            lambda_metric=lambda_metric,
            infonce_mode=infonce_mode,
            ranknet_max_pairs_per_batch=ranknet_max_pairs_per_batch,
        )

        if progress_callback:
            progress_callback(20.0)

        # Add MetricsCallback for training metrics collection (20% -> 90%)
        task_id = config.get('task_id')
        run_token = config.get("_run_token")
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

                def wrapped_progress(progress: float):
                    """Map 5-95% from MetricsCallback to 20-90% for decoder training."""
                    if progress_callback:
                        # MetricsCallback reports 5-95%, map to 20-90%
                        mapped = 20.0 + (progress - 5.0) / 90.0 * 70.0
                        progress_callback(max(20.0, min(90.0, mapped)))

                metrics_callback = MetricsCallback(
                    output_dir=str(output_dir),
                    task_id=task_id,
                    progress_callback=wrapped_progress,
                    db_update_callback=db_update_callback,
                )
                trainer.add_callback(metrics_callback)
                logger.info(f"Added MetricsCallback for task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to add MetricsCallback: {e}")
                # Fallback to simple progress callback
                if progress_callback:
                    from transformers import TrainerCallback

                    class ProgressCallback(TrainerCallback):
                        def on_step_end(self, args, state, control, **kwargs):
                            if state.max_steps and state.max_steps > 0:
                                train_progress = state.global_step / state.max_steps
                                progress_callback(20.0 + train_progress * 70.0)

                    trainer.add_callback(ProgressCallback())
        elif progress_callback:
            # Fallback if no task_id
            from transformers import TrainerCallback

            class ProgressCallback(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.max_steps > 0:
                        train_progress = state.global_step / state.max_steps
                        progress_callback(20.0 + train_progress * 70.0)

            trainer.add_callback(ProgressCallback())

        resume_from = config.get('resume_from_checkpoint')
        self._run_trainer_train(trainer, resume_from, "SFT")

        if progress_callback:
            progress_callback(95.0)

        # Save
        save_dir = str(output_dir / "final")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

        if progress_callback:
            progress_callback(100.0)

        # Test metrics after training
        if test_dataset is not None:
            self._update_task_stage("evaluating")
        test_metrics_after = self._evaluate_test_metrics(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            max_length=max_length,
            batch_size=eval_batch_size,
        )

        log_history = getattr(getattr(trainer, 'state', None), 'log_history', None)
        final_metrics = extract_final_loss_metrics(log_history)

        final_metrics = self._merge_test_metrics(final_metrics, test_metrics_before, test_metrics_after)

        return TrainingResult(
            model=model,
            save_dir=save_dir,
            final_metrics=final_metrics,
            task_id=config.get('task_id'),
        )

    def _train_rl(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """Run RL training using qwen3-rerank-trainer."""
        from transformers import AutoTokenizer
        from qwen3_rerank_trainer import RLTrainer, load_sft_model
        from qwen3_rerank_trainer.training import RLRerankDataset, RLCollator

        config = self.config
        rl_config = config.get('rl_config') or {}

        # Model paths
        sft_model_path = config.get('sft_checkpoint_path') or config['base_model_path']
        output_dir = Path(config.get('output_dir') or './output')
        output_dir.mkdir(parents=True, exist_ok=True)

        loss_type = rl_config.get('loss_type', self.training_method)
        reward_type = rl_config.get('reward_type', 'rank_based')
        reward_k = rl_config.get('reward_k', 10)
        scale_rewards = rl_config.get('scale_rewards', False)
        num_iterations = rl_config.get('num_iterations', 1)
        chunk_size = rl_config.get('chunk_size', 0)
        dpo_beta = rl_config.get('beta', 0.1)
        reference_free = rl_config.get('reference_free', False)

        base_model_path = config['base_model_path']
        logger.info(f"Loading SFT model from {sft_model_path}")
        model = load_sft_model(
            sft_model_path,
            base_model_path,
            trust_remote_code=settings.allow_model_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=settings.allow_model_remote_code,
            padding_side='left',
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if progress_callback:
            progress_callback(10.0)

        # Dataset - support dataset_configs or fallback to single path
        loss_config = config.get('loss_config') or {}
        n_docs = rl_config.get('n_docs', loss_config.get('n_docs', 8))
        n_pos = loss_config.get('n_pos', 1)
        max_length = rl_config.get('max_length', loss_config.get('max_length', 4096))
        seed = config.get('seed', 42)

        dataset_configs = config.get('dataset_configs')
        if dataset_configs:
            train_configs = [
                cfg for cfg in dataset_configs
                if self._normalize_split(cfg.get('split', 'train')) == 'train'
            ]
            if not train_configs:
                train_configs = dataset_configs
            data_path = _resolve_data_path(train_configs[0]['path'], "train")
            max_samples = train_configs[0].get('max_samples') or 0
            if len(train_configs) > 1:
                logger.warning("RL training uses only the first train dataset, ignoring remaining train datasets")
            if len(dataset_configs) > 1 and len(train_configs) == 1:
                logger.warning(f"RL training uses only the first dataset, ignoring {len(dataset_configs)-1} additional dataset(s)")
        else:
            data_path = _resolve_data_path(config['train_dataset_path'], "train")
            max_samples = config.get('max_samples', 0)

        train_dataset = RLRerankDataset(
            data_path, tokenizer=tokenizer,
            n_docs=n_docs,
            max_length=max_length,
            max_samples=max_samples,
        )

        test_dataset = None
        if dataset_configs:
            test_dataset = self._load_datasets(
                dataset_configs, tokenizer=tokenizer,
                n_docs=n_docs, n_pos=n_pos,
                max_length=max_length, seed=seed,
                split_filter='test',
            )
            if test_dataset:
                logger.info(f"Test samples: {len(test_dataset)}")
        else:
            test_data_path = config.get('test_dataset_path')
            if test_data_path and os.path.exists(test_data_path):
                from qwen3_rerank_trainer import RerankDataset
                test_dataset = RerankDataset(
                    test_data_path, tokenizer=tokenizer,
                    n_docs=n_docs, n_pos=n_pos,
                    max_length=max_length, seed=seed,
                )
                logger.info(f"Loaded test dataset from {test_data_path}: {len(test_dataset)} samples")

        collator = RLCollator(tokenizer, max_length=max_length)

        if progress_callback:
            progress_callback(15.0)

        # Test metrics before RL
        eval_batch_size = config.get('per_device_eval_batch_size') or config.get('per_device_train_batch_size') or 4
        test_metrics_before = self._evaluate_test_metrics(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            max_length=max_length,
            batch_size=eval_batch_size,
        )

        # Create RL trainer
        trainer = RLTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=collator,
            output_dir=str(output_dir),
            loss_type=loss_type,
            kl_coef=rl_config.get('kl_coef', 0.1),
            clip_range=rl_config.get('clip_range', 0.2),
            reward_type=reward_type,
            reward_k=reward_k,
            scale_rewards=scale_rewards,
            num_iterations=num_iterations,
            chunk_size=chunk_size,
            beta=dpo_beta,
            reference_free=reference_free,
            learning_rate=config.get('learning_rate', 5e-6),
            num_train_epochs=config.get('num_train_epochs', 1),
            per_device_train_batch_size=config.get('per_device_train_batch_size', 2),
        )

        if progress_callback:
            progress_callback(20.0)

        resume_from = config.get('resume_from_checkpoint')
        self._run_trainer_train(trainer, resume_from, f"RL ({loss_type})")

        save_dir = str(output_dir / "final")
        trainer.save_model(save_dir)
        logger.info(f"Model saved to {save_dir}")

        if progress_callback:
            progress_callback(100.0)

        # Test metrics after RL
        if test_dataset is not None:
            self._update_task_stage("evaluating")
        test_metrics_after = self._evaluate_test_metrics(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            max_length=max_length,
            batch_size=eval_batch_size,
        )

        log_history = getattr(getattr(trainer, 'state', None), 'log_history', None)
        final_metrics = extract_final_loss_metrics(log_history)

        final_metrics = self._merge_test_metrics(final_metrics, test_metrics_before, test_metrics_after)

        return TrainingResult(
            model=model,
            save_dir=save_dir,
            final_metrics=final_metrics,
            task_id=config.get('task_id'),
        )
