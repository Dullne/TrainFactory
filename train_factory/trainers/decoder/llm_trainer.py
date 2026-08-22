"""
LLM Trainer for general-purpose LLM fine-tuning.

Supports SFT, DPO, and ORPO training methods via HuggingFace TRL.
Automatically maps various data formats (QA, instruction, preference) to
the format expected by each trainer.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset, concatenate_datasets

from ...config import settings
from ...data.preprocessors import DPOPreprocessor
from ...tuners.policy import canonicalize_tuner_config
from ...utils.training_metrics import extract_final_loss_metrics
from ..base.training_result import TrainingResult
from .decoder_reranker_trainer import _resolve_data_path

logger = logging.getLogger(__name__)


DEEPSPEED_PRESETS = {
    "zero2": {
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },
    "zero3": {
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },
    "zero2_offload": {
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },
    "zero3_offload": {
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },
}


class LLMTrainer:
    """
    LLM Trainer supporting SFT, DPO, and ORPO training methods.

    Uses HuggingFace TRL library:
    - SFT: trl.SFTTrainer (supervised fine-tuning)
    - DPO: trl.DPOTrainer (direct preference optimization)
    - ORPO: trl.ORPOTrainer (odds ratio preference optimization)

    Data format auto-mapping:
    - SFT: accepts query/answer, instruction/output, messages, conversations
    - DPO/ORPO: accepts query/pos/neg, prompt/chosen/rejected
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

    def train(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """Execute training pipeline based on training_method."""
        self._update_task_stage("preparing")
        if self.training_method == 'sft':
            return self._train_sft(progress_callback)
        elif self.training_method == 'dpo':
            return self._train_dpo(progress_callback)
        elif self.training_method == 'orpo':
            return self._train_orpo(progress_callback)
        else:
            raise ValueError(f"Unsupported LLM training method: {self.training_method}")

    # ========== Shared Helpers ==========

    def _load_model_and_tokenizer(self):
        """Load AutoModelForCausalLM + AutoTokenizer, apply LoRA if configured."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = self.config
        model_path = config['base_model_path']

        # 默认关闭 bf16（与 API 链路一致）：直连库调用在旧卡（<Ampere）上
        # 不会因默认 bf16 而失败/精度异常；支持 bf16 时由配置显式开启。
        use_bf16 = config.get('bf16', False)
        use_fp16 = config.get('fp16', False)
        dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=settings.allow_model_remote_code,
            padding_side='right',
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # DeepSpeed manages device placement; avoid device_map="auto" when using it
        use_deepspeed = bool(config.get('deepspeed'))
        device_map = None if use_deepspeed else "auto"

        logger.info(f"Loading model from {model_path} (dtype={dtype}, device_map={device_map})")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=settings.allow_model_remote_code,
            torch_dtype=dtype,
            **({"device_map": device_map} if device_map else {}),
        )
        gradient_checkpointing = config.get('gradient_checkpointing')
        if gradient_checkpointing is None:
            gradient_checkpointing = True
        if gradient_checkpointing and hasattr(model, "config"):
            model.config.use_cache = False

        # Apply LoRA if configured
        if config.get('use_lora', False):
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=config.get('lora_r', 16),
                lora_alpha=config.get('lora_alpha', 32),
                lora_dropout=config.get('lora_dropout', 0.0),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer

    def _load_datasets(
        self,
        dataset_configs: List[Dict],
        split_filter: Optional[str] = None,
    ) -> Optional[Dataset]:
        """Load datasets from configs, merging multiple sources.

        Args:
            dataset_configs: List of {path, max_samples, split}
            split_filter: Only load configs matching this split (train/eval/test)

        Returns:
            Merged HuggingFace Dataset or None if no matching datasets found
        """
        from datasets import load_dataset

        all_datasets = []

        for dc in dataset_configs:
            dc_split = self._normalize_split(dc.get('split', 'train'))
            if split_filter and dc_split != split_filter:
                continue

            data_path = dc['path']
            max_samples = dc.get('max_samples')

            try:
                resolved = _resolve_data_path(data_path, split=dc_split)
                p = Path(resolved)

                if p.is_file():
                    ext = p.suffix.lower()
                    if ext in ('.jsonl', '.json'):
                        ds = load_dataset('json', data_files=str(p), split='train')
                    elif ext == '.parquet':
                        ds = load_dataset('parquet', data_files=str(p), split='train')
                    elif ext == '.csv':
                        ds = load_dataset('csv', data_files=str(p), split='train')
                    elif ext == '.arrow':
                        ds = Dataset.from_file(str(p))
                    else:
                        ds = load_dataset('json', data_files=str(p), split='train')
                elif p.is_dir():
                    ds = load_dataset('arrow', data_dir=str(p), split='train')
                else:
                    # Try as HuggingFace Hub dataset
                    ds = load_dataset(data_path, split=dc_split)

                if max_samples and max_samples > 0 and len(ds) > max_samples:
                    ds = ds.select(range(max_samples))

                logger.info(f"Loaded {len(ds)} samples from {data_path} (split={dc_split})")
                all_datasets.append(ds)

            except Exception as e:
                logger.warning(f"Failed to load dataset {data_path}: {e}")
                continue

        if not all_datasets:
            return None

        if len(all_datasets) == 1:
            return all_datasets[0]

        return concatenate_datasets(all_datasets)

    @staticmethod
    def _normalize_split(name: Optional[str]) -> str:
        split = (name or 'train').lower()
        if split in ('val', 'validation', 'dev'):
            return 'eval'
        if split in ('train', 'training'):
            return 'train'
        return split

    def _normalize_sft_dataset(self, dataset: Dataset) -> Dataset:
        """Normalize various formats to TRL SFT format with 'messages' column.

        Supports:
        - messages: [{role, content}] → pass through
        - conversations: ShareGPT format → convert
        - query + answer: QA format → convert
        - instruction + output: Alpaca format → convert

        Dirty samples (missing/None required fields, non-dict ShareGPT turns)
        are filtered out or defensively stringified so a single bad row no
        longer crashes the whole dataset.map (mirrors the DPO preprocessor's
        filter_invalid approach).
        """
        columns = dataset.column_names
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        allowed_roles = {"system", "user", "assistant", "tool", "function"}

        def normalize_native_messages(messages):
            """Validate native messages without discarding template extensions."""
            if not isinstance(messages, list) or not messages:
                return []

            normalized = []
            for turn in messages:
                # Native messages can contain a tool-call chain. Reject the
                # whole sample instead of silently removing one broken turn.
                if not isinstance(turn, dict):
                    return []

                raw_role = str(turn.get("role", "") or "").strip().lower()
                role = role_map.get(raw_role, raw_role)
                if role not in allowed_roles:
                    return []

                message = dict(turn)
                message["role"] = role

                tool_calls = message.get("tool_calls")
                has_tool_calls = tool_calls is not None
                if has_tool_calls:
                    if (
                        role != "assistant"
                        or not isinstance(tool_calls, list)
                        or not tool_calls
                        or any(not isinstance(call, dict) or not call for call in tool_calls)
                    ):
                        return []

                function_call = message.get("function_call")
                has_function_call = function_call is not None
                if has_function_call and (
                    role != "assistant"
                    or not isinstance(function_call, dict)
                    or not function_call
                ):
                    return []

                tool_call_id = message.get("tool_call_id")
                if tool_call_id is not None and (
                    role != "tool"
                    or not isinstance(tool_call_id, str)
                    or not tool_call_id.strip()
                ):
                    return []

                name = message.get("name")
                if name is not None and (
                    not isinstance(name, str) or not name.strip()
                ):
                    return []

                content = message.get("content")
                if content is None:
                    if not (has_tool_calls or has_function_call):
                        return []
                elif isinstance(content, str):
                    if not content.strip() and not (has_tool_calls or has_function_call):
                        return []
                elif isinstance(content, (list, dict)):
                    if not content and not (has_tool_calls or has_function_call):
                        return []
                else:
                    return []

                # Legacy function-result messages identify the invoked
                # function by name and carry its serialized result as text.
                if role == "function" and (
                    not isinstance(name, str)
                    or not name.strip()
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    return []

                normalized.append(message)

            return normalized

        def normalize_sharegpt_turns(messages):
            if not isinstance(messages, list):
                return []

            normalized = []
            for turn in messages:
                if not isinstance(turn, dict):
                    continue
                raw_role = str(turn.get("from", "") or "").strip().lower()
                role = role_map.get(raw_role, raw_role)
                if role not in allowed_roles:
                    continue
                raw_content = turn.get("value")
                if raw_content is None:
                    continue
                content = str(raw_content)
                if not content.strip():
                    continue
                normalized.append({
                    "role": role,
                    "content": content,
                })
            return normalized

        def has_training_conversation(sample):
            messages = sample.get("messages")
            if not isinstance(messages, list) or not messages:
                return False
            roles = {message.get("role") for message in messages if isinstance(message, dict)}
            return "user" in roles and "assistant" in roles

        def filter_training_conversations(mapped_dataset, source_format: str):
            original_count = len(mapped_dataset)
            filtered_dataset = mapped_dataset.filter(has_training_conversation)
            rejected_count = original_count - len(filtered_dataset)
            if rejected_count:
                logger.warning(
                    "Rejected %d invalid %s SFT sample(s); conversations must "
                    "contain valid user and assistant turns",
                    rejected_count,
                    source_format,
                )
            return filtered_dataset

        # Already in messages format
        if 'messages' in columns:
            dataset = dataset.map(
                lambda sample: {
                    "messages": normalize_native_messages(sample.get("messages"))
                }
            )
            return filter_training_conversations(dataset, "messages")

        # ShareGPT format
        if 'conversations' in columns:
            def convert_sharegpt(sample):
                return {
                    "messages": normalize_sharegpt_turns(sample.get("conversations"))
                }

            dataset = dataset.map(convert_sharegpt, remove_columns=columns)
            return filter_training_conversations(dataset, "ShareGPT")

        # QA format (from generation pipeline)
        if 'query' in columns and 'answer' in columns:
            system_prompt = self.config.get('system_prompt', '')

            dataset = dataset.filter(lambda s: s.get('query') and s.get('answer'))

            def convert_qa(sample):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                query = str(sample.get('query', ''))
                answer = str(sample.get('answer', ''))
                if sample.get('chunk_content'):
                    user_content = f"Context: {sample['chunk_content']}\n\nQuestion: {query}"
                else:
                    user_content = query
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": answer})
                return {"messages": messages}

            return dataset.map(convert_qa, remove_columns=columns)

        # Instruction/Alpaca format
        if 'instruction' in columns and 'output' in columns:
            system_prompt = self.config.get('system_prompt', '')

            dataset = dataset.filter(lambda s: s.get('instruction') and s.get('output'))

            def convert_instruction(sample):
                messages = []
                sys_msg = sample.get('system') or system_prompt
                if sys_msg:
                    messages.append({"role": "system", "content": str(sys_msg)})
                user_content = str(sample.get('instruction', ''))
                if sample.get('input'):
                    user_content += f"\n{sample['input']}"
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": str(sample.get('output', ''))})
                return {"messages": messages}

            return dataset.map(convert_instruction, remove_columns=columns)

        raise ValueError(
            f"Cannot detect SFT format from columns: {columns}. "
            "Expected one of: messages, conversations, query+answer, instruction+output"
        )

    def _normalize_preference_dataset(self, dataset: Dataset, tokenizer: Any) -> Dataset:
        """Normalize preference datasets via the shared DPO preprocessor."""
        max_length = self.config.get('max_length') or 2048
        preprocessor = DPOPreprocessor(
            tokenizer=tokenizer,
            max_length=max_length,
            config=self.config,
        )
        return preprocessor.preprocess(dataset)

    def _setup_metrics_callback(self, trainer, progress_callback, config, output_dir):
        """Attach MetricsCallback for progress & metrics tracking."""
        task_id = config.get('task_id')
        run_token = config.get("_run_token")

        if task_id:
            try:
                from ...monitoring.metrics_callback import MetricsCallback
                from ...storage.services.training_task_service import training_task_service

                def db_update_callback(metrics: dict):
                    try:
                        training_task_service.update_task_metrics(
                            task_id,
                            metrics,
                            run_token=run_token,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update task metrics in DB: {e}")

                def wrapped_progress(progress: float):
                    if progress_callback:
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
                self._add_simple_progress_callback(trainer, progress_callback)
        elif progress_callback:
            self._add_simple_progress_callback(trainer, progress_callback)

    @staticmethod
    def _add_simple_progress_callback(trainer, progress_callback):
        """Add a simple progress callback when MetricsCallback is unavailable."""
        from transformers import TrainerCallback

        class ProgressCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.max_steps and state.max_steps > 0:
                    train_progress = state.global_step / state.max_steps
                    progress_callback(20.0 + train_progress * 70.0)

        trainer.add_callback(ProgressCallback())

    def _build_result(self, model, save_dir, trainer, config) -> TrainingResult:
        """Build TrainingResult from trainer state."""
        log_history = getattr(getattr(trainer, 'state', None), 'log_history', None)
        final_metrics = extract_final_loss_metrics(log_history)

        return TrainingResult(
            model=model,
            save_dir=save_dir,
            final_metrics=final_metrics,
            task_id=config.get('task_id'),
        )

    def _get_common_training_args(self, config: Dict, output_dir: Path, eval_dataset) -> Dict:
        """Extract common training arguments from config."""
        gradient_checkpointing = config.get('gradient_checkpointing')
        if gradient_checkpointing is None:
            gradient_checkpointing = True

        args = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": config.get('per_device_train_batch_size', 4),
            "gradient_accumulation_steps": config.get('gradient_accumulation_steps', 1),
            "learning_rate": config.get('learning_rate', 2e-5),
            "num_train_epochs": config.get('num_train_epochs', 3),
            "warmup_ratio": config.get('warmup_ratio', 0.1),
            "logging_steps": config.get('logging_steps', 1),
            "save_steps": config.get('save_steps', 500),
            "save_total_limit": config.get('save_total_limit', 3),
            "bf16": config.get('bf16', True),
            "fp16": config.get('fp16', False),
            "report_to": config.get('report_to', 'none'),
            "seed": config.get('seed', 42),
            "gradient_checkpointing": gradient_checkpointing,
            "eval_strategy": config.get('eval_strategy', 'no') if eval_dataset else 'no',
            "eval_steps": config.get('eval_steps'),
            "save_strategy": config.get('save_strategy', 'steps'),
        }

        # DeepSpeed configuration
        ds_preset = config.get('deepspeed')
        if ds_preset and ds_preset in DEEPSPEED_PRESETS:
            args["deepspeed"] = DEEPSPEED_PRESETS[ds_preset]
            logger.info(f"Using DeepSpeed preset: {ds_preset}")

        return args

    def _load_and_normalize_datasets(
        self,
        mode: str,
        tokenizer: Optional[Any] = None,
    ):
        """Load datasets and normalize for the given mode (sft or preference).

        Returns:
            (train_dataset, eval_dataset) tuple
        """
        config = self.config
        dataset_configs = config.get('dataset_configs')
        if not dataset_configs:
            data_path = config.get('train_dataset_path')
            if not data_path:
                raise ValueError("No dataset_configs or train_dataset_path provided")
            dataset_configs = [{'path': data_path, 'split': 'train'}]

        train_dataset = self._load_datasets(dataset_configs, split_filter='train')
        if train_dataset is None:
            raise ValueError("No training datasets found (split='train')")

        eval_dataset = self._load_datasets(dataset_configs, split_filter='eval')

        if mode == 'sft':
            normalize_fn = self._normalize_sft_dataset
        else:
            if tokenizer is None:
                raise ValueError("Preference dataset normalization requires a tokenizer")

            def normalize_fn(dataset: Dataset) -> Dataset:
                return self._normalize_preference_dataset(dataset, tokenizer)

        train_dataset = normalize_fn(train_dataset)
        logger.info(f"Train dataset: {len(train_dataset)} samples, columns: {train_dataset.column_names}")

        if eval_dataset is not None:
            eval_dataset = normalize_fn(eval_dataset)
            logger.info(f"Eval dataset: {len(eval_dataset)} samples")

        return train_dataset, eval_dataset

    # ========== Training Methods ==========

    def _train_sft(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """SFT training using trl.SFTTrainer."""
        from trl import SFTConfig, SFTTrainer

        config = self.config
        output_dir = Path(config.get('output_dir') or './output')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model & tokenizer
        model, tokenizer = self._load_model_and_tokenizer()
        if progress_callback:
            progress_callback(10.0)

        # Load & normalize datasets
        train_dataset, eval_dataset = self._load_and_normalize_datasets('sft')
        if progress_callback:
            progress_callback(15.0)

        # Build config
        max_length = config.get('max_length') or 2048
        common_args = self._get_common_training_args(config, output_dir, eval_dataset)

        sft_config = SFTConfig(
            **common_args,
            max_length=max_length,
            dataset_text_field=None,  # Use messages format with chat template
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        self._setup_metrics_callback(trainer, progress_callback, config, output_dir)

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

        return self._build_result(model, save_dir, trainer, config)

    def _train_dpo(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """DPO training using trl.DPOTrainer."""
        from trl import DPOConfig, DPOTrainer

        config = self.config
        output_dir = Path(config.get('output_dir') or './output')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model & tokenizer
        model, tokenizer = self._load_model_and_tokenizer()
        if progress_callback:
            progress_callback(10.0)

        # Load & normalize datasets
        train_dataset, eval_dataset = self._load_and_normalize_datasets(
            'preference',
            tokenizer=tokenizer,
        )
        if progress_callback:
            progress_callback(15.0)

        # DPO-specific params
        rl_config = config.get('rl_config') or {}
        beta = rl_config.get('beta', 0.1)
        max_length = config.get('max_length') or 2048

        common_args = self._get_common_training_args(config, output_dir, eval_dataset)
        # DPO typically uses smaller LR
        if 'learning_rate' not in config:
            common_args['learning_rate'] = 5e-7

        dpo_config = DPOConfig(
            **common_args,
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_length // 2,
        )

        # For LoRA, DPO uses the base model as reference automatically
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        self._setup_metrics_callback(trainer, progress_callback, config, output_dir)

        resume_from = config.get('resume_from_checkpoint')
        self._run_trainer_train(trainer, resume_from, "DPO")

        if progress_callback:
            progress_callback(95.0)

        # Save
        save_dir = str(output_dir / "final")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

        if progress_callback:
            progress_callback(100.0)

        return self._build_result(model, save_dir, trainer, config)

    def _train_orpo(self, progress_callback: Optional[Callable] = None) -> TrainingResult:
        """ORPO training using trl.ORPOTrainer."""
        from trl import ORPOConfig, ORPOTrainer

        config = self.config
        output_dir = Path(config.get('output_dir') or './output')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model & tokenizer
        model, tokenizer = self._load_model_and_tokenizer()
        if progress_callback:
            progress_callback(10.0)

        # Load & normalize datasets
        train_dataset, eval_dataset = self._load_and_normalize_datasets(
            'preference',
            tokenizer=tokenizer,
        )
        if progress_callback:
            progress_callback(15.0)

        # ORPO-specific params
        rl_config = config.get('rl_config') or {}
        beta = rl_config.get('beta', 0.1)  # ORPO lambda parameter
        max_length = config.get('max_length') or 2048

        common_args = self._get_common_training_args(config, output_dir, eval_dataset)
        # ORPO uses moderate LR
        if 'learning_rate' not in config:
            common_args['learning_rate'] = 8e-6

        orpo_config = ORPOConfig(
            **common_args,
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_length // 2,
        )

        # ORPO does not need a reference model
        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        self._setup_metrics_callback(trainer, progress_callback, config, output_dir)

        resume_from = config.get('resume_from_checkpoint')
        self._run_trainer_train(trainer, resume_from, "ORPO")

        if progress_callback:
            progress_callback(95.0)

        # Save
        save_dir = str(output_dir / "final")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

        if progress_callback:
            progress_callback(100.0)

        return self._build_result(model, save_dir, trainer, config)
