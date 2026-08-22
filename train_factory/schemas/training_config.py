"""
Training Parameters Module for TrainFactory.

Defines training configuration using Pydantic v2.
Separates official HuggingFace training arguments from custom parameters.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# === Enumerations ===

class ReportTo(str, Enum):
    """Report tool enumeration."""
    NONE = "none"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    SWANLAB = "swanlab"


class OptimType(str, Enum):
    """Optimizer type enumeration."""
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAFACTOR = "adafactor"


class LRSchedulerType(str, Enum):
    """Learning rate scheduler type enumeration."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class EvalStrategy(str, Enum):
    """Evaluation strategy enumeration."""
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SaveStrategy(str, Enum):
    """Save strategy enumeration."""
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class LoggingStrategy(str, Enum):
    """Logging strategy enumeration."""
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EmbeddingDataFormat(str, Enum):
    """Embedding training data format enumeration."""
    AUTO = "auto"
    MNR_PAIRS = "mnr_pairs"
    MNR_TRIPLETS = "mnr_triplets"
    MNR_MULTI_NEGATIVES = "mnr_multi_negatives"
    COSINE_PAIRS = "cosine_pairs"


class EmbeddingLossName(str, Enum):
    """Embedding training loss function enumeration."""
    AUTO = "auto"
    MULTIPLE_NEGATIVES = "MultipleNegativesRankingLoss"
    COSINE = "CosineSimilarityLoss"
    COSENT = "CoSENTLoss"
    DYNAMIC_EXPLICIT_NEGATIVES = "DynamicExplicitNegativesRankingLoss"


# === Configuration Classes ===

class LoRAConfig(BaseModel):
    """LoRA fine-tuning configuration."""

    model_config = ConfigDict(extra='forbid')

    use_lora: bool = Field(default=False, description="Enable LoRA fine-tuning")
    r: int = Field(default=16, ge=1, le=512, description="LoRA rank parameter")
    lora_alpha: Optional[int] = Field(default=None, ge=1, le=1024, description="LoRA alpha parameter")
    lora_dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="LoRA dropout ratio")
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj"],
        description="LoRA target modules"
    )
    task_type: Optional[str] = Field(default=None, description="LoRA task type")
    bias: str = Field(default="none", description="Bias update type")

    def to_peft_config_params(self) -> Dict[str, Any]:
        """Convert to PEFT LoraConfig parameters."""
        params = {
            "r": self.r,
            "lora_alpha": self.lora_alpha or (self.r * 2),
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "target_modules": self.target_modules
        }
        if self.task_type is not None:
            from peft import TaskType
            params["task_type"] = getattr(TaskType, self.task_type, None)
        return params

    def model_post_init(self, __context) -> None:
        """Set dependent defaults after initialization."""
        if self.lora_alpha is None:
            object.__setattr__(self, 'lora_alpha', self.r * 2)


class HuggingFaceTrainingArgs(BaseModel):
    """
    HuggingFace official training arguments.
    These are passed directly to CrossEncoder/SentenceTransformerTrainingArguments.
    """

    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    # === Basic Training Parameters ===
    output_dir: Optional[str] = Field(default=None, description="Output directory")
    overwrite_output_dir: bool = Field(default=False, description="Overwrite output directory")
    num_train_epochs: float = Field(default=3.0, gt=0, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=8, ge=1, description="Batch size per device for training")
    per_device_eval_batch_size: int = Field(default=8, ge=1, description="Batch size per device for evaluation")

    # === Learning Rate and Optimizer ===
    learning_rate: float = Field(default=5e-5, gt=0, description="Learning rate")
    warmup_ratio: float = Field(default=0.0, ge=0, le=1, description="Warmup ratio")
    warmup_steps: int = Field(default=0, ge=0, description="Warmup steps")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    adam_beta1: float = Field(default=0.9, ge=0, le=1, description="Adam beta1")
    adam_beta2: float = Field(default=0.999, ge=0, le=1, description="Adam beta2")
    adam_epsilon: float = Field(default=1e-8, gt=0, description="Adam epsilon")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Max gradient norm")

    # === Training Strategy ===
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    eval_strategy: EvalStrategy = Field(default=EvalStrategy.NO, description="Evaluation strategy")
    save_strategy: SaveStrategy = Field(default=SaveStrategy.STEPS, description="Save strategy")
    logging_strategy: LoggingStrategy = Field(default=LoggingStrategy.STEPS, description="Logging strategy")
    logging_steps: float = Field(default=1, gt=0, description="Logging steps")
    max_steps: int = Field(default=-1, description="Max training steps, -1 means use epochs")

    # === Step Parameters ===
    eval_steps: Optional[float] = Field(default=None, gt=0, description="Evaluation steps")
    save_steps: float = Field(default=500, gt=0, description="Save steps")
    save_total_limit: Optional[int] = Field(default=None, ge=1, description="Max number of checkpoints to keep")

    # === Data Loading ===
    dataloader_num_workers: int = Field(default=0, ge=0, description="Number of data loader workers")
    dataloader_drop_last: bool = Field(default=False, description="Drop last incomplete batch")

    # === Mixed Precision ===
    bf16: bool = Field(default=False, description="Use bf16 mixed precision")
    fp16: bool = Field(default=False, description="Use fp16 mixed precision")
    gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")

    # === Model Evaluation and Saving ===
    load_best_model_at_end: Optional[bool] = Field(default=False, description="Load best model at end")
    metric_for_best_model: Optional[str] = Field(default=None, description="Metric for best model")
    greater_is_better: Optional[bool] = Field(default=None, description="Whether greater metric is better")

    # === DeepSpeed ===
    deepspeed: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="DeepSpeed config dict or path")

    # === System ===
    seed: int = Field(default=42, description="Random seed")
    disable_tqdm: Optional[bool] = Field(default=None, description="Disable progress bar")
    report_to: Optional[Union[ReportTo, str, List[str]]] = Field(default=ReportTo.NONE, description="Report tool")
    run_name: Optional[str] = Field(default=None, description="Run name")

    # === Learning Rate Scheduler ===
    lr_scheduler_type: LRSchedulerType = Field(default=LRSchedulerType.LINEAR, description="LR scheduler type")

    # === Optimizer ===
    optim: OptimType = Field(default=OptimType.ADAMW_TORCH, description="Optimizer type")

    # === Logging ===
    logging_dir: Optional[str] = Field(default=None, description="Logging directory")
    resume_from_checkpoint: Optional[str] = Field(default=None, description="Resume from checkpoint")

    @field_validator('report_to')
    @classmethod
    def validate_report_to(cls, v):
        """Validate report_to setting."""
        if v is None:
            return v
        if isinstance(v, ReportTo):
            return v.value
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot be enabled simultaneously")

    def to_dict(self) -> Dict[str, Any]:
        """Get training arguments dictionary for HuggingFace TrainingArguments."""
        training_args_dict = self.model_dump(exclude_unset=True)

        # Handle report_to parameter
        if 'report_to' in training_args_dict:
            report_to_value = training_args_dict['report_to']
            if hasattr(report_to_value, 'value'):
                report_to_value = report_to_value.value
            if isinstance(report_to_value, str):
                if report_to_value in ['none', '']:
                    training_args_dict['report_to'] = []
                else:
                    training_args_dict['report_to'] = [report_to_value]
            elif isinstance(report_to_value, list):
                filtered_list = [item for item in report_to_value if item not in ['none', '']]
                training_args_dict['report_to'] = filtered_list if filtered_list else []
        else:
            training_args_dict['report_to'] = []

        return training_args_dict


class SystemConfig(BaseModel):
    """
    System internal configuration - not passed to official TrainingArguments.
    """

    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    # === SwanLab Configuration ===
    swanlab_api_key: Optional[str] = Field(default=None, description="SwanLab API key")
    swanlab_project: Optional[str] = Field(default=None, description="SwanLab project name")

    # === Dataset Sampling ===
    train_sample_size: Optional[int] = Field(default=-1, ge=-1, description="Train dataset sample limit, -1 for no limit")
    eval_sample_size: Optional[int] = Field(default=-1, ge=-1, description="Eval dataset sample limit, -1 for no limit")
    test_sample_size: Optional[int] = Field(default=-1, ge=-1, description="Test dataset sample limit, -1 for no limit")

    # === Embedding Training Config ===
    embedding_data_format: Optional[str] = Field(default="auto", description="Embedding data format")
    embedding_loss_name: Optional[str] = Field(default="auto", description="Embedding loss function name")
    embedding_loss_temperature: Optional[float] = Field(default=None, description="Loss temperature parameter")
    embedding_label_normalization: Optional[str] = Field(default="auto", description="Label normalization strategy")

    # === Path Configuration ===
    train_dataset_path: Optional[str] = Field(default=None, description="Train dataset path")
    base_model_path: Optional[str] = Field(default=None, description="Base model path")
    model_type: Optional[str] = Field(default=None, description="Model type")
    task_id: Optional[str] = Field(default=None, description="Task ID")
    task_name: Optional[str] = Field(default=None, description="Task name")
    HF_subset: Optional[str] = Field(default=None, description="HuggingFace dataset subset")
    device: Optional[str] = Field(default=None, description="Device info")

    # === Model Parameters ===
    max_seq_length: Optional[int] = Field(default=512, gt=0, description="Max sequence length")


class TrainingConfig(BaseModel):
    """
    Complete training configuration combining all config classes.
    """

    model_config = ConfigDict(
        extra='allow',
        validate_assignment=True,
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    training_args: HuggingFaceTrainingArgs = Field(default_factory=HuggingFaceTrainingArgs)
    system_config: SystemConfig = Field(default_factory=SystemConfig)
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)

    def get_training_args_dict(self) -> Dict[str, Any]:
        """Get official training arguments dictionary."""
        return self.training_args.to_dict()

    def get_custom_params_dict(self) -> Dict[str, Any]:
        """Get custom parameters dictionary."""
        custom_dict = self.system_config.model_dump(exclude_unset=True)
        custom_dict['lora_config'] = self.lora_config.model_dump(exclude_unset=True)
        return custom_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from a flat dictionary."""
        obj_copy = config_dict.copy()

        # Handle LoRA config
        if 'lora_config' in obj_copy and isinstance(obj_copy['lora_config'], dict):
            lora_dict = obj_copy.pop('lora_config')
            lora_dict = {k: v for k, v in lora_dict.items() if v is not None}
        else:
            lora_dict = {}
            lora_keys = ['use_lora', 'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules']
            for key in lora_keys:
                if key in obj_copy:
                    value = obj_copy.pop(key)
                    if value is not None:
                        if key == 'lora_r':
                            lora_dict['r'] = value
                        elif key == 'lora_target_modules':
                            lora_dict['target_modules'] = value
                        else:
                            lora_dict[key] = value

        # Separate parameters by field names
        training_fields = set(HuggingFaceTrainingArgs.model_fields.keys())
        system_fields = set(SystemConfig.model_fields.keys())

        training_dict = {k: v for k, v in obj_copy.items() if k in training_fields}
        system_dict = {k: v for k, v in obj_copy.items() if k in system_fields}

        # Create sub-config objects
        training_args = HuggingFaceTrainingArgs.model_validate(training_dict)
        system_config = SystemConfig.model_validate(system_dict)
        lora_config = LoRAConfig.model_validate(lora_dict)

        return cls(
            training_args=training_args,
            system_config=system_config,
            lora_config=lora_config
        )


class TrainingParametersManager:
    """Training parameters manager."""

    def __init__(self):
        self._training_config: Optional[TrainingConfig] = None

    def load_from_config(self, config_dict: Dict[str, Any]) -> 'TrainingParametersManager':
        """Load parameters from configuration dictionary."""
        try:
            self._training_config = TrainingConfig.from_dict(config_dict)
            logger.info("Training parameters loaded successfully")
        except Exception as e:
            logger.error(f"Training parameters validation failed: {e}")
            raise ValueError(f"Invalid training parameters: {e}")
        return self

    def get_training_args_dict(self) -> Dict[str, Any]:
        """Get official training arguments dictionary."""
        if self._training_config is None:
            raise ValueError("Training parameters not initialized, call load_from_config first")
        return self._training_config.get_training_args_dict()

    def get_custom_params_dict(self) -> Dict[str, Any]:
        """Get custom parameters dictionary."""
        if self._training_config is None:
            raise ValueError("Training parameters not initialized, call load_from_config first")
        return self._training_config.get_custom_params_dict()

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration object."""
        if self._training_config is None:
            raise ValueError("Training parameters not initialized, call load_from_config first")
        return self._training_config
