"""
Training Metrics Callback for HuggingFace Transformers.

Integrates with Transformers Trainer to capture and store training metrics
during the training loop. Works with both SentenceTransformers and standard
HuggingFace Transformers.
"""

from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Dict, Any, Optional, Callable
import logging
import math

from .training_metrics import get_training_metrics_logger, cleanup_training_metrics_logger, TrainingMetricsLogger
from ..utils.strict_json import sanitize_json_value
from ..utils.training_metrics import extract_final_loss_metrics

logger = logging.getLogger(__name__)


class MetricsCallback(TrainerCallback):
    """
    Callback to capture training metrics and write to TrainingMetricsLogger.

    Captures:
    - Step-wise loss (train_loss, eval_loss)
    - Current step / total steps
    - Current epoch
    - Learning rate
    """

    def __init__(
        self,
        output_dir: str,
        task_id: str,
        progress_callback: Optional[Callable[[float], None]] = None,
        db_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            output_dir: Training output directory
            task_id: Training task ID
            progress_callback: Optional callback for progress updates (0-100%)
            db_update_callback: Optional callback to update database with current metrics
        """
        self.output_dir = output_dir
        self.task_id = task_id
        self.progress_callback = progress_callback
        self.db_update_callback = db_update_callback
        self.metrics_logger: Optional[TrainingMetricsLogger] = None
        self._last_logged_step = -1
        self._last_eval_logged_step = -1
        self._run_start_step = 0
        self._epoch_start_step = 0
        self._pending_epoch = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize metrics logger when training starts."""
        self._run_start_step = self._epoch_start_step = state.global_step
        self._pending_epoch = None
        self._last_logged_step = self._last_eval_logged_step = -1
        try:
            self.metrics_logger = get_training_metrics_logger(self.output_dir, self.task_id)

            # Save metadata
            self.metrics_logger.save_metadata({
                "max_steps": state.max_steps,
                "num_train_epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "logging_steps": args.logging_steps,
            })

            logger.info(f"MetricsCallback initialized for task {self.task_id}, max_steps={state.max_steps}")

            # Initial progress
            if self.progress_callback:
                self.progress_callback(5.0)

        except Exception as e:
            logger.error(f"Failed to initialize MetricsCallback: {e}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] = None, **kwargs):
        """Capture metrics on each logging step."""
        if not self.metrics_logger or logs is None:
            return

        # Avoid duplicate logging for same step (track train/eval separately)
        is_eval_log = any(key.startswith("eval_") for key in logs.keys())
        if is_eval_log:
            if state.global_step == self._last_eval_logged_step:
                return
            self._last_eval_logged_step = state.global_step
        else:
            if state.global_step == self._last_logged_step:
                return
            self._last_logged_step = state.global_step

        try:
            # Build metrics dict
            metrics = {}

            # Preserve diverged metric fields as null without polluting JSON/DB.
            # "loss" is reported during training, "train_loss" at end of training
            if "loss" in logs:
                metrics["train_loss"] = logs["loss"]
            elif "train_loss" in logs:
                metrics["train_loss"] = logs["train_loss"]
            if "eval_loss" in logs:
                metrics["eval_loss"] = logs["eval_loss"]
            if "learning_rate" in logs:
                metrics["learning_rate"] = logs["learning_rate"]

            # Add any evaluation metrics (eval_*)
            for key, value in logs.items():
                if key.startswith("eval_") and key != "eval_loss" and key != "eval_runtime":
                    metrics[key] = value

            metrics = sanitize_json_value(metrics)

            if metrics:
                # Calculate epoch from global_step
                epoch = state.epoch if state.epoch else None

                # Save to JSONL file
                self.metrics_logger.save_loss_record(
                    step=state.global_step,
                    metrics=metrics,
                    epoch=epoch
                )

                # Update database with current metrics
                if self.db_update_callback:
                    # Calculate current epoch (1-indexed for display)
                    # epoch=0.5 -> 1, epoch=1.0 -> 1, epoch=1.5 -> 2
                    current_epoch = None
                    if epoch is not None:
                        current_epoch = max(1, math.ceil(epoch)) if epoch > 0 else 1
                    update_data = {
                        "current_step": state.global_step,
                        "total_steps": state.max_steps,
                        "current_epoch": current_epoch,
                        "total_epochs": args.num_train_epochs,
                        **metrics
                    }
                    try:
                        self.db_update_callback(update_data)
                    except Exception as e:
                        logger.warning(f"Failed to update DB metrics: {e}")

        except Exception as e:
            logger.error(f"Error in on_log: {e}")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Update progress after each step."""
        if self.progress_callback and state.max_steps and state.max_steps > 0:
            # Calculate progress (reserve 5% for init and 5% for saving)
            train_progress = state.global_step / state.max_steps
            progress = 5.0 + train_progress * 90.0  # 5% - 95%
            self.progress_callback(progress)

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._finalize_pending_epoch(state)
        self._epoch_start_step = state.global_step

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Trainer emits epoch-strategy loss/evaluation logs AFTER this callback.
        # Flush at the next epoch begin (or train end) to include those records.
        self._pending_epoch = (
            max(1, math.ceil(state.epoch or 0)),
            self._epoch_start_step,
            state.global_step,
        )

    def _finalize_pending_epoch(self, state: TrainerState):
        if not self.metrics_logger or self._pending_epoch is None:
            return

        try:
            current_epoch, start_step, end_step = self._pending_epoch
            self._pending_epoch = None
            steps = end_step - start_step
            if steps <= 0:
                return
            epoch_metrics = {
                "total_steps_in_epoch": steps,
                "avg_train_loss": None,
            }
            last_loss_step = self._run_start_step
            covered_steps, weighted_loss = 0, 0.0
            for record in state.log_history or []:
                step = record.get("step")
                if not isinstance(step, int) or isinstance(step, bool):
                    continue
                if start_step < step <= end_step and "eval_loss" in record:
                    epoch_metrics["eval_loss"] = record["eval_loss"]
                if "loss" not in record or not last_loss_step < step <= end_step:
                    continue
                loss = record["loss"]
                # A logged loss averages the optimizer steps since the previous
                # loss log. Cross-epoch intervals cannot be split accurately.
                if (
                    last_loss_step >= start_step
                    and isinstance(loss, (int, float))
                    and not isinstance(loss, bool)
                    and math.isfinite(loss)
                ):
                    interval = step - last_loss_step
                    covered_steps += interval
                    weighted_loss += loss * interval
                last_loss_step = step
            # Do not mislabel a partial/invalid sample as a full epoch average.
            if covered_steps == steps:
                epoch_metrics["avg_train_loss"] = weighted_loss / steps
            self.metrics_logger.finalize_epoch(
                current_epoch,
                sanitize_json_value(epoch_metrics),
            )

        except Exception as e:
            logger.error(f"Error finalizing epoch metrics: {e}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Finalize metrics when training ends."""
        try:
            self._finalize_pending_epoch(state)
            final_metrics = {
                "total_steps": state.global_step,
                "total_epochs": args.num_train_epochs,
            }

            final_metrics.update(
                extract_final_loss_metrics(state.log_history) or {}
            )

            # Cleanup and finalize
            cleanup_training_metrics_logger(self.task_id, final_metrics)

            if self.progress_callback:
                self.progress_callback(100.0)

            logger.info(f"Training metrics finalized for task {self.task_id}")

        except Exception as e:
            logger.error(f"Error in on_train_end: {e}")
