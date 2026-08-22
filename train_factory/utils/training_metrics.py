"""Helpers for extracting terminal training metrics from trainer logs."""

import math
from typing import Any, Dict, Iterable, Mapping, Optional


def _valid_metric_value(value: Any) -> bool:
    """Reject None/NaN/Inf/non-numeric values so a diverged final step can't
    overwrite an earlier valid loss with a dirty value."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def extract_final_loss_metrics(
    log_history: Optional[Iterable[Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Return the latest train and eval losses, even when logged separately."""
    history = list(log_history or [])
    metrics: Dict[str, Any] = {}

    for log in reversed(history):
        if "final_train_loss" not in metrics:
            if "train_loss" in log and _valid_metric_value(log["train_loss"]):
                metrics["final_train_loss"] = log["train_loss"]
            elif "loss" in log and _valid_metric_value(log["loss"]):
                metrics["final_train_loss"] = log["loss"]
        if (
            "final_eval_loss" not in metrics
            and "eval_loss" in log
            and _valid_metric_value(log["eval_loss"])
        ):
            metrics["final_eval_loss"] = log["eval_loss"]
        if "final_train_loss" in metrics and "final_eval_loss" in metrics:
            break

    return metrics or None
