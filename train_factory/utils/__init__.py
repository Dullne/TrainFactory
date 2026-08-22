"""
Utility functions for TrainFactory.

This package intentionally uses lazy exports to keep import side effects low.
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # common_utils
    "init_swanlab": (".common_utils", "init_swanlab"),
    "setup_logging": (".common_utils", "setup_logging"),
    # peft workaround
    "apply_peft_unboundlocal_workaround": (".peft_workaround", "apply_peft_unboundlocal_workaround"),
    # embedding recipe
    "EmbeddingRecipeSpec": (".embedding_recipe", "EmbeddingRecipeSpec"),
    "EmbeddingRecipeError": (".embedding_recipe", "EmbeddingRecipeError"),
    "detect_data_format": (".embedding_recipe", "detect_data_format"),
    "prepare_dataset_for_recipe": (".embedding_recipe", "prepare_dataset_for_recipe"),
    "build_embedding_loss": (".embedding_recipe", "build_embedding_loss"),
    "normalize_loss_name": (".embedding_recipe", "normalize_loss_name"),
    # evaluation (backward-compatible)
    "UnifiedEvaluator": ("..evaluation", "UnifiedEvaluator"),
    "create_evaluator": ("..evaluation", "create_evaluator"),
    "MetricRegistry": ("..evaluation", "MetricRegistry"),
    "MetricInfo": ("..evaluation", "MetricInfo"),
    "EvaluatorInfo": ("..evaluation", "EvaluatorInfo"),
    "MetricCategory": ("..evaluation", "MetricCategory"),
    "EvaluatorType": ("..evaluation", "EvaluatorType"),
    "TaskType": ("..evaluation", "TaskType"),
    "metric_registry": ("..evaluation", "metric_registry"),
    "get_metric_registry": ("..evaluation", "get_metric_registry"),
    "EvaluationResult": ("..evaluation", "EvaluationResult"),
    "EvaluationResultProcessor": ("..evaluation", "EvaluationResultProcessor"),
    "evaluation_result_processor": ("..evaluation", "evaluation_result_processor"),
    "get_evaluation_result_processor": ("..evaluation", "get_evaluation_result_processor"),
    "convert_to_standard_format": ("..evaluation", "convert_to_standard_format"),
    "split_by_source_id": ("..evaluation", "split_by_source_id"),
    # monitoring (backward-compatible)
    "TrainingMetricsLogger": ("..monitoring", "TrainingMetricsLogger"),
    "get_training_metrics_logger": ("..monitoring", "get_training_metrics_logger"),
    "cleanup_training_metrics_logger": ("..monitoring", "cleanup_training_metrics_logger"),
    "LifecycleManager": ("..monitoring", "LifecycleManager"),
    "lifecycle_manager": ("..monitoring", "lifecycle_manager"),
    "get_lifecycle_manager": ("..monitoring", "get_lifecycle_manager"),
    "ResourceMonitor": ("..monitoring", "ResourceMonitor"),
    "ResourceUsage": ("..monitoring", "ResourceUsage"),
    "resource_monitor": ("..monitoring", "resource_monitor"),
    "get_resource_monitor": ("..monitoring", "get_resource_monitor"),
}

__all__ = list(_EXPORTS.keys()) + [
    "LossManager",
    "get_loss_manager",
    "cleanup_loss_manager",
]


def __getattr__(name: str):
    if name == "LossManager":
        return __getattr__("TrainingMetricsLogger")
    if name == "get_loss_manager":
        return __getattr__("get_training_metrics_logger")
    if name == "cleanup_loss_manager":
        return __getattr__("cleanup_training_metrics_logger")

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
