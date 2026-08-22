"""
Evaluation utilities for TrainFactory.

The API server imports task-management helpers from this package during startup.
Keep heavier evaluator implementations lazy so dashboard startup does not depend
on optional sentence-transformers evaluator symbols.
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # evaluator
    "UnifiedEvaluator": (".evaluator", "UnifiedEvaluator"),
    "create_evaluator": (".evaluator", "create_evaluator"),
    # metric_registry
    "MetricRegistry": (".metric_registry", "MetricRegistry"),
    "MetricInfo": (".metric_registry", "MetricInfo"),
    "EvaluatorInfo": (".metric_registry", "EvaluatorInfo"),
    "MetricCategory": (".metric_registry", "MetricCategory"),
    "EvaluatorType": (".metric_registry", "EvaluatorType"),
    "TaskType": (".metric_registry", "TaskType"),
    "metric_registry": (".metric_registry", "metric_registry"),
    "get_metric_registry": (".metric_registry", "get_metric_registry"),
    # result
    "EvaluationResult": (".result", "EvaluationResult"),
    "EvaluationResultProcessor": (".result", "EvaluationResultProcessor"),
    "evaluation_result_processor": (".result", "evaluation_result_processor"),
    "get_evaluation_result_processor": (".result", "get_evaluation_result_processor"),
    "convert_to_standard_format": (".result", "convert_to_standard_format"),
    "split_by_source_id": (".result", "split_by_source_id"),
    # evaluation task service
    "EvaluationTaskService": ("..storage.services.evaluation_task_service", "EvaluationTaskService"),
    "evaluation_task_service": ("..storage.services.evaluation_task_service", "evaluation_task_service"),
    # evaluation runner
    "run_evaluation_task": (".evaluation_runner", "run_evaluation_task"),
    "cancel_evaluation": (".evaluation_runner", "cancel_evaluation"),
    "is_evaluation_cancelled": (".evaluation_runner", "is_cancelled"),
    "clear_evaluation_cancellation": (".evaluation_runner", "_cleanup_cancelled"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
