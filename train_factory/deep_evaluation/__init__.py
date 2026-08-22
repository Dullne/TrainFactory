"""
深度评估模块

基于 DeepEval 框架的模型评估能力，支持检索相关指标与答案质量评估。
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # evaluator
    "DeepEvaluator": (".evaluator", "DeepEvaluator"),
    "EvaluationResult": (".evaluator", "EvaluationResult"),
    "BatchEvaluationResult": (".evaluator", "BatchEvaluationResult"),
    "create_sample": (".evaluator", "create_sample"),
    # metrics
    "BaseMetric": (".metrics", "BaseMetric"),
    "MetricResult": (".metrics", "MetricResult"),
    "MetricCategory": (".metrics", "MetricCategory"),
    "MetricRegistry": (".metrics", "MetricRegistry"),
    "EvaluationSample": (".metrics", "EvaluationSample"),
    "ContextualPrecisionMetric": (".metrics", "ContextualPrecisionMetric"),
    "ContextualRecallMetric": (".metrics", "ContextualRecallMetric"),
    "ContextualRelevancyMetric": (".metrics", "ContextualRelevancyMetric"),
    "AnswerRelevancyMetric": (".metrics", "AnswerRelevancyMetric"),
    "FaithfulnessMetric": (".metrics", "FaithfulnessMetric"),
    # llm judge
    "LLMConfig": (".llm_judge", "LLMConfig"),
    "LLMJudge": (".llm_judge", "LLMJudge"),
    "LLMJudgePool": (".llm_judge", "LLMJudgePool"),
    "create_llm_judge": (".llm_judge", "create_llm_judge"),
    "create_llm_judge_from_dict": (".llm_judge", "create_llm_judge_from_dict"),
    # task runner
    "cancel_deep_evaluation": (".task_runner", "cancel_deep_evaluation"),
    "clear_deep_cancellation": (".task_runner", "_cleanup_cancelled"),
    "run_deep_evaluation_task": (".deep_evaluation_runner", "run_deep_evaluation_task"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
