"""
深度评估指标

提供类似 DeepEval 的深度评估指标，包括：
- ContextualPrecision: 上下文精准度
- ContextualRecall: 上下文召回率
- ContextualRelevancy: 上下文相关性
- AnswerRelevancy: 答案相关性
- Faithfulness: 事实一致性
"""

from .base import (
    BaseMetric,
    MetricResult,
    MetricCategory,
    MetricRegistry,
    EvaluationSample,
)
from .contextual_precision import ContextualPrecisionMetric
from .contextual_recall import ContextualRecallMetric
from .contextual_relevancy import ContextualRelevancyMetric
from .answer_relevancy import AnswerRelevancyMetric
from .faithfulness import FaithfulnessMetric

__all__ = [
    # 基类
    "BaseMetric",
    "MetricResult",
    "MetricCategory",
    "MetricRegistry",
    "EvaluationSample",
    # 具体指标
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
]
