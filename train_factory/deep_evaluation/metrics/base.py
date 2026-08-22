"""
深度评估指标基类

定义评估指标的抽象接口，所有具体指标都需要继承此基类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class MetricCategory(str, Enum):
    """指标类别"""
    RETRIEVAL = "retrieval"  # 检索质量
    GENERATION = "generation"  # 生成质量
    FAITHFULNESS = "faithfulness"  # 事实一致性
    RELEVANCY = "relevancy"  # 相关性


@dataclass
class MetricResult:
    """评估结果"""
    score: float  # 0.0 - 1.0
    reason: str  # 评分理由
    details: Optional[Dict[str, Any]] = None  # 详细信息
    skipped: bool = False  # 评估是否因失败而跳过（不计入统计）

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "score": self.score,
            "reason": self.reason,
            "details": self.details,
        }
        if self.skipped:
            result["skipped"] = True
        return result


@dataclass
class EvaluationSample:
    """评估样本"""
    input: str  # 用户问题/Query
    expected_output: Optional[str] = None  # 期望的答案
    actual_output: Optional[str] = None  # 实际生成的答案
    retrieval_context: List[str] = field(default_factory=list)  # 检索到的上下文
    metadata: Optional[Dict[str, Any]] = None  # 其他元数据

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "retrieval_context": self.retrieval_context,
            "metadata": self.metadata,
        }


class BaseMetric(ABC):
    """
    评估指标基类

    所有深度评估指标都需要继承此类并实现 evaluate 方法。
    """

    # 指标名称（子类必须定义）
    name: str = ""
    # 指标描述
    description: str = ""
    # 指标类别
    category: MetricCategory = MetricCategory.RELEVANCY
    # 是否需要 LLM 评估
    requires_llm: bool = True
    # 是否需要期望输出
    requires_expected_output: bool = False
    # 是否需要实际输出
    requires_actual_output: bool = False
    # 是否需要检索上下文
    requires_retrieval_context: bool = False

    def __init__(self, **kwargs):
        """
        初始化指标

        Args:
            **kwargs: 指标特定的配置参数
        """
        self.config = kwargs

    @abstractmethod
    async def evaluate(
        self,
        sample: EvaluationSample,
        llm_client: Optional[Any] = None,
        **kwargs,
    ) -> MetricResult:
        """
        评估单个样本

        Args:
            sample: 评估样本
            llm_client: LLM 客户端（如果 requires_llm=True 则必须提供）
            **kwargs: 指标特定的额外参数

        Returns:
            MetricResult: 评估结果
        """
        pass

    def validate_sample(self, sample: EvaluationSample) -> None:
        """
        验证样本是否满足指标要求

        Args:
            sample: 评估样本

        Raises:
            ValueError: 如果样本不满足要求
        """
        if self.requires_expected_output and not sample.expected_output:
            raise ValueError(f"Metric '{self.name}' requires expected_output")
        if self.requires_actual_output and not sample.actual_output:
            raise ValueError(f"Metric '{self.name}' requires actual_output")
        if self.requires_retrieval_context and not sample.retrieval_context:
            raise ValueError(f"Metric '{self.name}' requires retrieval_context")

    def get_info(self) -> Dict[str, Any]:
        """获取指标信息"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "requires_llm": self.requires_llm,
            "requires_expected_output": self.requires_expected_output,
            "requires_actual_output": self.requires_actual_output,
            "requires_retrieval_context": self.requires_retrieval_context,
        }


class MetricRegistry:
    """
    指标注册表

    管理所有可用的评估指标。
    """

    _metrics: Dict[str, type] = {}

    @classmethod
    def register(cls, metric_class: type) -> type:
        """
        注册指标

        用法:
            @MetricRegistry.register
            class MyMetric(BaseMetric):
                name = "my_metric"
                ...
        """
        if not issubclass(metric_class, BaseMetric):
            raise TypeError(f"{metric_class} must be a subclass of BaseMetric")
        if not metric_class.name:
            raise ValueError(f"{metric_class} must define a 'name' attribute")
        cls._metrics[metric_class.name] = metric_class
        return metric_class

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """获取指标类"""
        return cls._metrics.get(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[BaseMetric]:
        """创建指标实例"""
        metric_class = cls.get(name)
        if metric_class:
            return metric_class(**kwargs)
        return None

    @classmethod
    def list_metrics(cls) -> List[Dict[str, Any]]:
        """列出所有注册的指标"""
        return [
            metric_class(**{}).get_info()
            for metric_class in cls._metrics.values()
        ]

    @classmethod
    def get_metrics_by_category(cls, category: MetricCategory) -> List[str]:
        """获取指定类别的指标名称列表"""
        return [
            name
            for name, metric_class in cls._metrics.items()
            if metric_class.category == category
        ]
