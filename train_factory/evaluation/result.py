"""
标准化评估结果处理

提供统一的评估结果包装和处理功能
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .metric_registry import get_metric_registry, MetricRegistry


@dataclass
class EvaluationResult:
    """标准化评估结果"""
    source_id: str                    # 数据源ID
    dataset_name: str                # 数据集名称
    evaluator_name: str              # 评估器名称
    step: int                        # 训练步数
    epoch: Optional[float] = None    # 训练轮数
    metrics: Dict[str, float] = None # 评估指标
    metadata: Dict[str, Any] = None  # 元数据

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}

    def add_metric(self, name: str, value: float):
        """添加评估指标"""
        self.metrics[name] = value

    def get_standardized_metrics(self) -> Dict[str, float]:
        """获取标准化的指标名称 (eval_{source_id}_{metric})"""
        standardized = {}
        for metric_name, value in self.metrics.items():
            standardized_name = f"eval_{self.source_id}_{metric_name}"
            standardized[standardized_name] = value
        return standardized

    def get_frontend_format(self, registry: MetricRegistry = None) -> Dict[str, Any]:
        """获取前端格式的数据"""
        if registry is None:
            registry = get_metric_registry()

        # 获取元数据
        metric_names = list(self.metrics.keys())
        frontend_metadata = registry.get_frontend_metadata(metric_names)

        return {
            "step": self.step,
            "epoch": self.epoch,
            "source_id": self.source_id,
            "dataset_name": self.dataset_name,
            "evaluator_name": self.evaluator_name,
            "metrics": self.metrics,
            "standardized_metrics": self.get_standardized_metrics(),
            "metadata": {
                **self.metadata,
                **frontend_metadata
            }
        }


class EvaluationResultProcessor:
    """评估结果处理器"""

    def __init__(self):
        self.registry = get_metric_registry()

    def extract_evaluation_results_from_logs(self, loss_data: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """从loss日志数据中提取评估结果"""
        results = []

        for record in loss_data:
            step = record.get('step', 0)
            epoch = record.get('epoch')

            # 按数据源分组指标
            source_metrics = self._group_metrics_by_source(record)

            for source_id, metrics in source_metrics.items():
                if metrics:  # 只处理有指标的数据源
                    # 推断评估器信息
                    evaluator_name = self._infer_evaluator_name(metrics)

                    result = EvaluationResult(
                        source_id=source_id,
                        dataset_name=f"source_{source_id}",  # 默认名称，后续会被映射
                        evaluator_name=evaluator_name,
                        step=step,
                        epoch=epoch,
                        metrics=metrics
                    )

                    results.append(result)

        return results

    def _group_metrics_by_source(self, record: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """按数据源分组指标"""
        source_metrics = {}

        for key, value in record.items():
            if key.startswith('eval_') and isinstance(value, (int, float)):
                # 解析 eval_{source_id}_{metric_name} 格式
                parts = key[5:].split('_')  # 去掉 'eval_' 前缀
                if len(parts) >= 2 and parts[0].isdigit():
                    source_id = parts[0]
                    metric_name = '_'.join(parts[1:])

                    if source_id not in source_metrics:
                        source_metrics[source_id] = {}

                    source_metrics[source_id][metric_name] = value

        return source_metrics

    def _infer_evaluator_name(self, metrics: Dict[str, float]) -> str:
        """从指标推断评估器名称"""
        metric_names = list(metrics.keys())
        evaluator_type = self.registry.infer_evaluator_type_from_metrics(metric_names)

        if evaluator_type:
            # 查找匹配的评估器
            for evaluator_name, evaluator_info in self.registry.evaluators.items():
                if evaluator_info.evaluator_type == evaluator_type:
                    return evaluator_name

        return "UnknownEvaluator"

    def apply_dataset_mapping(self, results: List[EvaluationResult],
                            source_mapping: Dict[str, str]) -> List[EvaluationResult]:
        """应用数据集名称映射"""
        for result in results:
            if result.source_id in source_mapping:
                result.dataset_name = source_mapping[result.source_id]

        return results

    def convert_to_frontend_format(self, results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """转换为前端格式"""
        return [result.get_frontend_format(self.registry) for result in results]

    def enhance_loss_data_with_metadata(self, loss_data: List[Dict[str, Any]],
                                      source_mapping: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """为loss数据增强元数据"""
        enhanced_data = []

        for record in loss_data:
            enhanced_record = record.copy()

            # 提取评估指标
            eval_metrics = {}
            for key, value in record.items():
                if key.startswith('eval_') and isinstance(value, (int, float)):
                    eval_metrics[key] = value

            if eval_metrics:
                # 获取元数据
                metric_names = []
                for key in eval_metrics.keys():
                    # 从 eval_{source_id}_{metric_name} 提取 metric_name
                    parts = key[5:].split('_')  # 去掉 'eval_' 前缀
                    if len(parts) >= 2:
                        metric_name = '_'.join(parts[1:])
                        metric_names.append(metric_name)

                if metric_names:
                    frontend_metadata = self.registry.get_frontend_metadata(metric_names)
                    enhanced_record['evaluation_metadata'] = frontend_metadata

            enhanced_data.append(enhanced_record)

        return enhanced_data


# 全局处理器实例
evaluation_result_processor = EvaluationResultProcessor()


def get_evaluation_result_processor() -> EvaluationResultProcessor:
    """获取全局评估结果处理器"""
    return evaluation_result_processor


def convert_to_standard_format(eval_results: Dict[str, Any], data_source_mapping: Dict[str, str] = None) -> Dict[str, Any]:
    """
    将所有指标转换为标准格式 eval_{source_id}_{metric}

    Args:
        eval_results: 原始评估结果
        data_source_mapping: 数据集名称到source_id的映射

    Returns:
        转换后的标准格式结果
    """
    converted = {}

    # 处理数据集名称映射
    dataset_to_source = {}
    if data_source_mapping:
        for full_name, source_id in data_source_mapping.items():
            if '/' in full_name:
                dataset_name = full_name.split('/')[-1]
                dataset_to_source[dataset_name] = source_id
            dataset_to_source[full_name] = source_id

    for key, value in eval_results.items():
        if key.startswith('eval_'):
            # 检查是否已经是标准格式 eval_{数字}_*
            remaining = key[5:]  # 去掉'eval_'
            parts = remaining.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                # 已经是标准格式，直接保留
                converted[key] = value
            else:
                # 非标准格式，需要转换
                converted_key = None
                for dataset_name, source_id in dataset_to_source.items():
                    if dataset_name in key:
                        # 提取指标名
                        dataset_pos = key.find(dataset_name)
                        metric_suffix = key[dataset_pos + len(dataset_name):]
                        if metric_suffix.startswith('_'):
                            metric_suffix = metric_suffix[1:]

                        converted_key = f"eval_{source_id}_{metric_suffix}"
                        break

                if converted_key:
                    converted[converted_key] = value
                else:
                    # 无法转换，保持原样（如统一指标）
                    converted[key] = value
        else:
            # 非eval字段，直接保留
            converted[key] = value

    return converted


def split_by_source_id(eval_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    按数据源ID分离评估结果

    Args:
        eval_results: 标准格式的评估结果

    Returns:
        按source_id分组的结果
    """
    source_results = {}
    unified_metrics = {}

    # 收集所有source_id
    source_ids = set()
    for key in eval_results.keys():
        if key.startswith('eval_'):
            parts = key[5:].split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                source_ids.add(parts[0])

    for key, value in eval_results.items():
        if key.startswith('eval_'):
            parts = key[5:].split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                # 标准格式：分配到对应数据源
                source_id = parts[0]
                if source_id not in source_results:
                    source_results[source_id] = {}
                source_results[source_id][key] = value
            else:
                # 统一指标：添加到所有数据源
                metric_name = key[5:]  # 去掉eval_前缀
                unified_metrics[metric_name] = value
        else:
            # 非eval字段（如epoch）：添加到所有数据源
            unified_metrics[key] = value

    # 将统一指标添加到所有数据源
    for source_id in source_results.keys():
        source_results[source_id].update(unified_metrics)

    return source_results
