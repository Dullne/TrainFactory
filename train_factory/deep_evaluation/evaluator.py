"""
深度评估器核心

提供单样本评估和批量评估功能。
"""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .metrics import (
    BaseMetric,
    MetricResult,
    MetricRegistry,
    EvaluationSample,
)


class _ConcurrencyLimitedLLMClient:
    """Limit all chat calls made by one evaluator across threads and event loops."""

    _PERMIT_POLL_INTERVAL_SECONDS = 0.01

    def __init__(self, client: Any, concurrency: int) -> None:
        self._client = client
        self._permits = threading.BoundedSemaphore(concurrency)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def _acquire_permit(self) -> None:
        while not self._permits.acquire(blocking=False):
            await asyncio.sleep(self._PERMIT_POLL_INTERVAL_SECONDS)

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        await self._acquire_permit()
        try:
            return await self._client.chat(*args, **kwargs)
        finally:
            self._permits.release()


@dataclass
class EvaluationResult:
    """单样本评估结果"""
    sample: EvaluationSample
    metric_results: Dict[str, MetricResult]  # metric_name -> result
    overall_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "metric_results": {
                name: result.to_dict()
                for name, result in self.metric_results.items()
            },
            "overall_score": self.overall_score,
        }


@dataclass
class BatchEvaluationResult:
    """批量评估结果"""
    results: List[EvaluationResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


class DeepEvaluator:
    """
    深度评估器

    支持使用多个指标评估模型输出的质量。
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        llm_client: Optional[Any] = None,
        concurrency: int = 5,
        chunk_eval_mode: str = "batch",
    ):
        """
        初始化评估器

        Args:
            metrics: 要使用的指标名称列表，默认使用所有注册的指标
            llm_client: LLM 客户端，用于需要 LLM 评估的指标
            concurrency: 批量评估时的并发数
        """
        self.llm_client = llm_client
        self.concurrency = max(1, int(concurrency))
        self._limited_llm_client = (
            _ConcurrencyLimitedLLMClient(llm_client, self.concurrency)
            if llm_client is not None
            else None
        )
        self.chunk_eval_mode = chunk_eval_mode

        # 初始化指标
        self.metrics: Dict[str, BaseMetric] = {}
        if metrics:
            for metric_name in metrics:
                metric = MetricRegistry.create(metric_name)
                if metric:
                    self.metrics[metric_name] = metric
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
        else:
            # 使用所有注册的指标
            for info in MetricRegistry.list_metrics():
                metric = MetricRegistry.create(info["name"])
                if metric:
                    self.metrics[info["name"]] = metric

    async def evaluate(
        self,
        sample: EvaluationSample,
        metrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        评估单个样本

        Args:
            sample: 评估样本
            metrics: 要使用的指标列表（可选，覆盖初始化时的设置）

        Returns:
            EvaluationResult: 评估结果
        """
        # 确定使用的指标
        metrics_to_use = self.metrics
        if metrics:
            metrics_to_use = {
                name: self.metrics[name]
                for name in metrics
                if name in self.metrics
            }

        # 并行执行所有指标评估
        metric_results: Dict[str, MetricResult] = {}
        tasks = []

        for name, metric in metrics_to_use.items():
            tasks.append(self._evaluate_metric(name, metric, sample))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(metrics_to_use.keys(), results):
            if isinstance(result, Exception):
                metric_results[name] = MetricResult(
                    score=0.0,
                    reason=f"Evaluation error: {str(result)}",
                    details={"error": str(result)},
                    skipped=True,
                )
            else:
                metric_results[name] = result

        # 计算综合分数（简单平均，排除 skipped）
        scores = [r.score for r in metric_results.values() if not r.skipped]
        overall_score = sum(scores) / len(scores) if scores else None

        return EvaluationResult(
            sample=sample,
            metric_results=metric_results,
            overall_score=overall_score,
        )

    async def _evaluate_metric(
        self,
        name: str,
        metric: BaseMetric,
        sample: EvaluationSample,
    ) -> MetricResult:
        """执行单个指标评估"""
        try:
            # 检查样本是否满足指标要求
            if metric.requires_expected_output and not sample.expected_output:
                return MetricResult(
                    score=0.0,
                    reason=f"Metric '{name}' requires expected_output",
                    skipped=True,
                )
            if metric.requires_actual_output and not sample.actual_output:
                return MetricResult(
                    score=0.0,
                    reason=f"Metric '{name}' requires actual_output",
                    skipped=True,
                )
            if metric.requires_retrieval_context and not sample.retrieval_context:
                return MetricResult(
                    score=0.0,
                    reason=f"Metric '{name}' requires retrieval_context",
                    skipped=True,
                )

            # 执行评估
            return await metric.evaluate(
                sample,
                self._limited_llm_client,
                mode=self.chunk_eval_mode,
            )
        except Exception as e:
            return MetricResult(
                score=0.0,
                reason=f"Evaluation failed: {str(e)}",
                details={"error": str(e)},
                skipped=True,
            )

    async def evaluate_batch(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> BatchEvaluationResult:
        """
        批量评估样本

        Args:
            samples: 评估样本列表
            metrics: 要使用的指标列表
            progress_callback: 进度回调函数，签名: (completed, total) -> None

        Returns:
            BatchEvaluationResult: 批量评估结果
        """
        results: List[EvaluationResult] = []
        semaphore = asyncio.Semaphore(self.concurrency)
        completed = 0
        total = len(samples)

        async def evaluate_with_semaphore(sample: EvaluationSample) -> EvaluationResult:
            nonlocal completed
            async with semaphore:
                result = await self.evaluate(sample, metrics)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                return result

        # 并行执行评估
        tasks = [evaluate_with_semaphore(sample) for sample in samples]
        results = await asyncio.gather(*tasks)

        # 计算汇总统计
        summary = self._compute_summary(results, metrics)

        return BatchEvaluationResult(
            results=list(results),
            summary=summary,
        )

    def _compute_summary(
        self,
        results: List[EvaluationResult],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """计算批量评估汇总统计"""
        if not results:
            return {}

        # 确定使用的指标
        metric_names = metrics if metrics else list(self.metrics.keys())

        summary = {
            "total_samples": len(results),
            "metrics": {},
            "overall": {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
        }

        # 计算每个指标的统计（排除 skipped）
        for name in metric_names:
            valid_scores = []
            skipped_count = 0
            for r in results:
                if name in r.metric_results:
                    mr = r.metric_results[name]
                    if mr.skipped:
                        skipped_count += 1
                    else:
                        valid_scores.append(mr.score)
            metric_stat = {"count": len(valid_scores), "skipped": skipped_count}
            if valid_scores:
                metric_stat.update({
                    "mean": sum(valid_scores) / len(valid_scores),
                    "min": min(valid_scores),
                    "max": max(valid_scores),
                })
            summary["metrics"][name] = metric_stat

        # 计算整体统计（排除 skipped）
        overall_scores = [r.overall_score for r in results if r.overall_score is not None]
        if overall_scores:
            summary["overall"] = {
                "mean": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
            }

        return summary

    @classmethod
    def list_available_metrics(cls) -> List[Dict[str, Any]]:
        """列出所有可用的评估指标"""
        return MetricRegistry.list_metrics()


def create_sample(
    input: str,
    expected_output: Optional[str] = None,
    actual_output: Optional[str] = None,
    retrieval_context: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvaluationSample:
    """创建评估样本的便捷函数"""
    return EvaluationSample(
        input=input,
        expected_output=expected_output,
        actual_output=actual_output,
        retrieval_context=retrieval_context or [],
        metadata=metadata,
    )
