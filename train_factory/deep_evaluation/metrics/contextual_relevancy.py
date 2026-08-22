"""
上下文相关性指标 (Contextual Relevancy)

评估检索到的上下文与用户问题的相关程度。
与 Contextual Precision 不同，此指标不需要期望答案，
仅基于问题和检索上下文进行评估。
"""

import json
import re
from typing import Any, Optional

from .base import BaseMetric, MetricResult, MetricCategory, MetricRegistry, EvaluationSample


CONTEXTUAL_RELEVANCY_PROMPT = """你是一个专业的检索质量评估专家。请评估以下检索结果与用户问题的相关程度。

## 任务
判断检索到的上下文整体上是否与用户问题相关，能否为回答问题提供有价值的信息。

## 用户问题
{input}

## 检索到的上下文
{contexts}

## 评估要求
1. 评估每个上下文片段与问题的相关程度
2. 考虑上下文是否包含回答问题所需的关键信息
3. 综合评估所有上下文的整体相关性
4. 给出 0.0 到 1.0 之间的相关性分数：
   - 0.0-0.2: 完全不相关
   - 0.2-0.4: 略微相关
   - 0.4-0.6: 部分相关
   - 0.6-0.8: 较为相关
   - 0.8-1.0: 高度相关

请以 JSON 格式返回结果：
{{
    "relevancy_scores": [每个上下文片段的相关性分数列表, 0.0-1.0],
    "score": 0.0到1.0之间的整体相关性分数,
    "reason": "评估理由"
}}"""


@MetricRegistry.register
class ContextualRelevancyMetric(BaseMetric):
    """
    上下文相关性指标

    评估检索上下文与用户问题的相关程度。
    高相关性意味着检索到的内容对回答问题有帮助。

    与 ContextualPrecision 的区别：
    - ContextualPrecision: 需要 expected_output，评估检索结果中相关内容的比例
    - ContextualRelevancy: 不需要 expected_output，直接评估上下文与问题的相关性
    """

    name = "contextual_relevancy"
    description = "评估检索上下文与问题的相关程度"
    category = MetricCategory.RETRIEVAL
    requires_llm = True
    requires_expected_output = False
    requires_actual_output = False
    requires_retrieval_context = True

    async def evaluate(
        self,
        sample: EvaluationSample,
        llm_client: Optional[Any] = None,
        **kwargs,
    ) -> MetricResult:
        """评估上下文相关性"""
        self.validate_sample(sample)

        if not llm_client:
            raise ValueError("LLM client is required for contextual relevancy evaluation")

        if not sample.retrieval_context:
            return MetricResult(
                score=0.0,
                reason="No retrieval context provided",
                details={"relevancy_scores": [], "total_count": 0},
            )

        # 格式化上下文
        contexts_text = "\n\n".join([
            f"[Context {i}]\n{ctx}"
            for i, ctx in enumerate(sample.retrieval_context)
        ])

        # 构建 prompt
        prompt = CONTEXTUAL_RELEVANCY_PROMPT.format(
            input=sample.input,
            contexts=contexts_text,
        )

        # 调用 LLM
        try:
            response = await llm_client.chat(prompt)
            result = self._parse_response(response, len(sample.retrieval_context))
            return result
        except Exception as e:
            return MetricResult(
                score=0.0,
                reason=f"Evaluation failed: {str(e)}",
                details={"error": str(e)},
                skipped=True,
            )

    def _parse_response(self, response: str, total_contexts: int) -> MetricResult:
        """解析 LLM 响应"""
        # 清理响应
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```\w*\n?', '', response)
            response = re.sub(r'\n?```$', '', response)

        try:
            data = json.loads(response)
            relevancy_scores = data.get("relevancy_scores", [])
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")

            # 确保分数在有效范围内
            score = max(0.0, min(1.0, score))

            # 如果没有提供整体分数，计算平均值
            if score == 0.0 and relevancy_scores:
                score = sum(relevancy_scores) / len(relevancy_scores)
                score = max(0.0, min(1.0, score))

            return MetricResult(
                score=score,
                reason=reason,
                details={
                    "relevancy_scores": relevancy_scores,
                    "total_count": total_contexts,
                    "avg_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
                },
            )
        except json.JSONDecodeError:
            # 尝试从文本中提取分数
            score_match = re.search(r'"?score"?\s*[:：]\s*([0-9.]+)', response)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
                return MetricResult(
                    score=score,
                    reason="Score extracted from text",
                    details={"raw_response": response[:500]},
                )

            return MetricResult(
                score=0.0,
                reason="Failed to parse LLM response",
                details={"raw_response": response[:500]},
                skipped=True,
            )
