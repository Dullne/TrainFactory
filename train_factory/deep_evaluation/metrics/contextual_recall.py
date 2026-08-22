"""
上下文召回率指标 (Contextual Recall)

评估期望答案中的关键信息有多少被检索到的上下文覆盖。
"""

import json
import re
from typing import Any, Optional

from .base import BaseMetric, MetricResult, MetricCategory, MetricRegistry, EvaluationSample


CONTEXTUAL_RECALL_PROMPT = """你是一个专业的检索质量评估专家。请评估以下检索结果的召回率。

## 任务
判断检索到的上下文是否覆盖了回答问题所需的所有关键信息。

## 用户问题
{input}

## 期望答案（标准答案）
{expected_output}

## 检索到的上下文
{contexts}

## 评估要求
1. 从期望答案中提取关键信息点
2. 判断每个关键信息点是否被检索到的上下文覆盖
3. 计算覆盖的信息点比例作为召回率分数
4. 提供评估理由

请以 JSON 格式返回结果：
{{
    "key_points": ["关键信息点1", "关键信息点2", ...],
    "covered_points": ["被覆盖的信息点1", ...],
    "uncovered_points": ["未被覆盖的信息点1", ...],
    "score": 0.0到1.0之间的召回率分数,
    "reason": "评估理由"
}}"""


@MetricRegistry.register
class ContextualRecallMetric(BaseMetric):
    """
    上下文召回率指标

    衡量期望答案中的关键信息被检索上下文覆盖的程度。
    高召回率意味着检索到了回答问题所需的大部分信息。
    """

    name = "contextual_recall"
    description = "评估检索上下文的召回率，衡量关键信息的覆盖程度"
    category = MetricCategory.RETRIEVAL
    requires_llm = True
    requires_expected_output = True
    requires_retrieval_context = True

    async def evaluate(
        self,
        sample: EvaluationSample,
        llm_client: Optional[Any] = None,
        **kwargs,
    ) -> MetricResult:
        """评估上下文召回率"""
        self.validate_sample(sample)

        if not llm_client:
            raise ValueError("LLM client is required for contextual recall evaluation")

        if not sample.retrieval_context:
            return MetricResult(
                score=0.0,
                reason="No retrieval context provided",
                details={"covered_count": 0, "total_count": 0},
            )

        # 格式化上下文
        contexts_text = "\n\n".join([
            f"[Context {i}]\n{ctx}"
            for i, ctx in enumerate(sample.retrieval_context)
        ])

        # 构建 prompt
        prompt = CONTEXTUAL_RECALL_PROMPT.format(
            input=sample.input,
            expected_output=sample.expected_output or "(未提供)",
            contexts=contexts_text,
        )

        # 调用 LLM
        try:
            response = await llm_client.chat(prompt)
            result = self._parse_response(response)
            return result
        except Exception as e:
            return MetricResult(
                score=0.0,
                reason=f"Evaluation failed: {str(e)}",
                details={"error": str(e)},
                skipped=True,
            )

    def _parse_response(self, response: str) -> MetricResult:
        """解析 LLM 响应"""
        # 清理响应
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```\w*\n?', '', response)
            response = re.sub(r'\n?```$', '', response)

        try:
            data = json.loads(response)
            key_points = data.get("key_points", [])
            covered_points = data.get("covered_points", [])
            uncovered_points = data.get("uncovered_points", [])
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")

            # 确保分数在有效范围内
            score = max(0.0, min(1.0, score))

            return MetricResult(
                score=score,
                reason=reason,
                details={
                    "key_points": key_points,
                    "covered_points": covered_points,
                    "uncovered_points": uncovered_points,
                    "covered_count": len(covered_points),
                    "total_count": len(key_points),
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
