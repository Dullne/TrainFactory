"""
答案相关性指标 (Answer Relevancy)

评估生成的答案与用户问题的相关程度。
"""

import json
import re
from typing import Any, Optional

from .base import BaseMetric, MetricResult, MetricCategory, MetricRegistry, EvaluationSample


ANSWER_RELEVANCY_PROMPT = """你是一个专业的答案质量评估专家。请评估以下答案与问题的相关性。

## 任务
判断生成的答案是否直接回答了用户的问题，以及回答的完整程度。

## 用户问题
{input}

## 生成的答案
{actual_output}

## 参考答案（如有）
{expected_output}

## 评估要求
1. 判断答案是否直接回答了问题
2. 评估答案的完整性（是否覆盖了问题的所有方面）
3. 评估答案的针对性（是否有无关内容）
4. 综合评分

请以 JSON 格式返回结果：
{{
    "directly_answers": true/false,
    "completeness": "完整程度描述",
    "relevance": "相关性描述",
    "score": 0.0到1.0之间的相关性分数,
    "reason": "评估理由"
}}"""


@MetricRegistry.register
class AnswerRelevancyMetric(BaseMetric):
    """
    答案相关性指标

    衡量生成的答案与用户问题的相关程度。
    高相关性意味着答案直接、完整地回答了用户的问题。
    """

    name = "answer_relevancy"
    description = "评估生成答案与用户问题的相关程度"
    category = MetricCategory.RELEVANCY
    requires_llm = True
    requires_actual_output = True

    async def evaluate(
        self,
        sample: EvaluationSample,
        llm_client: Optional[Any] = None,
        **kwargs,
    ) -> MetricResult:
        """评估答案相关性"""
        self.validate_sample(sample)

        if not llm_client:
            raise ValueError("LLM client is required for answer relevancy evaluation")

        if not sample.actual_output:
            return MetricResult(
                score=0.0,
                reason="No actual output provided",
                details={},
            )

        # 构建 prompt
        prompt = ANSWER_RELEVANCY_PROMPT.format(
            input=sample.input,
            actual_output=sample.actual_output,
            expected_output=sample.expected_output or "(未提供)",
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
            directly_answers = data.get("directly_answers", False)
            completeness = data.get("completeness", "")
            relevance = data.get("relevance", "")
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")

            # 确保分数在有效范围内
            score = max(0.0, min(1.0, score))

            return MetricResult(
                score=score,
                reason=reason,
                details={
                    "directly_answers": directly_answers,
                    "completeness": completeness,
                    "relevance": relevance,
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
