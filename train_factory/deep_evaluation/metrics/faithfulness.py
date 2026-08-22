"""
事实一致性指标 (Faithfulness)

评估生成的答案是否忠实于检索到的上下文，不包含幻觉或无根据的信息。
"""

import json
import re
from typing import Any, Optional

from .base import BaseMetric, MetricResult, MetricCategory, MetricRegistry, EvaluationSample


FAITHFULNESS_PROMPT = """你是一个专业的事实核查专家。请评估以下答案与检索上下文的一致性。

## 任务
判断生成的答案是否完全基于检索到的上下文，不包含幻觉或无根据的信息。

## 用户问题
{input}

## 检索到的上下文
{contexts}

## 生成的答案
{actual_output}

## 评估要求
1. 提取答案中的关键声明/陈述
2. 判断每个声明是否可以从检索上下文中找到支持
3. 识别任何无法从上下文验证的信息（可能的幻觉）
4. 综合评分

请以 JSON 格式返回结果：
{{
    "claims": ["答案中的声明1", "答案中的声明2", ...],
    "supported_claims": ["有上下文支持的声明1", ...],
    "unsupported_claims": ["无上下文支持的声明1", ...],
    "score": 0.0到1.0之间的一致性分数,
    "reason": "评估理由"
}}"""


@MetricRegistry.register
class FaithfulnessMetric(BaseMetric):
    """
    事实一致性指标

    衡量生成的答案是否忠实于检索到的上下文。
    高一致性意味着答案中的信息都可以从上下文中找到依据，没有幻觉。
    """

    name = "faithfulness"
    description = "评估答案与检索上下文的事实一致性，检测幻觉"
    category = MetricCategory.FAITHFULNESS
    requires_llm = True
    requires_actual_output = True
    requires_retrieval_context = True

    async def evaluate(
        self,
        sample: EvaluationSample,
        llm_client: Optional[Any] = None,
        **kwargs,
    ) -> MetricResult:
        """评估事实一致性"""
        self.validate_sample(sample)

        if not llm_client:
            raise ValueError("LLM client is required for faithfulness evaluation")

        if not sample.actual_output:
            return MetricResult(
                score=0.0,
                reason="No actual output provided",
                details={},
            )

        if not sample.retrieval_context:
            return MetricResult(
                score=0.0,
                reason="No retrieval context provided for faithfulness check",
                details={"warning": "Cannot verify faithfulness without context"},
            )

        # 格式化上下文
        contexts_text = "\n\n".join([
            f"[Context {i}]\n{ctx}"
            for i, ctx in enumerate(sample.retrieval_context)
        ])

        # 构建 prompt
        prompt = FAITHFULNESS_PROMPT.format(
            input=sample.input,
            contexts=contexts_text,
            actual_output=sample.actual_output,
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
            claims = data.get("claims", [])
            supported_claims = data.get("supported_claims", [])
            unsupported_claims = data.get("unsupported_claims", [])
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")

            # 确保分数在有效范围内
            score = max(0.0, min(1.0, score))

            return MetricResult(
                score=score,
                reason=reason,
                details={
                    "claims": claims,
                    "supported_claims": supported_claims,
                    "unsupported_claims": unsupported_claims,
                    "supported_count": len(supported_claims),
                    "total_count": len(claims),
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
