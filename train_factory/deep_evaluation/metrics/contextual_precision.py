"""
上下文精准度指标 (Contextual Precision)

评估检索到的上下文中有多少比例是与回答问题相关的。
支持两种评估模式：batch / individual。
"""

import asyncio
import json
import logging
import re
from typing import Any, List, Optional

from .base import BaseMetric, MetricResult, MetricCategory, MetricRegistry, EvaluationSample

logger = logging.getLogger(__name__)

# ---------- batch 模式 prompt ----------
CONTEXTUAL_PRECISION_PROMPT = """你是一个专业的检索质量评估专家。请评估以下检索结果的精准度。

## 任务
判断每个检索到的上下文片段是否与回答用户问题相关。

## 用户问题
{input}

## 期望答案（参考）
{expected_output}

## 检索到的上下文
{contexts}

## 评估要求
1. 对每个上下文片段，判断它是否包含回答问题所需的信息
2. 计算相关片段的比例作为精准度分数
3. 提供评估理由

请以 JSON 格式返回结果：
{{
    "relevant_indices": [相关上下文的索引列表, 从0开始],
    "score": 0.0到1.0之间的精准度分数,
    "reason": "评估理由"
}}"""

# ---------- individual 模式 prompt ----------
INDIVIDUAL_PROMPT = """你是一个专业的检索质量评估专家。请判断以下上下文片段是否与回答用户问题相关。

## 用户问题
{input}

## 期望答案（参考）
{expected_output}

## 上下文片段
{context}

## 评估要求
判断该上下文片段是否包含回答问题所需的信息。

请以 JSON 格式返回：
{{"relevant": true, "reason": "评估理由"}}
或
{{"relevant": false, "reason": "评估理由"}}"""


@MetricRegistry.register
class ContextualPrecisionMetric(BaseMetric):
    """
    上下文精准度指标

    衡量检索结果中相关上下文的比例。
    高精准度意味着检索到的内容大部分都是有用的。

    支持两种评估模式（通过 mode 参数指定）：
    - batch: 所有 chunk 一次调用，返回 relevant_indices（默认，成本最低，但有位置偏差）
    - individual: 每个 chunk 单独一次调用（无位置偏差，最准确，N 倍成本）
    """

    name = "contextual_precision"
    description = "评估检索上下文的精准度，衡量检索结果中相关内容的比例"
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
        """评估上下文精准度

        Args:
            sample: 评估样本
            llm_client: LLM 客户端
            mode: 评估模式 "batch" | "individual"
        """
        mode = kwargs.get("mode", "batch")
        self.validate_sample(sample)

        if not llm_client:
            raise ValueError("LLM client is required for contextual precision evaluation")

        if not sample.retrieval_context:
            return MetricResult(
                score=0.0,
                reason="No retrieval context provided",
                details={"relevant_count": 0, "total_count": 0},
            )

        try:
            if mode == "individual":
                return await self._evaluate_individual(sample, llm_client)
            else:
                return await self._evaluate_batch(sample, llm_client)
        except Exception as e:
            return MetricResult(
                score=0.0,
                reason=f"Evaluation failed: {str(e)}",
                details={"error": str(e)},
                skipped=True,
            )

    # ------------------------------------------------------------------ batch
    async def _evaluate_batch(
        self, sample: EvaluationSample, llm_client: Any,
    ) -> MetricResult:
        """batch 模式：所有 chunk 一次调用，返回 relevant_indices"""
        contexts_text = self._format_contexts(sample.retrieval_context)
        prompt = CONTEXTUAL_PRECISION_PROMPT.format(
            input=sample.input,
            expected_output=sample.expected_output or "(未提供)",
            contexts=contexts_text,
        )
        response = await llm_client.chat(prompt)
        return self._parse_batch_response(response, len(sample.retrieval_context))

    # -------------------------------------------------------------- individual
    async def _evaluate_individual(
        self,
        sample: EvaluationSample,
        llm_client: Any,
    ) -> MetricResult:
        """individual 模式：每个 chunk 单独一次调用，无位置偏差"""
        total = len(sample.retrieval_context)
        relevant_indices: List[int] = []

        async def _eval_one(i: int, ctx: str):
            prompt = INDIVIDUAL_PROMPT.format(
                input=sample.input,
                expected_output=sample.expected_output or "(未提供)",
                context=ctx,
            )
            resp = await llm_client.chat(prompt)
            relevant, _ = self._parse_individual_response(resp)
            if relevant:
                relevant_indices.append(i)

        tasks = [_eval_one(i, ctx) for i, ctx in enumerate(sample.retrieval_context)]
        await asyncio.gather(*tasks)

        relevant_indices.sort()
        score = len(relevant_indices) / total if total > 0 else 0.0
        return MetricResult(
            score=score,
            reason=f"Individual evaluation: {len(relevant_indices)}/{total} relevant",
            details={
                "relevant_indices": relevant_indices,
                "relevant_count": len(relevant_indices),
                "total_count": total,
                "mode": "individual",
            },
        )

    # ================================================================ helpers

    @staticmethod
    def _format_contexts(contexts: List[str]) -> str:
        return "\n\n".join(
            f"[Context {i}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

    @staticmethod
    def _clean_response(response: str) -> str:
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```\w*\n?', '', response)
            response = re.sub(r'\n?```$', '', response)
        return response

    # ---- batch 解析 ----
    def _parse_batch_response(self, response: str, total_contexts: int) -> MetricResult:
        response = self._clean_response(response)
        try:
            data = json.loads(response)
            relevant_indices = data.get("relevant_indices", [])
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "")
            score = max(0.0, min(1.0, score))
            return MetricResult(
                score=score,
                reason=reason,
                details={
                    "relevant_indices": relevant_indices,
                    "relevant_count": len(relevant_indices),
                    "total_count": total_contexts,
                    "mode": "batch",
                },
            )
        except json.JSONDecodeError:
            score_match = re.search(r'"?score"?\s*[:：]\s*([0-9.]+)', response)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
                return MetricResult(
                    score=score,
                    reason="Score extracted from text",
                    details={"raw_response": response[:500], "mode": "batch"},
                )
            return MetricResult(
                score=0.0,
                reason="Failed to parse LLM response",
                details={"raw_response": response[:500], "mode": "batch"},
                skipped=True,
            )

    # ---- individual 单条解析 ----
    @staticmethod
    def _parse_individual_response(response: str) -> tuple:
        """返回 (relevant: bool, reason: str)"""
        response = ContextualPrecisionMetric._clean_response(response)
        try:
            data = json.loads(response)
            relevant = bool(data.get("relevant", False))
            reason = data.get("reason", "")
            return relevant, reason
        except json.JSONDecodeError:
            low = response.lower()
            if '"relevant": true' in low or '"relevant":true' in low:
                return True, "Parsed from text"
            return False, f"Parse failed: {response[:200]}"
