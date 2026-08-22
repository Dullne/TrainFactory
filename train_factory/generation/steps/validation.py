"""
校验步骤

校验生成的正负例质量。
"""

import re
from typing import Any, Dict, List, Optional

from .base import BaseStep, StepResult, GeneratedSample
from ..prompts import PromptManager

# CJK / Japanese kana / Hangul ranges. Text in these scripts has no spaces, so
# str.split() yields a single whole-string token and makes every lexical-overlap
# comparison empty. Detect them to switch to character-level tokenization.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def _lexical_tokens(text: str) -> set:
    """Tokenize text for lexical-overlap heuristics.

    Uses whitespace tokens for space-delimited languages, and additionally adds
    character unigrams + bigrams for CJK text (which has no spaces) so overlap
    is meaningful instead of being always empty.
    """
    text = (text or "").lower().strip()
    if not text:
        return set()
    tokens = set(text.split())
    if _CJK_RE.search(text):
        compact = re.sub(r"\s+", "", text)
        tokens.update(compact)  # unigrams (also covers single-char tokens)
        tokens.update(compact[i:i + 2] for i in range(len(compact) - 1))  # bigrams
    return tokens

# CJK / Japanese kana / Hangul ranges. Text in these scripts has no spaces, so
# str.split() yields a single whole-string token and makes every lexical-overlap
# comparison empty. Detect them to switch to character-level tokenization.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")




class ValidationStep(BaseStep):
    """正负例校验步骤"""

    name = "validation"
    description = "校验生成的正负例质量"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Any = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        super().__init__(config, llm_client)
        self.prompt_manager = prompt_manager or PromptManager()

    async def execute(self, input_data: Dict[str, Any]) -> StepResult:
        """
        校验正负例质量

        Args:
            input_data: 包含 samples (GeneratedSample 列表)

        Returns:
            StepResult: 校验结果，data 中包含过滤后的样本
        """
        samples: List[GeneratedSample] = input_data.get("samples", [])

        if not samples:
            return StepResult(
                success=True,
                data={"samples": []},
                details={"warning": "No samples to validate"},
            )

        if not self.llm_client:
            # 没有 LLM 客户端，跳过校验
            return StepResult(
                success=True,
                data={"samples": samples},
                details={"warning": "Validation skipped: no LLM client"},
            )

        try:
            validated_samples = []
            total_positives_removed = 0
            total_negatives_removed = 0

            for sample in samples:
                validated_sample, pos_removed, neg_removed = await self._validate_sample(sample)
                if validated_sample:
                    validated_samples.append(validated_sample)
                total_positives_removed += pos_removed
                total_negatives_removed += neg_removed

            return StepResult(
                success=len(validated_samples) > 0,
                data={"samples": validated_samples},
                details={
                    "original_count": len(samples),
                    "validated_count": len(validated_samples),
                    "positives_removed": total_positives_removed,
                    "negatives_removed": total_negatives_removed,
                },
            )

        except Exception as e:
            # 校验失败，返回原始样本
            return StepResult(
                success=True,
                data={"samples": samples},
                details={"warning": f"Validation failed: {str(e)}, returning original samples"},
            )

    async def _validate_sample(
        self,
        sample: GeneratedSample,
    ) -> tuple:
        """
        校验单个样本

        Returns:
            (validated_sample, positives_removed, negatives_removed)
        """
        valid_positives = []
        valid_negatives = []
        pos_removed = 0
        neg_removed = 0

        # 校验正例
        for pos in sample.positive_chunks:
            is_valid = await self._validate_chunk(
                sample.query,
                sample.answer,
                pos,
                is_positive=True,
            )
            if is_valid:
                valid_positives.append(pos)
            else:
                pos_removed += 1

        # 校验负例
        for neg in sample.negative_chunks:
            is_valid = await self._validate_chunk(
                sample.query,
                sample.answer,
                neg,
                is_positive=False,
            )
            if is_valid:
                valid_negatives.append(neg)
            else:
                neg_removed += 1

        # 如果正例和负例都为空，返回 None
        if not valid_positives and not valid_negatives:
            return None, pos_removed, neg_removed

        validated_sample = GeneratedSample(
            query=sample.query,
            answer=sample.answer,
            positive_chunks=valid_positives,
            negative_chunks=valid_negatives,
            role_config=sample.role_config,
            source_doc_id=sample.source_doc_id,
            metadata=sample.metadata,
        )

        return validated_sample, pos_removed, neg_removed

    async def _validate_chunk(
        self,
        query: str,
        answer: str,
        chunk: str,
        is_positive: bool,
    ) -> bool:
        """校验单个 chunk"""
        try:
            # 简单的启发式校验
            # 正例应该与答案有较高的词汇重叠
            # 负例应该与答案有一定相关性但不直接回答问题

            # 基本检查
            if len(chunk.strip()) < 20:
                return False

            # 使用 LLM 校验（可选，为了性能可以跳过）
            # 这里采用简单的启发式规则
            if is_positive:
                # 正例：检查是否包含答案中的关键词
                answer_words = _lexical_tokens(answer)
                chunk_words = _lexical_tokens(chunk)
                overlap = len(answer_words & chunk_words)
                # 阈值至少为 1，避免短答案下 //2==0 导致恒通过
                threshold = min(3, max(1, len(answer_words) // 2))
                return overlap >= threshold
            else:
                # 负例：检查是否与问题相关但不直接回答
                query_words = _lexical_tokens(query)
                chunk_words = _lexical_tokens(chunk)
                query_overlap = len(query_words & chunk_words)
                answer_words = _lexical_tokens(answer)
                answer_overlap = len(answer_words & chunk_words)
                # 与问题有关联，但不能太像答案（阈值至少为 1）
                return query_overlap >= 1 and answer_overlap < max(1, len(answer_words) // 2)

        except Exception:
            return True  # 校验失败默认通过
