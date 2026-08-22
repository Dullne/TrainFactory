"""
文档质量评估步骤

评估输入文档的质量，过滤低质量文档。
默认启用，use_llm=False 时仅做规则检查（零 LLM 开销）。
"""

import logging
import re
from typing import Any, Dict, Optional

from .base import BaseStep, StepResult, Document
from ..prompts import PromptManager
from ..clients.llm_client import extract_json_from_response

logger = logging.getLogger(__name__)


class DocQualityStep(BaseStep):
    """文档质量评估步骤"""

    name = "doc_quality"
    description = "评估文档质量，过滤低质量内容"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Any = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        super().__init__(config, llm_client)
        self.min_score = config.get("min_score", 0.6)
        self.use_llm = config.get("use_llm", True)
        self.min_length = config.get("min_length", 50)
        self.min_text_ratio = config.get("min_text_ratio", 0.3)
        self.min_unique_chars = config.get("min_unique_chars", 10)
        self.prompt_manager = prompt_manager or PromptManager()

    async def execute(self, input_data: Document) -> StepResult:
        """
        评估文档质量

        Args:
            input_data: 输入文档

        Returns:
            StepResult: 评估结果，data 中包含质量分数
        """
        doc = input_data
        text = doc.content.strip()

        # ── 规则检查（始终执行，零 LLM 开销）──

        # 1. 长度检查
        if len(text) < self.min_length:
            logger.debug(
                f"Doc quality: too short ({len(text)} < {self.min_length}), "
                f"doc_id={doc.doc_id}"
            )
            return StepResult(
                success=False,
                error=f"Document too short ({len(text)} chars)",
                details={"length": len(text), "min_length": self.min_length},
            )

        # 2. 有意义文本占比（中文字符 + 英文字母）
        meaningful = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbfa-zA-Z]', text))
        text_ratio = meaningful / len(text) if text else 0
        if text_ratio < self.min_text_ratio:
            logger.debug(
                f"Doc quality: low text ratio ({text_ratio:.2f} < {self.min_text_ratio}), "
                f"doc_id={doc.doc_id}"
            )
            return StepResult(
                success=False,
                error=f"Low text ratio ({text_ratio:.2f})",
                details={"text_ratio": text_ratio, "min_text_ratio": self.min_text_ratio},
            )

        # 3. 独立字符数量（过滤重复/无意义内容）
        unique_chars = len(set(text))
        if unique_chars < self.min_unique_chars:
            logger.debug(
                f"Doc quality: too few unique chars ({unique_chars} < {self.min_unique_chars}), "
                f"doc_id={doc.doc_id}"
            )
            return StepResult(
                success=False,
                error=f"Too few unique characters ({unique_chars})",
                details={"unique_chars": unique_chars, "min_unique_chars": self.min_unique_chars},
            )

        # 如果不使用 LLM，规则检查通过即可
        if not self.use_llm:
            return StepResult(
                success=True,
                data={"score": 1.0, "is_usable": True},
                details={
                    "method": "rule_based",
                    "length": len(text),
                    "text_ratio": round(text_ratio, 2),
                    "unique_chars": unique_chars,
                },
            )

        # ── LLM 评估（可选）──

        if not self.llm_client:
            return StepResult(
                success=True,
                data={"score": 1.0, "is_usable": True},
                details={"method": "no_llm_client", "warning": "LLM client not provided"},
            )

        try:
            prompt = self.prompt_manager.render(
                "doc_quality",
                document=doc.content[:3000],
            )

            response = await self.llm_client.chat(prompt)
            result = extract_json_from_response(response)

            if result:
                score = result.get("overall_score", 0.0)
                is_usable = result.get("is_usable", score >= self.min_score)

                return StepResult(
                    success=is_usable,
                    data=result,
                    error=None if is_usable else "Document quality below threshold",
                    details={"score": score, "threshold": self.min_score},
                )
            else:
                # 解析失败，跳过评估，不伪造分数
                return StepResult(
                    success=True,
                    data={"score": None, "is_usable": True, "evaluation_skipped": True},
                    details={"warning": "Failed to parse LLM response", "method": "skipped"},
                )

        except Exception as e:
            # 评估失败，跳过评估，不伪造分数
            return StepResult(
                success=True,
                data={"score": None, "is_usable": True, "evaluation_skipped": True},
                details={"warning": f"Quality evaluation failed: {str(e)}", "method": "skipped"},
            )
