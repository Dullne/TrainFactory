"""
文档处理器

编排各处理步骤，处理单个文档。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .steps import (
    DocQualityStep,
    KeypointGenStep,
    QAGenStep,
    PosNegGenStep,
    ValidationStep,
)
from .steps.base import Document, QAPair, GeneratedSample
from .prompts import PromptManager

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """处理器配置"""
    generation_mode: str = "qa_extraction"  # qa_extraction 或 qa_based（内部模式）
    steps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    custom_prompts: Optional[Dict[str, str]] = None


class DocumentProcessor:
    """
    文档处理器

    处理单个文档，按配置执行各处理步骤。
    支持两种模式：
    - qa_extraction: 从文档提取 QA 对（含 chunk 关联）
    - qa_based: 从已有 QA 生成正负例（内部模式，供 LLM fallback 使用）
    """

    def __init__(
        self,
        config: ProcessorConfig,
        llm_client: Any = None,
    ):
        """
        初始化处理器

        Args:
            config: 处理器配置
            llm_client: LLM 客户端
        """
        self.config = config
        self.llm_client = llm_client
        self.prompt_manager = PromptManager(config.custom_prompts)

        # 初始化各步骤
        self._init_steps()

    def _init_steps(self):
        """初始化处理步骤"""
        steps_config = self.config.steps

        self.doc_quality_step = DocQualityStep(
            steps_config.get("doc_quality", {"enabled": True}),
            self.llm_client,
            prompt_manager=self.prompt_manager,
        )

        self.keypoint_step = KeypointGenStep(
            steps_config.get("keypoint_gen", {"enabled": False}),
            self.llm_client,
            prompt_manager=self.prompt_manager,
        )

        self.qa_gen_step = QAGenStep(
            steps_config.get("qa_gen", {"enabled": True}),
            self.llm_client,
            prompt_manager=self.prompt_manager,
        )

        self.pos_neg_step = PosNegGenStep(
            steps_config.get("pos_neg_extraction", {"enabled": True}),
            self.llm_client,
            prompt_manager=self.prompt_manager,
        )

        self.validation_step = ValidationStep(
            steps_config.get("validation", {"enabled": False}),
            self.llm_client,
            prompt_manager=self.prompt_manager,
        )

    async def process(self, doc: Document) -> List[GeneratedSample]:
        """
        处理单个文档

        Args:
            doc: 输入文档

        Returns:
            生成的样本列表
        """
        if self.config.generation_mode == "qa_extraction":
            return await self._process_qa_extraction(doc)
        else:
            return await self._process_qa_based(doc)

    async def _process_qa_based(self, doc: Document) -> List[GeneratedSample]:
        """
        QA-based 模式处理

        假设输入文档包含 query 和 answer 字段
        流程: QA → 正负例生成(内部可选角色) → 校验
        """
        # 从文档元数据中获取 QA
        query = doc.metadata.get("query", "")
        answer = doc.metadata.get("answer", "")

        if not query or not answer:
            logger.debug("Document missing query or answer in qa_based mode")
            return []

        qa_pair = QAPair(query=query, answer=answer)
        qa_pairs = [{"qa": qa_pair, "role": None}]

        # 1. 正负例生成（use_role 由 pos_neg_step 内部处理）
        if not self.pos_neg_step.is_enabled():
            return [
                GeneratedSample(
                    query=qa_pair.query,
                    answer=qa_pair.answer,
                    source_doc_id=doc.doc_id,
                )
            ]

        result = await self.pos_neg_step.execute({
            "document": doc,
            "qa_pairs": qa_pairs,
        })

        if not result.success or not result.data:
            logger.debug(f"Pos/Neg generation failed: {result.error}")
            return []

        samples = result.data.get("samples", [])

        # 2. 校验
        if self.validation_step.is_enabled() and samples:
            result = await self.validation_step.execute({"samples": samples})
            if result.success and result.data:
                samples = result.data.get("samples", samples)

        return samples

    async def _process_qa_extraction(self, doc: Document) -> List[GeneratedSample]:
        """
        QA 提取模式

        从文档中提取 QA 对，记录 chunk_id 和 chunk_content。
        不运行正负例生成。

        流程: Document → 质量评估(可选) → 关键点提取(可选) → QA生成(内部可选角色)
        """
        # 1. 文档质量评估
        if self.doc_quality_step.is_enabled():
            result = await self.doc_quality_step.execute(doc)
            if not result.success:
                logger.debug(f"Document quality check failed: {result.error}")
                return []

        # 2. 关键点提取
        keypoints = []
        if self.keypoint_step.is_enabled():
            result = await self.keypoint_step.execute(doc)
            if result.success and result.data:
                keypoints = result.data.get("keypoints", [])

        # 3. QA 生成（use_role 由 qa_gen_step 内部处理）
        if not self.qa_gen_step.is_enabled():
            logger.debug("QA generation is disabled")
            return []

        result = await self.qa_gen_step.execute({
            "document": doc,
            "keypoints": keypoints,
        })

        if not result.success or not result.data:
            logger.debug(f"QA generation failed: {result.error}")
            return []

        qa_pairs = result.data.get("qa_pairs", [])
        if not qa_pairs:
            return []

        # 5. 构建带 chunk 关联的 QA 样本（不运行 pos_neg）
        return [
            GeneratedSample(
                query=qa_data["qa"].query,
                answer=qa_data["qa"].answer,
                role_config=qa_data.get("role"),
                source_doc_id=doc.doc_id,
                metadata={
                    "chunk_id": doc.doc_id,
                    "chunk_content": doc.content,
                },
            )
            for qa_data in qa_pairs
        ]
