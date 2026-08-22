"""
正负例生成步骤

生成正例和负例文本块。对齐原版 prompt 体系，带完整来源/长度上下文。
"""

from typing import Any, Dict, List, Optional

from .base import BaseStep, StepResult, Document, QAPair, RoleConfig, GeneratedSample
from ..prompts import PromptManager, LENGTH_TYPES, LENGTH_TYPE_NAMES, get_source_info, sample_source_config
from ..clients.llm_client import extract_json_from_response


class PosNegGenStep(BaseStep):
    """正负例生成步骤"""

    name = "pos_neg_extraction"
    description = "生成正例和负例文本块"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Any = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        super().__init__(config, llm_client)
        self.num_positive = config.get("num_positive", 5)
        self.num_negative = config.get("num_negative", 20)
        self.use_role = config.get("use_role", False)
        self.roles_per_doc = config.get("roles_per_doc", 3)
        self.default_length = config.get("default_length", "medium")
        self.prompt_manager = prompt_manager or PromptManager()

    async def execute(self, input_data: Dict[str, Any]) -> StepResult:
        """
        生成正负例

        Args:
            input_data: 包含 document, qa_pairs (带可选的 role)

        Returns:
            StepResult: 生成结果，data 中包含 GeneratedSample 列表
        """
        document: Document = input_data.get("document")
        qa_pairs: List[Dict] = input_data.get("qa_pairs", [])

        if not qa_pairs:
            return StepResult(
                success=False,
                error="QA pairs are required for pos/neg generation",
            )

        if not self.llm_client:
            return StepResult(
                success=False,
                error="LLM client required for pos/neg generation",
            )

        try:
            samples = []

            for qa_data in qa_pairs:
                qa: QAPair = qa_data.get("qa")
                role: Optional[RoleConfig] = qa_data.get("role")

                if not qa:
                    continue

                # use_role 开启且没有现成角色时，内部生成
                if self.use_role and not role:
                    role = await self._generate_role(qa)

                # 生成正例
                positives = await self._generate_positives(qa, role)

                # 生成负例
                negatives = await self._generate_negatives(qa, role)

                sample = GeneratedSample(
                    query=qa.query,
                    answer=qa.answer,
                    positive_chunks=positives,
                    negative_chunks=negatives,
                    role_config=role,
                    source_doc_id=document.doc_id if document else None,
                )
                samples.append(sample)

            return StepResult(
                success=len(samples) > 0,
                data={"samples": samples},
                details={
                    "sample_count": len(samples),
                    "avg_positives": sum(len(s.positive_chunks) for s in samples) / len(samples) if samples else 0,
                    "avg_negatives": sum(len(s.negative_chunks) for s in samples) / len(samples) if samples else 0,
                },
            )

        except Exception as e:
            return StepResult(
                success=False,
                error=f"Pos/Neg generation failed: {str(e)}",
            )

    async def _generate_role(self, qa: QAPair) -> Optional[RoleConfig]:
        """从 QA 对生成一个角色"""
        from .role_gen import RoleGenStep

        role_step = RoleGenStep(
            {"enabled": True, "roles_per_doc": self.roles_per_doc},
            self.llm_client,
            prompt_manager=self.prompt_manager,
        )
        result = await role_step.execute({"qa_pair": qa})
        if result.success and result.data:
            roles = result.data.get("roles", [])
            return roles[0] if roles else None
        return None

    def _get_render_kwargs(self, qa: QAPair, role: Optional[RoleConfig]) -> Dict[str, Any]:
        """构建 prompt 渲染参数"""
        if self.use_role and role:
            # 有角色：从角色配置获取来源和长度信息
            source_info = get_source_info(role.source_type)
            min_words, max_words = LENGTH_TYPES.get(role.length_type, (200, 500))
            return {
                "query": qa.query,
                "answer": qa.answer,
                "source_type": role.source_type,
                "source_name": source_info["name"],
                "source_description": source_info["description"],
                "source_style": source_info["style"],
                "character": role.character,
                "difficulty": role.difficulty,
                "length_name": LENGTH_TYPE_NAMES.get(role.length_type, "中等"),
                "min_words": min_words,
                "max_words": max_words,
            }
        else:
            # 无角色：随机采样来源+长度组合，增加多样性
            config = sample_source_config()
            return {
                "query": qa.query,
                "answer": qa.answer,
                "source_type": config["source_type"],
                "source_name": config["source_name"],
                "source_description": config["source_description"],
                "source_style": config["source_style"],
                "length_name": config["length_name"],
                "min_words": config["min_words"],
                "max_words": config["max_words"],
            }

    async def _generate_positives(
        self,
        qa: QAPair,
        role: Optional[RoleConfig],
    ) -> List[str]:
        """生成正例"""
        positives = []

        for i in range(self.num_positive):
            try:
                kwargs = self._get_render_kwargs(qa, role)
                prompt_name = "positive_chunk_with_role" if (self.use_role and role) else "positive_chunk"
                prompt = self.prompt_manager.render(prompt_name, **kwargs)

                response = await self.llm_client.chat(prompt)
                chunk = self._extract_chunk(response)
                if chunk and len(chunk) > 20:
                    positives.append(chunk)

            except Exception:
                continue

        return positives

    async def _generate_negatives(
        self,
        qa: QAPair,
        role: Optional[RoleConfig],
    ) -> List[str]:
        """生成负例"""
        negatives = []

        for i in range(self.num_negative):
            try:
                kwargs = self._get_render_kwargs(qa, role)
                prompt_name = "negative_chunk_with_role" if (self.use_role and role) else "negative_chunk"
                prompt = self.prompt_manager.render(prompt_name, **kwargs)

                response = await self.llm_client.chat(prompt)
                chunk = self._extract_chunk(response)
                if chunk and len(chunk) > 20:
                    negatives.append(chunk)

            except Exception:
                continue

        return negatives

    @staticmethod
    def _extract_chunk(response: str) -> Optional[str]:
        """从 LLM 响应中提取 chunk 内容，兼容 JSON 和纯文本"""
        if not response:
            return None
        text = response.strip()
        # 尝试 JSON 解析（新 prompt 返回 {"output": "..."}）
        result = extract_json_from_response(text)
        if result and isinstance(result, dict) and "output" in result:
            return result["output"].strip()
        # 回退到纯文本
        return text
