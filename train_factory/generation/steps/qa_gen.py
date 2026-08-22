"""
问答对生成步骤

从文档生成问答对。use_role 开启时内部自动生成角色。
"""

import re
from typing import Any, Dict, List, Optional

from .base import BaseStep, StepResult, Document, QAPair, RoleConfig
from ..prompts import PromptManager
from ..clients.llm_client import extract_json_from_response

# Patterns that indicate document-referencing queries (not standalone).
# These queries are useless for retrieval training because real users
# don't search with "文档中..." or "根据上文..." prefixes.
_DOC_REF_PATTERN = re.compile(
    r"^(文档|文中|文章|文本|上文|上述|本文|该文|原文|根据文档|根据文本|根据上文|据文档)"
)

# Pronouns and vague referents that, when dominant in a query, indicate
# the query is context-dependent and not a standalone retrieval query.
_PRONOUN_TOKENS = re.compile(
    r"(它|它们|他|他们|她|她们|其|该|这个|那个|这种|那种|此|前者|后者)"
)


def _is_low_quality_query(query: str) -> bool:
    """Check if a query is too context-dependent for retrieval training.

    Returns True if the query should be filtered out.
    """
    q = query.strip()
    if not q:
        return True
    # Rule 1: starts with document reference
    if _DOC_REF_PATTERN.search(q):
        return True
    # Rule 2: pronoun-dominated — query is short AND starts with or
    # heavily relies on pronouns (e.g. "它的架构是什么？")
    if len(q) < 30:
        pronouns = _PRONOUN_TOKENS.findall(q)
        if len(pronouns) >= 2:
            return True
        # Query starts with a bare pronoun
        if _PRONOUN_TOKENS.match(q):
            return True
    return False


class QAGenStep(BaseStep):
    """问答对生成步骤"""

    name = "qa_gen"
    description = "从文档生成问答对"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Any = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        super().__init__(config, llm_client)
        self.num_qa_per_doc = config.get("num_qa_per_doc", 1)
        self.use_role = config.get("use_role", False)
        self.roles_per_doc = config.get("roles_per_doc", 3)
        self.languages = config.get("languages", ["中文"])
        self.prompt_manager = prompt_manager or PromptManager()

    async def execute(self, input_data: Dict[str, Any]) -> StepResult:
        """
        生成问答对

        Args:
            input_data: 包含 document, keypoints (可选)

        Returns:
            StepResult: 生成结果，data 中包含 QA 对列表
        """
        document: Document = input_data.get("document")
        keypoints: List[str] = input_data.get("keypoints", [])

        if not document:
            return StepResult(
                success=False,
                error="Document is required for QA generation",
            )

        if not self.llm_client:
            return StepResult(
                success=False,
                error="LLM client required for QA generation",
            )

        try:
            qa_pairs = []

            if self.use_role:
                # 内部生成角色，每个角色生成 1 个 QA
                roles = await self._generate_roles(document)
                for role in roles:
                    qa = await self._generate_qa_with_role(document, role)
                    if qa:
                        qa_pairs.append({"qa": qa, "role": role})
            else:
                # 直接生成 QA
                qas = await self._generate_qa_direct(document, keypoints)
                qa_pairs = [{"qa": qa, "role": None} for qa in qas]

            return StepResult(
                success=len(qa_pairs) > 0,
                data={"qa_pairs": qa_pairs},
                details={"count": len(qa_pairs)},
            )

        except Exception as e:
            return StepResult(
                success=False,
                error=f"QA generation failed: {str(e)}",
            )

    async def _generate_roles(self, document: Document) -> List[RoleConfig]:
        """从文档内容生成角色"""
        from .role_gen import RoleGenStep

        role_step = RoleGenStep(
            {"enabled": True, "roles_per_doc": self.roles_per_doc},
            self.llm_client,
            prompt_manager=self.prompt_manager,
        )
        result = await role_step.execute({"document": document})
        if result.success and result.data:
            return result.data.get("roles", [])
        return []

    async def _generate_qa_direct(
        self,
        document: Document,
        keypoints: List[str],
    ) -> List[QAPair]:
        """直接从文档生成 QA"""
        try:
            prompt = self.prompt_manager.render(
                "qa_generation",
                document=document.content[:4000],
                keypoints="\n".join([f"- {kp}" for kp in keypoints]) if keypoints else "(无)",
                num_qa=self.num_qa_per_doc,
            )

            response = await self.llm_client.chat(prompt)
            result = extract_json_from_response(response)

            if result and "qa_pairs" in result:
                return [
                    QAPair(
                        query=item.get("query", ""),
                        answer=item.get("answer", ""),
                    )
                    for item in result["qa_pairs"]
                    if item.get("query") and item.get("answer")
                    and not _is_low_quality_query(item["query"])
                ]
            return []

        except Exception:
            return []

    async def _generate_qa_with_role(
        self,
        document: Document,
        role: RoleConfig,
    ) -> Optional[QAPair]:
        """基于角色生成 QA"""
        try:
            prompt = self.prompt_manager.render(
                "qa_generation_with_role",
                document=document.content[:4000],
                character=role.character,
                question_type=role.question_type,
                difficulty=role.difficulty,
            )

            response = await self.llm_client.chat(prompt)
            result = extract_json_from_response(response)

            if result and result.get("query") and result.get("answer"):
                if _is_low_quality_query(result["query"]):
                    return None
                return QAPair(
                    query=result["query"],
                    answer=result["answer"],
                )
            return None

        except Exception:
            return None
