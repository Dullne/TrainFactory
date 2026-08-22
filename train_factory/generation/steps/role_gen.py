"""
角色配置生成步骤

生成用户角色配置以增加数据多样性。
"""

from typing import Any, Dict, List, Optional

from .base import BaseStep, StepResult, Document, QAPair, RoleConfig
from ..prompts import PromptManager, SOURCE_TYPES
from ..clients.llm_client import extract_json_from_response

# 带描述的来源类型列表，供 prompt 注入
_SOURCE_TYPE_OPTIONS = "\n".join([
    f"- {k}: {v['name']}（{v['description']}）"
    for k, v in SOURCE_TYPES.items()
])


class RoleGenStep(BaseStep):
    """角色配置生成步骤"""

    name = "role_gen"
    description = "生成用户角色配置以增加多样性"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Any = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        super().__init__(config, llm_client)
        self.roles_per_doc = config.get("roles_per_doc", 3)
        self.prompt_manager = prompt_manager or PromptManager()

    async def execute(self, input_data: Dict[str, Any]) -> StepResult:
        """
        生成角色配置

        Args:
            input_data: 包含 document 和可选的 qa_pair

        Returns:
            StepResult: 生成结果，data 中包含角色配置列表
        """
        document = input_data.get("document")
        qa_pair = input_data.get("qa_pair")

        if not self.llm_client:
            return StepResult(
                success=False,
                error="LLM client required for role generation",
            )

        try:
            roles = []

            # 如果有 QA ��，基于 QA 生成角色
            if qa_pair:
                for _ in range(self.roles_per_doc):
                    role = await self._generate_role_from_qa(qa_pair)
                    if role:
                        roles.append(role)
            elif document:
                # 基于文档内容用 LLM 生成角色
                roles = await self._generate_roles_from_doc(document)

            return StepResult(
                success=len(roles) > 0,
                data={"roles": roles},
                details={"count": len(roles)},
            )

        except Exception as e:
            return StepResult(
                success=False,
                error=f"Role generation failed: {str(e)}",
            )

    async def _generate_role_from_qa(self, qa_pair: QAPair) -> RoleConfig:
        """基于 QA 对生成角色配置"""
        try:
            prompt = self.prompt_manager.render(
                "role_generation",
                query=qa_pair.query,
                answer=qa_pair.answer,
                max_roles=1,
            )

            response = await self.llm_client.chat(prompt)
            result = extract_json_from_response(response)

            if result:
                # 兼容数组和单对象两种返回格式
                if isinstance(result, list):
                    result = result[0] if result else {}
                return RoleConfig(
                    character=result.get("character", "普通用户"),
                    question_type=result.get("question_type", "factual"),
                    difficulty=result.get("difficulty", "intermediate"),
                    source_type=result.get("source_type", "documentation"),
                    length_type=result.get("length_type", "medium"),
                )
            return None

        except Exception:
            return None

    async def _generate_roles_from_doc(self, document: Document) -> List[RoleConfig]:
        """基于文档内容用 LLM 生成角色配置"""
        try:
            prompt = self.prompt_manager.render(
                "role_generation_from_doc",
                document=document.content[:4000],
                num_roles=self.roles_per_doc,
            )

            response = await self.llm_client.chat(prompt)
            result = extract_json_from_response(response)

            if result and "roles" in result:
                raw_roles = result["roles"]
            elif isinstance(result, list):
                raw_roles = result
            else:
                return []

            roles = []
            for r in raw_roles[:self.roles_per_doc]:
                roles.append(RoleConfig(
                    character=r.get("character", "普通用户"),
                    question_type=r.get("question_type", "factual"),
                    difficulty=r.get("difficulty", "intermediate"),
                    source_type=r.get("source_type", "documentation"),
                    length_type=r.get("length_type", "medium"),
                ))
            return roles

        except Exception:
            return []
