"""
关键点生成步骤

从文档中提取关键信息点。
"""

from typing import Any, Dict, Optional

from .base import BaseStep, StepResult, Document
from ..prompts import PromptManager
from ..clients.llm_client import extract_json_from_response


class KeypointGenStep(BaseStep):
    """关键点生成步骤"""

    name = "keypoint_gen"
    description = "从文档中提取关键信息点"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: Any = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        super().__init__(config, llm_client)
        self.max_keypoints = config.get("max_keypoints", 5)
        self.prompt_manager = prompt_manager or PromptManager()

    async def execute(self, input_data: Document) -> StepResult:
        """
        提取关键点

        Args:
            input_data: 输入文档

        Returns:
            StepResult: 提取结果，data 中包含关键点列表
        """
        doc = input_data

        if not self.llm_client:
            return StepResult(
                success=False,
                error="LLM client required for keypoint extraction",
            )

        try:
            prompt = self.prompt_manager.render(
                "keypoint_extraction",
                document=doc.content[:4000],  # 限制长度
                max_keypoints=self.max_keypoints,
            )

            response = await self.llm_client.chat(prompt)
            result = extract_json_from_response(response)

            if result and "keypoints" in result:
                keypoints = result["keypoints"][:self.max_keypoints]
                return StepResult(
                    success=True,
                    data={"keypoints": keypoints},
                    details={"count": len(keypoints)},
                )
            else:
                return StepResult(
                    success=False,
                    error="Failed to extract keypoints",
                    details={"raw_response": response[:500]},
                )

        except Exception as e:
            return StepResult(
                success=False,
                error=f"Keypoint extraction failed: {str(e)}",
            )
