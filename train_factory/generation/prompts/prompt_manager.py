"""
Prompt 管理器

管理和渲染 Prompt 模板，支持用户自定义覆盖。
"""

import logging
from typing import Dict, Optional

from . import default_prompts

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Prompt 管理器

    管理 Prompt 模板，支持用户自定义覆盖默认模板。
    """

    # 默认 Prompt 映射
    DEFAULT_PROMPTS = {
        "role_generation": default_prompts.ROLE_GENERATION_PROMPT,
        "role_generation_from_doc": default_prompts.ROLE_GENERATION_FROM_DOC_PROMPT,
        "keypoint_extraction": default_prompts.KEYPOINT_EXTRACTION_PROMPT,
        "qa_generation": default_prompts.QA_GENERATION_PROMPT,
        "qa_generation_with_role": default_prompts.QA_GENERATION_WITH_ROLE_PROMPT,
        "positive_chunk": default_prompts.POSITIVE_CHUNK_PROMPT,
        "positive_chunk_with_role": default_prompts.POSITIVE_CHUNK_WITH_ROLE_PROMPT,
        "negative_chunk": default_prompts.NEGATIVE_CHUNK_PROMPT,
        "negative_chunk_with_role": default_prompts.NEGATIVE_CHUNK_WITH_ROLE_PROMPT,
        "validation": default_prompts.VALIDATION_PROMPT,
        "doc_quality": default_prompts.DOC_QUALITY_PROMPT,
    }

    def __init__(self, custom_prompts: Optional[Dict[str, str]] = None):
        """
        初始化 Prompt 管理器

        Args:
            custom_prompts: 自定义 Prompt 字典，key 为 Prompt 名称，value 为模板字符串
        """
        self.prompts = self.DEFAULT_PROMPTS.copy()
        if custom_prompts:
            for name, template in custom_prompts.items():
                if template:  # 只有非空模板才覆盖
                    self.prompts[name] = template

    def get(self, name: str) -> str:
        """
        获取 Prompt 模板

        Args:
            name: Prompt 名称

        Returns:
            Prompt 模板字符串

        Raises:
            KeyError: 如果 Prompt 名称不存在
        """
        if name not in self.prompts:
            raise KeyError(f"Unknown prompt: {name}. Available: {list(self.prompts.keys())}")
        return self.prompts[name]

    def render(self, name: str, **kwargs) -> str:
        """
        渲染 Prompt

        Args:
            name: Prompt 名称
            **kwargs: 模板变量

        Returns:
            渲染后的 Prompt 字符串
        """
        template = self.get(name)
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError, ValueError) as e:
            # A custom/user-supplied template containing unescaped literal braces
            # (e.g. a pasted JSON example like {"qa_pairs": [...]}) makes
            # str.format treat them as replacement fields and raise. Log and
            # re-raise a clear error so the failure is never swallowed silently
            # into empty generation without any trace of the root cause.
            logger.error(
                "Failed to render prompt '%s': %s. If this is a custom prompt, "
                "escape literal braces by doubling them ('{{' and '}}').",
                name,
                e,
            )
            raise ValueError(
                f"Failed to render prompt '{name}': {e}. If using a custom "
                f"prompt, escape literal braces by doubling them ('{{{{'/'}}}}')."
            ) from e

    def set(self, name: str, template: str) -> None:
        """
        设置 Prompt 模板

        Args:
            name: Prompt 名称
            template: 模板字符串
        """
        self.prompts[name] = template

    def list_prompts(self) -> Dict[str, str]:
        """
        列出所有可用的 Prompt

        Returns:
            Prompt 名称到模板的字典
        """
        return self.prompts.copy()

    @classmethod
    def get_source_types(cls) -> dict:
        """获取可用的文档来源类型"""
        return default_prompts.SOURCE_TYPES.copy()

    @classmethod
    def get_length_types(cls) -> dict:
        """获取可用的长度类型"""
        return default_prompts.LENGTH_TYPES.copy()
