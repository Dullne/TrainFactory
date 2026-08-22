"""
后处理基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..steps.base import GeneratedSample


@dataclass
class PostProcessResult:
    """后处理结果"""
    success: bool
    samples: List[GeneratedSample]
    removed_count: int = 0
    added_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class PostProcessorBase(ABC):
    """后处理器基类"""

    name: str = ""
    description: str = ""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    async def process(self, samples: List[GeneratedSample]) -> PostProcessResult:
        """
        处理样本

        Args:
            samples: 输入样本列表

        Returns:
            PostProcessResult: 处理结果
        """
        pass

    def is_enabled(self) -> bool:
        return self.enabled
