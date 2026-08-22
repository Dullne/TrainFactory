"""
处理步骤基类

定义处理步骤的抽象接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepResult:
    """步骤执行结果"""
    success: bool
    data: Any = None  # 步骤输出数据
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """输入文档"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None


@dataclass
class QAPair:
    """问答对"""
    query: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleConfig:
    """角色配置"""
    character: str  # 用户角色描述
    question_type: str  # 问题类型
    difficulty: str  # 难度级别
    source_type: str  # 文档来源类型
    length_type: str  # 期望长度类型

    def to_dict(self) -> Dict[str, str]:
        return {
            "character": self.character,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "source_type": self.source_type,
            "length_type": self.length_type,
        }


@dataclass
class GeneratedSample:
    """生成的样本"""
    query: str
    answer: str
    positive_chunks: List[str] = field(default_factory=list)
    negative_chunks: List[str] = field(default_factory=list)
    role_config: Optional[RoleConfig] = None
    source_doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "positive_chunks": self.positive_chunks,
            "negative_chunks": self.negative_chunks,
            "role_config": self.role_config.to_dict() if self.role_config else None,
            "source_doc_id": self.source_doc_id,
            "metadata": self.metadata,
        }


class BaseStep(ABC):
    """
    处理步骤基类

    所有处理步骤都需要继承此类并实现 execute 方法。
    """

    name: str = ""
    description: str = ""

    def __init__(self, config: Dict[str, Any], llm_client: Any = None):
        """
        初始化步骤

        Args:
            config: 步骤配置
            llm_client: LLM 客户端（如果需要）
        """
        self.config = config
        self.llm_client = llm_client
        self.enabled = config.get("enabled", True)

    @abstractmethod
    async def execute(self, input_data: Any) -> StepResult:
        """
        执行步骤

        Args:
            input_data: 输入数据（类型取决于具体步骤）

        Returns:
            StepResult: 执行结果
        """
        pass

    def is_enabled(self) -> bool:
        """检查步骤是否启用"""
        return self.enabled
