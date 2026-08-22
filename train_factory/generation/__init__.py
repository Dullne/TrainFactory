"""
数据集生成模块

提供从知识库文档生成训练/评估数据集的能力。
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # clients
    "LLMClient": (".clients", "LLMClient"),
    "LLMClientPool": (".clients", "LLMClientPool"),
    "LLMConfig": (".clients", "LLMConfig"),
    "EmbeddingClient": (".clients", "EmbeddingClient"),
    "EmbeddingConfig": (".clients", "EmbeddingConfig"),
    "RerankClient": (".clients", "RerankClient"),
    "RerankConfig": (".clients", "RerankConfig"),
    # prompts
    "PromptManager": (".prompts", "PromptManager"),
    "SOURCE_TYPES": (".prompts", "SOURCE_TYPES"),
    "LENGTH_TYPES": (".prompts", "LENGTH_TYPES"),
    # pipeline
    "DatasetGenerationPipeline": (".pipeline", "DatasetGenerationPipeline"),
    "PipelineConfig": (".pipeline", "PipelineConfig"),
    "PipelineResult": (".pipeline", "PipelineResult"),
    # processor
    "DocumentProcessor": (".processor", "DocumentProcessor"),
    "ProcessorConfig": (".processor", "ProcessorConfig"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
