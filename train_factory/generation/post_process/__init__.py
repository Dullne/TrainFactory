"""
后处理模块

提供数据生成后的后处理功能，包括：
- 去重 (dedup)
- 数据增广 (augmentation)
- 证据操作 (evidence_ops)
- 难负例挖掘 (mining)

所有后处理模块都是独立可选的。
"""

from .dedup import DedupProcessor
from .base import PostProcessorBase, PostProcessResult

__all__ = [
    "PostProcessorBase",
    "PostProcessResult",
    "DedupProcessor",
]
