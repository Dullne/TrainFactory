"""
去重处理器

基于 query 文本哈希去除完全重复的样本。
"""

from typing import Any, Dict, List, Set
import hashlib

from .base import PostProcessorBase, PostProcessResult
from ..steps.base import GeneratedSample


class DedupProcessor(PostProcessorBase):
    """基于 query 文本哈希的去重处理器"""

    name = "dedup"
    description = "基于 query 文本哈希去除完全重复的样本"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    async def process(self, samples: List[GeneratedSample]) -> PostProcessResult:
        if not samples:
            return PostProcessResult(success=True, samples=[], removed_count=0)

        deduped_samples = self._dedup_by_hash(samples)
        removed_count = len(samples) - len(deduped_samples)

        return PostProcessResult(
            success=True,
            samples=deduped_samples,
            removed_count=removed_count,
            details={
                "original_count": len(samples),
                "deduped_count": len(deduped_samples),
            },
        )

    def _dedup_by_hash(self, samples: List[GeneratedSample]) -> List[GeneratedSample]:
        """基于完整样本内容哈希去重（query+answer+正/负例+来源，MD5 完全匹配）。

        仅按 query 去重会误删「同一 query 配不同正/负文档」的合法样本，因此这里
        对整条样本内容做指纹，只有真正完全重复的样本才会被去除。
        """
        seen_hashes: Set[str] = set()
        deduped = []

        for sample in samples:
            fingerprint = "\x1f".join([
                (sample.query or "").lower().strip(),
                (sample.answer or "").lower().strip(),
                "|".join(sorted(sample.positive_chunks or [])),
                "|".join(sorted(sample.negative_chunks or [])),
                sample.source_doc_id or "",
            ])
            h = hashlib.md5(fingerprint.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped.append(sample)

        return deduped
