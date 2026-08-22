"""
Embedding 相似度筛选步骤

Phase 2 (doc_to_training / qa_to_training) 的核心步骤：
1. 批量向量化 query 和 chunk
2. 计算余弦相似度，过滤高相似度对（不需要训练）
3. 同时将 chunk 向量入库 Milvus

向量库复用：若 Milvus collection 已存在（如前次运行入库），
自动从 Milvus 取回已有向量，跳过重复 embedding 计算。
换阈值重跑时只需重新计算相似度，无需重建向量库。
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..clients.embedding_client import EmbeddingClient
from ..clients.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class EmbeddingFilterStep:
    """
    Embedding 相似度筛选 + Milvus 入库

    对 QA 数据集中每个 (query, chunk) 对计算 embedding 余弦相似度。
    相似度 > threshold 的对被过滤掉（embedding 已经能很好检索，不需要训练）。
    threshold >= 1.0 时跳过过滤，只计算 chunk 向量入库 Milvus。
    同时将所有 chunk 的 embedding 入库到 Milvus。

    支持两种使用模式:
    - execute(): 全量处理（QA 数据集路径）
    - execute_batch() + finalize_stats(): 增量流式处理（文档→QA→筛选流水线）
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        milvus_client: Optional[MilvusClient] = None,
        threshold: float = 0.85,
        collection_name: Optional[str] = None,
        default_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.embedding_client = embedding_client
        self.milvus_client = milvus_client
        self.threshold = threshold
        self.collection_name = collection_name
        self.default_metadata = default_metadata

        # chunk embedding 缓存（跨批次 + Milvus 复用）
        self._chunk_emb_cache: Dict[str, np.ndarray] = {}
        self._milvus_cache_loaded: bool = False

        # 增量模式内部状态
        self._unique_chunks: Dict[str, str] = {}  # chunk_id -> chunk_content（跨批次去重）
        self._collection_created: bool = False
        self._total_pairs: int = 0
        self._filtered_out: int = 0
        self._kept: int = 0
        self._milvus_inserted: int = 0
        self._milvus_cached: int = 0
        self._all_similarities: List[float] = []

    def _load_milvus_cache(self, chunk_ids: List[str]) -> None:
        """从 Milvus 预加载已有向量到缓存"""
        if not self.milvus_client or not self.collection_name:
            return
        if not chunk_ids:
            return

        try:
            existing = self.milvus_client.get_vectors_by_ids(
                self.collection_name, chunk_ids,
            )
            if existing:
                self._chunk_emb_cache.update(existing)
                self._milvus_cached += len(existing)
                logger.info(
                    f"  Milvus cache hit: {len(existing)}/{len(chunk_ids)} chunks "
                    f"from '{self.collection_name}'"
                )
        except Exception as e:
            logger.warning(f"  Milvus cache load failed: {e}")

    def _separate_new_chunks(
        self, unique_chunk_ids: List[str], unique_chunks: Dict[str, str],
    ) -> Tuple[List[str], List[str]]:
        """将 chunks 分为已缓存和需要新 embed 的"""
        new_ids = [cid for cid in unique_chunk_ids if cid not in self._chunk_emb_cache]
        new_texts = [unique_chunks[cid] for cid in new_ids]
        return new_ids, new_texts

    async def _embed_and_cache(
        self, new_ids: List[str], new_texts: List[str],
    ) -> Optional[np.ndarray]:
        """Embed 新 chunks 并更新缓存，返回原始 embeddings 用于 Milvus 插入"""
        if not new_ids:
            return None
        logger.info(f"  Embedding {len(new_ids)} new chunks...")
        embeddings = await self.embedding_client.embed(new_texts)
        for cid, emb in zip(new_ids, embeddings):
            self._chunk_emb_cache[cid] = emb
        return embeddings

    def _compute_similarities(
        self,
        qa_records: List[Dict[str, Any]],
        query_embeddings: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[int], List[float], int]:
        """计算 query-chunk 相似度并筛选"""
        similarities = []
        filtered_records = []
        filtered_indices = []

        for i, record in enumerate(qa_records):
            q_emb = query_embeddings[i]
            c_emb = self._chunk_emb_cache[record["chunk_id"]]

            q_norm = np.linalg.norm(q_emb)
            c_norm = np.linalg.norm(c_emb)
            if q_norm > 0 and c_norm > 0:
                sim = float(np.dot(q_emb, c_emb) / (q_norm * c_norm))
            else:
                sim = 0.0
            similarities.append(sim)

            if sim <= self.threshold:
                filtered_records.append(record)
                filtered_indices.append(i)

        filtered_out = len(qa_records) - len(filtered_records)
        return filtered_records, filtered_indices, similarities, filtered_out

    def _insert_new_to_milvus(
        self,
        new_ids: List[str],
        new_texts: List[str],
        new_embeddings: Optional[np.ndarray],
    ) -> Tuple[Optional[str], int]:
        """将新 chunks 插入 Milvus"""
        if not new_ids or new_embeddings is None:
            # 没有新 chunks，但 collection 可能已存在
            if self.milvus_client and self.collection_name:
                if self.milvus_client.collection_exists(self.collection_name):
                    return self.collection_name, 0
            return None, 0

        if not self.milvus_client or not self.collection_name:
            return None, 0

        try:
            dim = (
                new_embeddings.shape[1]
                if len(new_embeddings.shape) > 1
                else len(new_embeddings[0])
            )
            self.milvus_client.create_collection(self.collection_name, dim)
            metadata_list = [dict(self.default_metadata) for _ in new_ids] if self.default_metadata else None
            inserted = self.milvus_client.insert_vectors(
                collection_name=self.collection_name,
                chunk_ids=new_ids,
                chunk_contents=new_texts,
                vectors=new_embeddings,
                metadata=metadata_list,
            )
            logger.info(f"  Milvus: inserted {inserted} chunks into '{self.collection_name}'")
            return self.collection_name, inserted
        except Exception:
            for chunk_id in new_ids:
                self._chunk_emb_cache.pop(chunk_id, None)
            logger.exception("  Milvus ingestion failed")
            raise

    async def pre_index_all_chunks(
        self,
        documents: list,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Phase 0: 预索引所有文档 chunk 到 Milvus，填充内部缓存。

        将每个文档视为一个 chunk（chunk_id = doc_id, chunk_content = doc.content），
        批量 embed 并入库 Milvus。执行完后 _chunk_emb_cache 和 _unique_chunks
        已包含所有文档，后续 execute_batch 不再需要 embed chunk 或插入 Milvus。

        Args:
            documents: 文档列表，需有 doc_id 和 content 属性
            progress_callback: 进度回调 (completed, total)

        Returns:
            新插入 Milvus 的 chunk 数量
        """
        # 1. 收集 unique chunks: doc_id → doc.content
        unique_chunks: Dict[str, str] = {}
        for doc in documents:
            if doc.doc_id and doc.content:
                if doc.doc_id not in unique_chunks:
                    unique_chunks[doc.doc_id] = doc.content

        # 更新内部 _unique_chunks 状态
        self._unique_chunks.update(unique_chunks)

        chunk_ids = list(unique_chunks.keys())

        if not chunk_ids:
            if progress_callback:
                progress_callback(0, 0)
            return 0

        total = len(chunk_ids)
        logger.info(f"  Pre-indexing {total} document chunks into Milvus")

        # 2. 从 Milvus 加载已有向量（断点续传友好）
        self._load_milvus_cache(chunk_ids)
        self._milvus_cache_loaded = True

        # 3. 找出需要新 embed 的 chunks
        new_ids, new_texts = self._separate_new_chunks(chunk_ids, unique_chunks)

        if not new_ids:
            # 全部已缓存，无需 embed
            logger.info(f"  All {total} chunks already cached, skipping embed")
            if progress_callback:
                progress_callback(total, total)
            return 0

        logger.info(f"  Need to embed {len(new_ids)}/{total} new chunks")

        # 4. 分批 embed（_embed_and_cache 内部已处理 batch_size）
        new_embeddings = await self._embed_and_cache(new_ids, new_texts)

        # 5. 入库 Milvus
        _, inserted = self._insert_new_to_milvus(new_ids, new_texts, new_embeddings)
        # Count Phase 0 inserts toward the running total so finalize_stats reports
        # the collection instead of None (otherwise the freshly-created collection
        # is orphaned: unregistered, unlinked, not reusable by later runs).
        self._milvus_inserted += inserted

        if progress_callback:
            progress_callback(total, total)

        logger.info(
            f"  Pre-index complete: {inserted} new chunks inserted, "
            f"{total - len(new_ids)} cached"
        )
        return inserted


    async def execute(
        self,
        qa_records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        全量筛选和入库（qa_to_training 使用）

        流程：
        1. 收集 unique chunks
        2. 从 Milvus 取回已有向量（跳过重复 embedding）
        3. 只 embed 新 chunks
        4. 计算 query-chunk 相似度 + 筛选
        5. 只对新 chunks 执行 Milvus insert
        """
        total = len(qa_records)
        if total == 0:
            return [], {"total_pairs": 0, "filtered_out": 0, "kept": 0}

        skip_filter = self.threshold >= 1.0
        logger.info(
            f"EmbeddingFilterStep: processing {total} QA pairs, "
            f"threshold={self.threshold}{' (skip filter, index only)' if skip_filter else ''}"
        )

        # 1. 提取 unique chunks
        queries = [r["query"] for r in qa_records]
        chunk_ids = [r["chunk_id"] for r in qa_records]
        unique_chunks: Dict[str, str] = {}
        for cid, content in zip(chunk_ids, [r["chunk_content"] for r in qa_records]):
            if cid not in unique_chunks:
                unique_chunks[cid] = content
        unique_chunk_ids = list(unique_chunks.keys())

        logger.info(f"  Unique chunks: {len(unique_chunk_ids)}, Total QA pairs: {total}")

        # 2. 从 Milvus 预加载已有向量
        self._load_milvus_cache(unique_chunk_ids)

        # 3. 只 embed 新 chunks
        new_ids, new_texts = self._separate_new_chunks(unique_chunk_ids, unique_chunks)
        new_embeddings = await self._embed_and_cache(new_ids, new_texts)

        if skip_filter:
            filtered_records = list(qa_records)
            filtered_indices = list(range(total))
            similarities: List[float] = []
            filtered_out = 0
            logger.info(f"  Filter skipped (threshold={self.threshold}), kept all {total} pairs")
        else:
            # 4. embed queries + 计算相似度
            logger.info("  Embedding queries...")
            query_embeddings = await self.embedding_client.embed(queries)
            filtered_records, filtered_indices, similarities, filtered_out = (
                self._compute_similarities(qa_records, query_embeddings)
            )
            logger.info(
                f"  Filter result: total={total}, filtered_out={filtered_out} "
                f"(sim>{self.threshold}), kept={len(filtered_records)}"
            )

        # 5. 只对新 chunks 入库 Milvus
        milvus_collection, milvus_inserted = self._insert_new_to_milvus(
            new_ids, new_texts, new_embeddings,
        )

        stats = {
            "total_pairs": total,
            "filtered_out": filtered_out,
            "kept": len(filtered_records),
            "unique_chunks": len(unique_chunk_ids),
            "milvus_collection": milvus_collection,
            "milvus_inserted": milvus_inserted,
            "milvus_cached": self._milvus_cached,
            "threshold": self.threshold,
            "similarity_avg": float(np.mean(similarities)) if similarities else 0,
            "similarity_min": float(np.min(similarities)) if similarities else 0,
            "similarity_max": float(np.max(similarities)) if similarities else 0,
            "filtered_indices": filtered_indices,
        }

        return filtered_records, stats

    async def execute_batch(
        self,
        qa_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        增量处理一批 QA 记录（doc_to_training 流式模式）

        首次调用时从 Milvus 预加载已有向量，后续批次通过内存缓存复用。

        Args:
            qa_records: 一批 QA 记录

        Returns:
            通过筛选的记录列表
        """
        batch_size = len(qa_records)
        if batch_size == 0:
            return []

        self._total_pairs += batch_size
        skip_filter = self.threshold >= 1.0

        # 1. 提取 queries 和 chunks
        queries = [r["query"] for r in qa_records]
        chunk_ids = [r["chunk_id"] for r in qa_records]
        chunk_contents = [r["chunk_content"] for r in qa_records]

        # 2. 找出本批次新 chunks（跨批次去重）
        new_chunk_ids = []
        new_chunk_texts = []
        for cid, content in zip(chunk_ids, chunk_contents):
            if cid not in self._unique_chunks:
                self._unique_chunks[cid] = content
                new_chunk_ids.append(cid)
                new_chunk_texts.append(content)

        # 3. 首次调用时从 Milvus 预加载已有向量
        if not self._milvus_cache_loaded and new_chunk_ids:
            self._load_milvus_cache(new_chunk_ids)
            self._milvus_cache_loaded = True

        # 4. 只 embed 不在缓存中的新 chunks
        truly_new_ids = [cid for cid in new_chunk_ids if cid not in self._chunk_emb_cache]
        truly_new_texts = [self._unique_chunks[cid] for cid in truly_new_ids]
        new_embeddings = await self._embed_and_cache(truly_new_ids, truly_new_texts)

        if skip_filter:
            filtered_records = list(qa_records)
            self._kept += batch_size
        else:
            # 5. embed queries + 计算相似度
            query_embeddings = await self.embedding_client.embed(queries)
            filtered_records = []
            for i, record in enumerate(qa_records):
                q_emb = query_embeddings[i]
                c_emb = self._chunk_emb_cache[record["chunk_id"]]

                q_norm = np.linalg.norm(q_emb)
                c_norm = np.linalg.norm(c_emb)
                if q_norm > 0 and c_norm > 0:
                    sim = float(np.dot(q_emb, c_emb) / (q_norm * c_norm))
                else:
                    sim = 0.0
                self._all_similarities.append(sim)

                if sim <= self.threshold:
                    filtered_records.append(record)
                    self._kept += 1
                else:
                    self._filtered_out += 1

        # 6. 入库 Milvus（仅真正新的 chunks）
        if truly_new_ids and new_embeddings is not None and self.milvus_client and self.collection_name:
            try:
                if not self._collection_created:
                    dim = (
                        new_embeddings.shape[1]
                        if len(new_embeddings.shape) > 1
                        else len(new_embeddings[0])
                    )
                    self.milvus_client.create_collection(self.collection_name, dim)
                    self._collection_created = True

                metadata_list = [dict(self.default_metadata) for _ in truly_new_ids] if self.default_metadata else None
                inserted = self.milvus_client.insert_vectors(
                    collection_name=self.collection_name,
                    chunk_ids=truly_new_ids,
                    chunk_contents=truly_new_texts,
                    vectors=new_embeddings,
                    metadata=metadata_list,
                )
                self._milvus_inserted += inserted
                logger.debug(
                    f"  Milvus batch insert: {inserted} new chunks (total: {self._milvus_inserted})"
                )
            except Exception:
                for chunk_id in truly_new_ids:
                    self._chunk_emb_cache.pop(chunk_id, None)
                    self._unique_chunks.pop(chunk_id, None)
                logger.exception("  Milvus batch ingestion failed")
                raise

        logger.debug(
            f"  EmbeddingFilterStep batch: {batch_size} pairs, "
            f"kept={len(filtered_records)}, new_chunks={len(truly_new_ids)}, "
            f"cached={len(new_chunk_ids) - len(truly_new_ids)}"
        )

        return filtered_records

    def finalize_stats(self) -> Dict[str, Any]:
        """
        返回增量处理的全局累积统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_pairs": self._total_pairs,
            "filtered_out": self._filtered_out,
            "kept": self._kept,
            "unique_chunks": len(self._unique_chunks),
            "milvus_collection": self.collection_name if self._milvus_inserted > 0 or self._milvus_cached > 0 else None,
            "milvus_inserted": self._milvus_inserted,
            "milvus_cached": self._milvus_cached,
            "threshold": self.threshold,
            "similarity_avg": float(np.mean(self._all_similarities)) if self._all_similarities else 0,
            "similarity_min": float(np.min(self._all_similarities)) if self._all_similarities else 0,
            "similarity_max": float(np.max(self._all_similarities)) if self._all_similarities else 0,
        }
