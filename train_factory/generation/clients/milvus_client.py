"""
Milvus 向量数据库客户端

提供 chunk 向量入库能力，用于 Phase 2 训练数据生成。
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MilvusConfig:
    """Milvus 连接配置"""
    host: str = ""
    port: int = 19530
    token: Optional[str] = None

    def __post_init__(self):
        if not self.host:
            self.host = os.environ.get("MILVUS_HOST", "localhost")
        if not self.token:
            self.token = os.environ.get("MILVUS_TOKEN", None)
        port_env = os.environ.get("MILVUS_PORT")
        if port_env and self.port == 19530:
            self.port = int(port_env)


class MilvusClient:
    """
    Milvus 向量数据库客户端

    用于将 chunk 向量入库，支持 collection 创建和批量插入。
    """

    def __init__(self, config: Optional[MilvusConfig] = None):
        self.config = config or MilvusConfig()
        self._connected = False

    def connect(self) -> None:
        """建立 Milvus 连接"""
        from pymilvus import connections

        alias = f"tf_{id(self)}"
        connect_params: Dict[str, Any] = {
            "alias": alias,
            "host": self.config.host,
            "port": self.config.port,
        }
        if self.config.token:
            connect_params["token"] = self.config.token

        connections.connect(**connect_params)
        self._alias = alias
        self._connected = True
        logger.info(f"Connected to Milvus at {self.config.host}:{self.config.port}")

    def close(self) -> None:
        """断开 Milvus 连接"""
        if self._connected:
            from pymilvus import connections
            try:
                connections.disconnect(self._alias)
            except Exception:
                pass
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def collection_exists(self, collection_name: str) -> bool:
        """检查 collection 是否存在"""
        from pymilvus import utility
        return utility.has_collection(collection_name, using=self._alias)

    def create_collection(
        self,
        collection_name: str,
        dim: int,
        metric_type: Optional[str] = None,
        enable_hybrid: bool = False,
    ) -> bool:
        """
        创建 collection

        Schema:
        - chunk_id: VARCHAR(256), primary key, 源 chunk 标识
        - chunk_content: VARCHAR(65535), chunk 文本内容
        - vector: FLOAT_VECTOR(dim), embedding 向量
        - sparse_vector: SPARSE_FLOAT_VECTOR (仅 enable_hybrid=True，BM25 自动填充)

        Args:
            collection_name: collection 名称
            dim: 向量维度
            metric_type: 距离度量类型，默认 COSINE
            enable_hybrid: 是否启用混合搜索（BM25 稀疏向量），创建后不可更改

        Returns:
            本次调用实际创建集合时返回 True；名称已存在时返回 False。
        """
        from pymilvus import (
            Collection, CollectionSchema, FieldSchema, DataType,
        )

        if self.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists, skipping creation")
            return False

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(
                name="chunk_content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                **({"enable_analyzer": True, "analyzer_params": {"type": "chinese"}} if enable_hybrid else {}),
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        functions = []
        if enable_hybrid:
            from pymilvus import Function, FunctionType

            fields.append(
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
            )
            functions.append(
                Function(
                    name="bm25_func",
                    input_field_names=["chunk_content"],
                    output_field_names=["sparse_vector"],
                    function_type=FunctionType.BM25,
                )
            )

        schema = CollectionSchema(
            fields=fields,
            functions=functions,
            description="TrainFactory chunk vectors",
        )

        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self._alias,
        )

        # 创建 Dense 索引 (IVF_FLAT)
        metric = (metric_type or "COSINE").upper()
        index_params = {
            "metric_type": metric,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index("vector", index_params)

        # 创建 Sparse 索引 (BM25)
        if enable_hybrid:
            collection.create_index(
                "sparse_vector",
                {"metric_type": "BM25", "index_type": "AUTOINDEX"},
            )

        logger.info(
            f"Created collection '{collection_name}' with dim={dim}"
            f"{', hybrid=True' if enable_hybrid else ''}"
        )
        return True

    def insert_vectors(
        self,
        collection_name: str,
        chunk_ids: List[str],
        chunk_contents: List[str],
        vectors: np.ndarray,
        batch_size: int = 1000,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        批量插入 chunk 向量（自动跳过已存在的 chunk_id）

        Args:
            collection_name: collection 名称
            chunk_ids: chunk 标识列表
            chunk_contents: chunk 文本内容列表
            vectors: embedding 向量数组，shape (n, dim)
            batch_size: 每批插入数量
            metadata: 每条 chunk 的元数据字典列表（可选）

        Returns:
            插入的总条数
        """
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        collection.load()

        # 检查集合是否有 metadata 字段
        has_metadata_field = any(f.name == "metadata" for f in collection.schema.fields)

        # 查询已存在的 chunk_ids
        existing = set()
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            # json.dumps 转义引号/反斜杠（repr 直拼在含 ' 或 \ 的 chunk_id
            # 时生成非法/可注入表达式——与 search_similar 的修复保持一致）
            expr = f'chunk_id in {json.dumps(batch, ensure_ascii=False)}'
            results = collection.query(expr=expr, output_fields=["chunk_id"])
            existing.update(r["chunk_id"] for r in results)

        # 过滤掉已存在的
        new_indices = [i for i, cid in enumerate(chunk_ids) if cid not in existing]
        if not new_indices:
            logger.info(f"All {len(chunk_ids)} chunks already exist in '{collection_name}', skipping insert")
            return 0

        # 批量插入新 chunks
        total_inserted = 0
        for i in range(0, len(new_indices), batch_size):
            batch_idx = new_indices[i:i + batch_size]
            data = [
                [chunk_ids[j] for j in batch_idx],
                [chunk_contents[j] for j in batch_idx],
                vectors[batch_idx].tolist(),
            ]
            if has_metadata_field:
                if metadata:
                    data.append([metadata[j] for j in batch_idx])
                else:
                    data.append([{} for _ in batch_idx])
            collection.insert(data)
            total_inserted += len(batch_idx)

        collection.flush()
        logger.info(
            f"Inserted {total_inserted} new vectors into '{collection_name}' "
            f"(skipped {len(existing)} existing)"
        )
        return total_inserted

    def get_vectors_by_ids(
        self,
        collection_name: str,
        chunk_ids: List[str],
        batch_size: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        按 chunk_id 批量检索已存储的向量

        Args:
            collection_name: collection 名称
            chunk_ids: 要检索的 chunk_id 列表
            batch_size: 每批查询数量

        Returns:
            Dict[chunk_id, vector]: 已存储的向量映射
        """
        from pymilvus import Collection

        if not self.collection_exists(collection_name):
            return {}

        collection = Collection(collection_name, using=self._alias)
        collection.load()

        result: Dict[str, np.ndarray] = {}
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            # json.dumps 转义引号/反斜杠（repr 直拼在含 ' 或 \ 的 chunk_id
            # 时生成非法/可注入表达式——与 search_similar 的修复保持一致）
            expr = f'chunk_id in {json.dumps(batch, ensure_ascii=False)}'
            records = collection.query(
                expr=expr,
                output_fields=["chunk_id", "vector"],
            )
            for r in records:
                result[r["chunk_id"]] = np.array(r["vector"])

        return result

    def search_similar(
        self,
        collection_name: str,
        query_vectors: np.ndarray,
        top_k: int = 10,
        exclude_chunk_ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        向量相似度检索

        Args:
            collection_name: collection 名称
            query_vectors: 查询向量，shape (n, dim)
            top_k: 每个查询返回的最大结果数
            exclude_chunk_ids: 需要排除的 chunk_id 列表（排除自身）
            filter_expr: 元数据过滤表达式，如 metadata["dataset_id"] == "ds_123"

        Returns:
            List[List[Dict]]: 每个查询的结果列表，
            每个 Dict 包含: chunk_id, chunk_content, score
        """
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        collection.load()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

        # 构建过滤表达式（合并 exclude + filter_expr）
        exprs = []
        if exclude_chunk_ids:
            # Serialize via json so ids containing quotes/backslashes are escaped
            # (manual f'"{cid}"' produced malformed/injectable Milvus expressions
            # that raise a parse error and silently drop the record on retrieval).
            import json as _json
            ids_str = _json.dumps(list(exclude_chunk_ids), ensure_ascii=False)
            exprs.append(f"chunk_id not in {ids_str}")
        if filter_expr:
            exprs.append(f"({filter_expr})")
        expr = " && ".join(exprs) if exprs else None

        results = collection.search(
            data=query_vectors.tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "chunk_content"],
        )

        output = []
        for hits in results:
            query_results = []
            for hit in hits:
                query_results.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "chunk_content": hit.entity.get("chunk_content"),
                    "score": hit.score,
                })
            output.append(query_results)
        return output

    # ── 混合搜索 ────────────────────────────────────────────

    def supports_hybrid(self, collection_name: str) -> bool:
        """检查集合是否支持混合搜索（含 sparse_vector 字段）"""
        from pymilvus import Collection

        if not self.collection_exists(collection_name):
            return False
        collection = Collection(collection_name, using=self._alias)
        return any(f.name == "sparse_vector" for f in collection.schema.fields)

    def hybrid_search(
        self,
        collection_name: str,
        search_mode: str = "hybrid",
        query_vectors: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        top_k: int = 10,
        ranker: str = "rrf",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合搜索：支持 dense / sparse / hybrid 三种模式

        Args:
            collection_name: collection 名称
            search_mode: "dense" | "sparse" | "hybrid"
            query_vectors: 查询向量 shape (1, dim)，dense/hybrid 必需
            query_text: 查询文本，sparse/hybrid 必需
            top_k: 返回结果数
            ranker: 融合策略 "rrf" | "weighted"
            dense_weight: 加权融合时 dense 权重
            sparse_weight: 加权融合时 sparse 权重

        Returns:
            结果列表，每项包含 chunk_id, chunk_content, score
        """
        from pymilvus import Collection, AnnSearchRequest, RRFRanker, WeightedRanker

        collection = Collection(collection_name, using=self._alias)
        collection.load()

        if search_mode == "sparse":
            if query_text is None:
                raise ValueError("sparse/hybrid search requires query_text")
            # Sparse 单路检索直接调用 search，保留 BM25 原始分数语义。
            results = collection.search(
                data=[query_text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=top_k,
                expr=filter_expr,
                output_fields=["chunk_id", "chunk_content"],
            )
            output = []
            for hit in results[0]:
                output.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "chunk_content": hit.entity.get("chunk_content"),
                    "score": hit.score,
                })
            return output

        if search_mode == "dense":
            if query_vectors is None:
                raise ValueError("dense/hybrid search requires query_vectors")
            results = collection.search(
                data=query_vectors.tolist(),
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=filter_expr,
                output_fields=["chunk_id", "chunk_content"],
            )
            output = []
            for hit in results[0]:
                output.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "chunk_content": hit.entity.get("chunk_content"),
                    "score": hit.score,
                })
            return output

        reqs = []

        # Dense 请求
        if search_mode in ("dense", "hybrid"):
            if query_vectors is None:
                raise ValueError("dense/hybrid search requires query_vectors")
            dense_req = AnnSearchRequest(
                data=query_vectors.tolist(),
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=filter_expr,
            )
            reqs.append(dense_req)

        # Sparse 请求
        if search_mode in ("sparse", "hybrid"):
            if query_text is None:
                raise ValueError("sparse/hybrid search requires query_text")
            sparse_req = AnnSearchRequest(
                data=[query_text],
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=top_k,
                expr=filter_expr,
            )
            reqs.append(sparse_req)

        # Hybrid 模式仅支持双路请求，融合策略按 ranker 选择。
        if ranker == "weighted":
            fused_ranker = WeightedRanker(dense_weight, sparse_weight)
        else:
            fused_ranker = RRFRanker(k=60)

        results = collection.hybrid_search(
            reqs=reqs,
            rerank=fused_ranker,
            limit=top_k,
            output_fields=["chunk_id", "chunk_content"],
        )

        output = []
        for hit in results[0]:
            output.append({
                "chunk_id": hit.entity.get("chunk_id"),
                "chunk_content": hit.entity.get("chunk_content"),
                "score": hit.score,
            })
        return output

    # ── 集合管理方法 ──────────────────────────────────────────

    def list_collections(self) -> List[str]:
        """列出所有 collection 名称"""
        from pymilvus import utility
        return utility.list_collections(using=self._alias)

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        获取 collection 详细信息

        Returns:
            包含 name, description, num_entities, dim, schema, indexes, load_state 的字典，
            collection 不存在时返回 None
        """
        from pymilvus import Collection, utility

        if not utility.has_collection(collection_name, using=self._alias):
            return None

        collection = Collection(collection_name, using=self._alias)

        # 解析 schema
        schema_info = []
        dim = None
        for f in collection.schema.fields:
            info: Dict[str, Any] = {
                "name": f.name,
                "dtype": str(f.dtype).split(".")[-1],
                "is_primary": f.is_primary,
            }
            if hasattr(f, "max_length") and f.max_length:
                info["max_length"] = f.max_length
            if hasattr(f, "dim") and f.dim:
                info["dim"] = f.dim
                dim = f.dim
            schema_info.append(info)

        # 解析索引
        index_info = []
        for idx in collection.indexes:
            index_info.append({
                "field_name": idx.field_name,
                "index_type": idx.params.get("index_type"),
                "metric_type": idx.params.get("metric_type"),
                "params": dict(idx.params),
            })

        # 加载状态
        try:
            load_state = str(utility.load_state(collection_name, using=self._alias))
            # pymilvus 返回类似 <LoadState: Loaded>，提取值
            if "Loaded" in load_state:
                load_state = "Loaded"
            elif "Loading" in load_state:
                load_state = "Loading"
            else:
                load_state = "NotLoad"
        except Exception:
            load_state = "NotLoad"

        # 检查功能支持
        field_names = {f.name for f in collection.schema.fields}
        hybrid_enabled = "sparse_vector" in field_names
        metadata_enabled = "metadata" in field_names

        return {
            "name": collection_name,
            "description": collection.schema.description or "",
            "num_entities": collection.num_entities,
            "dim": dim,
            "schema": schema_info,
            "indexes": index_info,
            "load_state": load_state,
            "hybrid_enabled": hybrid_enabled,
            "metadata_enabled": metadata_enabled,
        }

    def drop_collection(self, collection_name: str) -> bool:
        """
        删除 collection

        Returns:
            True 如果成功删除，False 如果 collection 不存在
        """
        from pymilvus import utility

        if not utility.has_collection(collection_name, using=self._alias):
            return False
        utility.drop_collection(collection_name, using=self._alias)
        logger.info(f"Dropped collection '{collection_name}'")
        return True

    def query_entities(
        self,
        collection_name: str,
        offset: int = 0,
        limit: int = 20,
        output_fields: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        分页浏览 collection 中的实体

        Args:
            collection_name: collection 名称
            offset: 起始偏移量
            limit: 返回数量（最大 100）
            output_fields: 输出字段，默认排除 vector

        Returns:
            (实体列表, 总数量)
        """
        from pymilvus import Collection

        collection = Collection(collection_name, using=self._alias)
        collection.load()

        total = collection.num_entities

        if output_fields is None:
            output_fields = [
                f.name for f in collection.schema.fields
                if f.name not in ("vector", "sparse_vector")
            ]

        limit = min(limit, 100)

        results = collection.query(
            expr="chunk_id != ''",
            output_fields=output_fields,
            offset=offset,
            limit=limit,
        )
        return results, total

    def load_collection(self, collection_name: str) -> None:
        """加载 collection 到内存"""
        from pymilvus import Collection
        collection = Collection(collection_name, using=self._alias)
        collection.load()
        logger.info(f"Loaded collection '{collection_name}'")

    def release_collection(self, collection_name: str) -> None:
        """从内存释放 collection"""
        from pymilvus import Collection
        collection = Collection(collection_name, using=self._alias)
        collection.release()
        logger.info(f"Released collection '{collection_name}'")

    def flush_collection(self, collection_name: str) -> None:
        """刷新 collection 确保数据持久化"""
        from pymilvus import Collection
        collection = Collection(collection_name, using=self._alias)
        collection.flush()
        logger.info(f"Flushed collection '{collection_name}'")

    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        """
        清理 collection 名称，确保符合 Milvus 命名规则

        规则：字母、数字、下划线，以字母开头，最长 255 字符
        """
        # 替换非法字符为下划线
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # 确保以字母开头
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'tf_' + sanitized
        # 去除连续下划线
        sanitized = re.sub(r'_+', '_', sanitized)
        # 截断长度
        return sanitized[:255]
