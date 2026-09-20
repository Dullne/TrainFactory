"""
Embedding 客户端

提供异步批量 Embedding 调用能力。
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx
import numpy as np

from ...storage.services.outbound_endpoint_policy import create_pinned_async_client
from ...storage.services.inference_authorization_service import authorize_inference_model


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    endpoint: str
    model: str
    api_key: Optional[str] = None
    batch_size: int = 32
    concurrency: int = 20
    timeout: int = 60
    max_retries: int = 3
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key": "***" if self.api_key else None,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


class EmbeddingClient:
    """
    Embedding 客户端

    支持 OpenAI 兼容的 Embedding API 格式，带并发控制。
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def __aenter__(self):
        self._client = self._new_http_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._new_http_client()
        return self._client

    def _new_http_client(self) -> httpx.AsyncClient:
        return create_pinned_async_client(
            self.config.endpoint,
            self.config.user_id,
            timeout=self.config.timeout,
        )

    async def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        获取文本的 Embedding

        Args:
            texts: 单个文本或文本列表

        Returns:
            Embedding 向量数组，shape 为 (n_texts, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # 分批并发处理
        batches = [
            texts[i:i + self.config.batch_size]
            for i in range(0, len(texts), self.config.batch_size)
        ]

        async def _embed_with_semaphore(batch: List[str]) -> List[List[float]]:
            async with self._semaphore:
                return await self._embed_batch(batch)

        results = await asyncio.gather(*[_embed_with_semaphore(b) for b in batches])

        all_embeddings = []
        for batch_result in results:
            all_embeddings.extend(batch_result)

        # Fail loudly on a short/incomplete server response instead of letting a
        # downstream zip(ids, embeddings) silently truncate and leave chunk_ids
        # uncached (which later surfaces as an opaque KeyError).
        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(all_embeddings)} vectors for "
                f"{len(texts)} inputs (incomplete API response)."
            )

        return np.array(all_embeddings)

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """嵌入一批文本"""
        await asyncio.to_thread(
            authorize_inference_model, self.config.endpoint,
            self.config.model, self.config.user_id,
        )
        client = self._get_client()
        endpoint = self.config.endpoint.rstrip('/')
        if endpoint.endswith('/v1/embeddings') or endpoint.endswith('/embeddings'):
            url = endpoint
        elif endpoint.endswith('/v1'):
            url = f"{endpoint}/embeddings"
        else:
            url = f"{endpoint}/v1/embeddings"

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "input": texts,
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                data = response.json()
                # 校验 schema 并按索引排序；部分 OpenAI 兼容服务不返回 'data'
                # 或 per-item 'index'，此时回退到服务端返回顺序而非直接崩溃。
                items = data.get("data") if isinstance(data, dict) else data
                if not isinstance(items, list):
                    raise RuntimeError(
                        f"Embedding API returned unexpected schema: {str(data)[:200]}"
                    )
                if all(isinstance(it, dict) and "index" in it for it in items):
                    items = sorted(items, key=lambda x: x["index"])
                return [it["embedding"] for it in items]

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
            except httpx.TransportError as e:
                # 仅重试传输层错误（超时/连接失败）；schema 异常是确定性的，立即失败
                last_error = e
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Embedding API call failed after {self.config.max_retries} retries: {last_error}")

    async def similarity(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        计算 query 与 documents 的相似度

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            相似度分数列表
        """
        # 嵌入 query 和 documents
        all_texts = [query] + documents
        embeddings = await self.embed(all_texts)

        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # 计算余弦相似度
        # 零向量（空文本/embedding 服务异常输出）归一化会产生 NaN 并污染下游评分与缓存：
        # 直接失败而非兜底成 0 相似度，让调用方感知上游异常。
        query_norm = float(np.linalg.norm(query_emb))
        if query_norm <= 0:
            raise RuntimeError(
                "Query embedding is a zero vector (empty text or malformed "
                "embedding response); cannot compute similarity"
            )
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        zero_docs = np.where(doc_norms.ravel() <= 0)[0].tolist()
        if zero_docs:
            raise RuntimeError(
                f"Document embedding(s) at index {zero_docs} are zero vectors "
                "(empty text or malformed embedding response); cannot compute similarity"
            )
        query_unit = query_emb / query_norm
        doc_units = doc_embs / doc_norms
        similarities = np.dot(doc_units, query_unit)

        return similarities.tolist()

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
