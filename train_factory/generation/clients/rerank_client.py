"""
Rerank 客户端

提供异步 Rerank 调用能力。
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ...storage.services.outbound_endpoint_policy import create_pinned_async_client
from ...storage.services.inference_authorization_service import authorize_inference_model


def _resolve_rerank_text(item: Dict[str, Any], documents: List[str]) -> str:
    """Resolve a rerank result's document text across API variants.

    Handles an in-range integer 'index' into the original documents, a 'document'
    field that is either a plain string or a {'text': ...} object, and missing or
    out-of-range indices (returns '' instead of raising AttributeError/IndexError).
    """
    idx = item.get("index")
    if isinstance(idx, int) and 0 <= idx < len(documents):
        return documents[idx]
    doc = item.get("document")
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return doc.get("text", "") or ""
    return ""




@dataclass
class RerankConfig:
    """Rerank 配置"""
    endpoint: str
    model: str
    api_key: Optional[str] = None
    top_k: int = 10
    batch_size: int = 64  # 单次请求最大文档数
    concurrency: int = 10
    timeout: int = 60
    max_retries: int = 3
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key": "***" if self.api_key else None,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "concurrency": self.concurrency,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class RerankResult:
    """Rerank 结果"""
    index: int
    score: float
    text: str


class RerankClient:
    """
    Rerank 客户端

    支持常见的 Rerank API 格式。
    """

    def __init__(self, config: RerankConfig):
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

    def _build_rerank_urls(self) -> List[str]:
        endpoint = self.config.endpoint.rstrip("/")
        if endpoint.endswith("/v1/rerank") or endpoint.endswith("/rerank"):
            return [endpoint]

        urls: List[str] = []
        if endpoint.endswith("/v1"):
            urls.append(f"{endpoint}/rerank")
        else:
            urls.append(f"{endpoint}/rerank")
            urls.append(f"{endpoint}/v1/rerank")

        seen = set()
        unique_urls = []
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            unique_urls.append(url)
        return unique_urls

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        重排序文档

        当 documents 超过 batch_size 时自动分批请求，合并后取 top_k。

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回 top-k 结果（默认使用配置值）

        Returns:
            排序后的结果列表
        """
        top_k = top_k or self.config.top_k

        if len(documents) <= self.config.batch_size:
            return await self._rerank_single(query, documents, top_k)

        # 分批 rerank，合并结果
        all_results: List[RerankResult] = []
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i:i + self.config.batch_size]
            batch_results = await self._rerank_single(
                query, batch_docs, min(top_k, len(batch_docs))
            )
            # 修正 index 为原始文档列表中的位置
            for r in batch_results:
                all_results.append(RerankResult(
                    index=r.index + i,
                    score=r.score,
                    text=r.text,
                ))

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    async def _rerank_single(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> List[RerankResult]:
        """单批次 rerank 请求"""
        await asyncio.to_thread(
            authorize_inference_model, self.config.endpoint,
            self.config.model, self.config.user_id,
        )
        client = self._get_client()
        urls = self._build_rerank_urls()

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            for url in urls:
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()

                    data = response.json()
                    results = []

                    # 解析响应（支持多种 API 格式）
                    if "results" in data:
                        # Cohere/BGE 风格
                        for item in data["results"]:
                            results.append(RerankResult(
                                index=item.get("index", 0),
                                score=item.get("relevance_score", item.get("score", 0.0)),
                                text=_resolve_rerank_text(item, documents),
                            ))
                    elif "data" in data:
                        # OpenAI 风格
                        for item in data["data"]:
                            results.append(RerankResult(
                                index=item.get("index", 0),
                                score=item.get("score", 0.0),
                                text=_resolve_rerank_text(item, documents),
                            ))

                    # 按分数降序排序
                    results.sort(key=lambda x: x.score, reverse=True)
                    return results[:top_k]

                except httpx.HTTPStatusError as e:
                    last_error = e
                    if e.response.status_code == 404:
                        continue
                    if e.response.status_code >= 500:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
                except httpx.TransportError as e:
                    # 仅重试传输层错误（超时/连接失败）；解析错误立即失败
                    last_error = e
                    await asyncio.sleep(2 ** attempt)
                    break

        raise RuntimeError(f"Rerank API call failed after {self.config.max_retries} retries: {last_error}")

    async def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        获取所有文档的相关性分数

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            分数列表（与输入文档顺序对应）
        """
        # 获取所有文档的 rerank 结果
        results = await self.rerank(query, documents, top_k=len(documents))

        # 按原始索引构建分数列表
        scores = [0.0] * len(documents)
        for result in results:
            scores[result.index] = result.score

        return scores

    async def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankResult]]:
        """
        批量重排序

        Args:
            queries: 查询列表
            documents_list: 每个查询对应的文档列表
            top_k: 每个查询返回的 top-k 数量

        Returns:
            每个查询的重排序结果
        """
        async def _rerank_with_semaphore(query: str, docs: List[str]) -> List[RerankResult]:
            async with self._semaphore:
                return await self.rerank(query, docs, top_k)

        tasks = [
            _rerank_with_semaphore(query, docs)
            for query, docs in zip(queries, documents_list)
        ]
        return await asyncio.gather(*tasks)

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
