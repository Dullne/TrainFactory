"""Reranker client that applies TrainFactory's outbound request policy."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_VERSIONED_API_PATH = re.compile(r"/v[1-9]\d*$")


def _normalize_rerank_endpoint(endpoint: str, framework: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/rerank"):
        return normalized
    if framework == "vllm" or _VERSIONED_API_PATH.search(normalized):
        return normalized + "/rerank"
    return normalized + "/v1/rerank"


def request_user_outbound(
    method: str,
    url: str,
    user_id: Optional[str],
    **kwargs: Any,
):
    """Import lazily so the evaluation package remains cheap to import."""
    from ..storage.services.outbound_endpoint_policy import (
        request_user_outbound as secure_request,
    )

    return secure_request(method, url, user_id, **kwargs)


class SecureAPIReranker:
    """API reranker compatible with qwen3-rerank-trainer evaluators."""

    def __init__(
        self,
        endpoint: str,
        model: str = "Qwen3-Reranker-4B",
        timeout: int = 30,
        batch_size: int = 100,
        max_concurrency: int = 10,
        inference_framework: str = "",
        instruction: str = "",
        user_id: Optional[str] = None,
    ) -> None:
        framework = (inference_framework or "").lower()
        endpoint = _normalize_rerank_endpoint(endpoint, framework)
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.inference_framework = inference_framework.lower()
        self.instruction = instruction
        self.user_id = user_id

    @staticmethod
    def _extract_results(
        data: Any,
        num_documents: int,
    ) -> Tuple[List[int], Dict[int, float]]:
        results: Any = None
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            if "results" in data:
                results = data["results"]
            elif isinstance(data.get("data"), dict):
                results = data["data"].get("results", [])

        if not results:
            return (
                list(range(num_documents)),
                {index: 0.0 for index in range(num_documents)},
            )

        scores: Dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index", -1)
            score = item.get("relevance_score", item.get("score", 0.0))
            if isinstance(index, int) and 0 <= index < num_documents:
                scores[index] = float(score)

        ranking = sorted(
            scores,
            key=lambda index: (scores[index], index),
            reverse=True,
        )
        for index in range(num_documents):
            if index not in scores:
                scores[index] = 0.0
                ranking.append(index)
        return ranking, scores

    def _request_batch(
        self,
        query: str,
        documents: List[str],
    ) -> Tuple[List[int], Dict[int, float]]:
        if not documents:
            return [], {}

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if isinstance(self.instruction, str) and self.instruction.strip():
            instruction_field = {
                "vllm": "instruction",
                "sglang": "instruct",
            }.get(self.inference_framework)
            if instruction_field:
                payload[instruction_field] = self.instruction

        response = request_user_outbound(
            "POST",
            self.endpoint,
            self.user_id,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._extract_results(response.json(), len(documents))

    def rerank(
        self,
        query: str,
        documents: List[str],
    ) -> Tuple[List[int], Dict[int, float]]:
        if len(documents) <= self.batch_size:
            return self._request_batch(query, documents)

        scores: Dict[int, float] = {}
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            _, batch_scores = self._request_batch(query, batch)
            for local_index, score in batch_scores.items():
                scores[start + local_index] = score

        ranking = sorted(
            scores,
            key=lambda index: (scores[index], index),
            reverse=True,
        )
        return ranking, scores

    def rerank_batch(
        self,
        items: List[Tuple[str, List[str]]],
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
    ) -> List[Tuple[List[int], Dict[int, float]]]:
        del show_progress, progress_desc
        expanded_items: List[Tuple[str, List[str]]] = []
        item_mapping: List[Tuple[int, int]] = []
        for item_index, (query, documents) in enumerate(items):
            if len(documents) <= self.batch_size:
                expanded_items.append((query, documents))
                item_mapping.append((item_index, 0))
                continue
            for start in range(0, len(documents), self.batch_size):
                expanded_items.append(
                    (query, documents[start : start + self.batch_size])
                )
                item_mapping.append((item_index, start))

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            raw_results = list(
                executor.map(
                    lambda item: self._request_batch(item[0], item[1]),
                    expanded_items,
                )
            )

        merged_scores: List[Dict[int, float]] = [{} for _ in items]
        for batch_index, (_, batch_scores) in enumerate(raw_results):
            item_index, offset = item_mapping[batch_index]
            for local_index, score in batch_scores.items():
                merged_scores[item_index][offset + local_index] = score

        return [
            (
                sorted(
                    scores,
                    key=lambda index: (scores[index], index),
                    reverse=True,
                ),
                scores,
            )
            for scores in merged_scores
        ]

    def test_connection(self) -> bool:
        try:
            ranking, _ = self.rerank(
                "test query",
                ["document 1", "document 2"],
            )
            if ranking and len(ranking) == 2:
                logger.info("API connection test passed: %s", self.endpoint)
                return True
            logger.warning("API returned unexpected result: %s", ranking)
        except Exception as exc:
            logger.error("API connection test failed (%s)", type(exc).__name__)
        return False
