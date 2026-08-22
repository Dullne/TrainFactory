"""
SGLang API client for model deployment and LoRA adapter management.

Supports:
- Model health check and info
- LoRA adapter hot-loading (load/unload/list)
- Embeddings and rerank inference

SGLang requires --enable-lora flag at startup for LoRA support.
"""

import logging
from typing import Optional, List, Dict, Any

import requests

from ..storage.services.outbound_endpoint_policy import request_user_outbound

logger = logging.getLogger(__name__)


class SGLangClient:
    """SGLang API client with LoRA support."""

    def __init__(
        self,
        endpoint: str,
        timeout: int = 30,
        user_id: Optional[str] = None,
    ):
        """
        Initialize SGLang client.

        Args:
            endpoint: SGLang server endpoint (e.g., http://localhost:8000)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.user_id = user_id

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        return request_user_outbound(
            method,
            f"{self.endpoint}{path}",
            self.user_id,
            **kwargs,
        )

    # ==================== Health & Info ====================

    def health_check(self) -> bool:
        """Check if SGLang server is healthy."""
        try:
            resp = self._request("GET", "/health")
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"SGLang health check failed: {e}")
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about loaded models."""
        try:
            resp = self._request("GET", "/v1/models")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get SGLang model info: {e}")
            return None

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get SGLang server information."""
        try:
            resp = self._request("GET", "/get_server_info")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"Failed to get SGLang server info: {e}")
            return None

    # ==================== LoRA Management ====================

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
    ) -> Dict[str, Any]:
        """
        Load a LoRA adapter into the running model.

        Requires --enable-lora flag at server startup.

        Args:
            lora_name: Unique name for the adapter (used in inference requests)
            lora_path: Path to the LoRA adapter weights

        Returns:
            Response dict with status

        Raises:
            RuntimeError: If loading fails
        """
        try:
            # SGLang 动态 LoRA 的现行接口：POST /load_lora_adapter，
            # 字段为 lora_name/lora_path（旧 /v1/loras + name/path 已不匹配）
            resp = self._request(
                "POST",
                "/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                },
                timeout=self.timeout * 2,  # Loading may take longer
            )
            resp.raise_for_status()
            try:
                result = resp.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                result = {"status": "ok", "message": resp.text}
            logger.info(f"Loaded LoRA adapter '{lora_name}' from {lora_path}")
            return result
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if e.response else str(e)
            logger.error(f"Failed to load LoRA adapter '{lora_name}': {error_detail}")
            raise RuntimeError(f"Failed to load LoRA adapter: {error_detail}")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter '{lora_name}': {e}")
            raise RuntimeError(f"Failed to load LoRA adapter: {e}")

    def unload_lora_adapter(self, lora_name: str) -> Dict[str, Any]:
        """
        Unload a LoRA adapter from the running model.

        Args:
            lora_name: Name of the adapter to unload

        Returns:
            Response dict with status

        Raises:
            RuntimeError: If unloading fails
        """
        try:
            # SGLang 现行接口：POST /unload_lora_adapter（body 传 lora_name）
            resp = self._request(
                "POST",
                "/unload_lora_adapter",
                json={"lora_name": lora_name},
            )
            resp.raise_for_status()
            try:
                result = resp.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                result = {"status": "success"}
            logger.info(f"Unloaded LoRA adapter '{lora_name}'")
            return result
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if e.response else str(e)
            logger.error(f"Failed to unload LoRA adapter '{lora_name}': {error_detail}")
            raise RuntimeError(f"Failed to unload LoRA adapter: {error_detail}")
        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter '{lora_name}': {e}")
            raise RuntimeError(f"Failed to unload LoRA adapter: {e}")

    def list_lora_adapters(self) -> List[Dict[str, Any]]:
        """
        List currently loaded LoRA adapters.

        兼容两种 SGLang 形态（按镜像实际路由探测）：
        - 本仓库锁定的 nightly（commit 53992403，见 docs 与 server 源码核对）：
          无 /list_loras 路由，已加载 adapter 以带 parent 字段的条目出现在
          GET /v1/models（OpenAI 兼容接口，base model 无 parent）。
        - 新版 SGLang：GET /list_loras 直接返回 adapter 列表。

        Returns:
            List of loaded adapter info dicts (name/path/status)
        """
        # Strategy 1: /v1/models —— adapter 条目带 parent 字段，base model 没有
        try:
            resp = self._request("GET", "/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                adapters = []
                for item in data.get("data", []):
                    if item.get("parent") is not None:
                        adapters.append({
                            "name": item.get("id"),
                            "path": item.get("root"),
                            "status": "loaded",
                        })
                return adapters
        except Exception as e:
            logger.debug(f"/v1/models listing failed, trying /list_loras: {e}")

        # Strategy 2: /list_loras（新版 SGLang）
        try:
            resp = self._request("GET", "/list_loras")
            resp.raise_for_status()
            data = resp.json()
            loras = data if isinstance(data, list) else data.get("loras", [])
            return [
                {
                    "name": lora.get("name") or lora.get("id"),
                    "path": lora.get("path"),
                    "status": lora.get("status", "loaded"),
                }
                for lora in loras
            ]
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {e}")
            return []

    # ==================== Inference ====================

    def embeddings(
        self,
        inputs: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts.

        Args:
            inputs: List of texts to embed
            model: Optional model/adapter name (uses default if not specified)

        Returns:
            List of embedding vectors
        """
        try:
            payload = {"input": inputs}
            if model:
                payload["model"] = model

            resp = self._request(
                "POST",
                "/v1/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding failed: {e}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to query.

        Note: For Qwen3-Reranker, use custom chat template to disable thinking.

        Args:
            query: Query text
            documents: List of documents to rerank
            model: Optional model/adapter name
            top_n: Optional limit on number of results

        Returns:
            List of {index, relevance_score} dicts, sorted by score descending
        """
        try:
            # SGLang uses /v1/rerank endpoint
            payload = {
                "query": query,
                "documents": documents,
            }
            if model:
                payload["model"] = model
            if top_n:
                payload["top_n"] = top_n

            resp = self._request(
                "POST",
                "/v1/rerank",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            raise RuntimeError(f"Rerank failed: {e}")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            model: Optional model/adapter name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            if model:
                payload["model"] = model

            resp = self._request(
                "POST",
                "/v1/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise RuntimeError(f"Generation failed: {e}")


# Convenience function for creating clients
def create_sglang_client(
    endpoint: str,
    timeout: int = 30,
    user_id: Optional[str] = None,
) -> SGLangClient:
    """Create a SGLang client instance."""
    return SGLangClient(endpoint=endpoint, timeout=timeout, user_id=user_id)
