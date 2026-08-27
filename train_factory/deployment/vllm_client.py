"""
vLLM API client for model deployment and LoRA adapter management.

Supports:
- Model health check and info
- LoRA adapter hot-loading (load/unload/list)
- Embeddings and rerank inference

vLLM requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 for runtime LoRA updates.
"""

import logging
from typing import Optional, List, Dict, Any

import requests

from ..storage.services.outbound_endpoint_policy import request_user_outbound

logger = logging.getLogger(__name__)

class VLLMClient:
    """vLLM OpenAI-compatible API client with LoRA support."""

    def __init__(
        self,
        endpoint: str,
        timeout: int = 30,
        user_id: Optional[str] = None,
    ):
        """
        Initialize vLLM client.

        Args:
            endpoint: vLLM server endpoint (e.g., http://localhost:8000)
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
        """Check if vLLM server is healthy."""
        try:
            resp = self._request("GET", "/health")
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about loaded models."""
        try:
            resp = self._request("GET", "/v1/models")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get vLLM model info: {e}")
            return None

    def get_version(self) -> Optional[str]:
        """Get vLLM version."""
        try:
            resp = self._request("GET", "/version")
            resp.raise_for_status()
            return resp.json().get("version")
        except Exception as e:
            logger.debug(f"Failed to get vLLM version: {e}")
            return None

    # ==================== LoRA Management ====================

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
    ) -> Dict[str, Any]:
        """
        Load a LoRA adapter into the running model.

        Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 environment variable.

        Args:
            lora_name: Unique name for the adapter (used in inference requests)
            lora_path: Path to the LoRA adapter weights

        Returns:
            Response dict with status

        Raises:
            RuntimeError: If loading fails
        """
        try:
            resp = self._request(
                "POST",
                "/v1/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                },
                timeout=self.timeout * 2,  # Loading may take longer
            )
            resp.raise_for_status()
            # vLLM may return plain text (not JSON) on success
            try:
                result = resp.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                result = {"status": "ok", "message": resp.text}
            logger.info(f"Loaded LoRA adapter '{lora_name}' from {lora_path}")
            return result
        except requests.exceptions.HTTPError as e:
            # NOTE: Response.__bool__() returns False for 4xx/5xx, so use
            # ``is not None`` to correctly access the response body.
            error_detail = e.response.text if e.response is not None else str(e)
            # vLLM returns 400 when adapter name already exists — treat as success
            if e.response is not None and e.response.status_code == 400 and ("already loaded" in error_detail.lower() or "already exists" in error_detail.lower()):
                logger.warning(f"LoRA adapter '{lora_name}' already loaded, treating as success")
                return {"status": "already_loaded"}
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
            resp = self._request(
                "POST",
                "/v1/unload_lora_adapter",
                json={"lora_name": lora_name},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            try:
                result = resp.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                result = {"status": "ok", "message": resp.text}
            logger.info(f"Unloaded LoRA adapter '{lora_name}'")
            return result
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if e.response is not None else str(e)
            logger.error(f"Failed to unload LoRA adapter '{lora_name}': {error_detail}")
            raise RuntimeError(f"Failed to unload LoRA adapter: {error_detail}")
        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter '{lora_name}': {e}")
            raise RuntimeError(f"Failed to unload LoRA adapter: {e}")

    def list_lora_adapters(self) -> List[Dict[str, Any]]:
        """
        List currently loaded LoRA adapters.

        Returns:
            List of loaded adapter info dicts

        Note:
            vLLM's /v1/models endpoint returns all models including LoRA adapters.
            Adapters have 'parent' field pointing to the base model.
        """
        try:
            resp = self._request("GET", "/v1/models")
            resp.raise_for_status()
            data = resp.json()

            # Filter to only LoRA adapters (have 'parent' field)
            adapters = []
            for model in data.get("data", []):
                if model.get("parent"):
                    adapters.append({
                        "name": model.get("id"),
                        "parent": model.get("parent"),
                        "created": model.get("created"),
                        "owned_by": model.get("owned_by"),
                    })
            return adapters
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
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using the raw Cohere-compatible wire schema."""
        try:
            payload = {
                "query": query,
                "documents": documents,
            }
            if model:
                payload["model"] = model
            if top_n:
                payload["top_n"] = top_n
            if isinstance(instruction, str) and instruction.strip():
                payload["instruction"] = instruction

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
            logger.error("Failed to rerank documents (%s)", type(e).__name__)
            raise RuntimeError("Rerank failed") from e

    def score(
        self,
        text_1: str,
        text_2: str,
        model: Optional[str] = None,
    ) -> float:
        """
        Score the relevance between two texts.

        Args:
            text_1: First text (usually query)
            text_2: Second text (usually document)
            model: Optional model/adapter name

        Returns:
            Relevance score (0-1)
        """
        try:
            payload = {
                "text_1": text_1,
                "text_2": text_2,
            }
            if model:
                payload["model"] = model

            resp = self._request(
                "POST",
                "/v1/score",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Score endpoint returns {"score": float}
            return data.get("score", 0.0)
        except Exception as e:
            logger.error(f"Failed to score texts: {e}")
            raise RuntimeError(f"Score failed: {e}")


# Convenience function for creating clients
def create_vllm_client(
    endpoint: str,
    timeout: int = 30,
    user_id: Optional[str] = None,
) -> VLLMClient:
    """Create a vLLM client instance."""
    return VLLMClient(endpoint=endpoint, timeout=timeout, user_id=user_id)
