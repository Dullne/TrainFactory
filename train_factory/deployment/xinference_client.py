"""
Xinference API client for model deployment.

Provides methods to interact with Xinference server for launching, managing,
and querying embedding/reranker models.
"""

import logging
from typing import Optional, List, Dict, Any

import requests

from ..storage.services.outbound_endpoint_policy import request_user_outbound

logger = logging.getLogger(__name__)


class XinferenceClient:
    """Client for Xinference API."""

    def __init__(
        self,
        endpoint: str,
        timeout: int = 30,
        user_id: Optional[str] = None,
    ):
        """
        Initialize Xinference client.

        Args:
            endpoint: Xinference server endpoint (e.g., http://localhost:9997)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.user_id = user_id

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make HTTP request to Xinference API."""
        url = f"{self.endpoint}{path}"
        kwargs.setdefault("timeout", self.timeout)

        try:
            response = request_user_outbound(
                method,
                url,
                self.user_id,
                **kwargs,
            )
            response.raise_for_status()
            if response.text:
                return response.json()
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Xinference API error: {method} {url} - {e}")
            raise

    # ==================== Model Operations ====================

    def launch_model(
        self,
        model_uid: str,
        model_path: str,
        model_type: str = "embedding",
        replica: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Launch a model on Xinference.

        Args:
            model_uid: Unique identifier for the model
            model_path: Path to the model files
            model_type: Type of model ('embedding' or 'reranker')
            replica: Number of replicas
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            **kwargs: Additional model-specific parameters

        Returns:
            Model launch response
        """
        # Map internal model_type to Xinference model_type
        # Internal 'reranker' -> Xinference 'rerank'
        xinference_model_type = model_type
        if model_type == "reranker":
            xinference_model_type = "rerank"

        # Normalize dtype args for model-engine compatibility.
        # vLLM does not accept torch_dtype, keep dtype as-is there.
        model_engine = kwargs.get("model_engine")
        if model_engine == "vllm":
            if "torch_dtype" in kwargs and "dtype" not in kwargs:
                kwargs["dtype"] = kwargs.pop("torch_dtype")
            elif "torch_dtype" in kwargs and "dtype" in kwargs:
                kwargs.pop("torch_dtype")
        elif "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")

        payload = {
            "model_uid": model_uid,
            "model_path": model_path,
            "model_type": xinference_model_type,
            "replica": replica,
            **kwargs,
        }

        # Only add gpu_memory_utilization for embedding models
        # Reranker models in Xinference don't support this parameter
        if xinference_model_type != "rerank":
            payload["gpu_memory_utilization"] = gpu_memory_utilization

        # Use the unified model launch endpoint
        # Note: /v1/models/embedding and /v1/models/rerank are for inference, not launch
        path = "/v1/models"

        logger.info(f"Launching model {model_uid} on Xinference: {model_path}")
        return self._request("POST", path, json=payload)

    def terminate_model(self, model_uid: str) -> Dict[str, Any]:
        """
        Terminate a running model.

        Args:
            model_uid: Model UID to terminate

        Returns:
            Termination response
        """
        logger.info(f"Terminating model {model_uid}")
        return self._request("DELETE", f"/v1/models/{model_uid}")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all running models.

        Returns:
            List of model information
        """
        result = self._request("GET", "/v1/models")
        return result if isinstance(result, list) else result.get("data", [])

    def get_model(self, model_uid: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_uid: Model UID to query

        Returns:
            Model information or None if not found
        """
        try:
            return self._request("GET", f"/v1/models/{model_uid}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def get_model_status(self, model_uid: str) -> str:
        """
        Get the status of a model.

        Args:
            model_uid: Model UID to query

        Returns:
            Status string: 'running', 'starting', 'stopped', or 'unknown'
        """
        model = self.get_model(model_uid)
        if model:
            return model.get("status", "running")
        return "stopped"

    # ==================== Health Check ====================

    def health_check(self) -> bool:
        """
        Check if Xinference server is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self._request("GET", "/v1/models")
            return True
        except Exception as e:
            logger.warning(f"Xinference health check failed: {e}")
            return False

    # ==================== Inference (Optional) ====================

    def embed(
        self,
        model_uid: str,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            model_uid: Model UID to use
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        payload = {
            "model": model_uid,
            "input": texts,
        }
        result = self._request("POST", "/v1/embeddings", json=payload)
        return [item["embedding"] for item in result.get("data", [])]

    def rerank(
        self,
        model_uid: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query.

        Args:
            model_uid: Model UID to use
            query: Query string
            documents: List of documents to rerank
            top_n: Optional number of top results to return

        Returns:
            List of reranked results with scores
        """
        payload = {
            "model": model_uid,
            "query": query,
            "documents": documents,
        }
        if top_n:
            payload["top_n"] = top_n

        result = self._request("POST", "/v1/rerank", json=payload)
        return result.get("results", [])
