"""
External Model Configuration Entity.

Manages API configurations for external model services (LLM, Embedding, Rerank).
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict
from sqlalchemy import Column, JSON, Text, Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from train_factory.core.time_utils import now_naive


class ModelConfigDB(SQLModel, table=True):
    """
    External model configuration table.

    Stores API configurations for various model providers like OpenAI, Azure,
    Xinference, or custom endpoints.
    """
    __tablename__ = "model_configs"

    # Unique constraint: same user cannot have duplicate config names
    __table_args__ = (
        UniqueConstraint('user_id', 'config_name', name='uq_config_user_name'),
        Index('idx_config_user_type', 'user_id', 'model_type'),
        Index('idx_config_provider_status', 'provider', 'status'),
    )

    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)

    # Unique identifier
    config_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        max_length=36,
        index=True,
        unique=True
    )

    # Configuration name (user-friendly identifier)
    config_name: str = Field(max_length=255, index=True)

    # Display name for UI
    display_name: Optional[str] = Field(default=None, max_length=256)

    # Source type: external_api (from provider), local_deployed (from registry/deployment)
    source_type: str = Field(default="external_api", max_length=32, index=True)

    # === Local Deployment Link (for source_type=local_deployed) ===

    # Reference to model_registry (for locally trained models)
    registry_id: Optional[str] = Field(default=None, max_length=36, index=True)

    # Reference to deployments (for deployed models)
    deployment_id: Optional[str] = Field(default=None, max_length=36, index=True)

    # Container name (for local deployed models)
    container_name: Optional[str] = Field(default=None, max_length=255)

    # Inference framework: vllm, sglang, xinference (for local deployed models)
    inference_framework: Optional[str] = Field(default=None, max_length=32)

    # Model type: llm, embedding, rerank
    model_type: str = Field(max_length=50, index=True)

    # Provider: openai, azure, xinference, ollama, custom, etc.
    provider: str = Field(max_length=50, index=True)

    # === Connection Settings ===

    # API endpoint URL
    api_endpoint: str = Field(max_length=1024)

    # API key (encrypted or plain, depending on security requirements)
    api_key: Optional[str] = Field(default=None, max_length=512)

    # Model name/ID used in API calls
    model_name: str = Field(max_length=255)

    # === Model Properties ===

    # Embedding dimension (for embedding models)
    embedding_dim: int = Field(default=1024)

    # Maximum input length (tokens)
    max_input_length: Optional[int] = Field(default=None)

    # === Provider-specific Settings ===

    # For Azure: deployment name, API version, etc.
    # For Xinference: model_uid, etc.
    # Stored as JSON for flexibility
    provider_config: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON)
    )

    # === Model Parameters ===

    # Default parameters for API calls
    # For LLM: temperature, max_tokens, top_p, etc.
    # For Embedding: dimensions, encoding_format, etc.
    # For Rerank: top_n, return_documents, etc.
    default_params: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON)
    )

    # === Metadata ===

    description: Optional[str] = Field(default=None, sa_column=Column(Text))

    tags: Optional[List[str]] = Field(
        default=None,
        sa_column=Column(JSON)
    )

    # === Status ===

    # active, inactive, error
    status: str = Field(default="active", max_length=50, index=True)

    # Whether this is the default config for its model_type
    is_default: bool = Field(default=False)

    # Last connectivity check result
    last_check_status: Optional[str] = Field(default=None, max_length=50)
    last_check_time: Optional[datetime] = Field(default=None)
    last_check_error: Optional[str] = Field(default=None, sa_column=Column(Text))

    # === Multi-user ===

    user_id: Optional[str] = Field(default=None, max_length=64, index=True)

    # === Timestamps ===

    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "config_name": "openai-gpt4",
                "model_type": "llm",
                "provider": "openai",
                "api_endpoint": "https://api.openai.com/v1",
                "api_key": "sk-...",
                "model_name": "gpt-4",
                "provider_config": {},
                "default_params": {
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "description": "OpenAI GPT-4 for general tasks",
                "tags": ["production", "gpt4"],
                "status": "active",
                "is_default": True
            }
        }
    )
