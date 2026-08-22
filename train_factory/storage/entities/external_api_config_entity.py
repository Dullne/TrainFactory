"""
External API configuration entity.

Stores reusable external API connection details (URL + auth)
that can be referenced by multiple sync configurations.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Index, Text
from sqlmodel import Column, Field, JSON, SQLModel

from ...core.time_utils import sync_now_naive


_SENSITIVE_AUTH_KEY_PARTS = (
    "authorization",
    "cookie",
    "token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "credential",
    "private_key",
    "privatekey",
)


def _mask_sensitive_auth_value(value: Any) -> Any:
    if isinstance(value, dict):
        masked: Dict[Any, Any] = {}
        for key, child in value.items():
            normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
            if any(part in normalized_key for part in _SENSITIVE_AUTH_KEY_PARTS):
                masked[key] = "***"
            else:
                masked[key] = _mask_sensitive_auth_value(child)
        return masked
    if isinstance(value, list):
        return [_mask_sensitive_auth_value(child) for child in value]
    return value


class ExternalApiConfigDB(SQLModel, table=True):
    """External API connection configuration."""

    __tablename__ = "external_api_configs"

    __table_args__ = (
        Index("idx_ext_api_user", "user_id"),
        Index("idx_ext_api_status", "status"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    config_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        index=True, unique=True, max_length=36,
    )
    config_name: str = Field(max_length=255)
    user_id: str = Field(max_length=64)

    # Connection details
    api_url: str = Field(max_length=1024)
    auth_config: Dict[str, Any] = Field(sa_column=Column(JSON))

    # Optional metadata
    description: str = Field(default="", sa_column=Column(Text))
    status: str = Field(default="active", max_length=32)  # active / inactive

    # Timestamps
    created_at: datetime = Field(default_factory=sync_now_naive)
    updated_at: datetime = Field(default_factory=sync_now_naive)

    def to_dict(self, mask_sensitive: bool = True) -> Dict[str, Any]:
        """Convert to dictionary with optional sensitive data masking."""
        auth = self.auth_config or {}
        if mask_sensitive:
            auth = _mask_sensitive_auth_value(auth)

        return {
            "config_id": self.config_id,
            "config_name": self.config_name,
            "user_id": self.user_id,
            "api_url": self.api_url,
            "auth_config": auth,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
