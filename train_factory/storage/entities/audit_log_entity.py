"""
Audit log entity for tracking user operations.

Provides a record of all user actions for security and compliance.
"""

from datetime import datetime
from typing import Optional, Dict, Any
import uuid

from sqlmodel import SQLModel, Field, Column, Index
from sqlalchemy import JSON

from train_factory.core.time_utils import now_naive


class AuditLogDB(SQLModel, table=True):
    """Audit log database model for tracking user operations."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index('idx_audit_user_time', 'user_id', 'created_at'),
        Index('idx_audit_created_at', 'created_at'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        max_length=36,
        index=True,
    )

    # User information
    user_id: str = Field(max_length=36, index=True)
    username: Optional[str] = Field(default=None, max_length=255)

    # Action information
    action: str = Field(max_length=50, description="Action type: create, read, update, delete, login, logout, etc.")
    resource_type: str = Field(max_length=50, description="Resource type: training_task, model, deployment, dataset, config")
    resource_id: Optional[str] = Field(default=None, max_length=36, description="ID of the affected resource")

    # Request details
    method: str = Field(max_length=10, description="HTTP method: GET, POST, PUT, DELETE")
    endpoint: str = Field(max_length=500, description="API endpoint path")
    request_body: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="Request body (sensitive data should be redacted)"
    )

    # Response details
    status_code: int = Field(description="HTTP response status code")
    response_summary: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Brief summary of the response"
    )

    # Client information
    ip_address: Optional[str] = Field(default=None, max_length=45, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, max_length=500, description="User agent string")

    # Metadata
    extra_data: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
        description="Additional context data"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=now_naive)


# Predefined action types
class AuditAction:
    """Predefined audit action types."""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    TOKEN_REFRESH = "token_refresh"

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"

    # Special operations
    START = "start"
    STOP = "stop"
    DEPLOY = "deploy"
    DOWNLOAD = "download"


# Predefined resource types
class AuditResource:
    """Predefined audit resource types."""
    TRAINING_TASK = "training_task"
    GENERATION_TASK = "generation_task"
    EVALUATION_TASK = "evaluation_task"
    DEEP_EVALUATION_TASK = "deep_evaluation_task"
    SYNC_TASK = "sync_task"
    EXTERNAL_API_CONFIG = "external_api_config"
    MODEL = "model"
    DEPLOYMENT = "deployment"
    DATASET = "dataset"
    MODEL_CONFIG = "model_config"
    USER = "user"
