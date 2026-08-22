"""
User database entity.

Stores user account information for authentication.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field
import uuid

from train_factory.core.time_utils import now_naive


class UserDB(SQLModel, table=True):
    """User database model for authentication."""

    __tablename__ = "users"

    # === Primary Key ===
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()), unique=True, index=True)

    # === Account Info ===
    username: str = Field(max_length=64, unique=True, index=True)
    email: Optional[str] = Field(default=None, max_length=256, unique=True, index=True)
    hashed_password: str = Field(max_length=256)

    # === Status ===
    is_active: bool = Field(default=True, index=True)
    is_admin: bool = Field(default=False)
    token_version: int = Field(default=0)

    # === Timestamps ===
    created_at: datetime = Field(default_factory=now_naive)
    updated_at: datetime = Field(default_factory=now_naive)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes password)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "token_version": self.token_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
