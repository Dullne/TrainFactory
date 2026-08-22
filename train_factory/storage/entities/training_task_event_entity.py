"""
Training task event database entity.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON
from sqlalchemy import Index
import uuid

from train_factory.core.time_utils import now_naive


class TrainingTaskEventDB(SQLModel, table=True):
    """Training task event log model."""

    __tablename__ = "training_task_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True, unique=True)
    task_id: str = Field(max_length=36, index=True)
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)
    event_type: str = Field(max_length=64, index=True)
    payload: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now_naive, index=True)

    __table_args__ = (
        Index('idx_task_event_task_time', 'task_id', 'created_at'),
    )
