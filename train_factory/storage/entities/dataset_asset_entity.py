"""
Dataset asset database entity.

Tracks individual files (splits, versions) that belong to a dataset.
A single dataset may have multiple assets (train.jsonl, eval.jsonl, etc.).
"""

from datetime import datetime
from typing import Any, Dict, Optional
from sqlmodel import SQLModel, Field
import uuid

from train_factory.core.time_utils import now_naive


class DatasetAssetDB(SQLModel, table=True):
    """One file/object belonging to a dataset version."""

    __tablename__ = "dataset_assets"

    id: Optional[int] = Field(default=None, primary_key=True)
    asset_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        index=True,
        unique=True,
        max_length=36,
    )
    dataset_id: str = Field(max_length=36, index=True)
    version: int = Field(default=1)

    # data / split_train / split_eval / split_test
    asset_type: str = Field(default="data", max_length=32)

    # s3://trainfactory/datasets/{user}/{ds_id}/v1/train.jsonl
    storage_uri: str = Field(max_length=2048)

    file_format: Optional[str] = Field(default=None, max_length=32)
    compression: Optional[str] = Field(default=None, max_length=16)
    row_count: Optional[int] = Field(default=None)
    byte_size: Optional[int] = Field(default=None)
    checksum: Optional[str] = Field(default=None, max_length=128)

    created_at: datetime = Field(default_factory=now_naive)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "asset_type": self.asset_type,
            "storage_uri": self.storage_uri,
            "file_format": self.file_format,
            "compression": self.compression,
            "row_count": self.row_count,
            "byte_size": self.byte_size,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
