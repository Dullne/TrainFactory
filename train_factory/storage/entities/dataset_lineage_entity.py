"""
Dataset lineage edge database entity.

Records directed edges in the dataset lineage graph:
  from_dataset_id  --[relation_type]--> to_dataset_id

Each edge is optionally linked to the task (sync / generation / training)
that produced the transformation.
"""

from datetime import datetime
import hashlib
import json
from typing import Any, Dict, Optional
from sqlalchemy import UniqueConstraint, event
from sqlmodel import SQLModel, Field, Column, JSON
import uuid

from train_factory.core.time_utils import now_naive


def _public_lineage_params(value: Any) -> Any:
    """Remove durable worker credentials from public lineage metadata."""
    if isinstance(value, dict):
        return {
            key: _public_lineage_params(item)
            for key, item in value.items()
            if not str(key).lower().endswith("run_token")
        }
    if isinstance(value, list):
        return [_public_lineage_params(item) for item in value]
    return value


def build_dataset_lineage_edge_key(
    from_dataset_id: Optional[str],
    to_dataset_id: str,
    relation_type: str,
) -> str:
    """Build the stable identity for a semantic lineage edge."""
    canonical = json.dumps(
        [from_dataset_id, to_dataset_id, relation_type],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DatasetLineageEdgeDB(SQLModel, table=True):
    """A single directed edge in the dataset lineage DAG."""

    __tablename__ = "dataset_lineage_edges"
    __table_args__ = (
        UniqueConstraint("edge_key", name="uq_lineage_edge_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    edge_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        index=True,
        unique=True,
        max_length=36,
    )
    edge_key: Optional[str] = Field(
        default=None,
        max_length=64,
        nullable=False,
    )

    # NULL for root edges (e.g. sync_fetched has no upstream dataset)
    from_dataset_id: Optional[str] = Field(default=None, max_length=36, index=True)
    to_dataset_id: str = Field(max_length=36, index=True)

    # sync_fetched / merged / qa_extracted / qa_filtered
    # training_generated / deep_eval_generated / training_input
    relation_type: str = Field(max_length=32)

    # sync / generation / training
    op_task_type: Optional[str] = Field(default=None, max_length=32)
    op_task_id: Optional[str] = Field(default=None, max_length=36, index=True)

    # Optional snapshot of the transformation parameters
    op_params: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=now_naive)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "from_dataset_id": self.from_dataset_id,
            "to_dataset_id": self.to_dataset_id,
            "relation_type": self.relation_type,
            "op_task_type": self.op_task_type,
            "op_task_id": self.op_task_id,
            "op_params": _public_lineage_params(self.op_params),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@event.listens_for(DatasetLineageEdgeDB, "before_insert")
def _populate_dataset_lineage_edge_key(_mapper, _connection, target) -> None:
    target.edge_key = build_dataset_lineage_edge_key(
        target.from_dataset_id,
        target.to_dataset_id,
        target.relation_type,
    )
