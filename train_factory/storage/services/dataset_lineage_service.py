"""
Dataset lineage service — CRUD + graph queries for dataset_lineage_edges.

Supports recording directed edges between datasets and querying the
lineage graph upstream / downstream / both directions.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from ..database import get_engine
from ..entities.dataset_lineage_entity import (
    DatasetLineageEdgeDB,
    build_dataset_lineage_edge_key,
)

logger = logging.getLogger(__name__)

_EDGE_KEY_UNIQUE_NAMES = {
    "uq_lineage_edge_key",
    "ix_dataset_lineage_edges_edge_key",
}


def _is_edge_key_unique_conflict(exc: IntegrityError) -> bool:
    """Recognize only the database constraint protecting semantic edges."""
    original = exc.orig
    message = str(original).lower()
    if (
        "unique constraint failed" in message
        and "dataset_lineage_edges.edge_key" in message
    ):
        return True

    args = getattr(original, "args", ())
    error_code = args[0] if args else None
    if error_code == 1062 and any(
        name in message for name in _EDGE_KEY_UNIQUE_NAMES
    ):
        return True

    diagnostic = getattr(original, "diag", None)
    constraint_name = getattr(diagnostic, "constraint_name", None)
    sqlstate = (
        getattr(original, "sqlstate", None)
        or getattr(original, "pgcode", None)
    )
    return (
        sqlstate == "23505"
        and constraint_name in _EDGE_KEY_UNIQUE_NAMES
    )


class DatasetLineageService:
    """Service for managing dataset lineage edges."""

    def __init__(self):
        self.engine = None

    def _get_engine(self):
        if self.engine is None:
            self.engine = get_engine()
        return self.engine

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def _create_edge_in_session(
        session: Session,
        from_dataset_id: Optional[str],
        to_dataset_id: str,
        relation_type: str,
        op_task_type: Optional[str],
        op_task_id: Optional[str],
        op_params: Optional[Dict[str, Any]],
    ) -> Tuple[DatasetLineageEdgeDB, bool]:
        """Insert one semantic edge, reconciling a concurrent winner."""
        edge_key = build_dataset_lineage_edge_key(
            from_dataset_id,
            to_dataset_id,
            relation_type,
        )
        edge = DatasetLineageEdgeDB(
            edge_key=edge_key,
            from_dataset_id=from_dataset_id,
            to_dataset_id=to_dataset_id,
            relation_type=relation_type,
            op_task_type=op_task_type,
            op_task_id=op_task_id,
            op_params=op_params,
        )
        try:
            with session.begin_nested():
                session.add(edge)
                session.flush()
            return edge, True
        except IntegrityError as exc:
            if not _is_edge_key_unique_conflict(exc):
                raise
            existing = session.exec(
                select(DatasetLineageEdgeDB).where(
                    DatasetLineageEdgeDB.edge_key == edge_key
                ).with_for_update()
            ).one_or_none()
            if existing is None:
                raise
            return existing, False

    def create_edge(
        self,
        from_dataset_id: Optional[str],
        to_dataset_id: str,
        relation_type: str,
        op_task_type: Optional[str] = None,
        op_task_id: Optional[str] = None,
        op_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a lineage edge: from_dataset → to_dataset."""
        with Session(self._get_engine()) as session:
            edge, created = self._create_edge_in_session(
                session,
                from_dataset_id,
                to_dataset_id,
                relation_type,
                op_task_type,
                op_task_id,
                op_params,
            )
            session.commit()
            session.refresh(edge)
            if not created:
                return edge.to_dict()
            from_tag = (from_dataset_id or "root")[:8]
            logger.info(
                f"Lineage edge: {from_tag}→{to_dataset_id[:8]} "
                f"({relation_type})"
            )
            return edge.to_dict()

    def create_edges_bulk(
        self,
        from_dataset_ids: List[str],
        to_dataset_id: str,
        relation_type: str,
        op_task_type: Optional[str] = None,
        op_task_id: Optional[str] = None,
        op_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Create multiple edges pointing to the same to_dataset_id.

        幂等：同 (from, to, relation_type) 的边已存在时跳过，防止生成任务
        重启/重跑积累重复 lineage 边。
        """
        unique_sources = sorted(
            set(from_dataset_ids),
            key=lambda source: build_dataset_lineage_edge_key(
                source,
                to_dataset_id,
                relation_type,
            ),
        )
        with Session(self._get_engine()) as session:
            edges: List[DatasetLineageEdgeDB] = []
            for from_dataset_id in unique_sources:
                edge, created = self._create_edge_in_session(
                    session,
                    from_dataset_id,
                    to_dataset_id,
                    relation_type,
                    op_task_type,
                    op_task_id,
                    op_params,
                )
                if created:
                    edges.append(edge)
            session.commit()
            for edge in edges:
                session.refresh(edge)
            return [edge.to_dict() for edge in edges]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_upstream(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get edges where *dataset_id* is the target (who produced it)."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetLineageEdgeDB).where(
                DatasetLineageEdgeDB.to_dataset_id == dataset_id
            )
            return [e.to_dict() for e in session.exec(stmt).all()]

    def get_downstream(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get edges where *dataset_id* is the source (what it produced)."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetLineageEdgeDB).where(
                DatasetLineageEdgeDB.from_dataset_id == dataset_id
            )
            return [e.to_dict() for e in session.exec(stmt).all()]

    def get_edges_by_task(self, op_task_id: str) -> List[Dict[str, Any]]:
        """Get all edges created by a specific task."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetLineageEdgeDB).where(
                DatasetLineageEdgeDB.op_task_id == op_task_id
            )
            return [e.to_dict() for e in session.exec(stmt).all()]

    def get_lineage_graph(
        self,
        dataset_id: str,
        direction: str = "both",
        depth: int = 3,
    ) -> Dict[str, Any]:
        """Build a lineage sub-graph around *dataset_id*.

        Args:
            dataset_id: Starting dataset.
            direction: "upstream", "downstream", or "both".
            depth: Maximum traversal depth.

        Returns:
            {"nodes": [dataset_id, ...], "edges": [edge_dict, ...]}
        """
        visited_nodes: Set[str] = {dataset_id}
        collected_edges: List[Dict[str, Any]] = []

        with Session(self._get_engine()) as session:
            if direction in ("upstream", "both"):
                self._traverse(
                    session, dataset_id, "upstream", depth,
                    visited_nodes, collected_edges,
                )
            if direction in ("downstream", "both"):
                self._traverse(
                    session, dataset_id, "downstream", depth,
                    visited_nodes, collected_edges,
                )

        return {
            "nodes": sorted(visited_nodes),
            "edges": collected_edges,
        }

    def _traverse(
        self,
        session: Session,
        dataset_id: str,
        direction: str,
        remaining_depth: int,
        visited: Set[str],
        edges: List[Dict[str, Any]],
    ):
        """Recursive BFS traversal of lineage graph."""
        if remaining_depth <= 0:
            return

        if direction == "upstream":
            stmt = select(DatasetLineageEdgeDB).where(
                DatasetLineageEdgeDB.to_dataset_id == dataset_id
            )
        else:
            stmt = select(DatasetLineageEdgeDB).where(
                DatasetLineageEdgeDB.from_dataset_id == dataset_id
            )

        results = session.exec(stmt).all()
        next_ids: List[str] = []

        for edge in results:
            edge_dict = edge.to_dict()
            # Avoid duplicate edges
            if not any(e["edge_id"] == edge_dict["edge_id"] for e in edges):
                edges.append(edge_dict)

            neighbor = (
                edge.from_dataset_id if direction == "upstream"
                else edge.to_dataset_id
            )
            if not neighbor:
                continue
            if neighbor not in visited:
                visited.add(neighbor)
                next_ids.append(neighbor)

        for nid in next_ids:
            self._traverse(session, nid, direction, remaining_depth - 1, visited, edges)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_edges_for_dataset(self, dataset_id: str) -> int:
        """Delete all edges involving *dataset_id* (as source or target)."""
        with Session(self._get_engine()) as session:
            stmt = select(DatasetLineageEdgeDB).where(
                (DatasetLineageEdgeDB.from_dataset_id == dataset_id)
                | (DatasetLineageEdgeDB.to_dataset_id == dataset_id)
            )
            results = session.exec(stmt).all()
            count = len(results)
            for edge in results:
                session.delete(edge)
            session.commit()
            if count:
                logger.info(f"Deleted {count} lineage edges for dataset {dataset_id}")
            return count


dataset_lineage_service = DatasetLineageService()
