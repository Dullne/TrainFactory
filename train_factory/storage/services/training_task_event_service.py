"""
Training task event service for audit-style event logging.
"""

import logging
from typing import Optional, Dict, Any, List

from sqlmodel import select

from ..database import get_session
from ..entities.training_task_event_entity import TrainingTaskEventDB

logger = logging.getLogger(__name__)


class TrainingTaskEventService:
    """Service for training task event log operations."""

    def log_event(
        self,
        task_id: str,
        event_type: str,
        user_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Record a new event. Returns event_id if created."""
        with get_session() as session:
            event = TrainingTaskEventDB(
                task_id=task_id,
                user_id=user_id,
                event_type=event_type,
                payload=payload,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            logger.debug(f"Logged task event {event.event_id} ({event_type}) for {task_id}")
            return event.event_id

    def list_events(
        self,
        task_id: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List recent events for a task."""
        with get_session() as session:
            statement = (
                select(TrainingTaskEventDB)
                .where(TrainingTaskEventDB.task_id == task_id)
                .order_by(TrainingTaskEventDB.created_at.desc())
                .limit(limit)
            )
            events = session.exec(statement).all()
            return [
                {
                    "event_id": e.event_id,
                    "task_id": e.task_id,
                    "user_id": e.user_id,
                    "event_type": e.event_type,
                    "payload": e.payload,
                    "created_at": e.created_at,
                }
                for e in events
            ]


# Global service instance
training_task_event_service = TrainingTaskEventService()
