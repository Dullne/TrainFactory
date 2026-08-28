"""
CRUD service for external API configurations.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy import or_
from sqlmodel import Session, select

from ..database import get_engine
from ..entities.deployment_entity import DeploymentDB
from ..entities.external_api_config_entity import ExternalApiConfigDB
from ..entities.external_sync_entity import ExternalSyncTaskDB
from ...core.time_utils import sync_now_naive

logger = logging.getLogger(__name__)


def _deployment_api_config_reference(config_id: str):
    """Match normalized and legacy JSON deployment API-config bindings."""
    return or_(
        DeploymentDB.external_api_config_id == config_id,
        DeploymentDB.config["external_api_config_id"].as_string() == config_id,
    )


def _utcnow_naive() -> datetime:
    """Return sync-mode timestamp as naive datetime for DB compatibility."""
    return sync_now_naive()


class ExternalApiConfigService:
    """Service for managing external API configurations."""

    def __init__(self):
        self.engine = None

    def _get_engine(self):
        if self.engine is None:
            self.engine = get_engine()
        return self.engine

    @staticmethod
    def _lock_references(
        session: Session,
        config_id: str,
    ) -> tuple[list[DeploymentDB], list[ExternalSyncTaskDB]]:
        """Lock API consumers in API -> deployments -> sync tasks order."""
        deployments = list(
            session.exec(
                select(DeploymentDB)
                .where(_deployment_api_config_reference(config_id))
                .order_by(DeploymentDB.deployment_id)
                .with_for_update()
            ).all()
        )
        sync_tasks = list(
            session.exec(
                select(ExternalSyncTaskDB)
                .where(ExternalSyncTaskDB.external_api_config_id == config_id)
                .order_by(ExternalSyncTaskDB.task_id)
                .with_for_update()
            ).all()
        )
        return deployments, sync_tasks

    # ─── CRUD ───

    def create_config(
        self,
        config_name: str,
        user_id: str,
        api_url: str,
        auth_config: Dict[str, Any],
        description: str = "",
    ) -> Dict[str, Any]:
        config = ExternalApiConfigDB(
            config_name=config_name,
            user_id=user_id,
            api_url=api_url,
            auth_config=auth_config,
            description=description,
        )
        with Session(self._get_engine()) as session:
            session.add(config)
            session.commit()
            session.refresh(config)
            logger.info(f"Created external API config: {config.config_id} ({config_name})")
            return config.to_dict()

    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        with Session(self._get_engine()) as session:
            stmt = select(ExternalApiConfigDB).where(
                ExternalApiConfigDB.config_id == config_id
            )
            config = session.exec(stmt).first()
            return config.to_dict() if config else None

    def get_config_raw(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get config with unmasked auth data (for internal use)."""
        with Session(self._get_engine()) as session:
            stmt = select(ExternalApiConfigDB).where(
                ExternalApiConfigDB.config_id == config_id
            )
            config = session.exec(stmt).first()
            if not config:
                return None
            return config.to_dict(mask_sensitive=False)

    def list_configs(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        with Session(self._get_engine()) as session:
            stmt = select(ExternalApiConfigDB)
            count_stmt = select(ExternalApiConfigDB)

            if user_id:
                stmt = stmt.where(ExternalApiConfigDB.user_id == user_id)
                count_stmt = count_stmt.where(ExternalApiConfigDB.user_id == user_id)
            if status:
                stmt = stmt.where(ExternalApiConfigDB.status == status)
                count_stmt = count_stmt.where(ExternalApiConfigDB.status == status)

            total = len(session.exec(count_stmt).all())
            stmt = stmt.order_by(ExternalApiConfigDB.created_at.desc())
            stmt = stmt.offset(offset).limit(limit)
            configs = session.exec(stmt).all()
            return [c.to_dict() for c in configs], total

    def update_config(self, config_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalApiConfigDB)
                .where(ExternalApiConfigDB.config_id == config_id)
                .with_for_update()
            )
            config = session.exec(stmt).first()
            if not config:
                return None

            if (
                "api_url" in kwargs
                and (kwargs.get("api_url") or "").strip()
                != (config.api_url or "").strip()
            ):
                deployments, sync_tasks = self._lock_references(
                    session,
                    config_id,
                )
                if deployments or sync_tasks:
                    raise ValueError(
                        "External API URL cannot change while referenced by a "
                        "sync task or deployment; create a new API config"
                    )

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            config.updated_at = _utcnow_naive()

            session.add(config)
            session.commit()
            session.refresh(config)
            return config.to_dict()

    def delete_config(self, config_id: str) -> bool:
        with Session(self._get_engine()) as session:
            stmt = (
                select(ExternalApiConfigDB)
                .where(ExternalApiConfigDB.config_id == config_id)
                .with_for_update()
            )
            config = session.exec(stmt).first()
            if not config:
                return False
            deployments, sync_tasks = self._lock_references(
                session,
                config_id,
            )
            if deployments or sync_tasks:
                raise ValueError(
                    "Cannot delete: this API config is referenced by sync "
                    "configurations or deployments"
                )
            session.delete(config)
            session.commit()
            logger.info(f"Deleted external API config: {config_id}")
            return True

    def is_referenced_by_sync(self, config_id: str) -> bool:
        """Check if this API config is referenced by sync tasks or deployments."""
        from ..entities.external_sync_entity import ExternalSyncTaskDB
        from ..entities.deployment_entity import DeploymentDB

        with Session(self._get_engine()) as session:
            sync_stmt = select(ExternalSyncTaskDB).where(
                ExternalSyncTaskDB.external_api_config_id == config_id
            )
            if session.exec(sync_stmt).first() is not None:
                return True

            deployment_stmt = select(DeploymentDB).where(
                _deployment_api_config_reference(config_id)
            )
            return session.exec(deployment_stmt).first() is not None


external_api_config_service = ExternalApiConfigService()
