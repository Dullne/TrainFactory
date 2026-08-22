"""
Model registry service for database operations.
"""

import logging
import math
import numbers
import shutil
from ...utils.path_utils import hash_path
import os
from typing import Optional, List, Dict, Any, Iterable, Tuple
from train_factory.core.time_utils import now_naive

from sqlalchemy import update

from sqlmodel import select, or_, func

from ..database import get_session
from ..entities.model_registry_entity import ModelRegistryDB, ModelVersionDB
from ..entities.deployment_entity import DeploymentDB
from ..entities.model_config_entity import ModelConfigDB
from ...config import settings
from ...core.remote_download_security import resolve_managed_artifact_directory

logger = logging.getLogger(__name__)

_UNSET_USER_ID = object()


def _normalize_artifact_path(path: Optional[str]) -> Optional[str]:
    """Return a stable local path for exact and containment comparisons."""
    if not isinstance(path, str) or not path.strip():
        return None
    return os.path.normcase(
        os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    )


def _is_same_or_descendant(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _remove_deployment_container(deployment: DeploymentDB, model_id: str) -> None:
    """Remove a managed deployment runtime before deleting its database row."""
    from ...deployment.deployment_service import deployment_service
    from ...deployment.docker_deployer import docker_deployer

    if deployment_service._is_unmanaged_binding(deployment):
        return

    deploy_mode = deployment_service._normalize_deploy_mode(deployment.deploy_mode)
    if deploy_mode == "container":
        if not deployment.container_name:
            return
        deployment_service._require_managed_container(deployment)
        try:
            removed = docker_deployer.remove_container(deployment.container_name)
        except Exception as stop_err:
            raise RuntimeError(
                "Cannot delete model because container cleanup failed "
                f"for deployment {deployment.deployment_id}"
            ) from stop_err
        if not removed:
            raise RuntimeError(
                "Cannot delete model because container cleanup failed "
                f"for deployment {deployment.deployment_id}"
            )
        logger.warning(
            f"Stopped container {deployment.container_name} for model {model_id}"
        )
        return

    if deployment.status not in {"running", "starting", "restarting", "stopping"}:
        return
    if not deployment.model_uid:
        return
    try:
        client = deployment_service._get_xinference_client(
            deployment.xinference_endpoint,
            user_id=deployment.user_id,
        )
        client.terminate_model(deployment.model_uid)
    except Exception as stop_err:
        raise RuntimeError(
            "Cannot delete model because shared runtime cleanup failed "
            f"for deployment {deployment.deployment_id}"
        ) from stop_err
    logger.warning(
        f"Terminated shared model {deployment.model_uid} for model {model_id}"
    )


class ModelRegistryService:
    """Service for model registry database operations."""

    def _split_metrics(self, metrics: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
        """Split metrics into numeric and raw buckets."""
        if not metrics or not isinstance(metrics, dict):
            return None, None
        numeric: Dict[str, float] = {}
        raw: Dict[str, Any] = {}
        for key, value in metrics.items():
            num = self._normalize_metric_value(value)
            if num is not None:
                numeric[key] = num
            else:
                raw[key] = value
        return (numeric or None), (raw or None)

    def _normalize_metric_value(self, value: Any) -> Optional[float]:
        """Coerce a metric value to float when possible; return None otherwise."""
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, numbers.Number):
            num = float(value)
            return num if math.isfinite(num) else None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("%"):
                text = text[:-1].strip()
            try:
                num = float(text)
            except ValueError:
                return None
            return num if math.isfinite(num) else None
        return None

    def _normalize_metrics(self, metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Normalize metrics to numeric values; drop invalid entries."""
        if not metrics or not isinstance(metrics, dict):
            return None
        normalized: Dict[str, float] = {}
        for key, value in metrics.items():
            num = self._normalize_metric_value(value)
            if num is not None:
                normalized[key] = num
        return normalized or None

    def _get_path_unique_key_for_model(self, model: ModelRegistryDB) -> Optional[str]:
        """Determine a stable salt for model_path hashing when needed."""
        if model.source_type == "external_bind":
            extra = model.extra_metadata or {}
            if isinstance(extra, dict):
                bound_uid = extra.get("bound_model_uid")
                if bound_uid:
                    return str(bound_uid)
            if model.model_name:
                return str(model.model_name)
            return str(model.model_id)
        return None

    def _model_to_dict(self, model: ModelRegistryDB) -> Dict[str, Any]:
        """Convert model ORM object to dictionary."""
        return {
            "id": model.id,
            "model_id": model.model_id,
            "model_name": model.model_name,
            "version": model.version,
            "model_type": model.model_type,
            "source_task_id": model.source_task_id,
            "base_model_path": model.base_model_path,
            "model_path": model.model_path,
            "description": model.description,
            "tags": model.tags,
            "category": model.category,
            "extra_metadata": model.extra_metadata,
            "source_type": model.source_type,
            "is_adapter": bool(model.is_adapter),
            "metrics": self._normalize_metrics(model.metrics),
            "file_size": model.file_size,
            "status": model.status,
            "is_latest": model.is_latest,
            "user_id": model.user_id,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    def _version_to_dict(self, version: ModelVersionDB) -> Dict[str, Any]:
        """Convert version ORM object to dictionary."""
        return {
            "id": version.id,
            "version_id": version.version_id,
            "model_id": version.model_id,
            "version": version.version,
            "model_path": version.model_path,
            "changelog": version.changelog,
            "metrics": self._normalize_metrics(version.metrics),
            "created_at": version.created_at,
        }

    # ==================== CRUD Operations ====================

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        version: str = "v1.0.0",
        source_task_id: Optional[str] = None,
        base_model_path: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        source_type: Optional[str] = None,
        path_unique_key: Optional[str] = None,
        is_adapter: bool = False,
    ) -> Dict[str, Any]:
        """Register a new model. Returns dict with model info."""
        with get_session() as session:
            normalized_metrics, raw_metrics = self._split_metrics(metrics)
            extra_metadata_payload = extra_metadata
            if raw_metrics:
                if isinstance(extra_metadata, dict):
                    extra_metadata_payload = dict(extra_metadata)
                else:
                    extra_metadata_payload = {}
                extra_metadata_payload["_metrics_raw"] = raw_metrics
            # Calculate file size if path exists
            file_size = None
            if os.path.exists(model_path):
                if os.path.isfile(model_path):
                    file_size = os.path.getsize(model_path)
                elif os.path.isdir(model_path):
                    file_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(model_path)
                        for filename in filenames
                    )

            # Mark previous versions as not latest（原子 UPDATE，防并发注册
            # 产生两个 is_latest=True）
            statement = update(ModelRegistryDB).where(
                ModelRegistryDB.model_name == model_name,
                ModelRegistryDB.is_latest.is_(True),
            )
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            session.exec(statement.values(is_latest=False))

            # Determine status: if model_path exists locally, set to "available"
            if status is None:
                if os.path.exists(model_path):
                    status = "available"
                else:
                    status = "registered"

            # Create new model registry entry
            model = ModelRegistryDB(
                model_name=model_name,
                model_path=model_path,
                model_path_hash=hash_path(model_path, path_unique_key),
                model_type=model_type,
                version=version,
                source_task_id=source_task_id,
                base_model_path=base_model_path,
                description=description,
                tags=tags,
                category=category,
                extra_metadata=extra_metadata_payload,
                metrics=normalized_metrics,
                file_size=file_size,
                user_id=user_id,
                is_latest=True,
                status=status,
                source_type=source_type or "trained",
                is_adapter=is_adapter,
            )
            session.add(model)
            # Flush to get model_id without committing
            session.flush()
            session.refresh(model)

            # Also create a version entry
            version_entry = ModelVersionDB(
                model_id=model.model_id,
                version=version,
                model_path=model_path,
                changelog="Initial registration",
                metrics=normalized_metrics,
            )
            session.add(version_entry)

            # Single commit for both model and version
            session.commit()

            logger.info(f"Registered model: {model.model_id} ({model_name} {version})")
            return self._model_to_dict(model)

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by model_id. Returns dict or None."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
            model = session.exec(statement).first()
            if model:
                return self._model_to_dict(model)
            return None

    def get_model_by_path(
        self, model_path: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get model by model_path."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(
                ModelRegistryDB.model_path_hash == hash_path(model_path)
            )
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            model = session.exec(statement).first()
            return self._model_to_dict(model) if model else None

    def list_models_referencing_artifact_paths(
        self,
        artifact_paths: Iterable[Optional[str]],
        *,
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """List same-tenant models stored at or below artifact paths."""
        normalized_roots = {
            normalized
            for path in artifact_paths
            if (normalized := _normalize_artifact_path(path)) is not None
        }
        if not normalized_roots:
            return []

        with get_session() as session:
            models = session.exec(
                select(ModelRegistryDB).where(
                    ModelRegistryDB.user_id == user_id
                )
            ).all()
            references = []
            for model in models:
                normalized_model_path = _normalize_artifact_path(model.model_path)
                if normalized_model_path and any(
                    _is_same_or_descendant(normalized_model_path, root)
                    for root in normalized_roots
                ):
                    references.append(self._model_to_dict(model))
            return references

    def get_model_by_source_task(
        self,
        task_id: str,
        user_id: Any = _UNSET_USER_ID,
    ) -> Optional[Dict[str, Any]]:
        """Get model by source training task ID."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(
                ModelRegistryDB.source_task_id == task_id
            )
            if user_id is not _UNSET_USER_ID:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            model = session.exec(statement).first()
            return self._model_to_dict(model) if model else None

    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
        latest_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List models with optional filters.

        Returns:
            Tuple of (models, total_count)
        """
        with get_session() as session:
            # Build filter conditions
            conditions = []
            if model_type:
                conditions.append(ModelRegistryDB.model_type == model_type)
            if status:
                conditions.append(ModelRegistryDB.status == status)
            if category:
                conditions.append(ModelRegistryDB.category == category)
            if user_id:
                conditions.append(ModelRegistryDB.user_id == user_id)
            if latest_only:
                conditions.append(ModelRegistryDB.is_latest.is_(True))

            # Get total count
            count_stmt = select(func.count()).select_from(ModelRegistryDB)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Get paginated data
            statement = select(ModelRegistryDB)
            for cond in conditions:
                statement = statement.where(cond)
            statement = statement.order_by(ModelRegistryDB.created_at.desc())
            statement = statement.offset(offset).limit(limit)
            models = session.exec(statement).all()
            return [self._model_to_dict(model) for model in models], total

    def update_model(
        self,
        model_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Update model information."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
            model = session.exec(statement).first()
            if model:
                if description is not None:
                    model.description = description
                if tags is not None:
                    model.tags = tags
                if category is not None:
                    model.category = category
                if extra_metadata is not None:
                    model.extra_metadata = extra_metadata
                if status is not None:
                    model.update_status(status)
                else:
                    model.updated_at = now_naive()
                session.add(model)
                session.commit()
                logger.info(f"Updated model {model_id}")
                return True
            return False

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate model-registry statistics (user-scoped, filter-independent).

        Used by the list endpoint so dashboard cards reflect real totals rather
        than only the current page.
        """
        with get_session() as session:
            def _grouped(column):
                stmt = select(column, func.count()).group_by(column)
                if user_id:
                    stmt = stmt.where(ModelRegistryDB.user_id == user_id)
                # row[0] = group value, row[1] = count; skip null/empty keys
                return {row[0]: row[1] for row in session.exec(stmt).all() if row[0]}

            count_stmt = select(func.count()).select_from(ModelRegistryDB)
            if user_id:
                count_stmt = count_stmt.where(ModelRegistryDB.user_id == user_id)
            total = session.exec(count_stmt).one()

            return {
                "total": total,
                "by_status": _grouped(ModelRegistryDB.status),
                "by_type": _grouped(ModelRegistryDB.model_type),
            }

    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete a model and its versions.

        Args:
            model_id: The model ID to delete
            force: If True, also delete related deployments and configs

        Returns:
            True if deleted, False if model not found

        Raises:
            ValueError: If model has active deployments or configs and force=False
        """
        with get_session() as session:
            model_statement = select(ModelRegistryDB).where(
                ModelRegistryDB.model_id == model_id
            )
            model = session.exec(model_statement).first()
            if not model:
                return False

            # Check for active deployments
            deployment_stmt = select(DeploymentDB).where(DeploymentDB.model_id == model_id)
            deployments = session.exec(deployment_stmt).all()

            # Check for model configs
            config_stmt = select(ModelConfigDB).where(ModelConfigDB.registry_id == model_id)
            configs = session.exec(config_stmt).all()

            if deployments or configs:
                if not force:
                    deps = []
                    if deployments:
                        deps.append(f"{len(deployments)} deployment(s)")
                    if configs:
                        deps.append(f"{len(configs)} config(s)")
                    raise ValueError(
                        f"Cannot delete model {model_id}: has {', '.join(deps)}. "
                        "Use force=True to delete anyway."
                    )

            expected_path = resolve_managed_artifact_directory(
                settings.models_dir,
                model_id,
            )
            try:
                stored_path = resolve_managed_artifact_directory(
                    settings.models_dir,
                    model_id,
                    model.model_path,
                )
            except ValueError:
                if model.source_type == "downloaded":
                    raise
                stored_path = None

            if deployments or configs:
                # Stop external runtimes before removing model files. A failed
                # runtime cleanup must leave both storage and DB records retryable.
                for deployment in deployments:
                    _remove_deployment_container(deployment, model_id)
                    session.delete(deployment)
                    logger.warning(f"Force deleted deployment {deployment.deployment_id} for model {model_id}")
                for config in configs:
                    session.delete(config)
                    logger.warning(f"Force deleted config {config.config_id} for model {model_id}")

            if stored_path == expected_path and stored_path.exists():
                try:
                    shutil.rmtree(stored_path)
                except OSError as exc:
                    raise RuntimeError(
                        f"Model {model_id} storage cleanup failed; record retained"
                    ) from exc

            # Delete versions first
            version_statement = select(ModelVersionDB).where(ModelVersionDB.model_id == model_id)
            versions = session.exec(version_statement).all()
            for version in versions:
                session.delete(version)

            # Delete model
            session.delete(model)
            session.commit()
            logger.info(f"Deleted model {model_id}")
            return True

    # ==================== Version Management ====================

    def add_version(
        self,
        model_id: str,
        version: str,
        model_path: str,
        changelog: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        set_as_latest: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Add a new version to an existing model."""
        with get_session() as session:
            normalized_metrics, raw_metrics = self._split_metrics(metrics)
            # Get the model
            model_statement = select(ModelRegistryDB).where(ModelRegistryDB.model_id == model_id)
            model = session.exec(model_statement).first()
            if not model:
                return None

            # Create version entry
            version_entry = ModelVersionDB(
                model_id=model_id,
                version=version,
                model_path=model_path,
                changelog=changelog,
                metrics=normalized_metrics,
            )
            session.add(version_entry)

            # Update model if setting as latest
            if set_as_latest:
                # Mark all other versions of this model as not latest（原子 UPDATE）
                session.exec(
                    update(ModelRegistryDB)
                    .where(
                        ModelRegistryDB.model_name == model.model_name,
                        ModelRegistryDB.user_id == model.user_id,
                        ModelRegistryDB.is_latest.is_(True),
                    )
                    .values(is_latest=False)
                )

                path_unique_key = self._get_path_unique_key_for_model(model)
                if path_unique_key is None and model.model_path_hash:
                    # Preserve salted hashes for legacy duplicate paths.
                    if model.model_path_hash != hash_path(model.model_path):
                        path_unique_key = str(model.model_id or model.id)

                if raw_metrics:
                    extra_metadata_payload = dict(model.extra_metadata or {})
                    extra_metadata_payload["_metrics_raw"] = raw_metrics
                    model.extra_metadata = extra_metadata_payload

                model.version = version
                model.model_path = model_path
                model.model_path_hash = hash_path(model_path, path_unique_key)
                model.metrics = normalized_metrics
                model.is_latest = True
                model.updated_at = now_naive()
                session.add(model)

            session.commit()
            session.refresh(version_entry)
            logger.info(f"Added version {version} to model {model_id}")
            return self._version_to_dict(version_entry)

    def get_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a model."""
        with get_session() as session:
            statement = select(ModelVersionDB).where(ModelVersionDB.model_id == model_id)
            statement = statement.order_by(ModelVersionDB.created_at.desc())
            versions = session.exec(statement).all()
            return [self._version_to_dict(v) for v in versions]

    # ==================== Search & Filter ====================

    def search_models(
        self,
        query: str,
        model_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search models by name or description."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(
                or_(
                    ModelRegistryDB.model_name.contains(query),
                    ModelRegistryDB.description.contains(query),
                )
            )
            if model_type:
                statement = statement.where(ModelRegistryDB.model_type == model_type)
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            statement = statement.limit(limit)
            models = session.exec(statement).all()
            return [self._model_to_dict(model) for model in models]

    def get_models_by_tag(
        self,
        tag: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get models with a specific tag."""
        # Note: JSON array containment query varies by database
        # For MySQL, we use JSON_CONTAINS
        with get_session() as session:
            statement = select(ModelRegistryDB)
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            # Filter in Python for simplicity (works across databases)
            models = session.exec(statement).all()
            result = []
            for model in models:
                if model.tags and tag in model.tags:
                    result.append(self._model_to_dict(model))
            return result

    def get_models_by_category(
        self,
        category: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get models in a specific category."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(ModelRegistryDB.category == category)
            if user_id:
                statement = statement.where(ModelRegistryDB.user_id == user_id)
            models = session.exec(statement).all()
            return [self._model_to_dict(model) for model in models]

    # ==================== Model Comparison ====================

    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple models."""
        with get_session() as session:
            statement = select(ModelRegistryDB).where(ModelRegistryDB.model_id.in_(model_ids))
            models = session.exec(statement).all()

            comparison = {
                "models": [],
                "metrics_keys": set(),
            }

            for model in models:
                normalized_metrics = self._normalize_metrics(model.metrics) or {}
                model_info = {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "version": model.version,
                    "model_type": model.model_type,
                    "metrics": normalized_metrics,
                }
                comparison["models"].append(model_info)
                if normalized_metrics:
                    comparison["metrics_keys"].update(normalized_metrics.keys())

            comparison["metrics_keys"] = list(comparison["metrics_keys"])
            return comparison

    # ==================== Register from Task ====================

    def register_from_task(
        self,
        task_id: str,
        model_name: str,
        version: str = "v1.0.0",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Register a model from a completed training task."""
        from .training_task_service import training_task_service

        existing = self.get_model_by_source_task(task_id, user_id=user_id)
        if existing:
            logger.info(f"Model already registered for task {task_id}")
            return existing

        task = training_task_service.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return None

        if task['status'] != 'succeeded':
            logger.error(f"Task {task_id} has not succeeded, status: {task['status']}")
            return None

        if not task['final_model_path']:
            logger.error(f"Task {task_id} has no final model path")
            return None

        if user_id and task.get('user_id') and task['user_id'] != user_id:
            logger.warning(f"User {user_id} not authorized to register task {task_id}")
            return None

        training_params = task.get('training_params') or {}
        is_adapter = bool(task.get('is_lora') or training_params.get('use_lora'))

        model = self.register_model(
            model_name=model_name,
            model_path=task['final_model_path'],
            model_type=task['model_type'],
            version=version,
            source_task_id=task_id,
            base_model_path=task['base_model_path'],
            description=description or task['description'],
            tags=tags,
            category=category,
            metrics=task['final_metrics'],
            user_id=user_id or task['user_id'],
            is_adapter=is_adapter,
        )
        if model:
            training_task_service.update_task_trained_model_registry_id(task_id, model['model_id'])
        return model


# Global service instance
model_registry_service = ModelRegistryService()
