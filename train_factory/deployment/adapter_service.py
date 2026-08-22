"""
LoRA Adapter management service.

Handles loading and unloading of LoRA adapters on vLLM and SGLang deployments.
Xinference does not support runtime adapter loading.
"""

import logging
import os
import posixpath
from typing import Optional, List, Dict, Any

from sqlmodel import select

from ..storage.database import get_session
from ..storage.entities.deployment_entity import DeploymentDB
from ..storage.entities.loaded_adapter_entity import LoadedAdapterDB
from ..storage.services.model_registry_service import model_registry_service
from .vllm_client import VLLMClient
from .sglang_client import SGLangClient

logger = logging.getLogger(__name__)


class AdapterService:
    """Service for managing LoRA adapters on deployments."""

    def _get_inference_client(self, deployment: DeploymentDB):
        """Get the appropriate inference client for the deployment."""
        framework = deployment.inference_framework or "xinference"
        endpoint = deployment.xinference_endpoint

        if framework == "vllm":
            return VLLMClient(endpoint, user_id=deployment.user_id)
        elif framework == "sglang":
            return SGLangClient(endpoint, user_id=deployment.user_id)
        else:
            raise ValueError(f"Framework '{framework}' does not support adapter hot-loading")

    def _adapter_to_dict(self, adapter: LoadedAdapterDB) -> Dict[str, Any]:
        """Convert adapter entity to dictionary."""
        return adapter.to_dict()

    # ==================== Load/Unload Operations ====================

    def load_adapter(
        self,
        deployment_id: str,
        adapter_name: str,
        adapter_path: str,
        source_task_id: Optional[str] = None,
        source_model_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a LoRA adapter onto a running deployment.

        Args:
            deployment_id: Deployment to load adapter on
            adapter_name: Unique name for the adapter (used in inference)
            adapter_path: Path to adapter weights
            source_task_id: Training task that created this adapter
            source_model_id: Model registry ID if adapter is registered
            user_id: User ID for isolation

        Returns:
            Loaded adapter info dict

        Raises:
            ValueError: If deployment not found, not running, or doesn't support LoRA
            RuntimeError: If loading fails
        """
        with get_session() as session:
            # Get deployment
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            if deployment.status != "running":
                raise ValueError(f"Deployment is not running (status: {deployment.status})")

            if not deployment.enable_lora:
                raise ValueError("Deployment does not have LoRA enabled")

            framework = deployment.inference_framework or "xinference"
            if framework == "xinference":
                raise ValueError("Xinference does not support runtime adapter loading")

            # Embedding adapter hot-loading is only guaranteed on vLLM.
            try:
                model = model_registry_service.get_model(deployment.model_id)
                if model and model.get("model_type") == "embedding" and framework != "vllm":
                    raise ValueError(
                        f"Embedding adapter hot-loading is only supported on vLLM, "
                        f"current framework: {framework}"
                    )
            except ValueError:
                raise
            except Exception as e:
                logger.warning(f"Failed to verify deployment model type for adapter loading: {e}")

            # Clean up stale records (failed/stuck loading) for this adapter name
            # First try to unload from inference service, then update DB
            stale = session.exec(
                select(LoadedAdapterDB).where(
                    LoadedAdapterDB.deployment_id == deployment_id,
                    LoadedAdapterDB.adapter_name == adapter_name,
                    LoadedAdapterDB.status.in_(["failed", "loading"])
                )
            ).all()
            if stale:
                try:
                    client = self._get_inference_client(deployment)
                    client.unload_lora_adapter(adapter_name)
                    logger.info(f"Unloaded stale adapter '{adapter_name}' from service before retry")
                except Exception:
                    pass  # May not exist on service, that's fine
                for s in stale:
                    s.force_status("unloaded", "Cleaned up before retry")
                    session.add(s)
                session.commit()
                logger.info(f"Cleaned up {len(stale)} stale adapter records for '{adapter_name}'")

            # Check if adapter already loaded — treat as success
            existing = session.exec(
                select(LoadedAdapterDB).where(
                    LoadedAdapterDB.deployment_id == deployment_id,
                    LoadedAdapterDB.adapter_name == adapter_name,
                    LoadedAdapterDB.status.in_(["loading", "loaded"])
                )
            ).first()
            if existing and existing.status == "loaded":
                logger.info(f"Adapter '{adapter_name}' already loaded on deployment {deployment_id}, returning existing")
                return self._adapter_to_dict(existing)
            if existing and existing.status == "loading":
                # Stuck in loading state — clean up and retry
                existing.force_status("unloaded", "Cleaned up stuck loading state")
                session.add(existing)
                session.commit()

            # Check max_loras limit
            loaded_count = session.exec(
                select(LoadedAdapterDB).where(
                    LoadedAdapterDB.deployment_id == deployment_id,
                    LoadedAdapterDB.status == "loaded"
                )
            ).all()
            if len(loaded_count) >= deployment.max_loras:
                raise ValueError(
                    f"Maximum number of adapters ({deployment.max_loras}) reached. "
                    "Unload an adapter first."
                )

            # Create adapter record
            adapter = LoadedAdapterDB(
                deployment_id=deployment_id,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                source_task_id=source_task_id,
                source_model_id=source_model_id,
                user_id=user_id,
                status="loading",
            )
            session.add(adapter)
            session.commit()
            session.refresh(adapter)

            try:
                # Get client and load adapter
                client = self._get_inference_client(deployment)
                client.load_lora_adapter(adapter_name, adapter_path)

                # Update status to loaded
                adapter.update_status("loaded")
                session.add(adapter)
                session.commit()
                session.refresh(adapter)

                logger.info(f"Loaded adapter '{adapter_name}' on deployment {deployment_id}")
                return self._adapter_to_dict(adapter)

            except Exception as e:
                # Update status to failed
                adapter.update_status("failed", str(e))
                session.add(adapter)
                session.commit()
                logger.error(f"Failed to load adapter '{adapter_name}': {e}")
                raise RuntimeError(f"Failed to load adapter: {e}")

    def unload_adapter(
        self,
        deployment_id: str,
        adapter_name: str,
    ) -> bool:
        """
        Unload a LoRA adapter from a deployment.

        Args:
            deployment_id: Deployment ID
            adapter_name: Name of the adapter to unload

        Returns:
            True if unloaded successfully

        Raises:
            ValueError: If deployment or adapter not found
            RuntimeError: If unloading fails
        """
        with get_session() as session:
            # Get deployment
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment:
                raise ValueError(f"Deployment not found: {deployment_id}")

            # Get adapter record
            adapter = session.exec(
                select(LoadedAdapterDB).where(
                    LoadedAdapterDB.deployment_id == deployment_id,
                    LoadedAdapterDB.adapter_name == adapter_name,
                    LoadedAdapterDB.status == "loaded"
                )
            ).first()

            if not adapter:
                # DB has no loaded record, but adapter may still exist on the service
                # (e.g. DB got corrupted after a failed load/unload cycle).
                # Try to unload from service directly.
                try:
                    client = self._get_inference_client(deployment)
                    client.unload_lora_adapter(adapter_name)
                    logger.info(
                        f"Unloaded orphaned adapter '{adapter_name}' from service "
                        f"(no DB loaded record)"
                    )
                except Exception:
                    pass  # Adapter may genuinely not exist on service
                # Also clean up any non-loaded DB records
                stale = session.exec(
                    select(LoadedAdapterDB).where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        LoadedAdapterDB.adapter_name == adapter_name,
                        LoadedAdapterDB.status.in_(["loading", "failed"])
                    )
                ).all()
                for s in stale:
                    s.force_status("unloaded")
                    session.add(s)
                if stale:
                    session.commit()
                return True

            # Update status to unloading
            adapter.update_status("unloading")
            session.add(adapter)
            session.commit()

            try:
                # Get client and unload adapter
                client = self._get_inference_client(deployment)
                client.unload_lora_adapter(adapter_name)

                # Update status to unloaded
                adapter.update_status("unloaded")
                session.add(adapter)
                session.commit()

                logger.info(f"Unloaded adapter '{adapter_name}' from deployment {deployment_id}")
                return True

            except Exception as e:
                # Revert status to loaded on failure
                adapter.force_status("loaded", str(e))
                session.add(adapter)
                session.commit()
                logger.error(f"Failed to unload adapter '{adapter_name}': {e}")
                raise RuntimeError(f"Failed to unload adapter: {e}")

    # ==================== Query Operations ====================

    def list_active_training_output_consumers(
        self,
        task_id: str,
        final_model_path: Optional[str],
        *,
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """List active adapters that still consume a training task's output."""

        def normalize_path(path: Optional[str]) -> Optional[str]:
            if not isinstance(path, str) or not path.strip():
                return None
            return posixpath.normpath(path.strip().replace("\\", "/"))

        normalized_output_path = normalize_path(final_model_path)
        with get_session() as session:
            query = select(LoadedAdapterDB).where(
                LoadedAdapterDB.status.in_(["loading", "loaded", "unloading"])
            )
            if user_id is None:
                query = query.where(LoadedAdapterDB.user_id.is_(None))
            else:
                query = query.where(LoadedAdapterDB.user_id == user_id)

            consumers = []
            for adapter in session.exec(query).all():
                path_matches = (
                    normalized_output_path is not None
                    and normalize_path(adapter.adapter_path) == normalized_output_path
                )
                if adapter.source_task_id == task_id or path_matches:
                    consumers.append(self._adapter_to_dict(adapter))
            return consumers

    def list_loaded_adapters(
        self,
        deployment_id: str,
        include_unloaded: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List adapters loaded on a deployment.

        Args:
            deployment_id: Deployment ID
            include_unloaded: Include unloaded/failed adapters

        Returns:
            List of adapter info dicts
        """
        with get_session() as session:
            query = select(LoadedAdapterDB).where(
                LoadedAdapterDB.deployment_id == deployment_id
            )
            if not include_unloaded:
                query = query.where(LoadedAdapterDB.status.in_(["loading", "loaded"]))

            adapters = session.exec(query.order_by(LoadedAdapterDB.loaded_at.desc())).all()
            return [self._adapter_to_dict(a) for a in adapters]

    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """Get adapter by ID."""
        with get_session() as session:
            adapter = session.exec(
                select(LoadedAdapterDB).where(LoadedAdapterDB.adapter_id == adapter_id)
            ).first()
            if adapter:
                return self._adapter_to_dict(adapter)
            return None

    def get_available_adapters(
        self,
        base_model_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get available adapters that can be loaded.

        Searches training tasks with is_lora=True and model registry with is_adapter=True.

        Args:
            base_model_id: Filter by base model ID
            user_id: Filter by user ID

        Returns:
            List of available adapter info dicts
        """
        from ..storage.entities.training_task_entity import TrainingTaskDB

        available = []
        base_model_path_filter: Optional[str] = None
        base_model_id_cache: Dict[str, Optional[str]] = {}

        if base_model_id:
            # API contract: base_model_id is a model registry ID.
            base_model = model_registry_service.get_model(base_model_id)
            if not base_model:
                return []
            base_model_path = base_model.get("model_path")
            if not base_model_path:
                return []
            base_model_path_filter = os.path.normpath(base_model_path)

        def _resolve_base_model_id(base_model_path: Optional[str]) -> Optional[str]:
            if not base_model_path:
                return None
            norm_path = os.path.normpath(base_model_path)
            if norm_path in base_model_id_cache:
                return base_model_id_cache[norm_path]
            model = model_registry_service.get_model_by_path(norm_path, user_id=user_id)
            model_id = model.get("model_id") if model else None
            base_model_id_cache[norm_path] = model_id
            return model_id

        with get_session() as session:
            # Get adapters from completed training tasks
            task_query = select(TrainingTaskDB).where(
                TrainingTaskDB.is_lora.is_(True),
                TrainingTaskDB.status == "succeeded",
                TrainingTaskDB.final_model_path.isnot(None)
            )
            if user_id:
                task_query = task_query.where(TrainingTaskDB.user_id == user_id)

            tasks = session.exec(task_query).all()
            for task in tasks:
                task_base_path = task.base_model_path or ""
                if base_model_path_filter and os.path.normpath(task_base_path) != base_model_path_filter:
                    continue
                available.append({
                    "source": "training_task",
                    "source_id": task.task_id,
                    "name": f"task-{task.task_id[:8]}",
                    "path": task.final_model_path,
                    "base_model_id": _resolve_base_model_id(task_base_path),
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "description": f"From training task: {task.task_name or task.task_id}",
                })

            # Get adapters from model registry
            all_models, _ = model_registry_service.list_models(user_id=user_id)
            models = [m for m in all_models if m.get('is_adapter')]
            for model in models:
                model_base_path = model.get("base_model_path", "")
                if base_model_path_filter and os.path.normpath(model_base_path) != base_model_path_filter:
                    continue

                available.append({
                    "source": "model_registry",
                    "source_id": model['model_id'],
                    "name": model['model_name'],
                    "path": model['model_path'],
                    "base_model_id": _resolve_base_model_id(model_base_path),
                    "created_at": model.get('created_at'),
                    "description": model.get('description'),
                })

        return available

    # ==================== Sync Operations ====================

    def sync_loaded_adapters(self, deployment_id: str) -> int:
        """
        Sync loaded adapters with actual state from inference service.

        Args:
            deployment_id: Deployment ID

        Returns:
            Number of adapters updated
        """
        with get_session() as session:
            # Get deployment
            statement = select(DeploymentDB).where(DeploymentDB.deployment_id == deployment_id)
            deployment = session.exec(statement).first()
            if not deployment or deployment.status != "running":
                return 0

            framework = deployment.inference_framework or "xinference"
            if framework == "xinference":
                return 0

            try:
                # Get actual loaded adapters from service
                client = self._get_inference_client(deployment)
                actual_adapters = client.list_lora_adapters()
                actual_names = {a.get("name") for a in actual_adapters}

                # Get DB records marked as loaded
                db_adapters = session.exec(
                    select(LoadedAdapterDB).where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        LoadedAdapterDB.status == "loaded"
                    )
                ).all()
                db_loaded_names = {a.adapter_name for a in db_adapters}

                updated_count = 0

                # 1. DB says loaded but service doesn't have it → mark unloaded
                for adapter in db_adapters:
                    if adapter.adapter_name not in actual_names:
                        adapter.force_status("unloaded", "Adapter not found on service (auto-sync)")
                        session.add(adapter)
                        updated_count += 1
                        logger.info(f"Auto-sync: adapter '{adapter.adapter_name}' marked as unloaded")

                # 2. Service has it but DB doesn't → restore or create DB record
                for name in actual_names:
                    if name in db_loaded_names:
                        continue
                    # Check if there's a recent unloaded/failed record we can restore
                    prev = session.exec(
                        select(LoadedAdapterDB).where(
                            LoadedAdapterDB.deployment_id == deployment_id,
                            LoadedAdapterDB.adapter_name == name,
                        ).order_by(LoadedAdapterDB.loaded_at.desc())
                    ).first()
                    if prev and prev.status in ("unloaded", "failed"):
                        prev.force_status("loaded")
                        prev.unloaded_at = None
                        session.add(prev)
                        updated_count += 1
                        logger.info(f"Auto-sync: restored adapter '{name}' to loaded from {prev.status}")
                    elif not prev:
                        # No record at all, create one with sentinel path
                        new_adapter = LoadedAdapterDB(
                            deployment_id=deployment_id,
                            adapter_name=name,
                            adapter_path="<unknown:auto-sync>",
                            status="loaded",
                        )
                        session.add(new_adapter)
                        updated_count += 1
                        logger.info(f"Auto-sync: created DB record for adapter '{name}' found on service")

                session.commit()
                return updated_count

            except Exception as e:
                logger.warning(f"Failed to sync adapters for deployment {deployment_id}: {e}")
                return 0


# Global service instance
adapter_service = AdapterService()
