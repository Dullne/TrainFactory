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

from ..config import settings
from ..core.remote_download_security import resolve_managed_artifact_directory
from ..storage.database import get_session
from ..storage.entities.deployment_entity import DeploymentDB
from ..storage.entities.deployment_replica_entity import DeploymentReplicaDB
from ..storage.entities.loaded_adapter_entity import LoadedAdapterDB
from ..storage.entities.model_registry_entity import (
    MODEL_DELETE_INTENT_METADATA_KEY,
    ModelRegistryDB,
)
from ..storage.services.model_artifact_membership_service import (
    lock_model_artifact_membership,
)
from ..storage.services.model_registry_service import model_registry_service
from .vllm_client import VLLMClient
from .sglang_client import SGLangClient
from .deployment_service import (
    DeploymentReplicaNotFoundError,
    DeploymentService,
    ReplicaOperationClaim,
    ReplicaOperationBusyError,
    ReplicaOperationLostError,
    deployment_service,
)

logger = logging.getLogger(__name__)


class AdapterService:
    """Service for managing LoRA adapters on deployments."""

    def __init__(
        self,
        deployment_lifecycle_service: DeploymentService = deployment_service,
    ) -> None:
        self._deployment_lifecycle_service = deployment_lifecycle_service

    @staticmethod
    def _require_claim_owner(
        session,
        claim: ReplicaOperationClaim | None,
        *,
        lock: bool = True,
    ) -> DeploymentDB | None:
        if claim is None:
            return None
        replica_condition = DeploymentDB.replica_operation_replica_id.is_(None)
        if claim.replica_id is not None:
            replica_condition = (
                DeploymentDB.replica_operation_replica_id == claim.replica_id
            )
        statement = select(DeploymentDB).where(
            DeploymentDB.deployment_id == claim.deployment_id,
            DeploymentDB.replica_operation_token == claim.token,
            DeploymentDB.replica_operation_generation == claim.generation,
            DeploymentDB.replica_operation_kind == claim.operation,
            replica_condition,
        )
        if lock:
            statement = statement.with_for_update()
        owner = session.exec(statement).first()
        if owner is None:
            session.rollback()
            raise ReplicaOperationLostError(
                "deployment replica operation ownership was lost"
            )
        return owner

    @staticmethod
    def _lock_unknown_adapter_membership(session) -> None:
        """Serialize a conservative runtime-only adapter membership write."""
        lock_model_artifact_membership(session)
        models = list(
            session.exec(
                select(ModelRegistryDB)
                .order_by(ModelRegistryDB.model_id)
                .with_for_update()
            ).all()
        )
        for model in models:
            metadata = model.extra_metadata
            has_delete_intent = (
                isinstance(metadata, dict)
                and isinstance(
                    metadata.get(MODEL_DELETE_INTENT_METADATA_KEY),
                    dict,
                )
            )
            if model.status != "deleting" and not has_delete_intent:
                continue
            try:
                resolve_managed_artifact_directory(
                    settings.models_dir,
                    model.model_id,
                    model.model_path,
                )
            except ValueError:
                continue
            raise ReplicaOperationBusyError(
                "cannot synchronize an adapter with unknown source while a "
                "managed model deletion is in progress"
            )

    def _sync_parent_status_is_eligible(self, deployment: DeploymentDB) -> bool:
        if self._deployment_lifecycle_service._uses_replica_lifecycle(deployment):
            return deployment.status in {"running", "degraded"}
        return deployment.status == "running"

    def _require_sync_parent_owner(
        self,
        session,
        deployment_id: str,
        *,
        claim: ReplicaOperationClaim | None,
        deployment_replica_id: str | None,
        user_id: str | None,
        expected_replica_lifecycle: bool,
    ) -> DeploymentDB:
        if claim is not None:
            owner = self._require_claim_owner(session, claim)
        else:
            statement = select(DeploymentDB).where(
                DeploymentDB.deployment_id == deployment_id
            )
            if user_id is not None:
                statement = statement.where(DeploymentDB.user_id == user_id)
            owner = session.exec(statement.with_for_update()).first()
        if (
            owner is None
            or not self._sync_parent_status_is_eligible(owner)
            or owner.replica_operation_kind == "delete"
        ):
            session.rollback()
            raise ReplicaOperationLostError(
                "deployment was deleted or changed during adapter sync"
            )
        uses_replica_lifecycle = (
            self._deployment_lifecycle_service._uses_replica_lifecycle(owner)
        )
        if uses_replica_lifecycle != expected_replica_lifecycle:
            session.rollback()
            raise ReplicaOperationLostError(
                "deployment replica lifecycle changed during adapter sync"
            )
        if uses_replica_lifecycle:
            replica = session.exec(
                select(DeploymentReplicaDB)
                .where(
                    DeploymentReplicaDB.deployment_id == deployment_id,
                    DeploymentReplicaDB.replica_id == deployment_replica_id,
                    DeploymentReplicaDB.status == "running",
                    DeploymentReplicaDB.health_status == "HEALTHY",
                )
                .with_for_update()
            ).first()
            if replica is None:
                session.rollback()
                raise ReplicaOperationLostError(
                    "deployment replica changed during adapter sync"
                )
        return owner

    def _claim_adapter_operation(
        self,
        deployment_id: str,
        *,
        operation: str,
        deployment_replica_id: str | None,
        user_id: str | None,
    ) -> ReplicaOperationClaim:
        """Claim the exact parent generation for canonical and legacy runtimes."""
        return self._deployment_lifecycle_service._claim_replica_operation(
            deployment_id,
            operation=operation,
            replica_id=deployment_replica_id,
            user_id=user_id,
        )

    def _renew_claim_before_runtime(
        self,
        claim: ReplicaOperationClaim | None,
    ) -> None:
        if claim is not None:
            self._deployment_lifecycle_service._require_replica_operation_ownership(
                claim
            )

    @staticmethod
    def _runtime_adapter_name(item: object) -> str:
        if not isinstance(item, dict):
            return ""
        return str(item.get("name") or item.get("id") or "")

    @staticmethod
    def _runtime_adapter_path(item: object) -> str | None:
        if not isinstance(item, dict):
            return None
        value = (
            item.get("path")
            or item.get("adapter_path")
            or item.get("lora_path")
            or item.get("root")
        )
        if not isinstance(value, str) or not value.strip():
            return None
        # Runtime paths are paths inside the Linux inference service even when
        # the control plane itself runs on Windows.  Keep POSIX case semantics.
        return posixpath.normpath(value.strip().replace("\\", "/"))

    def _verify_runtime_adapter_absence(
        self,
        client,
        claim: ReplicaOperationClaim | None,
        adapter_name: str,
    ) -> None:
        """Require a successful unload or an authoritative list absence."""
        try:
            self._renew_claim_before_runtime(claim)
            client.unload_lora_adapter(adapter_name)
            return
        except ReplicaOperationLostError:
            raise
        except Exception as unload_error:
            try:
                self._renew_claim_before_runtime(claim)
                runtime_adapters = client.list_lora_adapters()
            except ReplicaOperationLostError:
                raise
            except Exception as list_error:
                raise RuntimeError(
                    "Failed to verify adapter runtime absence during unload recovery"
                ) from list_error
            if any(
                self._runtime_adapter_name(item) == adapter_name
                for item in runtime_adapters
            ):
                raise RuntimeError(
                    "Failed to verify adapter runtime absence during unload recovery"
                ) from unload_error

    def _load_runtime_with_reconciliation(
        self,
        client,
        claim: ReplicaOperationClaim | None,
        adapter_name: str,
        adapter_path: str,
    ) -> tuple[str, Exception | None]:
        """Return loaded, absent, or unknown after an uncertain load response."""
        try:
            self._renew_claim_before_runtime(claim)
            client.load_lora_adapter(adapter_name, adapter_path)
            return "loaded", None
        except ReplicaOperationLostError:
            raise
        except Exception as load_error:
            try:
                self._renew_claim_before_runtime(claim)
                runtime_adapters = client.list_lora_adapters()
            except ReplicaOperationLostError:
                raise
            except Exception:
                return "unknown", load_error
            matching = [
                item
                for item in runtime_adapters
                if self._runtime_adapter_name(item) == adapter_name
            ]
            if not matching:
                return "absent", load_error
            expected_path = self._runtime_adapter_path({"path": adapter_path})
            if any(
                self._runtime_adapter_path(item) == expected_path
                for item in matching
                if self._runtime_adapter_path(item) is not None
            ):
                return "loaded", load_error
            return "unknown", load_error

    def _finalize_adapter_loading(
        self,
        *,
        claim: ReplicaOperationClaim | None,
        adapter_id: str,
        deployment_id: str,
        bound_replica_id: str | None,
        runtime_state: str,
        runtime_error: Exception | None,
    ) -> Dict[str, Any]:
        with get_session() as session:
            self._require_claim_owner(session, claim)
            adapter = session.exec(
                select(LoadedAdapterDB)
                .where(
                    LoadedAdapterDB.adapter_id == adapter_id,
                    LoadedAdapterDB.deployment_id == deployment_id,
                    self._replica_binding_condition(bound_replica_id),
                    LoadedAdapterDB.status == "loading",
                )
                .with_for_update()
            ).first()
            if adapter is None:
                raise ReplicaOperationLostError(
                    "adapter plan changed before finalization"
                )
            if runtime_state == "loaded":
                adapter.update_status("loaded")
            elif runtime_state == "absent":
                adapter.update_status("failed", str(runtime_error))
            else:
                adapter.error_message = (
                    "adapter load result could not be verified: "
                    f"{runtime_error}"
                )
            session.add(adapter)
            result = self._adapter_to_dict(adapter)
            session.commit()
            return result

    def _finalize_stale_adapter_cleanup(
        self,
        *,
        claim: ReplicaOperationClaim | None,
        stale_ids: tuple[str, ...],
    ) -> None:
        if not stale_ids:
            return
        with get_session() as session:
            self._require_claim_owner(session, claim)
            stale = list(
                session.exec(
                    select(LoadedAdapterDB)
                    .where(
                        LoadedAdapterDB.adapter_id.in_(stale_ids),
                        LoadedAdapterDB.status.in_(["failed", "loading"]),
                    )
                    .order_by(LoadedAdapterDB.adapter_id)
                    .with_for_update()
                ).all()
            )
            if {adapter.adapter_id for adapter in stale} != set(stale_ids):
                raise ReplicaOperationLostError(
                    "stale adapter records changed before cleanup finalization"
                )
            for adapter in stale:
                adapter.force_status("unloaded", "Cleaned up before retry")
                session.add(adapter)
            session.commit()

    def _get_inference_client(
        self,
        deployment: DeploymentDB,
        *,
        endpoint: str | None = None,
    ):
        """Get the appropriate inference client for the deployment."""
        framework = deployment.inference_framework or "xinference"
        endpoint = endpoint or deployment.xinference_endpoint

        if framework == "vllm":
            return VLLMClient(endpoint, user_id=deployment.user_id)
        elif framework == "sglang":
            return SGLangClient(endpoint, user_id=deployment.user_id)
        else:
            raise ValueError(f"Framework '{framework}' does not support adapter hot-loading")

    def _adapter_to_dict(self, adapter: LoadedAdapterDB) -> Dict[str, Any]:
        """Convert adapter entity to dictionary."""
        return adapter.to_dict()

    @staticmethod
    def _resolve_replica_binding(
        session,
        deployment: DeploymentDB,
        deployment_replica_id: str | None,
        *,
        user_id: str | None,
        require_healthy: bool,
    ) -> DeploymentReplicaDB | None:
        if user_id is not None and deployment.user_id != user_id:
            raise ValueError("deployment not found")
        config = deployment.config or {}
        if not (
            isinstance(config, dict)
            and type(config.get("replica_schema_version")) is int
            and config.get("replica_schema_version") == 1
        ):
            if deployment_replica_id is not None:
                raise ValueError("deployment replica not found")
            return None
        replicas = list(
            session.exec(
                select(DeploymentReplicaDB)
                .where(
                    DeploymentReplicaDB.deployment_id
                    == deployment.deployment_id
                )
                .order_by(DeploymentReplicaDB.replica_index)
            ).all()
        )
        if deployment_replica_id is None:
            if len(replicas) != 1:
                raise ValueError(
                    "replica_id is required for multi-replica deployment"
                )
            replica = replicas[0]
        else:
            replica = next(
                (
                    item
                    for item in replicas
                    if item.replica_id == deployment_replica_id
                ),
                None,
            )
            if replica is None:
                raise ValueError("deployment replica not found")
        if require_healthy and not (
            replica.status == "running" and replica.health_status == "HEALTHY"
        ):
            raise ValueError("deployment replica is not running and healthy")
        return replica

    @staticmethod
    def _replica_binding_condition(deployment_replica_id: str | None):
        if deployment_replica_id is None:
            return LoadedAdapterDB.deployment_replica_id.is_(None)
        return LoadedAdapterDB.deployment_replica_id == deployment_replica_id

    def _claim_targets_legacy(
        self,
        claim: ReplicaOperationClaim,
    ) -> bool:
        with get_session() as session:
            deployment = self._require_claim_owner(
                session,
                claim,
                lock=False,
            )
            assert deployment is not None
            return not self._deployment_lifecycle_service._uses_replica_lifecycle(
                deployment
            )

    @staticmethod
    def _validate_legacy_adapter_parent(
        deployment: DeploymentDB,
    ) -> str:
        if deployment.status != "running":
            raise ValueError(
                f"Deployment is not running (status: {deployment.status})"
            )
        if not deployment.enable_lora:
            raise ValueError("Deployment does not have LoRA enabled")
        framework = deployment.inference_framework or "xinference"
        if framework == "xinference":
            raise ValueError(
                "Xinference does not support runtime adapter loading"
            )
        return framework

    def _validate_legacy_sglang_adapter_type(
        self,
        deployment: DeploymentDB,
        framework: str,
    ) -> None:
        if framework != "sglang":
            return
        try:
            model = model_registry_service.get_model(deployment.model_id)
        except Exception as exc:
            raise ValueError(
                "Unable to verify SGLang deployment model type for adapter loading"
            ) from exc
        model_type = str((model or {}).get("model_type") or "").strip().lower()
        if not model_type:
            raise ValueError(
                "Unable to verify SGLang deployment model type for adapter loading"
            )
        if model_type in {"rerank", "reranker", "decoder_reranker"}:
            raise ValueError(
                "SGLang /v1/rerank does not expose a LoRA adapter selector"
            )

    def _load_legacy_adapter_claimed(
        self,
        deployment_id: str,
        adapter_name: str,
        adapter_path: str,
        *,
        source_task_id: str | None,
        source_model_id: str | None,
        user_id: str | None,
        claim: ReplicaOperationClaim,
    ) -> Dict[str, Any]:
        with get_session() as session:
            deployment = self._require_claim_owner(
                session,
                claim,
                lock=False,
            )
            assert deployment is not None
            if self._deployment_lifecycle_service._uses_replica_lifecycle(
                deployment
            ):
                raise ReplicaOperationLostError(
                    "deployment changed to replica lifecycle during adapter load"
                )
            framework = self._validate_legacy_adapter_parent(deployment)
            runtime_deployment = deployment.model_copy(deep=True)
            stale_ids = tuple(
                adapter.adapter_id
                for adapter in session.exec(
                    select(LoadedAdapterDB).where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        LoadedAdapterDB.deployment_replica_id.is_(None),
                        LoadedAdapterDB.adapter_name == adapter_name,
                        LoadedAdapterDB.status.in_(["failed", "loading"]),
                    )
                ).all()
            )
        self._validate_legacy_sglang_adapter_type(
            runtime_deployment,
            framework,
        )

        if stale_ids:
            client = self._get_inference_client(runtime_deployment)
            self._verify_runtime_adapter_absence(
                client,
                claim,
                adapter_name,
            )
            self._finalize_stale_adapter_cleanup(
                claim=claim,
                stale_ids=stale_ids,
            )

        with get_session() as session:
            model_registry_service.lock_model_artifact_references(
                session,
                [adapter_path],
                required_model_ids=(
                    (source_model_id,) if source_model_id else ()
                ),
            )
            deployment = self._require_claim_owner(session, claim)
            assert deployment is not None
            if self._deployment_lifecycle_service._uses_replica_lifecycle(
                deployment
            ):
                raise ReplicaOperationLostError(
                    "deployment changed to replica lifecycle during adapter load"
                )
            self._validate_legacy_adapter_parent(deployment)
            existing = session.exec(
                select(LoadedAdapterDB).where(
                    LoadedAdapterDB.deployment_id == deployment_id,
                    LoadedAdapterDB.deployment_replica_id.is_(None),
                    LoadedAdapterDB.adapter_name == adapter_name,
                    LoadedAdapterDB.status == "loaded",
                )
            ).first()
            if existing is not None:
                return self._adapter_to_dict(existing)
            loaded_count = len(
                list(
                    session.exec(
                        select(LoadedAdapterDB).where(
                            LoadedAdapterDB.deployment_id == deployment_id,
                            LoadedAdapterDB.deployment_replica_id.is_(None),
                            LoadedAdapterDB.status == "loaded",
                        )
                    ).all()
                )
            )
            if loaded_count >= deployment.max_loras:
                raise ValueError(
                    f"Maximum number of adapters ({deployment.max_loras}) reached. "
                    "Unload an adapter first."
                )
            adapter = LoadedAdapterDB(
                deployment_id=deployment_id,
                deployment_replica_id=None,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                source_task_id=source_task_id,
                source_model_id=source_model_id,
                user_id=user_id,
                status="loading",
            )
            session.add(adapter)
            session.commit()
            adapter_id = adapter.adapter_id

        client = self._get_inference_client(runtime_deployment)
        runtime_state, runtime_error = self._load_runtime_with_reconciliation(
            client,
            claim,
            adapter_name,
            adapter_path,
        )
        result = self._finalize_adapter_loading(
            claim=claim,
            adapter_id=adapter_id,
            deployment_id=deployment_id,
            bound_replica_id=None,
            runtime_state=runtime_state,
            runtime_error=runtime_error,
        )
        if runtime_state == "loaded":
            return result
        if runtime_state == "absent":
            raise RuntimeError(f"Failed to load adapter: {runtime_error}") from runtime_error
        raise RuntimeError(
            "Failed to load adapter: runtime state could not be verified"
        ) from runtime_error

    def _unload_legacy_adapter_claimed(
        self,
        deployment_id: str,
        adapter_name: str,
        *,
        user_id: str | None,
        claim: ReplicaOperationClaim,
    ) -> bool:
        with get_session() as session:
            deployment = self._require_claim_owner(session, claim)
            assert deployment is not None
            if self._deployment_lifecycle_service._uses_replica_lifecycle(
                deployment
            ):
                raise ReplicaOperationLostError(
                    "deployment changed to replica lifecycle during adapter unload"
                )
            self._validate_legacy_adapter_parent(deployment)
            runtime_deployment = deployment.model_copy(deep=True)
            adapter = session.exec(
                select(LoadedAdapterDB)
                .where(
                    LoadedAdapterDB.deployment_id == deployment_id,
                    LoadedAdapterDB.deployment_replica_id.is_(None),
                    LoadedAdapterDB.adapter_name == adapter_name,
                    LoadedAdapterDB.status == "loaded",
                )
                .with_for_update()
            ).first()
            adapter_id = None
            if adapter is not None:
                adapter.update_status("unloading")
                session.add(adapter)
                adapter_id = adapter.adapter_id
            session.commit()

        client = self._get_inference_client(runtime_deployment)
        runtime_unload_error: Exception | None = None
        try:
            self._renew_claim_before_runtime(claim)
            client.unload_lora_adapter(adapter_name)
        except ReplicaOperationLostError:
            raise
        except Exception as exc:
            runtime_unload_error = exc
            try:
                self._renew_claim_before_runtime(claim)
                runtime_names = {
                    str(item.get("name") or item.get("id") or "")
                    for item in client.list_lora_adapters()
                    if isinstance(item, dict)
                }
                if adapter_name not in runtime_names:
                    runtime_unload_error = None
            except ReplicaOperationLostError:
                raise
            except Exception:
                pass

        if runtime_unload_error is not None:
            if adapter_id is not None:
                with get_session() as session:
                    self._require_claim_owner(session, claim)
                    adapter = session.exec(
                        select(LoadedAdapterDB)
                        .where(LoadedAdapterDB.adapter_id == adapter_id)
                        .with_for_update()
                    ).first()
                    if adapter is not None and adapter.status == "unloading":
                        adapter.force_status("loaded", str(runtime_unload_error))
                        session.add(adapter)
                    session.commit()
            raise RuntimeError(
                "Failed to verify adapter runtime absence during unload recovery"
            ) from runtime_unload_error

        with get_session() as session:
            self._require_claim_owner(session, claim)
            candidates = list(
                session.exec(
                    select(LoadedAdapterDB)
                    .where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        LoadedAdapterDB.deployment_replica_id.is_(None),
                        LoadedAdapterDB.adapter_name == adapter_name,
                        LoadedAdapterDB.status.in_(
                            ["loading", "failed", "unloading"]
                        ),
                    )
                    .with_for_update()
                ).all()
            )
            for adapter in candidates:
                adapter.force_status("unloaded")
                session.add(adapter)
            session.commit()
        return True

    # ==================== Load/Unload Operations ====================

    def load_adapter(
        self,
        deployment_id: str,
        adapter_name: str,
        adapter_path: str,
        source_task_id: Optional[str] = None,
        source_model_id: Optional[str] = None,
        deployment_replica_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        claim = self._claim_adapter_operation(
            deployment_id,
            operation="load_adapter",
            deployment_replica_id=deployment_replica_id,
            user_id=user_id,
        )
        try:
            return self._load_adapter_claimed(
                deployment_id,
                adapter_name,
                adapter_path,
                source_task_id=source_task_id,
                source_model_id=source_model_id,
                deployment_replica_id=deployment_replica_id,
                user_id=user_id,
                _claim=claim,
            )
        finally:
            if claim is not None:
                self._deployment_lifecycle_service._release_replica_operation(claim)

    def _load_canonical_adapter_claimed(
        self,
        deployment_id: str,
        adapter_name: str,
        adapter_path: str,
        *,
        source_task_id: str | None,
        source_model_id: str | None,
        deployment_replica_id: str | None,
        user_id: str | None,
        claim: ReplicaOperationClaim | None,
    ) -> Dict[str, Any]:
        """Load on a canonical replica without holding a Session across I/O."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            replica = self._resolve_replica_binding(
                session,
                deployment,
                deployment_replica_id,
                user_id=user_id,
                require_healthy=True,
            )
            bound_replica_id = replica.replica_id if replica is not None else None
            endpoint = replica.endpoint if replica is not None else None
            if replica is None and deployment.status != "running":
                raise ValueError(
                    f"Deployment is not running (status: {deployment.status})"
                )
            if not deployment.enable_lora:
                raise ValueError("Deployment does not have LoRA enabled")
            framework = deployment.inference_framework or "xinference"
            if framework == "xinference":
                raise ValueError(
                    "Xinference does not support runtime adapter loading"
                )
            runtime_deployment = deployment.model_copy(deep=True)
            self._require_claim_owner(session, claim, lock=False)
            stale_ids = tuple(
                adapter.adapter_id
                for adapter in session.exec(
                    select(LoadedAdapterDB).where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        self._replica_binding_condition(bound_replica_id),
                        LoadedAdapterDB.adapter_name == adapter_name,
                        LoadedAdapterDB.status.in_(["failed", "loading"]),
                    )
                ).all()
            )

        try:
            model = model_registry_service.get_model(runtime_deployment.model_id)
        except Exception as exc:
            if framework == "sglang":
                raise ValueError(
                    "Unable to verify SGLang deployment model type for adapter loading"
                ) from exc
            logger.warning(
                "Failed to verify deployment model type for adapter loading: %s",
                exc,
            )
            model = None
        model_type = str((model or {}).get("model_type") or "").strip().lower()
        if framework == "sglang" and not model_type:
            raise ValueError(
                "Unable to verify SGLang deployment model type for adapter loading"
            )
        if framework == "sglang" and model_type in {
            "rerank",
            "reranker",
            "decoder_reranker",
        }:
            raise ValueError(
                "SGLang /v1/rerank does not expose a LoRA adapter selector"
            )

        client = self._get_inference_client(
            runtime_deployment,
            endpoint=endpoint,
        )
        if stale_ids:
            self._verify_runtime_adapter_absence(client, claim, adapter_name)
            self._finalize_stale_adapter_cleanup(
                claim=claim,
                stale_ids=stale_ids,
            )

        with get_session() as session:
            model_registry_service.lock_model_artifact_references(
                session,
                [adapter_path],
                required_model_ids=(
                    (source_model_id,) if source_model_id else ()
                ),
            )
            owner = self._require_claim_owner(session, claim)
            if owner is None:
                owner = session.exec(
                    select(DeploymentDB)
                    .where(DeploymentDB.deployment_id == deployment_id)
                    .with_for_update()
                ).first()
            if owner is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            current_replica = self._resolve_replica_binding(
                session,
                owner,
                bound_replica_id,
                user_id=user_id,
                require_healthy=True,
            )
            current_bound_replica_id = (
                current_replica.replica_id if current_replica is not None else None
            )
            if current_bound_replica_id != bound_replica_id:
                raise ReplicaOperationLostError(
                    "deployment replica binding changed during adapter load"
                )
            existing = session.exec(
                select(LoadedAdapterDB)
                .where(
                    LoadedAdapterDB.deployment_id == deployment_id,
                    self._replica_binding_condition(bound_replica_id),
                    LoadedAdapterDB.adapter_name == adapter_name,
                    LoadedAdapterDB.status.in_(["loading", "loaded"]),
                )
                .with_for_update()
            ).first()
            if existing is not None and existing.status == "loaded":
                return self._adapter_to_dict(existing)
            if existing is not None:
                raise ReplicaOperationLostError(
                    "adapter loading state changed during retry preparation"
                )
            loaded_count = list(
                session.exec(
                    select(LoadedAdapterDB).where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        self._replica_binding_condition(bound_replica_id),
                        LoadedAdapterDB.status == "loaded",
                    )
                ).all()
            )
            if len(loaded_count) >= owner.max_loras:
                raise ValueError(
                    f"Maximum number of adapters ({owner.max_loras}) reached. "
                    "Unload an adapter first."
                )
            adapter = LoadedAdapterDB(
                deployment_id=deployment_id,
                deployment_replica_id=bound_replica_id,
                adapter_name=adapter_name,
                adapter_path=adapter_path,
                source_task_id=source_task_id,
                source_model_id=source_model_id,
                user_id=user_id,
                status="loading",
            )
            session.add(adapter)
            session.commit()
            adapter_id = adapter.adapter_id

        runtime_state, runtime_error = self._load_runtime_with_reconciliation(
            client,
            claim,
            adapter_name,
            adapter_path,
        )
        result = self._finalize_adapter_loading(
            claim=claim,
            adapter_id=adapter_id,
            deployment_id=deployment_id,
            bound_replica_id=bound_replica_id,
            runtime_state=runtime_state,
            runtime_error=runtime_error,
        )
        if runtime_state == "loaded":
            return result
        if runtime_state == "absent":
            raise RuntimeError(
                f"Failed to load adapter: {runtime_error}"
            ) from runtime_error
        raise RuntimeError(
            "Failed to load adapter: runtime state could not be verified"
        ) from runtime_error

    def _load_adapter_claimed(
        self,
        deployment_id: str,
        adapter_name: str,
        adapter_path: str,
        source_task_id: Optional[str] = None,
        source_model_id: Optional[str] = None,
        deployment_replica_id: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        _claim: ReplicaOperationClaim | None,
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
        if (
            _claim is not None
            and deployment_replica_id is None
            and self._claim_targets_legacy(_claim)
        ):
            return self._load_legacy_adapter_claimed(
                deployment_id,
                adapter_name,
                adapter_path,
                source_task_id=source_task_id,
                source_model_id=source_model_id,
                user_id=user_id,
                claim=_claim,
            )
        return self._load_canonical_adapter_claimed(
            deployment_id,
            adapter_name,
            adapter_path,
            source_task_id=source_task_id,
            source_model_id=source_model_id,
            deployment_replica_id=deployment_replica_id,
            user_id=user_id,
            claim=_claim,
        )

    def unload_adapter(
        self,
        deployment_id: str,
        adapter_name: str,
        deployment_replica_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        claim = self._claim_adapter_operation(
            deployment_id,
            operation="unload_adapter",
            deployment_replica_id=deployment_replica_id,
            user_id=user_id,
        )
        try:
            return self._unload_adapter_claimed(
                deployment_id,
                adapter_name,
                deployment_replica_id=deployment_replica_id,
                user_id=user_id,
                _claim=claim,
            )
        finally:
            if claim is not None:
                self._deployment_lifecycle_service._release_replica_operation(claim)

    def _unload_canonical_adapter_claimed(
        self,
        deployment_id: str,
        adapter_name: str,
        *,
        deployment_replica_id: str | None,
        user_id: str | None,
        claim: ReplicaOperationClaim | None,
    ) -> bool:
        """Unload on a canonical replica without holding a Session across I/O."""
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise ValueError(f"Deployment not found: {deployment_id}")
            replica = self._resolve_replica_binding(
                session,
                deployment,
                deployment_replica_id,
                user_id=user_id,
                require_healthy=True,
            )
            bound_replica_id = replica.replica_id if replica is not None else None
            endpoint = replica.endpoint if replica is not None else None
            runtime_deployment = deployment.model_copy(deep=True)
            self._require_claim_owner(session, claim)
            candidates = list(
                session.exec(
                    select(LoadedAdapterDB)
                    .where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        self._replica_binding_condition(bound_replica_id),
                        LoadedAdapterDB.adapter_name == adapter_name,
                        LoadedAdapterDB.status.in_(
                            ["loading", "loaded", "failed", "unloading"]
                        ),
                    )
                    .order_by(LoadedAdapterDB.adapter_id)
                    .with_for_update()
                ).all()
            )
            candidate_ids = tuple(adapter.adapter_id for adapter in candidates)
            loaded = next(
                (adapter for adapter in candidates if adapter.status == "loaded"),
                None,
            )
            loaded_id = loaded.adapter_id if loaded is not None else None
            if loaded is not None:
                loaded.update_status("unloading")
                session.add(loaded)
            session.commit()

        client = self._get_inference_client(
            runtime_deployment,
            endpoint=endpoint,
        )
        try:
            self._verify_runtime_adapter_absence(client, claim, adapter_name)
        except ReplicaOperationLostError:
            raise
        except Exception as runtime_error:
            if loaded_id is not None:
                with get_session() as session:
                    self._require_claim_owner(session, claim)
                    current = session.exec(
                        select(LoadedAdapterDB)
                        .where(
                            LoadedAdapterDB.adapter_id == loaded_id,
                            LoadedAdapterDB.deployment_id == deployment_id,
                            self._replica_binding_condition(bound_replica_id),
                            LoadedAdapterDB.status == "unloading",
                        )
                        .with_for_update()
                    ).first()
                    if current is None:
                        raise ReplicaOperationLostError(
                            "adapter unload plan changed during recovery"
                        ) from runtime_error
                    current.force_status("loaded", str(runtime_error))
                    session.add(current)
                    session.commit()
            raise RuntimeError(
                "Failed to verify adapter runtime absence during unload recovery"
            ) from runtime_error

        if candidate_ids:
            with get_session() as session:
                self._require_claim_owner(session, claim)
                current_candidates = list(
                    session.exec(
                        select(LoadedAdapterDB)
                        .where(LoadedAdapterDB.adapter_id.in_(candidate_ids))
                        .order_by(LoadedAdapterDB.adapter_id)
                        .with_for_update()
                    ).all()
                )
                if {
                    adapter.adapter_id for adapter in current_candidates
                } != set(candidate_ids):
                    raise ReplicaOperationLostError(
                        "adapter unload plan changed before finalization"
                    )
                for adapter in current_candidates:
                    if (
                        adapter.deployment_id != deployment_id
                        or adapter.deployment_replica_id != bound_replica_id
                        or adapter.adapter_name != adapter_name
                        or adapter.status
                        not in {"loading", "loaded", "failed", "unloading"}
                    ):
                        raise ReplicaOperationLostError(
                            "adapter unload plan changed before finalization"
                        )
                    adapter.force_status("unloaded")
                    session.add(adapter)
                session.commit()
        return True

    def _unload_adapter_claimed(
        self,
        deployment_id: str,
        adapter_name: str,
        deployment_replica_id: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        _claim: ReplicaOperationClaim | None,
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
        if (
            _claim is not None
            and deployment_replica_id is None
            and self._claim_targets_legacy(_claim)
        ):
            return self._unload_legacy_adapter_claimed(
                deployment_id,
                adapter_name,
                user_id=user_id,
                claim=_claim,
            )
        return self._unload_canonical_adapter_claimed(
            deployment_id,
            adapter_name,
            deployment_replica_id=deployment_replica_id,
            user_id=user_id,
            claim=_claim,
        )

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
        deployment_replica_id: str | None = None,
        user_id: str | None = None,
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
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                raise ValueError("deployment not found")
            replica = self._resolve_replica_binding(
                session,
                deployment,
                deployment_replica_id,
                user_id=user_id,
                require_healthy=False,
            )
            bound_replica_id = replica.replica_id if replica is not None else None
            query = select(LoadedAdapterDB).where(
                LoadedAdapterDB.deployment_id == deployment_id,
                self._replica_binding_condition(bound_replica_id),
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

    def sync_loaded_adapters(
        self,
        deployment_id: str,
        deployment_replica_id: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        best_effort: bool = False,
    ) -> int:
        """
        Sync loaded adapters with actual state from inference service.

        Args:
            deployment_id: Deployment ID

        Returns:
            Number of adapters updated
        """
        with get_session() as session:
            deployment = session.exec(
                select(DeploymentDB).where(
                    DeploymentDB.deployment_id == deployment_id
                )
            ).first()
            if deployment is None:
                if best_effort:
                    return 0
                raise DeploymentReplicaNotFoundError("deployment not found")
            if not self._sync_parent_status_is_eligible(deployment):
                return 0
            expected_replica_lifecycle = (
                self._deployment_lifecycle_service._uses_replica_lifecycle(
                    deployment
                )
            )

        try:
            claim = self._claim_adapter_operation(
                deployment_id,
                operation="load_adapter",
                deployment_replica_id=deployment_replica_id,
                user_id=user_id,
            )
        except (DeploymentReplicaNotFoundError, ReplicaOperationBusyError) as exc:
            if not best_effort:
                raise
            logger.warning(
                "Skipped adapter sync claim for deployment %s: %s",
                deployment_id,
                exc,
            )
            return 0
        try:
            return self._sync_loaded_adapters_claimed(
                deployment_id,
                deployment_replica_id=deployment_replica_id,
                user_id=user_id,
                _claim=claim,
                expected_replica_lifecycle=expected_replica_lifecycle,
                best_effort=best_effort,
            )
        finally:
            if claim is not None:
                self._deployment_lifecycle_service._release_replica_operation(claim)

    def _sync_loaded_adapters_short_transactions(
        self,
        deployment_id: str,
        deployment_replica_id: str | None,
        user_id: str | None,
        *,
        claim: ReplicaOperationClaim | None,
        expected_replica_lifecycle: bool,
        best_effort: bool,
    ) -> int:
        """Reconcile a runtime snapshot without holding a Session during I/O."""
        preflight_complete = False
        try:
            with get_session() as session:
                deployment = self._require_claim_owner(
                    session,
                    claim,
                    lock=False,
                )
                if deployment is None:
                    raise ReplicaOperationLostError(
                        "deployment replica operation ownership was lost"
                    )
                if not self._sync_parent_status_is_eligible(deployment):
                    session.rollback()
                    raise ReplicaOperationLostError(
                        "deployment was deleted or changed during adapter sync"
                    )
                if (
                    self._deployment_lifecycle_service._uses_replica_lifecycle(
                        deployment
                    )
                    != expected_replica_lifecycle
                ):
                    session.rollback()
                    raise ReplicaOperationLostError(
                        "deployment replica lifecycle changed during adapter sync"
                    )
                replica = self._resolve_replica_binding(
                    session,
                    deployment,
                    deployment_replica_id,
                    user_id=user_id,
                    require_healthy=True,
                )
                bound_replica_id = (
                    replica.replica_id if replica is not None else None
                )
                endpoint = replica.endpoint if replica is not None else None
                if (deployment.inference_framework or "xinference") == "xinference":
                    return 0
                runtime_deployment = deployment.model_copy(deep=True)
                self._require_claim_owner(session, claim, lock=False)
            preflight_complete = True

            client = self._get_inference_client(
                runtime_deployment,
                endpoint=endpoint,
            )
            self._renew_claim_before_runtime(claim)
            actual_adapters = client.list_lora_adapters()
            if not isinstance(actual_adapters, list):
                raise RuntimeError("adapter runtime list returned an invalid response")
            actual_names = {
                self._runtime_adapter_name(adapter)
                for adapter in actual_adapters
                if self._runtime_adapter_name(adapter)
            }

            with get_session() as session:
                adapter_query = (
                    select(LoadedAdapterDB)
                    .where(
                        LoadedAdapterDB.deployment_id == deployment_id,
                        self._replica_binding_condition(bound_replica_id),
                    )
                    .order_by(
                        LoadedAdapterDB.loaded_at.desc(),
                        LoadedAdapterDB.adapter_id.desc(),
                    )
                )
                snapshot_adapters = list(session.exec(adapter_query).all())
                snapshot_signature = tuple(
                    sorted(
                        (
                            adapter.adapter_id,
                            adapter.adapter_name,
                            adapter.adapter_path,
                            adapter.source_model_id,
                            adapter.status,
                        )
                        for adapter in snapshot_adapters
                    )
                )
                snapshot_loaded_names = {
                    adapter.adapter_name
                    for adapter in snapshot_adapters
                    if adapter.status == "loaded"
                }
                snapshot_restore = []
                for name in actual_names - snapshot_loaded_names:
                    previous = next(
                        (
                            adapter
                            for adapter in snapshot_adapters
                            if adapter.adapter_name == name
                        ),
                        None,
                    )
                    if previous is not None and previous.status in {
                        "unloaded",
                        "failed",
                    }:
                        snapshot_restore.append(previous)
                snapshot_names_to_create = {
                    name
                    for name in actual_names - snapshot_loaded_names
                    if not any(
                        adapter.adapter_name == name
                        for adapter in snapshot_adapters
                    )
                }
                unknown_membership_change = bool(snapshot_names_to_create) or any(
                    adapter.adapter_path == "<unknown:auto-sync>"
                    for adapter in snapshot_restore
                )
                if unknown_membership_change:
                    self._lock_unknown_adapter_membership(session)
                if snapshot_restore:
                    model_registry_service.lock_model_artifact_references(
                        session,
                        [adapter.adapter_path for adapter in snapshot_restore],
                        required_model_ids=tuple(
                            adapter.source_model_id
                            for adapter in snapshot_restore
                            if adapter.source_model_id
                        ),
                        membership_gate_locked=unknown_membership_change,
                    )

                self._require_sync_parent_owner(
                    session,
                    deployment_id,
                    claim=claim,
                    deployment_replica_id=bound_replica_id,
                    user_id=user_id,
                    expected_replica_lifecycle=expected_replica_lifecycle,
                )
                locked_adapters = list(
                    session.exec(adapter_query.with_for_update()).all()
                )
                locked_signature = tuple(
                    sorted(
                        (
                            adapter.adapter_id,
                            adapter.adapter_name,
                            adapter.adapter_path,
                            adapter.source_model_id,
                            adapter.status,
                        )
                        for adapter in locked_adapters
                    )
                )
                if locked_signature != snapshot_signature:
                    raise ReplicaOperationLostError(
                        "adapter records changed during runtime synchronization"
                    )

                loaded_adapters = [
                    adapter
                    for adapter in locked_adapters
                    if adapter.status == "loaded"
                ]
                db_loaded_names = {
                    adapter.adapter_name for adapter in loaded_adapters
                }
                adapters_to_unload = [
                    adapter
                    for adapter in loaded_adapters
                    if adapter.adapter_name not in actual_names
                ]
                adapters_to_restore: list[LoadedAdapterDB] = []
                adapter_names_to_create: list[str] = []
                for name in actual_names - db_loaded_names:
                    previous = next(
                        (
                            adapter
                            for adapter in locked_adapters
                            if adapter.adapter_name == name
                        ),
                        None,
                    )
                    if previous is not None and previous.status in {
                        "unloaded",
                        "failed",
                    }:
                        adapters_to_restore.append(previous)
                    elif previous is None:
                        adapter_names_to_create.append(name)

                for adapter in adapters_to_unload:
                    adapter.force_status(
                        "unloaded",
                        "Adapter not found on service (auto-sync)",
                    )
                    session.add(adapter)
                for adapter in adapters_to_restore:
                    adapter.force_status("loaded")
                    adapter.unloaded_at = None
                    session.add(adapter)
                for name in adapter_names_to_create:
                    session.add(
                        LoadedAdapterDB(
                            deployment_id=deployment_id,
                            deployment_replica_id=bound_replica_id,
                            adapter_name=name,
                            adapter_path="<unknown:auto-sync>",
                            status="loaded",
                            user_id=user_id,
                        )
                    )
                session.commit()
                return (
                    len(adapters_to_unload)
                    + len(adapters_to_restore)
                    + len(adapter_names_to_create)
                )
        except Exception as exc:
            if not preflight_complete and isinstance(exc, ValueError):
                raise
            if not best_effort:
                raise
            logger.warning(
                "Failed to sync adapters for deployment %s: %s",
                deployment_id,
                exc,
            )
            return 0

    def _sync_loaded_adapters_claimed(
        self,
        deployment_id: str,
        deployment_replica_id: Optional[str],
        user_id: Optional[str],
        *,
        _claim: ReplicaOperationClaim | None,
        expected_replica_lifecycle: bool,
        best_effort: bool,
    ) -> int:
        return self._sync_loaded_adapters_short_transactions(
            deployment_id,
            deployment_replica_id,
            user_id,
            claim=_claim,
            expected_replica_lifecycle=expected_replica_lifecycle,
            best_effort=best_effort,
        )

# Global service instance
adapter_service = AdapterService()
