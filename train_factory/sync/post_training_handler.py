"""
Post-training handler: triggered after training task completes.

Manages LoRA adapter lifecycle:
1. Auto-discover compatible deployment (or use explicit base_deployment_id)
2. Unload old adapter (if any)
3. Load new adapter onto deployment
4. Update sync task with current adapter info

Also provides `load_adapter_for_training()` for manual retry from the API.
"""

import logging
import threading
from typing import Any, Dict, Optional

from ..enums.sync_status import SyncStatus, SyncTrainingStatus

logger = logging.getLogger(__name__)

# Per-config lock to prevent concurrent adapter load/unload operations
# on the same sync task (e.g. auto callback vs manual retry).
_config_locks: Dict[str, threading.Lock] = {}
_config_locks_guard = threading.Lock()


def _get_config_lock(config_id: str) -> threading.Lock:
    """Get or create a per-config threading lock."""
    with _config_locks_guard:
        if config_id not in _config_locks:
            _config_locks[config_id] = threading.Lock()
        return _config_locks[config_id]


def _find_compatible_deployment(
    base_model_path: str,
    user_id: Optional[str] = None,
    external_api_config_id: Optional[str] = None,
) -> Optional[str]:
    """Auto-discover a compatible deployment for adapter hot-loading.

    Finds a running deployment that:
    1. Serves the same base model (by model_path → model_id lookup)
    2. Has enable_lora=True
    3. Uses a framework that supports hot-loading (sglang or vllm)
    4. (If external_api_config_id given) belongs to the same external API config

    Returns deployment_id if found, None otherwise.
    """
    from ..storage.services.model_registry_service import model_registry_service
    from ..deployment.deployment_service import deployment_service

    # Step 1: Resolve base_model_path → registered model_id
    model = model_registry_service.get_model_by_path(base_model_path)
    if not model:
        logger.debug(f"No registered model found for path: {base_model_path}")
        return None

    model_id = model["model_id"]

    # Step 2: Find running deployments for this model
    deployments, _ = deployment_service.list_deployments(
        model_id=model_id,
        status="running",
        user_id=user_id,
    )

    # Step 3: Filter for LoRA-enabled + compatible framework + matching api config
    for dep in deployments:
        if external_api_config_id:
            dep_api_config = _get_deployment_external_api_config_id(dep)
            # Tenant isolation: when sync task is bound to an external API config,
            # deployment must be explicitly bound to the same config.
            if dep_api_config != external_api_config_id:
                continue
        if dep.get("enable_lora") and dep.get("inference_framework") in ("sglang", "vllm"):
            logger.info(
                f"Auto-discovered compatible deployment: {dep['deployment_id'][:8]} "
                f"(framework={dep['inference_framework']}, model={model_id[:8]})"
            )
            return dep["deployment_id"]

    logger.debug(f"No compatible deployment found for model {model_id[:8]}")
    return None


def _get_deployment_external_api_config_id(dep: Dict[str, Any]) -> str:
    """Read deployment external_api_config_id with backward compatibility."""
    column_value = (dep.get("external_api_config_id") or "").strip()
    if column_value:
        return column_value
    config = dep.get("config") or {}
    raw = config.get("external_api_config_id")
    return raw.strip() if isinstance(raw, str) else ""


def _resolve_deployment_id(config: Dict[str, Any], tag: str) -> Optional[str]:
    """Resolve deployment ID from sync task config."""
    from ..deployment.deployment_service import deployment_service

    user_id = config.get("user_id")
    external_api_config_id = (config.get("external_api_config_id") or "").strip()

    deployment_id = config.get("base_deployment_id")
    if deployment_id:
        dep = deployment_service.get_deployment(deployment_id)
        if not dep:
            logger.warning(f"[sync:{tag}] base_deployment_id not found: {deployment_id}")
            deployment_id = None
        elif user_id and dep.get("user_id") != user_id:
            logger.warning(
                f"[sync:{tag}] base_deployment_id belongs to another user, skip adapter loading"
            )
            deployment_id = None
        elif external_api_config_id:
            dep_api_config = _get_deployment_external_api_config_id(dep)
            if dep_api_config != external_api_config_id:
                logger.warning(
                    f"[sync:{tag}] base_deployment_id api_config mismatch, skip adapter loading "
                    f"(deployment={dep_api_config or '<empty>'}, task={external_api_config_id})"
                )
                deployment_id = None

    if not deployment_id:
        base_model_path = (config.get("training_config") or {}).get("base_model_path")
        if base_model_path:
            deployment_id = _find_compatible_deployment(
                base_model_path,
                user_id=user_id,
                external_api_config_id=external_api_config_id or None,
            )

    return deployment_id


def _resolve_target_deployment_id(
    config: Dict[str, Any],
    target: Dict[str, Any],
    tag: str,
) -> Optional[str]:
    """Resolve a target deployment without crossing task or tenant boundaries."""
    from ..deployment.deployment_service import deployment_service

    config_id = config.get("task_id")
    if target.get("task_id") != config_id:
        raise PermissionError("Training target belongs to another sync task")

    deployment_id = target.get("base_deployment_id")
    if not deployment_id:
        return _resolve_deployment_id(config, tag)

    deployment = deployment_service.get_deployment(deployment_id)
    if not deployment:
        raise ValueError(f"Deployment not found: {deployment_id}")

    user_id = config.get("user_id")
    if user_id and deployment.get("user_id") != user_id:
        raise PermissionError(f"Deployment {deployment_id} belongs to another user")

    expected_api_config = (config.get("external_api_config_id") or "").strip()
    if expected_api_config:
        deployment_api_config = _get_deployment_external_api_config_id(deployment)
        if deployment_api_config != expected_api_config:
            raise PermissionError(
                f"Deployment {deployment_id} belongs to another API tenant"
            )

    return deployment_id


def load_adapter_for_training(
    config_id: str,
    training_task_id: str,
    final_model_path: str,
    model_registry_id: Optional[str] = None,
    replace: bool = True,
    target_id: Optional[str] = None,
):
    """Load adapter for a completed sync training.

    Shared by both the automatic post-training callback and manual retry.

    Args:
        replace: If True, unload all existing adapters before loading.
                 If False, load alongside existing adapters.
    """
    from ..storage.services.external_sync_service import external_sync_service
    from ..deployment.adapter_service import AdapterService

    tag = config_id[:8]
    lock = _get_config_lock(config_id)

    if not lock.acquire(timeout=5):
        raise RuntimeError(
            f"[sync:{tag}] Another adapter operation is in progress, please retry later"
        )

    try:
        # Reload latest config/training after acquiring lock to avoid stale snapshots
        # when auto-callback and manual retry run concurrently.
        config = external_sync_service.get_task_raw(config_id)
        if not config:
            raise ValueError(f"Sync task not found: {config_id}")

        sync_training = external_sync_service.get_training_by_task_id(training_task_id)
        if not sync_training or sync_training.get("task_id") != config_id:
            raise ValueError(f"Sync training not found for task: {training_task_id}")
        target_id = target_id or sync_training.get("target_id")

        external_sync_service.update_task(config_id, status=SyncStatus.LOADING_ADAPTER)
        if target_id:
            from ..storage.services.external_sync_service import external_sync_service as _ess
            _target = _ess.get_training_target_raw(target_id)
            if not _target:
                raise ValueError(f"Training target not found: {target_id}")
            deployment_id = _resolve_target_deployment_id(config, _target, tag)
        else:
            deployment_id = _resolve_deployment_id(config, tag)
        if not deployment_id:
            logger.info(f"[sync:{tag}] No compatible deployment found, skipping adapter loading")
            external_sync_service.update_training_status(
                training_task_id, SyncTrainingStatus.COMPLETED,
                output_adapter_path=final_model_path,
                output_model_registry_id=model_registry_id,
            )
            external_sync_service.update_task(config_id, status=SyncStatus.IDLE)
            return

        adapter_service = AdapterService()

        # 1. Sync DB with actual vLLM state
        try:
            adapter_service.sync_loaded_adapters(deployment_id)
        except Exception as e:
            logger.warning(f"[sync:{tag}] Failed to sync adapters: {e}")

        if replace:
            # Unload ALL loaded adapters before loading new one
            loaded_adapters = adapter_service.list_loaded_adapters(deployment_id)
            failed_unloads = []
            for loaded in loaded_adapters:
                adapter_name_to_unload = loaded.get("adapter_name")
                try:
                    adapter_service.unload_adapter(deployment_id, adapter_name_to_unload)
                    logger.info(f"[sync:{tag}] Unloaded adapter: {adapter_name_to_unload}")
                except Exception as e:
                    logger.warning(f"[sync:{tag}] Failed to unload adapter '{adapter_name_to_unload}': {e}")
                    failed_unloads.append(adapter_name_to_unload or "<unknown>")

            # Keep DB/runtime state aligned: if any unload failed, abort replace flow.
            if failed_unloads:
                failed_text = ", ".join(failed_unloads)
                raise RuntimeError(
                    f"Failed to unload adapter(s): {failed_text}. "
                    "Please retry after inference service recovers."
                )

            # Clear sync task adapter reference
            if config.get("current_adapter_name"):
                external_sync_service.update_task(
                    config_id,
                    current_adapter_name=None,
                    current_adapter_id=None,
                )
            # Mark ALL adapter_loaded training records as adapter_unloaded
            # （排除本次要加载的训练，避免 replace 流程与并发加载互相覆盖）
            external_sync_service.mark_all_trainings_adapter_unloaded(
                config_id,
                exclude_training_task_id=training_task_id,
            )

        # 2. Load new adapter
        # sync-{config_id[:8]}-{training_task_id[:8]}: traceable to both sync task and training
        if target_id:
            from ..storage.services.external_sync_service import external_sync_service as _ess2
            _tgt = _ess2.get_training_target_raw(target_id)
            target_tag = _tgt["model_type"][:4] if _tgt else "unkn"
            adapter_name = f"sync-{config_id[:8]}-{target_tag}-{training_task_id[:8]}"
        else:
            adapter_name = f"sync-{config_id[:8]}-{training_task_id[:8]}"
        user_id = config.get("user_id") or ""

        result = adapter_service.load_adapter(
            deployment_id=deployment_id,
            adapter_name=adapter_name,
            adapter_path=final_model_path,
            source_task_id=training_task_id,
            user_id=user_id,
        )

        logger.info(f"[sync:{tag}] Loaded new adapter: {adapter_name} (id={result.get('adapter_id', '')[:8]})")

        # 3. Update sync task
        latest_config = external_sync_service.get_task_raw(config_id) or config
        current_total_trainings = int(latest_config.get("total_trainings", 0) or 0)
        training_round = int(sync_training.get("training_round", 0) or 0)
        # Adapter reload/retry for the same training should not increase the round counter.
        target_total_trainings = max(current_total_trainings, training_round)
        external_sync_service.update_task(
            config_id,
            current_adapter_name=adapter_name,
            current_adapter_id=result.get("adapter_id"),
            current_training_id=training_task_id,
            total_trainings=target_total_trainings,
            status=SyncStatus.IDLE,
        )

        if target_id:
            from ..storage.services.external_sync_service import external_sync_service as _ess3
            _ess3.update_training_target(
                target_id,
                current_adapter_name=adapter_name,
                current_adapter_id=result.get("adapter_id"),
            )

        # 4. Update training record
        external_sync_service.update_training_status(
            training_task_id,
            SyncTrainingStatus.ADAPTER_LOADED,
            output_adapter_path=final_model_path,
            output_model_registry_id=model_registry_id,
            loaded_adapter_name=adapter_name,
            loaded_adapter_id=result.get("adapter_id"),
        )

    except Exception as e:
        logger.exception(f"[sync:{tag}] Adapter loading failed")
        external_sync_service.update_training_status(
            training_task_id, SyncTrainingStatus.ADAPTER_LOAD_FAILED,
            output_adapter_path=final_model_path,
            output_model_registry_id=model_registry_id,
        )
        external_sync_service.update_task(
            config_id, status=SyncStatus.ERROR, error_message=f"Adapter loading failed: {e}"
        )
        raise
    finally:
        lock.release()


def on_training_completed(
    training_task_id: str,
    final_model_path: str,
    model_registry_id: Optional[str] = None,
):
    """Called after a training task completes successfully.

    Checks if this training was triggered by the sync system,
    and if so, manages adapter loading.
    """
    from ..storage.services.external_sync_service import external_sync_service

    # Check if this training is tracked by sync
    sync_training = external_sync_service.get_training_by_task_id(training_task_id)
    if not sync_training:
        return  # Not a sync-triggered training, ignore

    config_id = sync_training["task_id"]
    tag = config_id[:8]

    logger.info(
        f"[sync:{tag}] Training completed: task={training_task_id[:8]}, "
        f"model_path={final_model_path}"
    )

    try:
        completion = external_sync_service.complete_training_claim(
            training_task_id
        )
        if not completion.get("completed") and not completion.get(
            "already_completed"
        ):
            logger.warning(
                "[sync:%s] Training claim could not be completed: %s",
                tag,
                training_task_id[:8],
            )
            return completion
        if completion.get("deletion_pending"):
            logger.info(
                "[sync:%s] Skipping adapter load because task deletion is pending",
                tag,
            )
            return completion
        target_id = sync_training.get("target_id")

        load_adapter_for_training(
            config_id=config_id,
            training_task_id=training_task_id,
            final_model_path=final_model_path,
            model_registry_id=model_registry_id,
            replace=True,
            target_id=target_id,
        )

        if target_id:
            from ..storage.services.external_sync_service import external_sync_service as _ess4
            from ..enums.sync_status import TrainingTargetStatus
            target = _ess4.get_training_target_raw(target_id)
            if target:
                _ess4.update_training_target(
                    target_id,
                    status=TrainingTargetStatus.IDLE,
                    total_trainings=target["total_trainings"] + 1,
                )
            from .level2_handler import _schedule_next_training
            _schedule_next_training(config_id)
    except Exception:
        # Already logged and status updated inside load_adapter_for_training
        logger.debug(f"[sync:{tag}] Adapter loading failed (details logged above)")


def on_training_failed(training_task_id: str, error: str = ""):
    """Restore a sync training claim after any terminal training failure."""
    from ..storage.services.external_sync_service import external_sync_service

    reason = error or "Training failed"
    recovery = external_sync_service.fail_training_and_restore_claim(
        training_task_id,
        reason,
    )
    if recovery.get("recovered"):
        logger.warning(
            "Sync training %s failed (%s); restored %s samples",
            training_task_id[:8],
            reason[:200],
            recovery.get("restored_sample_count", 0),
        )
    return recovery


def unload_current_adapter(config_id: str):
    """Unload all adapters from the sync task's deployment."""
    from ..storage.services.external_sync_service import external_sync_service
    from ..deployment.adapter_service import AdapterService

    config = external_sync_service.get_task_raw(config_id)
    if not config:
        raise ValueError(f"Sync task not found: {config_id}")

    tag = config_id[:8]
    lock = _get_config_lock(config_id)

    if not lock.acquire(timeout=5):
        raise RuntimeError(
            f"[sync:{tag}] Another adapter operation is in progress, please retry later"
        )

    try:
        deployment_id = _resolve_deployment_id(config, tag)
        if not deployment_id:
            raise ValueError("No compatible deployment found")

        adapter_service = AdapterService()

        # Sync DB with vLLM state first, then unload ALL loaded adapters
        try:
            adapter_service.sync_loaded_adapters(deployment_id)
        except Exception as e:
            logger.warning(f"[sync:{tag}] Failed to sync adapters: {e}")

        loaded_adapters = adapter_service.list_loaded_adapters(deployment_id)
        if not loaded_adapters and not config.get("current_adapter_name"):
            raise ValueError("No adapter currently loaded")

        failed_unloads = []
        for loaded in loaded_adapters:
            name = loaded.get("adapter_name")
            try:
                adapter_service.unload_adapter(deployment_id, name)
                logger.info(f"[sync:{tag}] Unloaded adapter: {name}")
            except Exception as e:
                logger.warning(f"[sync:{tag}] Failed to unload adapter '{name}': {e}")
                failed_unloads.append(name or "<unknown>")

        # Do not clear DB state unless all unload operations succeed,
        # otherwise service/runtime and DB state may diverge.
        if failed_unloads:
            failed_text = ", ".join(failed_unloads)
            raise RuntimeError(
                f"Failed to unload adapter(s): {failed_text}. "
                "Please retry after inference service recovers."
            )

        # Clear adapter reference
        external_sync_service.update_task(
            config_id,
            current_adapter_name=None,
            current_adapter_id=None,
        )

        # Mark ALL adapter_loaded training records as adapter_unloaded
        external_sync_service.mark_all_trainings_adapter_unloaded(config_id)
    finally:
        lock.release()
