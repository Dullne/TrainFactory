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
    has_explicit_deployment = bool(deployment_id)
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

    if not deployment_id and not has_explicit_deployment:
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


def _resolve_target_deployment_binding(
    config: Dict[str, Any],
    target: Dict[str, Any],
    tag: str,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve deployment and replica as one indivisible binding pair."""
    deployment_id = _resolve_target_deployment_id(config, target, tag)
    if target.get("base_deployment_id"):
        return deployment_id, target.get("base_deployment_replica_id")
    return deployment_id, config.get("base_deployment_replica_id")


def _load_adapter_targets(config_id: str) -> list[Dict[str, Any]]:
    from ..storage.services.external_sync_service import external_sync_service

    return external_sync_service.list_training_targets_raw(
        config_id,
        is_active=None,
    )


def _adapter_reference_matches(
    owner: Dict[str, Any],
    adapter_id: Optional[str],
    adapter_name: Optional[str],
) -> bool:
    owner_adapter_id = owner.get("current_adapter_id")
    if adapter_id and owner_adapter_id:
        return owner_adapter_id == adapter_id
    return bool(
        adapter_name
        and owner.get("current_adapter_name") == adapter_name
    )


def _has_other_adapter_reference(
    targets: list[Dict[str, Any]],
    selected_target: Optional[Dict[str, Any]],
    adapter_id: Optional[str],
    adapter_name: Optional[str],
) -> bool:
    selected_target_id = (
        selected_target.get("target_id") if selected_target else None
    )
    return any(
        target.get("target_id") != selected_target_id
        and _adapter_reference_matches(target, adapter_id, adapter_name)
        for target in targets
    )


def _validate_adapter_training_owner(
    config: Dict[str, Any],
    selected_target: Optional[Dict[str, Any]],
    training: Dict[str, Any],
) -> None:
    """Require adapter source metadata to match the exact task and target."""
    expected_task_id = config.get("task_id")
    expected_target_id = (
        selected_target.get("target_id") if selected_target else None
    )
    if training.get("task_id") != expected_task_id:
        raise ValueError("Loaded adapter training ownership mismatch")
    if training.get("target_id") != expected_target_id:
        raise ValueError("Loaded adapter training target ownership mismatch")

    snapshot = training.get("target_config_snapshot") or {}
    if (
        "task_id" in snapshot
        and snapshot.get("task_id") != expected_task_id
    ):
        raise ValueError("Loaded adapter snapshot task ownership mismatch")
    if (
        "target_id" in snapshot
        and snapshot.get("target_id") != expected_target_id
    ):
        raise ValueError("Loaded adapter snapshot target ownership mismatch")


def _get_persisted_adapter_binding(
    adapter_service: Any,
    config: Dict[str, Any],
    selected_target: Optional[Dict[str, Any]],
) -> Optional[tuple[str, Optional[str]]]:
    """Read an owner's frozen runtime binding from adapter/training metadata."""
    from ..storage.services.external_sync_service import external_sync_service

    owner = selected_target or config
    adapter_id = owner.get("current_adapter_id")
    if not adapter_id:
        return None
    adapter = adapter_service.get_adapter(adapter_id)
    if not adapter:
        raise ValueError("Loaded adapter metadata not found")
    if owner.get("current_adapter_name") and adapter.get(
        "adapter_name"
    ) != owner.get("current_adapter_name"):
        raise ValueError("Loaded adapter name binding mismatch")
    if adapter.get("user_id") != config.get("user_id"):
        raise ValueError("Loaded adapter user binding mismatch")

    source_training_id = adapter.get("source_task_id")
    training = (
        external_sync_service.get_training_by_task_id(source_training_id)
        if source_training_id
        else None
    )
    if not training:
        raise ValueError("Loaded adapter training ownership mismatch")
    _validate_adapter_training_owner(config, selected_target, training)
    snapshot = training.get("target_config_snapshot") or {}
    if {
        "base_deployment_id",
        "base_deployment_replica_id",
    }.issubset(snapshot) and snapshot.get("base_deployment_id"):
        return (
            snapshot["base_deployment_id"],
            snapshot.get("base_deployment_replica_id"),
        )
    deployment_id = adapter.get("deployment_id")
    if not deployment_id:
        raise ValueError("Loaded adapter deployment binding is missing")
    return deployment_id, adapter.get("deployment_replica_id")


def _resolve_owned_runtime_adapter(
    *,
    adapter_service: Any,
    config: Dict[str, Any],
    selected_target: Optional[Dict[str, Any]],
    deployment_id: str,
    deployment_replica_id: Optional[str],
    loaded_adapters: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve and validate exactly one sync-owned adapter, or fail closed."""
    from ..storage.services.external_sync_service import external_sync_service

    owner = selected_target or config
    expected_adapter_id = owner.get("current_adapter_id")
    expected_adapter_name = owner.get("current_adapter_name")
    if not expected_adapter_id and not expected_adapter_name:
        raise ValueError("No adapter currently loaded")

    runtime_matches = [
        adapter
        for adapter in loaded_adapters
        if (
            adapter.get("adapter_id") == expected_adapter_id
            if expected_adapter_id
            else adapter.get("adapter_name") == expected_adapter_name
        )
    ]
    if len(runtime_matches) != 1:
        raise ValueError("Adapter runtime ownership mismatch")
    runtime_adapter = runtime_matches[0]
    adapter_id = runtime_adapter.get("adapter_id")
    if not adapter_id:
        raise ValueError("Loaded adapter metadata is missing adapter_id")
    persisted_adapter = adapter_service.get_adapter(adapter_id)
    if not persisted_adapter:
        raise ValueError("Loaded adapter metadata not found")

    expected_values = {
        "adapter_id": adapter_id,
        "adapter_name": expected_adapter_name or runtime_adapter.get("adapter_name"),
        "deployment_id": deployment_id,
        "deployment_replica_id": deployment_replica_id,
        "user_id": config.get("user_id"),
        "status": "loaded",
    }
    for field, expected in expected_values.items():
        if persisted_adapter.get(field) != expected:
            raise ValueError(f"Loaded adapter {field} binding mismatch")
        if runtime_adapter.get(field) != expected:
            raise ValueError(f"Adapter runtime {field} binding mismatch")

    source_training_id = persisted_adapter.get("source_task_id")
    if not source_training_id:
        raise ValueError("Loaded adapter ownership metadata is missing")
    if runtime_adapter.get("source_task_id") != source_training_id:
        raise ValueError("Adapter runtime training ownership mismatch")
    training = external_sync_service.get_training_by_task_id(source_training_id)
    if not training:
        raise ValueError("Loaded adapter training ownership mismatch")
    _validate_adapter_training_owner(config, selected_target, training)
    if training.get("loaded_adapter_id") != adapter_id:
        raise ValueError("Loaded adapter training id ownership mismatch")
    if training.get("loaded_adapter_name") != expected_values["adapter_name"]:
        raise ValueError("Loaded adapter training name ownership mismatch")
    if training.get("status") != SyncTrainingStatus.ADAPTER_LOADED:
        raise ValueError("Loaded adapter training status ownership mismatch")

    snapshot = training.get("target_config_snapshot") or {}
    if {
        "base_deployment_id",
        "base_deployment_replica_id",
    }.issubset(snapshot) and (
        snapshot.get("base_deployment_id") != deployment_id
        or snapshot.get("base_deployment_replica_id")
        != deployment_replica_id
    ):
        raise ValueError("Loaded adapter training binding mismatch")
    return persisted_adapter


def _select_unload_target(
    config: Dict[str, Any],
    targets: list[Dict[str, Any]],
    target_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Select an explicit target, with one-target legacy compatibility."""
    config_id = config.get("task_id")
    if target_id:
        target = next(
            (item for item in targets if item.get("target_id") == target_id),
            None,
        )
        if target is None:
            raise ValueError(f"Training target not found: {target_id}")
        if target.get("task_id") != config_id:
            raise PermissionError("Training target belongs to another sync task")
        return target

    current_targets = [
        target
        for target in targets
        if target.get("current_adapter_id") or target.get("current_adapter_name")
    ]
    if len(current_targets) > 1:
        raise ValueError(
            "target_id is required when multiple training targets have adapters"
        )
    if current_targets:
        return current_targets[0]
    if config.get("current_adapter_id") or config.get("current_adapter_name"):
        return None
    raise ValueError("No adapter currently loaded")


def _clear_binding_adapter_state(
    config: Dict[str, Any],
    targets: list[Dict[str, Any]],
    scoped_target_ids: tuple[Optional[str], ...],
    *,
    exclude_training_task_id: Optional[str] = None,
    loaded_adapter_id: Optional[str] = None,
    loaded_adapter_name: Optional[str] = None,
    adapter_still_referenced: bool = False,
) -> None:
    """Release selected owner references and recompute the task summary."""
    from ..storage.services.external_sync_service import external_sync_service

    scoped_ids = {target_id for target_id in scoped_target_ids if target_id}
    cleared_adapters = [
        dict(target)
        for target in targets
        if target.get("target_id") in scoped_ids
        and (target.get("current_adapter_id") or target.get("current_adapter_name"))
    ]
    for target in targets:
        if target.get("target_id") not in scoped_ids:
            continue
        target_updates: Dict[str, Any] = {
            "current_adapter_name": None,
            "current_adapter_id": None,
        }
        if target.get("current_training_id") != exclude_training_task_id:
            target_updates["current_training_id"] = None
        external_sync_service.update_training_target(
            target["target_id"],
            **target_updates,
        )
        target.update(target_updates)

    if not adapter_still_referenced:
        mark_kwargs: Dict[str, Any] = {
            "loaded_adapter_id": loaded_adapter_id,
            "loaded_adapter_name": loaded_adapter_name,
        }
        if exclude_training_task_id:
            mark_kwargs["exclude_training_task_id"] = (
                exclude_training_task_id
            )
        external_sync_service.mark_all_trainings_adapter_unloaded(
            config["task_id"],
            **mark_kwargs,
        )

    summary_was_cleared = None in scoped_target_ids or any(
        (
            config.get("current_adapter_id")
            and target.get("current_adapter_id") == config.get("current_adapter_id")
        )
        or (
            not config.get("current_adapter_id")
            and config.get("current_adapter_name")
            and target.get("current_adapter_name")
            == config.get("current_adapter_name")
        )
        for target in cleared_adapters
    )
    if not summary_was_cleared and (
        config.get("current_adapter_id") or config.get("current_adapter_name")
    ):
        return

    remaining = next(
        (
            target
            for target in targets
            if target.get("current_adapter_id") or target.get("current_adapter_name")
        ),
        None,
    )
    summary = {
        "current_adapter_name": (
            remaining.get("current_adapter_name") if remaining else None
        ),
        "current_adapter_id": (
            remaining.get("current_adapter_id") if remaining else None
        ),
        "current_training_id": (
            remaining.get("current_training_id")
            if remaining
            else exclude_training_task_id
        ),
    }
    external_sync_service.update_task(config["task_id"], **summary)
    config.update(summary)


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
        replace: If True, release only this task/target's current adapter
                 before loading. If False, load alongside it.
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
        training_target_id = sync_training.get("target_id")
        if target_id is not None and target_id != training_target_id:
            raise ValueError("Sync training target ownership mismatch")
        target_id = training_target_id
        binding_snapshot = sync_training.get("target_config_snapshot") or {}
        has_binding_snapshot = {
            "base_deployment_id",
            "base_deployment_replica_id",
        }.issubset(binding_snapshot)

        selected_target = None
        if target_id:
            from ..storage.services.external_sync_service import external_sync_service as _ess
            _target = _ess.get_training_target_raw(target_id)
            if not _target:
                raise ValueError(f"Training target not found: {target_id}")
            selected_target = _target
            binding_target = dict(_target)
            binding_config = dict(config)
            if has_binding_snapshot:
                binding_target["base_deployment_id"] = binding_snapshot.get(
                    "base_deployment_id"
                )
                binding_target["base_deployment_replica_id"] = (
                    binding_snapshot.get("base_deployment_replica_id")
                )
                binding_config["base_deployment_id"] = binding_snapshot.get(
                    "base_deployment_id"
                )
                binding_config["base_deployment_replica_id"] = (
                    binding_snapshot.get("base_deployment_replica_id")
                )
            deployment_id, deployment_replica_id = (
                _resolve_target_deployment_binding(
                    binding_config,
                    binding_target,
                    tag,
                )
            )
        else:
            binding_config = dict(config)
            if has_binding_snapshot:
                binding_config["base_deployment_id"] = binding_snapshot.get(
                    "base_deployment_id"
                )
                binding_config["base_deployment_replica_id"] = (
                    binding_snapshot.get("base_deployment_replica_id")
                )
            deployment_id = _resolve_deployment_id(binding_config, tag)
            deployment_replica_id = binding_config.get(
                "base_deployment_replica_id"
            )
        _validate_adapter_training_owner(config, selected_target, sync_training)
        external_sync_service.update_task(config_id, status=SyncStatus.LOADING_ADAPTER)
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
            adapter_service.sync_loaded_adapters(
                deployment_id,
                deployment_replica_id=deployment_replica_id,
                user_id=config.get("user_id"),
            )
        except Exception as e:
            logger.warning(f"[sync:{tag}] Failed to sync adapters: {e}")
            raise

        if replace:
            targets = _load_adapter_targets(config_id)
            if selected_target and not any(
                target.get("target_id") == selected_target.get("target_id")
                for target in targets
            ):
                targets.append(selected_target)
            owner = selected_target or config
            if owner.get("current_adapter_id") or owner.get(
                "current_adapter_name"
            ):
                loaded_adapters = adapter_service.list_loaded_adapters(
                    deployment_id,
                    deployment_replica_id=deployment_replica_id,
                    user_id=config.get("user_id"),
                )
                owned_adapter = _resolve_owned_runtime_adapter(
                    adapter_service=adapter_service,
                    config=config,
                    selected_target=selected_target,
                    deployment_id=deployment_id,
                    deployment_replica_id=deployment_replica_id,
                    loaded_adapters=loaded_adapters,
                )
                adapter_id_to_release = owned_adapter.get("adapter_id")
                adapter_name_to_release = owned_adapter.get("adapter_name")
                still_referenced = _has_other_adapter_reference(
                    targets,
                    selected_target,
                    adapter_id_to_release,
                    adapter_name_to_release,
                )
                if not still_referenced:
                    adapter_service.unload_adapter(
                        deployment_id,
                        adapter_name_to_release,
                        deployment_replica_id=deployment_replica_id,
                        user_id=config.get("user_id"),
                    )
                    logger.info(
                        "[sync:%s] Unloaded owned adapter: %s",
                        tag,
                        adapter_name_to_release,
                    )
                scoped_target_ids = (
                    (selected_target["target_id"],)
                    if selected_target
                    else (None,)
                )
                _clear_binding_adapter_state(
                    config,
                    targets,
                    scoped_target_ids,
                    exclude_training_task_id=training_task_id,
                    loaded_adapter_id=adapter_id_to_release,
                    loaded_adapter_name=adapter_name_to_release,
                    adapter_still_referenced=still_referenced,
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
            deployment_replica_id=deployment_replica_id,
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
                current_training_id=training_task_id,
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


def unload_current_adapter(config_id: str, target_id: Optional[str] = None):
    """Unload adapters from one target's deployment and replica binding."""
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
        targets = _load_adapter_targets(config_id)
        selected_target = _select_unload_target(config, targets, target_id)
        adapter_service = AdapterService()
        persisted_binding = _get_persisted_adapter_binding(
            adapter_service,
            config,
            selected_target,
        )
        if selected_target is not None:
            binding_target = dict(selected_target)
            binding_config = dict(config)
            if persisted_binding is not None:
                binding_target["base_deployment_id"] = persisted_binding[0]
                binding_target["base_deployment_replica_id"] = (
                    persisted_binding[1]
                )
                binding_config["base_deployment_id"] = persisted_binding[0]
                binding_config["base_deployment_replica_id"] = (
                    persisted_binding[1]
                )
            deployment_id, deployment_replica_id = (
                _resolve_target_deployment_binding(
                    binding_config,
                    binding_target,
                    tag,
                )
            )
        else:
            binding_config = dict(config)
            if persisted_binding is not None:
                binding_config["base_deployment_id"] = persisted_binding[0]
                binding_config["base_deployment_replica_id"] = (
                    persisted_binding[1]
                )
            deployment_id = _resolve_deployment_id(binding_config, tag)
            deployment_replica_id = binding_config.get(
                "base_deployment_replica_id"
            )
        if not deployment_id:
            raise ValueError("No compatible deployment found")

        # Refresh runtime metadata before validating the exact owned adapter.
        try:
            adapter_service.sync_loaded_adapters(
                deployment_id,
                deployment_replica_id=deployment_replica_id,
                user_id=config.get("user_id"),
            )
        except Exception as e:
            logger.warning(f"[sync:{tag}] Failed to sync adapters: {e}")
            raise

        loaded_adapters = adapter_service.list_loaded_adapters(
            deployment_id,
            deployment_replica_id=deployment_replica_id,
            user_id=config.get("user_id"),
        )
        owned_adapter = _resolve_owned_runtime_adapter(
            adapter_service=adapter_service,
            config=config,
            selected_target=selected_target,
            deployment_id=deployment_id,
            deployment_replica_id=deployment_replica_id,
            loaded_adapters=loaded_adapters,
        )
        adapter_id_to_release = owned_adapter.get("adapter_id")
        adapter_name_to_release = owned_adapter.get("adapter_name")
        still_referenced = _has_other_adapter_reference(
            targets,
            selected_target,
            adapter_id_to_release,
            adapter_name_to_release,
        )
        if not still_referenced:
            adapter_service.unload_adapter(
                deployment_id,
                adapter_name_to_release,
                deployment_replica_id=deployment_replica_id,
                user_id=config.get("user_id"),
            )
            logger.info(
                "[sync:%s] Unloaded owned adapter: %s",
                tag,
                adapter_name_to_release,
            )
        scoped_target_ids = (
            (selected_target["target_id"],)
            if selected_target
            else (None,)
        )
        _clear_binding_adapter_state(
            config,
            targets,
            scoped_target_ids,
            loaded_adapter_id=adapter_id_to_release,
            loaded_adapter_name=adapter_name_to_release,
            adapter_still_referenced=still_referenced,
        )
    finally:
        lock.release()
