"""
Level 2 handler: triggered after generation task completes.

Checks if accumulated training samples have reached the training threshold,
and if so, creates a training task with ALL historical training datasets.
"""

import logging
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from ..enums.sync_status import SyncStatus
from ..storage.services.background_task_admission_service import (
    background_task_admission_service,
)

logger = logging.getLogger(__name__)


def _validate_sync_training_parent_checkpoint(
    parent_task_id: Optional[str],
    checkpoint_path: Optional[str],
    sync_user_id: str,
) -> None:
    """Apply the API's parent ownership and checkpoint containment rules to sync."""
    from ..config.settings import get_settings

    if get_settings().auth_enabled and not sync_user_id:
        raise ValueError("Sync training requires an owning user")

    from ..api.routes.training_routes import (
        _validate_training_parent_checkpoint,
    )

    _validate_training_parent_checkpoint(
        parent_task_id,
        checkpoint_path,
        {"user_id": sync_user_id, "is_admin": False},
    )


def _build_sync_target_output_dir(
    config_id: str,
    target_id: str,
    training_round: int,
    output_root: Optional[Path] = None,
) -> str:
    """Build a sync target output path under the configured training output root."""
    if output_root is None:
        from ..config.settings import get_settings
        output_root = get_settings().output_dir

    normalized_output_root = str(output_root).replace("\\", "/")
    return str(
        PurePosixPath(normalized_output_root)
        / "sync"
        / config_id
        / target_id
        / f"round_{training_round}"
    )


def on_generation_completed(
    generation_task_id: str,
    output_dataset_id: Optional[str],
    output_sample_count: int,
):
    """Called after a generation task completes successfully.

    Checks if this generation was triggered by the sync system,
    and if the training threshold has been reached.
    """
    from ..storage.services.external_sync_service import external_sync_service

    completion = external_sync_service.complete_generation_and_consume_batches(
        generation_task_id,
        output_dataset_id=output_dataset_id,
        output_sample_count=output_sample_count,
    )
    if not completion.get("tracking_found"):
        return completion
    if not completion.get("completed"):
        return completion

    config_id = completion["task_id"]
    tag = config_id[:8]

    logger.info(
        f"[sync:{tag}] Generation completed: task={generation_task_id[:8]}, "
        f"output_dataset={output_dataset_id}, samples={output_sample_count}"
    )

    # The completion transaction has committed before any training is claimed.
    config = external_sync_service.get_task_raw(config_id)
    if not config:
        return completion

    if completion.get("uses_training_targets"):
        _schedule_next_training(config_id)
        return completion

    # 注：此前的 `if all_targets:` 分支引用了未定义变量（NameError），且与
    # uses_training_targets 分支冗余——已删除。legacy 单阈值逻辑如下。
    training_threshold = int(config.get("training_threshold", 0) or 0)
    if training_threshold > 0 and config["pending_training_samples"] >= training_threshold:
        if config.get("status") in (SyncStatus.TRAINING, SyncStatus.LOADING_ADAPTER):
            logger.info(
                f"[sync:{tag}] Training already in progress (status={config.get('status')}), skip trigger"
            )
            return completion
        logger.info(
            f"[sync:{tag}] Level 2 threshold reached: "
            f"{config['pending_training_samples']} >= {training_threshold}"
        )
        _trigger_training(config)
        _schedule_next_training(config_id)
        return completion

    # Training threshold not reached (or disabled) — restore IDLE so worker
    # The transaction already restored IDLE so the worker may run the next cycle.
    logger.info(f"[sync:{tag}] Generation done, training threshold not reached, back to IDLE")
    return completion


def on_generation_failed(generation_task_id: str, error: str = ""):
    """Atomically restore sync state after a generation task fails."""
    from ..storage.services.external_sync_service import external_sync_service

    reason = error or "Generation failed"
    recovery = external_sync_service.fail_generation_and_restore_batches(
        generation_task_id,
        reason,
    )
    if recovery.get("recovered"):
        logger.warning(
            "Generation %s failed (%s); restored %s batches / %s records",
            generation_task_id[:8],
            reason[:200],
            recovery.get("restored_batch_count", 0),
            recovery.get("restored_record_count", 0),
        )
    return recovery


def on_qa_phase_completed(
    generation_task_id: str,
    qa_dataset_id: str,
    qa_sample_count: int,
):
    """Called when Phase 1 (QA extraction) completes.

    Records QA output. Pending samples are credited only once the containing
    generation is complete and the dataset is visible to the training launcher.
    """
    from ..storage.services.external_sync_service import external_sync_service

    qa_result = external_sync_service.update_generation_qa_output(
        generation_task_id, qa_dataset_id, qa_sample_count
    )
    if not qa_result.get("tracking_found"):
        return qa_result

    config_id = qa_result["task_id"]
    tag = config_id[:8]

    logger.info(
        f"[sync:{tag}] QA phase completed: task={generation_task_id[:8]}, "
        f"qa_dataset={qa_dataset_id}, samples={qa_sample_count}"
    )

    if qa_result.get("schedule_training"):
        _schedule_next_training(config_id)
    return qa_result


def _trigger_training(config: Dict[str, Any]):
    """Run a legacy auto-training launch and recover the parent on failure."""
    from ..storage.services.external_sync_service import external_sync_service

    config_id = config["task_id"]
    tag = config_id[:8]
    try:
        _launch_legacy_training(config)
    except Exception as exc:
        logger.exception(f"[sync:{tag}] Failed to trigger training: {exc}")
        external_sync_service.compare_and_set_task_status(
            config_id,
            SyncStatus.TRAINING,
            SyncStatus.ERROR,
            error_message=f"Trigger training failed: {exc}",
        )


def _launch_legacy_training(config: Dict[str, Any]):
    """Level 2: Create a training task with all accumulated training datasets."""
    from ..storage.services.external_sync_service import external_sync_service
    from ..storage.services.training_task_service import training_task_service
    from ..storage.services.dataset_service import dataset_service
    from ..config.settings import get_settings

    config_id = config["task_id"]
    tag = config_id[:8]

    # Collect ALL completed generation datasets
    all_gens = external_sync_service.get_all_completed_generation_datasets(config_id)
    if not all_gens:
        logger.warning(f"[sync:{tag}] No completed generation datasets found")
        external_sync_service.update_task(config_id, status=SyncStatus.IDLE)
        return

    # Gather dataset paths from the registered training datasets
    dataset_configs = []
    total_samples = 0
    for gen in all_gens:
        ds_id = gen.get("output_dataset_id")
        if not ds_id:
            continue
        ds = dataset_service.get_dataset(ds_id)
        if not ds:
            logger.warning(f"[sync:{tag}] Dataset {ds_id} not found, skipping")
            continue
        storage_location = ds.get("storage_uri") or ds.get("storage_path")
        if not storage_location:
            continue
        dataset_configs.append({
            "path": storage_location,
            "max_samples": None,
            "split": "train",
        })
        total_samples += gen.get("output_sample_count", 0)

    if not dataset_configs:
        logger.warning(f"[sync:{tag}] No valid training datasets found")
        external_sync_service.update_task(config_id, status=SyncStatus.IDLE)
        return

    training_round = config.get("total_trainings", 0) + 1

    # Build training config from sync config
    train_cfg = config.get("training_config")
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    # Resolve base_model_path: deployment config takes precedence over training config
    base_model_path = ""
    if config.get("base_deployment_id"):
        base_model_path = _resolve_base_model_path(
            config["base_deployment_id"],
            tag,
            expected_user_id=config.get("user_id"),
            expected_external_api_config_id=config.get("external_api_config_id"),
        )
    if not base_model_path:
        base_model_path = train_cfg.get("base_model_path", "")

    if not base_model_path:
        logger.warning(
            f"[sync:{tag}] Training not configured: no base_model_path "
            f"(base_deployment_id={config.get('base_deployment_id')}, "
            f"training_config.base_model_path={train_cfg.get('base_model_path')}). "
            f"Staying idle — configure training model to enable auto-training."
        )
        external_sync_service.update_task(config_id, status=SyncStatus.IDLE)
        return

    model_type = train_cfg.get("model_type", "llm")
    training_method = train_cfg.get("training_method", "sft")

    # Check if there's a validation dataset
    has_eval_dataset = any(ds.get("split") in ("eval", "validation", "dev") for ds in dataset_configs)

    training_config = {
        "base_model_path": base_model_path,
        "model_type": model_type,
        "training_method": training_method,
        "dataset_configs": dataset_configs,
        "train_dataset_path": dataset_configs[0]["path"],
        "num_train_epochs": 3,
        "eval_strategy": "epoch" if has_eval_dataset else "no",
        "save_strategy": "epoch",
        "logging_steps": 1,
        "use_lora": True,
        "lora_config": {
            "use_lora": True,
            "r": train_cfg.get("lora_r", 16),
            "lora_alpha": train_cfg.get("lora_alpha", 32),
            "lora_dropout": train_cfg.get("lora_dropout", 0.0),
        },
    }

    # Copy additional training params (user config overrides defaults above)
    for key in ("num_train_epochs", "per_device_train_batch_size", "learning_rate",
                "warmup_ratio", "gradient_accumulation_steps", "max_length",
                "save_steps", "eval_steps", "logging_steps", "gpu_ids",
                "eval_strategy", "save_strategy",
                "embedding_loss_name", "reranker_loss_name", "bf16", "fp16",
                "rl_config", "sft_checkpoint_path", "parent_task_id"):
        if key in train_cfg:
            training_config[key] = train_cfg[key]

    # Default loss name based on model_type if not explicitly configured
    if "embedding_loss_name" not in training_config and model_type == "embedding":
        training_config["embedding_loss_name"] = "auto"
    if "reranker_loss_name" not in training_config and model_type in ("reranker", "decoder_reranker"):
        training_config["reranker_loss_name"] = "auto"

    # Loss config
    if "loss_config" in train_cfg:
        training_config["loss_config"] = train_cfg["loss_config"]

    training_config["user_id"] = config["user_id"]
    from ..api.routes.training_routes import _normalize_training_config_paths
    training_config = _normalize_training_config_paths(
        training_config,
        config["user_id"],
    )

    from ..enums.training_status import TrainingStatus

    task_id = str(uuid.uuid4())
    sync_training_created = False
    execution_lease = None
    lease_handed_off = False
    parent_guard = None
    try:
        task_name = f"sync-{tag}-train-r{training_round}"
        settings = get_settings()
        output_dir = str(settings.get_task_output_dir(task_id))
        training_config["output_dir"] = output_dir
        training_config["task_id"] = task_id

        claimed = external_sync_service.create_training_with_claim(
            task_id=config_id,
            training_task_id=task_id,
            user_id=config["user_id"],
            input_dataset_ids=[
                generation["output_dataset_id"]
                for generation in all_gens
                if generation.get("output_dataset_id")
            ],
            total_samples=total_samples,
            training_round=training_round,
            require_threshold=True,
        )
        if not claimed:
            logger.info("[sync:%s] Legacy training claim was not acquired", tag)
            return False
        sync_training_created = True

        parent_task_id = training_config.get("parent_task_id")
        if parent_task_id:
            parent_guard = background_task_admission_service.begin_deletion(
                "training",
                parent_task_id,
            )
        _validate_sync_training_parent_checkpoint(
            parent_task_id,
            training_config.get("sft_checkpoint_path"),
            config["user_id"],
        )

        _task_info, execution_lease = background_task_admission_service.admit_execution(
            "training",
            task_id,
            config["user_id"],
            training_task_service.create_task,
            task_id=task_id,
            task_name=task_name,
            model_path=training_config["base_model_path"],
            train_dataset_path=training_config["train_dataset_path"],
            training_params=training_config,
            description=f"Auto-triggered by sync task {config_id}, round {training_round}, {total_samples} total samples",
            user_id=config["user_id"],
            model_type=model_type,
            training_method=training_method,
            model_architecture=train_cfg.get("model_architecture"),
            rl_config=training_config.get("rl_config"),
            loss_config=training_config.get("loss_config"),
            parent_task_id=training_config.get("parent_task_id"),
            sft_checkpoint_path=training_config.get("sft_checkpoint_path"),
            output_dir=output_dir,
            require_managed_datasets=True,
        )
        training_task_service.update_task_output_dir(task_id, output_dir, training_config)
        if parent_guard is not None:
            parent_guard.release()
            parent_guard = None

        # Run training in background thread (training is sync, not async)
        from ..api.routes.training_routes import run_training_task
        import threading
        thread = threading.Thread(
            target=background_task_admission_service.run_sync,
            args=(execution_lease, run_training_task, task_id, training_config),
            daemon=True,
        )
        try:
            thread.start()
            lease_handed_off = True
        except Exception as exc:
            raise RuntimeError(f"Training thread failed to start: {exc}") from exc

        logger.info(
            f"[sync:{tag}] Training task created: {task_id}, "
            f"round={training_round}, datasets={len(dataset_configs)}, samples={total_samples}"
        )
    except Exception as exc:
        if parent_guard is not None:
            parent_guard.release()
        if execution_lease is not None and not lease_handed_off:
            execution_lease.release()
        if task_id:
            try:
                training_task_service.update_task_status(
                    task_id,
                    TrainingStatus.FAILED.value,
                    error_message=str(exc),
                )
            except Exception:
                logger.exception(
                    "[sync:%s] Could not mark failed training task %s",
                    tag,
                    task_id,
                )
        if sync_training_created:
            try:
                external_sync_service.fail_training_and_restore_claim(
                    task_id,
                    str(exc),
                )
            except Exception:
                logger.exception(
                    "[sync:%s] Could not mark failed sync training %s",
                    tag,
                    task_id,
                )
        raise
    return True


def _resolve_base_model_path(
    deployment_id: str,
    tag: str,
    *,
    expected_user_id: Optional[str] = None,
    expected_external_api_config_id: Optional[str] = None,
) -> str:
    """Resolve base model filesystem path from a deployment ID.

    Chain: deployment_id → model_id → model_registry → base_model_path
    """
    from ..deployment.deployment_service import deployment_service
    from ..storage.services.model_registry_service import model_registry_service

    deployment = deployment_service.get_deployment(deployment_id)
    if not deployment:
        logger.warning(f"[sync:{tag}] Deployment {deployment_id} not found")
        return ""

    if expected_user_id and deployment.get("user_id") != expected_user_id:
        raise PermissionError(
            f"Deployment {deployment_id} belongs to another user"
        )

    expected_api_config = (expected_external_api_config_id or "").strip()
    if expected_api_config:
        deployment_api_config = (
            deployment.get("external_api_config_id") or ""
        ).strip()
        if not deployment_api_config:
            raw_config = deployment.get("config") or {}
            deployment_api_config = (
                raw_config.get("external_api_config_id") or ""
            ).strip()
        if deployment_api_config != expected_api_config:
            raise PermissionError(
                f"Deployment {deployment_id} belongs to another API tenant"
            )

    model_id = deployment.get("model_id")
    if not model_id:
        logger.warning(f"[sync:{tag}] Deployment {deployment_id} has no model_id")
        return ""

    model = model_registry_service.get_model(model_id)
    if not model:
        logger.warning(f"[sync:{tag}] Model {model_id} not found in registry")
        return ""
    if expected_user_id and model.get("user_id") != expected_user_id:
        raise PermissionError(f"Model {model_id} belongs to another user")

    path = model.get("base_model_path") or model.get("model_path") or ""
    if path:
        logger.info(f"[sync:{tag}] Resolved base_model_path from deployment: {path}")
    return path


def _schedule_next_training(config_id: str):
    """Check all training targets and schedule the next eligible one."""
    from ..storage.services.external_sync_service import external_sync_service

    tag = config_id[:8]
    targets = external_sync_service.list_training_targets_raw(config_id)
    if any(
        target.get("status")
        in {"training", "loading_adapter"}
        for target in targets
    ):
        return
    next_target = next(
        (
            target
            for target in targets
            if target.get("status") in {"idle", "ready"}
            and int(target.get("training_threshold") or 0) > 0
            and int(target.get("pending_training_samples") or 0)
            >= int(target.get("training_threshold") or 0)
        ),
        None,
    )
    if not next_target:
        return

    logger.info(
        f"[sync:{tag}] Scheduling training for target '{next_target['target_name']}' "
        f"(priority={next_target['priority']}, pending={next_target['pending_training_samples']})"
    )
    _trigger_training_for_target(
        config_id,
        next_target,
        require_threshold=True,
    )


def _trigger_training_for_target(
    config_id: str,
    target: Dict[str, Any],
    *,
    require_threshold: bool = False,
) -> bool:
    """Create and launch training for a specific target."""
    return _launch_training_for_target(
        config_id,
        target,
        require_threshold=require_threshold,
    )


def _launch_training_for_target(
    config_id: str,
    target: Dict[str, Any],
    *,
    require_threshold: bool = False,
) -> bool:
    """Atomically claim a target, then create and launch its training task."""
    import threading
    from ..storage.services.external_sync_service import external_sync_service

    tag = config_id[:8]
    target_id = target["target_id"]
    target_name = target["target_name"]
    model_type = target.get("model_type", "embedding")
    training_method = target.get("training_method", "sft")
    train_cfg = target.get("training_config") or {}
    base_model_path = target.get("base_model_path", "")
    config = external_sync_service.get_task_raw(config_id)
    if not config:
        raise ValueError(f"Sync task not found: {config_id}")
    if target.get("task_id") and target.get("task_id") != config_id:
        raise PermissionError("Training target belongs to another sync task")

    if not base_model_path and target.get("base_deployment_id"):
        try:
            base_model_path = _resolve_base_model_path(
                target["base_deployment_id"],
                tag,
                expected_user_id=config.get("user_id"),
                expected_external_api_config_id=config.get("external_api_config_id"),
            )
        except Exception as e:
            logger.warning(f"[sync:{tag}] Cannot resolve base model from deployment: {e}")

    if not base_model_path:
        logger.error(f"[sync:{tag}] No base_model_path for target '{target_name}'")
        return False

    if target["data_phase"] == "qa":
        datasets = external_sync_service.get_all_completed_qa_datasets(config_id)
    else:
        datasets = external_sync_service.get_all_completed_generation_datasets(config_id)

    if not datasets:
        logger.warning(f"[sync:{tag}] No datasets available for target '{target_name}'")
        return False

    from ..storage.services.dataset_service import dataset_service

    dataset_configs = []
    total_samples = 0
    for ds in datasets:
        ds_info = dataset_service.get_dataset(ds["dataset_id"])
        if not ds_info:
            continue
        storage_location = ds_info.get("storage_uri") or ds_info.get("storage_path")
        if not storage_location:
            continue
        dataset_configs.append({
            "path": storage_location,
            "dataset_id": ds["dataset_id"],
            "sample_count": ds.get("sample_count", 0),
        })
        total_samples += ds.get("sample_count", 0)

    if not dataset_configs:
        logger.warning(f"[sync:{tag}] No valid dataset paths for target '{target_name}'")
        return False

    training_round = target.get("total_trainings", 0) + 1

    training_config = {
        "base_model_path": base_model_path,
        "model_type": model_type,
        "training_method": training_method,
        # 与 legacy 路径一致：run_training_task 无条件读取该键
        "train_dataset_path": dataset_configs[0]["path"],
        "lora_r": train_cfg.get("lora_r", 16),
        "lora_alpha": train_cfg.get("lora_alpha", 32),
        "lora_dropout": train_cfg.get("lora_dropout", 0.0),
        "num_train_epochs": train_cfg.get("num_train_epochs", 3),
        "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 16),
        "learning_rate": train_cfg.get("learning_rate", 2e-5),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.1),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 1),
        "dataset_configs": dataset_configs,
    }
    for key in (
        "max_length", "bf16", "fp16", "gpu_ids",
        "embedding_loss_name", "reranker_loss_name", "loss_config",
        "rl_config", "sft_checkpoint_path", "parent_task_id",
    ):
        if key in train_cfg:
            training_config[key] = train_cfg[key]

    sync_user_id = config.get("user_id", "system") if config else "system"
    training_config["user_id"] = sync_user_id

    from ..api.routes.training_routes import run_training_task, _normalize_training_config_paths
    from ..storage.services.training_task_service import training_task_service
    from ..enums.training_status import TrainingStatus

    task_id = str(uuid.uuid4())
    sync_training_created = False
    execution_lease = None
    lease_handed_off = False
    parent_guard = None
    try:
        training_config = _normalize_training_config_paths(
            training_config,
            sync_user_id,
        )
        task_name = f"sync-{tag}-{target_name[:16]}-r{training_round}"
        output_dir = _build_sync_target_output_dir(
            config_id,
            target_id,
            training_round,
        )
        training_config["output_dir"] = output_dir
        training_config["task_id"] = task_id

        claimed = external_sync_service.create_training_with_claim(
            task_id=config_id,
            training_task_id=task_id,
            user_id=sync_user_id,
            input_dataset_ids=[item["dataset_id"] for item in dataset_configs],
            total_samples=total_samples,
            training_round=training_round,
            target_id=target_id,
            require_threshold=require_threshold,
        )
        if not claimed:
            logger.info(
                "[sync:%s] Target '%s' was claimed by another worker",
                tag,
                target_name,
            )
            return False
        sync_training_created = True

        parent_task_id = training_config.get("parent_task_id")
        if parent_task_id:
            parent_guard = background_task_admission_service.begin_deletion(
                "training",
                parent_task_id,
            )
        _validate_sync_training_parent_checkpoint(
            parent_task_id,
            training_config.get("sft_checkpoint_path"),
            sync_user_id,
        )

        _task_info, execution_lease = background_task_admission_service.admit_execution(
            "training",
            task_id,
            sync_user_id,
            training_task_service.create_task,
            task_id=task_id,
            task_name=task_name,
            model_path=training_config["base_model_path"],
            train_dataset_path=training_config["train_dataset_path"],
            training_params=training_config,
            model_type=model_type,
            training_method=training_method,
            user_id=sync_user_id,
            rl_config=training_config.get("rl_config"),
            loss_config=training_config.get("loss_config"),
            parent_task_id=training_config.get("parent_task_id"),
            sft_checkpoint_path=training_config.get("sft_checkpoint_path"),
            output_dir=output_dir,
            require_managed_datasets=True,
        )
        training_task_service.update_task_output_dir(task_id, output_dir, training_config)
        if parent_guard is not None:
            parent_guard.release()
            parent_guard = None

        logger.info(
            f"[sync:{tag}] Training task created for target '{target_name}': "
            f"task_id={task_id[:8]}, datasets={len(dataset_configs)}, samples={total_samples}"
        )

        thread = threading.Thread(
            target=background_task_admission_service.run_sync,
            args=(execution_lease, run_training_task, task_id, training_config),
            daemon=True,
        )
        try:
            thread.start()
            lease_handed_off = True
        except Exception as exc:
            raise RuntimeError(f"Training thread failed to start: {exc}") from exc
    except Exception as exc:
        if parent_guard is not None:
            parent_guard.release()
        if execution_lease is not None and not lease_handed_off:
            execution_lease.release()
        if task_id:
            try:
                training_task_service.update_task_status(
                    task_id,
                    TrainingStatus.FAILED.value,
                    error_message=str(exc),
                )
            except Exception:
                logger.exception(
                    "[sync:%s] Could not mark failed training task %s",
                    tag,
                    task_id,
                )
        if sync_training_created:
            try:
                external_sync_service.fail_training_and_restore_claim(
                    task_id,
                    str(exc),
                )
            except Exception:
                logger.exception(
                    "[sync:%s] Could not mark failed sync training %s",
                    tag,
                    task_id,
                )
        raise

    return True
