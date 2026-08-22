"""Database services for TrainFactory.

Lower-case exports such as ``training_task_service`` intentionally use proxy
objects instead of relying on package attribute fallback. Otherwise
``from train_factory.storage.services import training_task_service`` resolves to
the submodule object, not the global service instance, which breaks callers at
runtime.
"""

import sys
from importlib import import_module
from types import ModuleType
from typing import Dict, Tuple, Any

_CLASS_EXPORTS: Dict[str, Tuple[str, str]] = {
    "TrainingTaskService": (".training_task_service", "TrainingTaskService"),
    "ModelRegistryService": (".model_registry_service", "ModelRegistryService"),
    "ModelConfigService": (".model_config_service", "ModelConfigService"),
    "DatasetService": (".dataset_service", "DatasetService"),
    "AuditLogService": (".audit_log_service", "AuditLogService"),
    "EvaluationTaskService": (".evaluation_task_service", "EvaluationTaskService"),
    "DeepEvaluationTaskService": (".deep_evaluation_task_service", "DeepEvaluationTaskService"),
    "TrainingTaskEventService": (".training_task_event_service", "TrainingTaskEventService"),
    "GenerationTaskService": (".generation_task_service", "GenerationTaskService"),
}

_INSTANCE_EXPORTS: Dict[str, Tuple[str, str]] = {
    "training_task_service": (".training_task_service", "training_task_service"),
    "model_registry_service": (".model_registry_service", "model_registry_service"),
    "model_config_service": (".model_config_service", "model_config_service"),
    "dataset_service": (".dataset_service", "dataset_service"),
    "audit_log_service": (".audit_log_service", "audit_log_service"),
    "evaluation_task_service": (".evaluation_task_service", "evaluation_task_service"),
    "deep_evaluation_task_service": (".deep_evaluation_task_service", "deep_evaluation_task_service"),
    "training_task_event_service": (".training_task_event_service", "training_task_event_service"),
    "generation_task_service": (".generation_task_service", "generation_task_service"),
}


def _load_export(module_name: str, attr_name: str) -> Any:
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)


class _LazyServiceProxy:
    """Resolve a service instance on first attribute access."""

    def __init__(self, module_name: str, attr_name: str):
        self._module_name = module_name
        self._attr_name = attr_name
        self._target: Any | None = None

    def _resolve(self) -> Any:
        if self._target is None:
            self._target = _load_export(self._module_name, self._attr_name)
        return self._target

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return repr(self._resolve())


class _ServicesModule(ModuleType):
    """Keep package-level service exports stable even after submodule imports."""

    def __getattribute__(self, name: str) -> Any:
        if name in _INSTANCE_EXPORTS:
            proxies = ModuleType.__getattribute__(self, "_SERVICE_PROXIES")
            return proxies[name]
        return ModuleType.__getattribute__(self, name)


_SERVICE_PROXIES: Dict[str, _LazyServiceProxy] = {}
for _name, (_module_name, _attr_name) in _INSTANCE_EXPORTS.items():
    _SERVICE_PROXIES[_name] = _LazyServiceProxy(_module_name, _attr_name)
    globals()[_name] = _SERVICE_PROXIES[_name]

sys.modules[__name__].__class__ = _ServicesModule

__all__ = list(_CLASS_EXPORTS.keys()) + list(_INSTANCE_EXPORTS.keys())


def __getattr__(name: str):
    if name in _CLASS_EXPORTS:
        module_name, attr_name = _CLASS_EXPORTS[name]
        return _load_export(module_name, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
