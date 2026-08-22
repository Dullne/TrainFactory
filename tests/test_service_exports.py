"""Regression tests for importing concrete storage service instances."""

import importlib
from types import ModuleType

from train_factory.storage.services.generation_task_service import generation_task_service
from train_factory.storage.services.training_task_service import training_task_service


def test_service_instance_imports_not_modules():
    """Service instance imports must never resolve to submodule objects."""
    assert not isinstance(training_task_service, ModuleType)
    assert not isinstance(generation_task_service, ModuleType)
    assert hasattr(training_task_service, "get_all_tasks")
    assert hasattr(generation_task_service, "get_all_tasks")


def test_package_level_service_export_survives_submodule_import():
    """Import order must not change package-level service exports."""
    services_pkg = importlib.import_module("train_factory.storage.services")
    importlib.import_module("train_factory.storage.services.training_task_service")

    exported = services_pkg.training_task_service

    assert not isinstance(exported, ModuleType)
    assert hasattr(exported, "get_all_tasks")
