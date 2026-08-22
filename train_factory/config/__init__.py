"""Lazy configuration exports for TrainFactory."""

from importlib import import_module
import sys
from types import ModuleType
from typing import Any

__all__ = ["Settings", "settings", "get_settings"]


def _settings_module():
    return import_module(".settings", __name__)


def __getattr__(name: str) -> Any:
    if name == "Settings":
        return _settings_module().Settings
    if name == "get_settings":
        return _settings_module().get_settings
    if name == "settings":
        return _settings_module().get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


class _ConfigModule(ModuleType):
    def __getattribute__(self, name: str) -> Any:
        if name == "settings":
            module = import_module(".settings", __name__)
            return module.get_settings()
        return super().__getattribute__(name)


sys.modules[__name__].__class__ = _ConfigModule
