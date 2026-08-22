"""
Authentication module for TrainFactory.

Provides JWT-based authentication for API endpoints.
"""

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "hash_password": (".password", "hash_password"),
    "verify_password": (".password", "verify_password"),
    "create_access_token": (".jwt_handler", "create_access_token"),
    "decode_token": (".jwt_handler", "decode_token"),
    "get_current_user": (".dependencies", "get_current_user"),
    "get_current_user_optional": (".dependencies", "get_current_user_optional"),
    "user_service": (".user_service", "user_service"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
