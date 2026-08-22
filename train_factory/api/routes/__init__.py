"""API routes for TrainFactory."""

__all__ = [
    "training_router",
    "registry_router",
    "deployment_router",
    "resource_router",
]


def __getattr__(name: str):
    """Lazily import route modules to keep package import lightweight."""
    mapping = {
        "training_router": (".training_routes", "router"),
        "registry_router": (".registry_routes", "router"),
        "deployment_router": (".deployment_routes", "router"),
        "resource_router": (".resource_routes", "router"),
    }
    if name in mapping:
        module_name, attr_name = mapping[name]
        from importlib import import_module

        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
