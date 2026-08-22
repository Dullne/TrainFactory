"""Storage layer for TrainFactory."""

__all__ = ["get_engine", "get_session", "init_db"]


def __getattr__(name: str):
    """Lazy export database helpers to avoid importing DB stack on package import."""
    if name in __all__:
        from . import database
        return getattr(database, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
