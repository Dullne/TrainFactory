"""Model registry status enumerations."""

from enum import Enum


class ModelStatus(str, Enum):
    """Model registry status."""

    REGISTERED = "registered"     # Model metadata registered, files not yet available
    AVAILABLE = "available"       # Model files exist locally, ready for deployment/training
    ARCHIVED = "archived"         # Model is archived and not available
