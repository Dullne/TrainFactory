"""Model config source enumerations."""

from enum import Enum


class ConfigSourceType(str, Enum):
    """Model config source type - where the model config comes from."""

    EXTERNAL_API = "external_api"     # External API (OpenAI, Azure, etc.)
    LOCAL_DEPLOYED = "local_deployed"  # Locally deployed model (from registry/deployment)
