"""Health status enumerations."""

from enum import Enum


class HealthStatus(str, Enum):
    """Service health status."""

    HEALTHY = "HEALTHY"       # Service is healthy and responding
    UNHEALTHY = "UNHEALTHY"   # Service is not healthy
    UNKNOWN = "UNKNOWN"       # Health status unknown (not checked yet)
