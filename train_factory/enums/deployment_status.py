"""Deployment status enumerations."""

from enum import Enum


class DeploymentStatus(str, Enum):
    """Deployment status."""

    PENDING = "pending"       # Waiting to start
    STARTING = "starting"     # Starting up
    RUNNING = "running"       # Running and serving
    RESTARTING = "restarting" # Restarting (reload or container restart)
    STOPPING = "stopping"     # Shutting down
    STOPPED = "stopped"       # Stopped
    FAILED = "failed"         # Failed to start or crashed
