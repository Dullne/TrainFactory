"""Training status enumerations."""

from enum import Enum


class TrainingStatus(str, Enum):
    """Training task status."""

    PENDING = "pending"           # Task created, waiting to start
    PREPARING = "preparing"       # Preparing resources (downloading model, loading data)
    RUNNING = "running"           # Training in progress
    EVALUATING = "evaluating"     # Post-training evaluation
    SUCCEEDED = "succeeded"       # Training completed successfully
    FAILED = "failed"             # Training failed
    STOPPED = "stopped"           # Training stopped by user
    CANCELLED = "cancelled"       # Training cancelled before starting


class ProcessStatus(str, Enum):
    """Training process status."""

    NOT_STARTED = "not_started"   # Process not started
    STARTING = "starting"         # Process is starting
    RUNNING = "running"           # Process is running
    COMPLETED = "completed"       # Process completed normally
    FAILED = "failed"             # Process failed
    KILLED = "killed"             # Process was killed
    UNKNOWN = "unknown"           # Process status unknown
