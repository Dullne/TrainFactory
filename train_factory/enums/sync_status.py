"""External sync related statuses."""


class SyncStatus:
    """Sync task status."""

    IDLE = "idle"
    SYNCING = "syncing"
    GENERATING = "generating"
    TRAINING = "training"
    LOADING_ADAPTER = "loading_adapter"
    ERROR = "error"
    DELETING = "deleting"
    DELETING_CASCADE = "deleting_cascade"


class BatchStatus:
    """Sync batch status."""

    FETCHED = "fetched"
    REGISTERED = "registered"
    GENERATION_QUEUED = "generation_queued"
    GENERATION_DONE = "generation_done"


class SyncGenerationStatus:
    """Sync generation tracking status."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class SyncTrainingStatus:
    """Sync training tracking status."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    ADAPTER_LOADED = "adapter_loaded"
    ADAPTER_UNLOADED = "adapter_unloaded"
    ADAPTER_LOAD_FAILED = "adapter_load_failed"
    ADAPTER_FAILED = "adapter_failed"


class TrainingTargetStatus:
    """Training target status within a sync task."""
    IDLE = "idle"
    READY = "ready"
    TRAINING = "training"
    LOADING_ADAPTER = "loading_adapter"
    ERROR = "error"

