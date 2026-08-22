"""Dataset status enumerations."""

from enum import Enum


class DatasetStatus(str, Enum):
    """Dataset processing status."""

    REGISTERED = "registered"     # Dataset metadata registered, files may not be ready
    UPLOADING = "uploading"       # Dataset file is being uploaded
    DOWNLOADING = "downloading"   # Dataset is being downloaded from remote
    PROCESSING = "processing"     # Dataset is being processed
    STAGING = "staging"           # Internal generated product; not consumable
    READY = "ready"               # Dataset is ready for use
    ERROR = "error"               # Dataset processing failed
    ARCHIVED = "archived"         # Dataset is archived
    DELETING = "deleting"         # Durable cleanup is in progress
