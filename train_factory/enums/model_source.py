"""Model source related enumerations."""

from enum import Enum


class ModelSourceType(str, Enum):
    """Model source type - where the model comes from."""

    TRAINED = "trained"         # Trained locally
    DOWNLOADED = "downloaded"   # Downloaded from remote (ModelScope/HuggingFace)
    UPLOADED = "uploaded"       # Uploaded by user


class DownloadStatus(str, Enum):
    """Model download status."""

    PENDING = "pending"           # Waiting to start download
    DOWNLOADING = "downloading"   # Currently downloading
    COMPLETED = "completed"       # Download completed successfully
    FAILED = "failed"             # Download failed
