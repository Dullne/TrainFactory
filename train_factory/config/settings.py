"""
TrainFactory Settings Module.

Centralized configuration management using pydantic-settings.
Supports environment variables, .env files, and YAML configuration.
"""

import os
from functools import lru_cache
from ipaddress import ip_network
from pathlib import Path
from threading import RLock
from typing import Optional

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .public_origin import (
    is_loopback_bind_address,
    is_loopback_public_origin,
    parse_public_origin,
)
from .secret_files import resolve_secret

ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_ENV_FILE = ROOT_DIR / ".env"

SINGLE_API_WORKER_ERROR = (
    "TrainFactory requires exactly one API worker because background sync "
    "managers run in-process; multi-process API mode is not supported."
)


def validate_api_worker_count(value: int) -> int:
    """Reject process counts that would duplicate in-process background managers."""
    if value != 1:
        raise ValueError(SINGLE_API_WORKER_ERROR)
    return value


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # String/repr output is safe. Structured callers must also pass
        # include_input=False when serializing Pydantic validation details.
        hide_input_in_errors=True,
        populate_by_name=True,
    )

    # === Database Settings ===
    mysql_url: str = Field(
        default="mysql+pymysql://trainfactory_app@localhost:3306/train_factory",
        description="MySQL connection URL",
        exclude=True,
        repr=False,
    )
    mysql_url_file: Optional[Path] = Field(
        default=None,
        validation_alias="MYSQL_URL_FILE",
        exclude=True,
        repr=False,
        description="File containing the MySQL connection URL",
    )
    db_pool_size: int = Field(default=5, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, description="Max overflow connections")
    db_echo: bool = Field(default=False, description="Echo SQL statements")
    db_auto_migrate: bool = Field(
        default=True,
        description="Auto-run Alembic migrations on startup",
    )

    # === Storage Paths ===
    base_dir: Path = Field(
        default_factory=lambda: Path("/app"),
        description="Base directory for the application",
    )
    training_cache: Path = Field(
        default_factory=lambda: Path("/app/cache"),
        description="Cache directory for training artifacts",
    )
    models_dir: Path = Field(
        default_factory=lambda: Path("/app/models"),
        description="Directory for storing models",
    )
    datasets_dir: Path = Field(
        default_factory=lambda: Path("/app/data/datasets"),
        description="Directory for storing datasets",
    )
    output_dir: Path = Field(
        default_factory=lambda: Path("/app/output"),
        description="Default output directory for training results",
    )
    path_mappings: str = Field(
        default="",
        description="Comma-separated host:container path mappings for local registrations",
    )

    # === HuggingFace Settings ===
    hf_home: Optional[str] = Field(
        default=None,
        description="HuggingFace home directory (HF_HOME)",
    )
    hf_datasets_cache: Optional[str] = Field(
        default=None,
        description="HuggingFace datasets cache directory",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API token",
    )

    # === ModelScope Settings ===
    modelscope_cache: Optional[str] = Field(
        default=None,
        description="ModelScope cache directory",
    )

    # === Remote Download Limits ===
    download_max_concurrent_global: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum concurrent remote downloads across the API process",
    )
    download_max_concurrent_per_user: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Maximum concurrent remote downloads for one user",
    )
    model_download_max_bytes: int = Field(
        default=30 * 1024**3,
        ge=1,
        description="Maximum remote model repository size in bytes",
    )
    dataset_download_max_bytes: int = Field(
        default=10 * 1024**3,
        ge=1,
        description="Maximum remote dataset repository size in bytes",
    )
    download_storage_max_bytes_global: int = Field(
        default=200 * 1024**3,
        ge=1,
        description="Maximum persisted bytes for all remote downloads",
    )
    download_storage_max_bytes_per_user: int = Field(
        default=60 * 1024**3,
        ge=1,
        description="Maximum persisted remote-download bytes for one user",
    )
    download_metadata_timeout_seconds: int = Field(
        default=15,
        ge=1,
        le=120,
        description="Timeout for remote repository metadata preflight requests",
    )
    download_max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum file-transfer workers used by one remote download",
    )

    # === Long-running API Task Limits ===
    background_task_max_active_global: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum active training, evaluation, and generation jobs globally",
    )
    background_task_max_active_per_user: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Maximum active training, evaluation, and generation jobs per user",
    )

    # === External Sync Resource Limits ===
    sync_storage_max_bytes_global: int = Field(
        default=20 * 1024**3,
        ge=1,
        description="Maximum persisted bytes below the external sync data root",
    )
    sync_storage_max_bytes_per_user: int = Field(
        default=5 * 1024**3,
        ge=1,
        description="Maximum persisted external sync bytes for one user",
    )
    sync_pending_max_batches_per_task: int = Field(
        default=1_000,
        ge=1,
        description="Maximum unconsumed external sync batches for one task",
    )
    sync_pending_max_records_per_task: int = Field(
        default=1_000_000,
        ge=1,
        description="Maximum unconsumed external sync records for one task",
    )
    sync_generation_max_input_bytes: int = Field(
        default=1024**3,
        ge=1,
        description="Maximum merged input bytes for one sync generation",
    )
    sync_max_record_bytes: int = Field(
        default=8 * 1024**2,
        ge=1,
        description="Maximum UTF-8 bytes for one external sync JSONL record",
    )
    sync_max_future_skew_seconds: int = Field(
        default=300,
        ge=0,
        le=86400,
        description="Maximum accepted future clock skew for external records",
    )
    sync_boundary_max_ids: int = Field(
        default=10_000,
        ge=1,
        description="Maximum deduplication keys persisted for one sync boundary",
    )
    sync_boundary_max_bytes: int = Field(
        default=4 * 1024**2,
        ge=1,
        description="Maximum encoded bytes persisted for sync boundary keys",
    )
    sync_historical_max_docs: int = Field(
        default=100_000,
        ge=1,
        description="Maximum documents materialized for adapter history backfill",
    )
    sync_historical_max_bytes: int = Field(
        default=64 * 1024**2,
        ge=1,
        description="Maximum raw history bytes materialized for adapter backfill",
    )

    # === Training Settings ===
    default_train_epochs: int = Field(default=3, description="Default number of training epochs")
    default_batch_size: int = Field(default=16, description="Default batch size per device")
    default_learning_rate: float = Field(default=2e-5, description="Default learning rate")
    default_warmup_ratio: float = Field(default=0.1, description="Default warmup ratio")
    training_allow_cpu_fallback: bool = Field(
        default=False,
        description=(
            "Allow automatic training to fall back to CPU when no GPU is available; "
            "disabled by default to prevent unbounded host resource use"
        ),
    )
    allow_model_remote_code: bool = Field(
        default=False,
        description=(
            "Allow model repositories to execute custom Python code while loading; "
            "set only through trusted operational configuration"
        ),
    )

    # === API Settings ===
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of API workers")
    debug: bool = Field(default=False, description="Enable debug mode (exposes detailed errors)")

    # === Logging Settings ===
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru format string",
    )
    app_timezone: str = Field(
        default="UTC",
        description="Application timezone name (IANA), used when generating naive timestamps",
    )

    # === SwanLab Settings (Optional) ===
    swanlab_api_key: Optional[str] = Field(default=None, description="SwanLab API key")
    swanlab_project: Optional[str] = Field(default="train-factory", description="SwanLab project name")

    # === Xinference Settings ===
    xinference_endpoint: Optional[str] = Field(
        default=None,
        description="Default Xinference service endpoint (defaults to http://xinference:9997 for Docker)"
    )
    xinference_default_replica: int = Field(default=1, description="Default replica count")
    xinference_default_gpu_memory: float = Field(default=0.9, description="Default GPU memory utilization")
    xinference_request_timeout: int = Field(default=120, description="Xinference API request timeout (seconds)")

    # === JWT Authentication Settings ===
    auth_enabled: bool = Field(default=True, description="Enable JWT authentication")
    jwt_secret_key: str = Field(
        default="dev-only-secret-key-must-change-in-production",
        description="Secret key for JWT (MUST set via JWT_SECRET_KEY env var in production)",
        exclude=True,
        repr=False,
    )
    jwt_secret_key_file: Optional[Path] = Field(
        default=None,
        validation_alias="JWT_SECRET_KEY_FILE",
        exclude=True,
        repr=False,
        description="File containing the JWT signing secret",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=1440,  # 24 hours
        gt=0,
        description="Access token expiration time in minutes"
    )
    auth_cookie_secure: bool = Field(
        default=False,
        description="Require HTTPS when sending the authentication cookie",
    )
    host_bind_address: str = Field(
        default="127.0.0.1",
        validation_alias="HOST_BIND_ADDRESS",
        description="Host address used for all published container ports",
    )
    public_base_url: AnyHttpUrl = Field(
        default="http://localhost:3000",
        validation_alias="PUBLIC_BASE_URL",
        validate_default=True,
        description="Canonical externally visible HTTP(S) origin",
    )
    allow_insecure_cookie_non_loopback_for_testing: bool = Field(
        default=False,
        validation_alias="TEST_ALLOW_INSECURE_AUTH_COOKIE_NON_LOOPBACK",
        description="Debug-only transport override for isolated tests",
    )
    default_admin_username: str = Field(
        default="admin",
        min_length=3,
        max_length=64,
        description="Username for the bootstrapped administrator",
    )
    default_admin_password: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Password for the bootstrapped administrator",
        exclude=True,
        repr=False,
    )
    default_admin_password_file: Optional[Path] = Field(
        default=None,
        validation_alias="DEFAULT_ADMIN_PASSWORD_FILE",
        exclude=True,
        repr=False,
        description="File containing the bootstrapped administrator password",
    )
    default_admin_email: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Email for the bootstrapped administrator",
    )
    self_registration_enabled: bool = Field(
        default=False,
        description="Allow users to create their own accounts",
    )

    # === Storage Backend ===
    storage_backend: str = Field(
        default="local",
        description="Default storage backend for datasets: 'local' or 's3'",
    )
    max_upload_size: int = Field(
        default=500 * 1024 * 1024,  # 500MB
        description="Maximum file upload size in bytes",
    )

    # === Storage Backend ===
    storage_backend: str = Field(
        default="local",
        description="Default storage backend for datasets: 'local' or 's3'",
    )
    max_upload_size: int = Field(
        default=500 * 1024 * 1024,  # 500MB
        description="Maximum file upload size in bytes",
    )

    # === MinIO / S3 Object Storage ===
    minio_endpoint: str = Field(
        default="minio:9000",
        description="MinIO/S3 endpoint (host:port)",
    )
    minio_access_key: Optional[str] = Field(
        default=None,
        description="MinIO/S3 access key",
    )
    minio_secret_key: Optional[str] = Field(
        default=None,
        description="MinIO/S3 secret key",
    )
    minio_bucket: str = Field(
        default="trainfactory",
        description="Default bucket for dataset storage",
    )
    minio_secure: bool = Field(
        default=False,
        description="Use HTTPS for MinIO connections",
    )
    local_cache_dir: Path = Field(
        default_factory=lambda: Path("/app/cache/datasets"),
        description="Local cache directory for datasets downloaded from object storage",
    )

    # === CORS Settings ===
    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins"
    )

    # === Security Settings ===
    rate_limit_enabled: bool = Field(default=True, description="Enable API rate limiting")
    rate_limit_login: str = Field(default="5/minute", description="Rate limit for login endpoint")
    rate_limit_register: str = Field(default="3/hour", description="Rate limit for register endpoint")
    rate_limit_default: str = Field(default="100/minute", description="Default rate limit for API endpoints")
    audit_log_retention_days: int = Field(
        default=90,
        ge=1,
        description="Number of days to retain audit log records",
    )
    audit_log_cleanup_interval_hours: int = Field(
        default=24,
        ge=1,
        description="Interval between audit log retention cleanups",
    )
    rate_limit_trusted_proxies: str = Field(
        default="",
        description="Comma-separated proxy IP addresses or CIDRs trusted to supply X-Forwarded-For",
    )

    @field_validator("api_workers")
    @classmethod
    def validate_api_workers(cls, value: int) -> int:
        return validate_api_worker_count(value)

    @field_validator("host_bind_address")
    @classmethod
    def validate_host_bind_address(cls, value: str) -> str:
        is_loopback_bind_address(value, "HOST_BIND_ADDRESS")
        return value

    @field_validator("public_base_url", mode="before")
    @classmethod
    def validate_public_base_url(cls, value: object) -> str:
        return parse_public_origin(value, "PUBLIC_BASE_URL")

    @field_validator("rate_limit_trusted_proxies")
    @classmethod
    def validate_rate_limit_trusted_proxies(cls, value: str) -> str:
        networks = []
        for entry in (item.strip() for item in value.split(",")):
            if not entry:
                continue
            try:
                network = ip_network(entry, strict=False)
            except ValueError as exc:
                raise ValueError(f"Invalid trusted proxy IP or CIDR: {entry}") from exc
            if network.prefixlen == 0:
                raise ValueError("Universal trusted proxy networks are not allowed")
            networks.append(str(network))
        return ",".join(networks)

    @model_validator(mode="after")
    def resolve_file_backed_secrets(self):
        secret_fields = (
            ("mysql_url", "mysql_url_file", "MYSQL_URL"),
            ("jwt_secret_key", "jwt_secret_key_file", "JWT_SECRET_KEY"),
            (
                "default_admin_password",
                "default_admin_password_file",
                "DEFAULT_ADMIN_PASSWORD",
            ),
        )
        for value_field, file_field, setting_name in secret_fields:
            direct_value = (
                getattr(self, value_field)
                if value_field in self.model_fields_set
                else None
            )
            resolved = resolve_secret(
                direct_value=direct_value,
                file_path=getattr(self, file_field),
                setting_name=setting_name,
            )
            if resolved is not None:
                if value_field == "default_admin_password" and len(resolved) > 128:
                    raise ValueError("DEFAULT_ADMIN_PASSWORD has invalid length")
                setattr(self, value_field, resolved)
        return self

    @model_validator(mode="after")
    def validate_download_concurrency(self):
        if self.download_max_concurrent_per_user > self.download_max_concurrent_global:
            raise ValueError(
                "Remote download per-user concurrency limit cannot exceed global limit"
            )
        return self

    @model_validator(mode="after")
    def validate_download_storage_quota(self):
        if (
            self.download_storage_max_bytes_per_user
            > self.download_storage_max_bytes_global
        ):
            raise ValueError(
                "Remote download per-user storage quota cannot exceed global quota"
            )
        return self

    @model_validator(mode="after")
    def validate_background_task_concurrency(self):
        if (
            self.background_task_max_active_per_user
            > self.background_task_max_active_global
        ):
            raise ValueError(
                "Background task per-user active limit cannot exceed global limit"
            )
        return self

    @model_validator(mode="after")
    def validate_sync_storage_quota(self):
        if self.sync_storage_max_bytes_per_user > self.sync_storage_max_bytes_global:
            raise ValueError(
                "External sync per-user storage quota cannot exceed global sync storage quota"
            )
        if (
            self.sync_generation_max_input_bytes
            > self.sync_storage_max_bytes_per_user
        ):
            raise ValueError(
                "External sync generation input limit cannot exceed the per-user "
                "sync storage quota"
            )
        if self.sync_max_record_bytes > self.sync_historical_max_bytes:
            raise ValueError(
                "External sync record limit cannot exceed the history window byte limit"
            )
        if self.sync_max_record_bytes > self.sync_generation_max_input_bytes:
            raise ValueError(
                "External sync record limit cannot exceed the generation input limit"
            )
        return self

    @model_validator(mode="after")
    def validate_s3_credentials(self):
        if self.storage_backend == "s3" and (
            not self.minio_access_key or not self.minio_secret_key
        ):
            raise ValueError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required when "
                "STORAGE_BACKEND=s3"
            )
        return self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_security_settings()
        self._setup_directories()
        self._setup_environment()

    def _check_security_settings(self):
        """Check security settings and refuse insecure production configurations."""
        import warnings

        loopback_bind = is_loopback_bind_address(
            self.host_bind_address,
            "HOST_BIND_ADDRESS",
        )
        public_origin = parse_public_origin(
            str(self.public_base_url),
            "PUBLIC_BASE_URL",
        )
        public_scheme = public_origin.split(":", 1)[0]
        public_origin_is_loopback = is_loopback_public_origin(public_origin)

        if self.allow_insecure_cookie_non_loopback_for_testing and not self.debug:
            raise RuntimeError(
                "TEST_ALLOW_INSECURE_AUTH_COOKIE_NON_LOOPBACK requires debug mode."
            )

        if self.auth_enabled:
            override_is_valid = (
                self.allow_insecure_cookie_non_loopback_for_testing
                and self.debug
                and not (loopback_bind and public_origin_is_loopback)
                and public_scheme == "http"
                and not self.auth_cookie_secure
            )
            transport_is_valid = (
                public_scheme == "http"
                and loopback_bind
                and public_origin_is_loopback
                and not self.auth_cookie_secure
            ) or (public_scheme == "https" and self.auth_cookie_secure)
            if not (override_is_valid or transport_is_valid):
                raise RuntimeError(
                    "Authentication deployment transport policy is invalid."
                )

        # Check JWT secret key
        public_placeholder_secrets = {
            "dev-only-secret-key-must-change-in-production",
            "dev-secret-change-me",
            "change-me-to-random-string",
        }
        if self.jwt_secret_key in public_placeholder_secrets:
            if not self.debug:
                # Refuse to start in production with the public default secret:
                # anyone who reads the repo can forge arbitrary user identities.
                # Debug mode still only warns so local dev without JWT_SECRET_KEY
                # keeps working.
                raise RuntimeError(
                    "JWT_SECRET_KEY must be set to a secure random value in "
                    "non-debug mode (current value is the public default)."
                )
            warnings.warn(
                "SECURITY WARNING: Using default JWT secret key in debug mode. "
                "Set JWT_SECRET_KEY for production use.",
                UserWarning,
                stacklevel=2,
            )
        if (
            not self.debug
            and len(self.jwt_secret_key.strip().encode("utf-8")) < 32
        ):
            raise RuntimeError(
                "JWT_SECRET_KEY must contain at least 32 bytes in non-debug mode."
            )

        # Check CORS settings
        if "*" in self.allowed_origins:
            if self.auth_enabled and not self.debug:
                raise RuntimeError(
                    "ALLOWED_ORIGINS cannot contain '*' when authentication is "
                    "enabled in non-debug mode."
                )
            warnings.warn(
                "SECURITY WARNING: CORS allows all origins (*). "
                "Set ALLOWED_ORIGINS to specific domains in production.",
                UserWarning,
                stacklevel=2
            )

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for path_attr in ["training_cache", "models_dir", "datasets_dir", "output_dir", "local_cache_dir"]:
            path = getattr(self, path_attr)
            if path:
                path.mkdir(parents=True, exist_ok=True)

    def _setup_environment(self):
        """Set up environment variables for external libraries."""
        # HuggingFace
        if self.hf_home:
            os.environ.setdefault("HF_HOME", str(self.hf_home))
        else:
            os.environ.setdefault("HF_HOME", str(self.training_cache / "hf_home"))

        if self.hf_datasets_cache:
            os.environ.setdefault("HF_DATASETS_CACHE", str(self.hf_datasets_cache))
        else:
            os.environ.setdefault("HF_DATASETS_CACHE", str(self.training_cache / "hf_datasets"))

        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)

        # ModelScope
        if self.modelscope_cache:
            os.environ.setdefault("MODELSCOPE_CACHE", str(self.modelscope_cache))
        else:
            os.environ.setdefault("MODELSCOPE_CACHE", str(self.training_cache / "modelscope"))

    @property
    def hf_cache_dir(self) -> str:
        """Get HuggingFace datasets cache directory."""
        return os.environ.get("HF_DATASETS_CACHE", str(self.training_cache / "hf_datasets"))

    def get_task_output_dir(self, task_id: str) -> Path:
        """Get output directory for a specific training task."""
        return self.output_dir / task_id


_SETTINGS_CACHE_LOCK = RLock()


@lru_cache()
def _get_settings_cached() -> Settings:
    return Settings()


def get_settings() -> Settings:
    """Get cached settings instance."""
    with _SETTINGS_CACHE_LOCK:
        return _get_settings_cached()


def _clear_settings_cache() -> None:
    with _SETTINGS_CACHE_LOCK:
        _get_settings_cached.cache_clear()


def _settings_cache_info():
    with _SETTINGS_CACHE_LOCK:
        return _get_settings_cached.cache_info()


get_settings.cache_clear = _clear_settings_cache  # type: ignore[attr-defined]
get_settings.cache_info = _settings_cache_info  # type: ignore[attr-defined]
