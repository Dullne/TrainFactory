"""Tests for security-sensitive Settings validation (JWT secret guard)."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from threading import Barrier, Event, Lock
import time

import pytest
from pydantic import AnyHttpUrl, ValidationError

from train_factory.config.settings import Settings

PUBLIC_PLACEHOLDER_SECRETS = (
    "dev-only-secret-key-must-change-in-production",
    "dev-secret-change-me",
    "change-me-to-random-string",
)


def _transport_settings(tmp_path, **overrides):
    values = {
        "_env_file": None,
        "debug": False,
        "auth_enabled": True,
        "jwt_secret_key": "test-only-secret-that-is-not-published",
        "host_bind_address": "127.0.0.1",
        "public_base_url": "http://localhost:3000",
        "auth_cookie_secure": False,
        "training_cache": tmp_path / "cache",
        "models_dir": tmp_path / "models",
        "datasets_dir": tmp_path / "datasets",
        "output_dir": tmp_path / "output",
        "local_cache_dir": tmp_path / "local-cache",
    }
    values.update(overrides)
    return Settings(**values)


def _parse_public_origin(value, setting_name):
    from train_factory.config.public_origin import parse_public_origin

    return parse_public_origin(value, setting_name)


@pytest.mark.parametrize(
    ("bind", "public_url", "secure_cookie"),
    (
        ("127.0.0.1", "http://localhost:3000", False),
        ("127.23.45.67", "http://localhost:3000", False),
        ("::1", "http://[::1]:3000", False),
        ("LOCALHOST", "http://localhost:3000", False),
        ("0.0.0.0", "https://train.example", True),
    ),
)
def test_authenticated_transport_accepts_safe_bind_url_cookie_matrix(
    tmp_path,
    bind,
    public_url,
    secure_cookie,
):
    settings = _transport_settings(
        tmp_path,
        host_bind_address=bind,
        public_base_url=public_url,
        auth_cookie_secure=secure_cookie,
    )

    assert settings.host_bind_address == bind
    assert str(settings.public_base_url).rstrip("/") == public_url
    assert settings.auth_cookie_secure is secure_cookie


def test_public_base_url_default_is_validated_as_any_http_url(tmp_path):
    settings = Settings(
        _env_file=None,
        debug=False,
        jwt_secret_key="test-only-secret-that-is-not-published",
        training_cache=tmp_path / "cache",
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        output_dir=tmp_path / "output",
        local_cache_dir=tmp_path / "local-cache",
    )

    assert isinstance(settings.public_base_url, AnyHttpUrl)
    assert str(settings.public_base_url) == "http://localhost:3000/"


@pytest.mark.parametrize(
    ("bind", "public_url", "secure_cookie"),
    (
        ("0.0.0.0", "http://host:3000", False),
        ("192.168.1.10", "http://host:3000", False),
        ("0.0.0.0", "http://host:3000", True),
        ("127.0.0.1", "https://train.example", False),
    ),
)
def test_authenticated_transport_rejects_unsafe_bind_url_cookie_matrix(
    tmp_path,
    bind,
    public_url,
    secure_cookie,
):
    with pytest.raises(RuntimeError, match="deployment transport policy"):
        _transport_settings(
            tmp_path,
            host_bind_address=bind,
            public_base_url=public_url,
            auth_cookie_secure=secure_cookie,
        )


def test_loopback_bind_rejects_public_http_origin(tmp_path):
    with pytest.raises(RuntimeError, match="deployment transport policy"):
        _transport_settings(
            tmp_path,
            host_bind_address="127.0.0.1",
            public_base_url="http://public.example:3000",
            auth_cookie_secure=False,
        )


def test_auth_disabled_still_validates_public_url_but_does_not_require_cookie(
    tmp_path,
):
    settings = _transport_settings(
        tmp_path,
        auth_enabled=False,
        host_bind_address="0.0.0.0",
        public_base_url="http://public.example:3000",
        auth_cookie_secure=False,
    )

    assert settings.auth_enabled is False


@pytest.mark.parametrize(
    "invalid_url",
    (
        "ftp://example.com",
        "https://user:password@example.com",
        "https://example.com/private",
    ),
)
def test_auth_disabled_rejects_invalid_public_url(tmp_path, invalid_url):
    with pytest.raises(ValidationError, match="PUBLIC_BASE_URL"):
        _transport_settings(
            tmp_path,
            auth_enabled=False,
            public_base_url=invalid_url,
        )


def test_non_loopback_insecure_cookie_override_requires_debug(tmp_path):
    with pytest.raises(RuntimeError, match="TEST_ALLOW_INSECURE"):
        _transport_settings(
            tmp_path,
            debug=False,
            allow_insecure_cookie_non_loopback_for_testing=True,
        )

    settings = _transport_settings(
        tmp_path,
        debug=True,
        host_bind_address="0.0.0.0",
        public_base_url="http://public.example:3000",
        auth_cookie_secure=False,
        allow_insecure_cookie_non_loopback_for_testing=True,
    )
    assert settings.allow_insecure_cookie_non_loopback_for_testing is True


def test_testing_override_does_not_allow_http_with_secure_cookie(tmp_path):
    with pytest.raises(RuntimeError, match="deployment transport policy"):
        _transport_settings(
            tmp_path,
            debug=True,
            host_bind_address="0.0.0.0",
            public_base_url="http://public.example:3000",
            auth_cookie_secure=True,
            allow_insecure_cookie_non_loopback_for_testing=True,
        )


@pytest.mark.parametrize(
    "invalid_bind",
    ("::ffff:127.0.0.1", "127.0.0.1/8", "internal-hostname"),
)
def test_host_bind_address_rejects_ambiguous_values_without_echo(
    tmp_path,
    invalid_bind,
):
    with pytest.raises(ValidationError) as exc_info:
        _transport_settings(
            tmp_path,
            host_bind_address=invalid_bind,
            public_base_url="https://train.example",
            auth_cookie_secure=True,
        )

    message = str(exc_info.value)
    assert "HOST_BIND_ADDRESS" in message
    assert invalid_bind not in message


def test_transport_rejection_occurs_before_directory_and_environment_side_effects(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("HF_HOME", raising=False)

    with pytest.raises(RuntimeError, match="deployment transport policy"):
        _transport_settings(
            tmp_path,
            host_bind_address="0.0.0.0",
            public_base_url="http://host:3000",
            auth_cookie_secure=False,
            hf_home="transport-policy-rejected-cache",
        )

    for name in ("cache", "models", "datasets", "output", "local-cache"):
        assert not (tmp_path / name).exists()
    assert "HF_HOME" not in os.environ


def test_settings_validation_hides_public_url_and_jwt_inputs(tmp_path):
    public_canary = "public-origin-canary.invalid"
    jwt_canary = "short-jwt-canary"

    with pytest.raises((ValidationError, RuntimeError)) as exc_info:
        _transport_settings(
            tmp_path,
            public_base_url=f"https://user:password@{public_canary}",
            jwt_secret_key=jwt_canary,
        )

    message = str(exc_info.value)
    assert public_canary not in message
    assert jwt_canary not in message
    if isinstance(exc_info.value, ValidationError):
        safe_errors = json.dumps(
            exc_info.value.errors(include_input=False),
            default=str,
        )
        safe_json = exc_info.value.json(include_input=False)
        assert public_canary not in safe_errors
        assert jwt_canary not in safe_errors
        assert public_canary not in safe_json
        assert jwt_canary not in safe_json


@pytest.mark.parametrize(
    ("value", "category"),
    (
        ("ftp://example.com", "scheme"),
        ("http://bad host", "syntax"),
        ("http://host\\private", "syntax"),
        ("https://user:password@example.com", "credentials"),
        ("https://example.com?private=query", "query"),
        ("https://example.com?", "query"),
        ("https://example.com#private-fragment", "fragment"),
        ("https://example.com#", "fragment"),
        ("https://example.com/private/path", "path"),
        ("HTTPS://Example.COM", "canonical"),
    ),
)
def test_public_origin_rejects_unsafe_or_noncanonical_values_without_echo(
    value,
    category,
):
    with pytest.raises(ValueError) as exc_info:
        _parse_public_origin(value, "PUBLIC_BASE_URL")

    message = str(exc_info.value)
    assert message == f"PUBLIC_BASE_URL has invalid {category}"
    assert value not in message
    assert exc_info.value.__cause__ is None


def test_public_origin_normalizes_only_a_root_trailing_slash():
    assert (
        _parse_public_origin("https://train.example/", "PUBLIC_BASE_URL")
        == "https://train.example"
    )


def test_get_settings_constructs_one_singleton_on_concurrent_first_access(
    monkeypatch,
):
    import train_factory.config as config

    settings_module = importlib.import_module("train_factory.config.settings")
    settings_module.get_settings.cache_clear()
    barrier = Barrier(12)
    construction_lock = Lock()
    construction_count = 0

    class FakeSettings:
        def __init__(self):
            nonlocal construction_count
            with construction_lock:
                construction_count += 1
            time.sleep(0.05)

    monkeypatch.setattr(settings_module, "Settings", FakeSettings)

    def load_settings(_index):
        barrier.wait(timeout=2)
        return config.settings

    try:
        with ThreadPoolExecutor(max_workers=12) as executor:
            instances = list(executor.map(load_settings, range(12)))

        assert construction_count == 1
        assert len({id(instance) for instance in instances}) == 1
        assert settings_module.get_settings.cache_info().currsize == 1
    finally:
        settings_module.get_settings.cache_clear()


def test_get_settings_cache_clear_waits_for_inflight_construction(monkeypatch):
    settings_module = importlib.import_module("train_factory.config.settings")
    settings_module.get_settings.cache_clear()
    construction_started = Event()
    allow_construction = Event()
    clear_finished = Event()

    class BlockingSettings:
        def __init__(self):
            construction_started.set()
            assert allow_construction.wait(timeout=2)

    monkeypatch.setattr(settings_module, "Settings", BlockingSettings)

    def clear_cache():
        settings_module.get_settings.cache_clear()
        clear_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            construction = executor.submit(settings_module.get_settings)
            assert construction_started.wait(timeout=2)
            clearing = executor.submit(clear_cache)
            assert not clear_finished.wait(timeout=0.1)
            allow_construction.set()
            construction.result(timeout=2)
            clearing.result(timeout=2)

        assert settings_module.get_settings.cache_info().currsize == 0
    finally:
        allow_construction.set()
        settings_module.get_settings.cache_clear()


@pytest.mark.parametrize("secret", PUBLIC_PLACEHOLDER_SECRETS)
def test_default_jwt_secret_raises_in_production(secret):
    """Non-debug + public default secret must refuse to construct Settings."""
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        Settings(_env_file=None, debug=False, jwt_secret_key=secret)


@pytest.mark.parametrize("secret", PUBLIC_PLACEHOLDER_SECRETS)
def test_default_jwt_secret_allowed_in_debug(secret):
    """Debug mode keeps the legacy warn-only behavior for local dev."""
    with pytest.warns(UserWarning, match="default JWT secret"):
        s = Settings(_env_file=None, debug=True, jwt_secret_key=secret)
    assert s.jwt_secret_key == secret


def test_custom_secret_allowed_in_production():
    """Non-debug with a real secret constructs normally."""
    secret = "a-real-random-secret-that-is-not-published-in-the-repository"
    s = Settings(_env_file=None, debug=False, jwt_secret_key=secret)
    assert s.jwt_secret_key == secret


@pytest.mark.parametrize(
    "secret",
    (
        " " * 40,
        "x",
        "x" * 31,
        "\u5bc6" * 10,
    ),
)
def test_short_jwt_secret_raises_in_production(tmp_path, secret):
    """Production JWT secrets must contain at least 32 UTF-8 bytes."""
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key=secret,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )


def test_jwt_secret_minimum_is_measured_in_utf8_bytes(tmp_path):
    secret = "\u5bc6" * 11

    settings = Settings(
        _env_file=None,
        debug=False,
        jwt_secret_key=secret,
        training_cache=tmp_path / "cache",
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        output_dir=tmp_path / "output",
        local_cache_dir=tmp_path / "local-cache",
    )

    assert settings.jwt_secret_key == secret


def test_auth_bootstrap_settings_have_safe_defaults():
    fields = Settings.model_fields

    assert fields["default_admin_username"].default == "admin"
    assert fields["default_admin_password"].default is None
    assert fields["default_admin_email"].default is None
    assert fields["self_registration_enabled"].default is False


def test_rate_limit_proxy_settings_have_safe_defaults():
    fields = Settings.model_fields

    assert fields["rate_limit_enabled"].default is True
    assert fields["rate_limit_trusted_proxies"].default == ""


def test_authenticated_production_rejects_wildcard_cors(tmp_path):
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        Settings(
            _env_file=None,
            debug=False,
            auth_enabled=True,
            jwt_secret_key="test-only-secret-that-is-not-published",
            allowed_origins="https://app.example,*",
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )


def test_debug_mode_keeps_wildcard_cors_warning(tmp_path):
    with pytest.warns(UserWarning, match="CORS allows all origins"):
        settings = Settings(
            _env_file=None,
            debug=True,
            auth_enabled=True,
            jwt_secret_key="debug-secret",
            allowed_origins="*",
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )

    assert settings.allowed_origins == "*"


def test_api_worker_setting_defaults_to_single_process():
    assert Settings.model_fields["api_workers"].default == 1


def test_sync_resource_limits_have_bounded_defaults():
    fields = Settings.model_fields

    assert fields["sync_storage_max_bytes_global"].default == 20 * 1024**3
    assert fields["sync_storage_max_bytes_per_user"].default == 5 * 1024**3
    assert fields["sync_pending_max_batches_per_task"].default == 1_000
    assert fields["sync_pending_max_records_per_task"].default == 1_000_000
    assert fields["sync_generation_max_input_bytes"].default == 1024**3
    assert fields["sync_max_record_bytes"].default == 8 * 1024**2
    assert fields["sync_max_future_skew_seconds"].default == 300
    assert fields["sync_boundary_max_ids"].default == 10_000
    assert fields["sync_boundary_max_bytes"].default == 4 * 1024**2
    assert fields["sync_historical_max_docs"].default == 100_000
    assert fields["sync_historical_max_bytes"].default == 64 * 1024**2


def test_sync_per_user_storage_quota_cannot_exceed_global_quota(tmp_path):
    with pytest.raises(ValidationError, match="sync storage quota"):
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            sync_storage_max_bytes_global=10,
            sync_storage_max_bytes_per_user=11,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )


def test_sync_generation_input_cannot_exceed_per_user_storage_quota(tmp_path):
    with pytest.raises(ValidationError, match="generation input limit"):
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            sync_storage_max_bytes_global=20,
            sync_storage_max_bytes_per_user=10,
            sync_generation_max_input_bytes=11,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )


@pytest.mark.parametrize("workers", (0, 2, 8))
def test_api_worker_setting_rejects_unsupported_process_counts(tmp_path, workers):
    with pytest.raises(ValidationError, match="exactly one API worker"):
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            api_workers=workers,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )


def test_object_store_credentials_have_no_public_defaults():
    fields = Settings.model_fields

    assert fields["minio_access_key"].default is None
    assert fields["minio_secret_key"].default is None


@pytest.mark.parametrize(
    ("access_key", "secret_key"),
    ((None, None), ("", ""), ("access", None), (None, "secret")),
)
def test_s3_storage_requires_both_object_store_credentials(
    tmp_path,
    access_key,
    secret_key,
):
    with pytest.raises(ValidationError, match="MINIO_ACCESS_KEY and MINIO_SECRET_KEY"):
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            storage_backend="s3",
            minio_access_key=access_key,
            minio_secret_key=secret_key,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )


def test_local_storage_does_not_require_object_store_credentials(tmp_path):
    settings = Settings(
        _env_file=None,
        debug=False,
        jwt_secret_key="test-only-secret-that-is-not-published",
        storage_backend="local",
        minio_access_key=None,
        minio_secret_key=None,
        training_cache=tmp_path / "cache",
        models_dir=tmp_path / "models",
        datasets_dir=tmp_path / "datasets",
        output_dir=tmp_path / "output",
        local_cache_dir=tmp_path / "local-cache",
    )

    assert settings.minio_access_key is None
    assert settings.minio_secret_key is None


@pytest.mark.parametrize(
    "trusted_proxies",
    ("not-a-cidr", "0.0.0.0/0", "::/0"),
)
def test_rate_limit_proxy_settings_reject_invalid_or_universal_networks(
    tmp_path,
    trusted_proxies,
):
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            rate_limit_trusted_proxies=trusted_proxies,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )

    assert exc_info.value.errors()[0]["loc"] == ("rate_limit_trusted_proxies",)


def test_auth_bootstrap_settings_define_length_constraints():
    fields = Settings.model_fields
    username_metadata = fields["default_admin_username"].metadata
    password_metadata = fields["default_admin_password"].metadata
    email_metadata = fields["default_admin_email"].metadata

    assert any(
        getattr(item, "min_length", None) == 3 for item in username_metadata
    )
    assert any(
        getattr(item, "max_length", None) == 64 for item in username_metadata
    )
    assert not any(hasattr(item, "min_length") for item in password_metadata)
    assert any(
        getattr(item, "max_length", None) == 128 for item in password_metadata
    )
    assert any(
        getattr(item, "max_length", None) == 256 for item in email_metadata
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("default_admin_username", "ab"),
        ("default_admin_username", "a" * 65),
        ("default_admin_password", "a" * 129),
        ("default_admin_email", "a" * 257),
    ),
)
def test_auth_bootstrap_settings_reject_invalid_lengths(field_name, value):
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            **{field_name: value},
        )

    assert exc_info.value.errors()[0]["loc"] == (field_name,)


@pytest.mark.parametrize("field_name", ("jwt_secret_key", "default_admin_password"))
def test_secret_settings_reject_explicit_empty_values_before_startup(tmp_path, field_name):
    values = {
        "_env_file": None,
        "debug": False,
        "jwt_secret_key": "test-only-secret-that-is-not-published",
        "training_cache": tmp_path / "cache",
        "models_dir": tmp_path / "models",
        "datasets_dir": tmp_path / "datasets",
        "output_dir": tmp_path / "output",
        "local_cache_dir": tmp_path / "local-cache",
        field_name: "",
    }

    with pytest.raises(ValidationError, match="empty value"):
        Settings(**values)

    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("invalid_minutes", (0, -1))
def test_jwt_access_token_expiry_must_be_positive(
    tmp_path,
    invalid_minutes,
):
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            jwt_access_token_expire_minutes=invalid_minutes,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )

    assert exc_info.value.errors()[0]["loc"] == (
        "jwt_access_token_expire_minutes",
    )
