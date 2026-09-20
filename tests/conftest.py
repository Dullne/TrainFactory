import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest


def _bootstrap_test_env() -> None:
    """Prepare writable runtime dirs and import path for pytest."""
    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = repo_root / ".runtime" / "pytest"
    paths = {
        "BASE_DIR": runtime_root / "app",
        "TRAINING_CACHE": runtime_root / "cache",
        "MODELS_DIR": runtime_root / "models",
        "DATASETS_DIR": runtime_root / "datasets",
        "OUTPUT_DIR": runtime_root / "output",
        "LOCAL_CACHE_DIR": runtime_root / "cache_datasets",
    }

    for key, value in paths.items():
        os.environ.setdefault(key, str(value))
        value.mkdir(parents=True, exist_ok=True)

    # Use a fixed non-default JWT secret so the production guard
    # (settings._check_security_settings) doesn't refuse to construct Settings
    # during tests. Real deployments must set JWT_SECRET_KEY themselves.
    os.environ.setdefault("JWT_SECRET_KEY", "test-suite-secret-not-for-production")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _install_sqlmodel_stub_if_needed() -> None:
    """Install a minimal sqlmodel stub for offline test environments."""
    if importlib.util.find_spec("sqlmodel"):
        return

    import sqlalchemy as sa
    from pydantic import BaseModel, ConfigDict
    from pydantic import Field as PydanticField
    from sqlalchemy.orm import Session as SASession

    stub = types.ModuleType("sqlmodel")

    class SQLModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", populate_by_name=True)
        metadata: ClassVar[sa.MetaData] = sa.MetaData()

        def __init_subclass__(cls, **kwargs):
            kwargs.pop("table", None)
            super().__init_subclass__()

    def Field(default=..., **kwargs):
        filtered = {}
        if "default_factory" in kwargs:
            filtered["default_factory"] = kwargs["default_factory"]
        if "alias" in kwargs:
            filtered["alias"] = kwargs["alias"]
        if "description" in kwargs:
            filtered["description"] = kwargs["description"]
        if "max_length" in kwargs:
            filtered["max_length"] = kwargs["max_length"]
        return PydanticField(default, **filtered)

    def Relationship(*args, **kwargs):  # noqa: ARG001
        return None

    stub.SQLModel = SQLModel
    stub.Field = Field
    stub.Relationship = Relationship
    stub.Session = SASession
    stub.create_engine = sa.create_engine
    stub.select = sa.select
    stub.delete = sa.delete
    stub.func = sa.func
    stub.text = sa.text
    stub.and_ = sa.and_
    stub.or_ = sa.or_
    stub.Column = sa.Column
    stub.JSON = sa.JSON
    stub.Index = sa.Index

    sys.modules["sqlmodel"] = stub


def _install_optional_dependency_stubs() -> None:
    """Install tiny stubs for optional deps missing in offline CI sandboxes."""
    if importlib.util.find_spec("loguru") is None:
        loguru_stub = types.ModuleType("loguru")

        class _NoOpLogger:
            def __getattr__(self, name):  # noqa: D401
                def _noop(*args, **kwargs):  # noqa: ARG001
                    return None

                return _noop

        loguru_stub.logger = _NoOpLogger()
        sys.modules["loguru"] = loguru_stub

    if importlib.util.find_spec("jose") is None:
        jose_stub = types.ModuleType("jose")

        class JWTError(Exception):
            pass

        class _JWT:
            @staticmethod
            def encode(*args, **kwargs):  # noqa: ARG001
                raise JWTError("python-jose is not installed")

            @staticmethod
            def decode(*args, **kwargs):  # noqa: ARG001
                raise JWTError("python-jose is not installed")

        jose_stub.JWTError = JWTError
        jose_stub.jwt = _JWT()
        sys.modules["jose"] = jose_stub

    if importlib.util.find_spec("slowapi") is None:
        slowapi_stub = types.ModuleType("slowapi")

        class Limiter:  # noqa: D401
            def __init__(self, *args, **kwargs):  # noqa: ARG001
                pass

        def _rate_limit_exceeded_handler(*args, **kwargs):  # noqa: ARG001
            return None

        slowapi_stub.Limiter = Limiter
        slowapi_stub._rate_limit_exceeded_handler = _rate_limit_exceeded_handler
        sys.modules["slowapi"] = slowapi_stub

        errors_mod = types.ModuleType("slowapi.errors")

        class RateLimitExceeded(Exception):
            pass

        errors_mod.RateLimitExceeded = RateLimitExceeded
        sys.modules["slowapi.errors"] = errors_mod

        util_mod = types.ModuleType("slowapi.util")

        def get_remote_address(*args, **kwargs):  # noqa: ARG001
            return "127.0.0.1"

        util_mod.get_remote_address = get_remote_address
        sys.modules["slowapi.util"] = util_mod


_bootstrap_test_env()
_install_sqlmodel_stub_if_needed()
_install_optional_dependency_stubs()


@pytest.fixture
def empty_inference_catalog(monkeypatch, tmp_path):
    """External-API route tests use the real gate with an isolated empty catalog."""
    from contextlib import contextmanager

    from sqlmodel import Session, create_engine

    from train_factory.storage.entities.deployment_entity import DeploymentDB
    from train_factory.storage.services import inference_authorization_service

    engine = create_engine(f"sqlite:///{tmp_path / 'inference-catalog.db'}")
    DeploymentDB.__table__.create(engine)

    @contextmanager
    def catalog_session():
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(inference_authorization_service, "get_session", catalog_session)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def ensure_event_loop():
    """Provide a default event loop for tests using asyncio.get_event_loop()."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)
