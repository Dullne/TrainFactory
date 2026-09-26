"""Explicit configuration and authentication for opt-in live sync tests."""

import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import pytest
import requests
from sqlalchemy.engine import make_url


@pytest.fixture(scope="session")
def sync_integration_settings():
    if os.getenv("TEST_SYNC_ISOLATED") != "1":
        pytest.fail("Live sync tests require TEST_SYNC_ISOLATED=1 and a disposable API/database")
    base = os.getenv("TEST_API_BASE", "").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path != "/api":
        pytest.fail("Set TEST_API_BASE explicitly to the isolated server's /api URL")
    if not os.getenv("TEST_SYNC_USER_ID") or not os.getenv("TEST_SYNC_OTHER_USER_ID"):
        pytest.fail("Set TEST_SYNC_USER_ID and TEST_SYNC_OTHER_USER_ID for the isolated test accounts")
    value = os.getenv("MYSQL_URL")
    source = os.getenv("MYSQL_URL_FILE")
    if bool(value) == bool(source):
        pytest.fail("Set exactly one of MYSQL_URL or MYSQL_URL_FILE for the isolated database")
    try:
        url = make_url(value if value else Path(source).read_text(encoding="utf-8").strip())
    except Exception:
        pytest.fail("The explicit sync test database URL could not be read", pytrace=False)
    if url.drivername != "mysql+pymysql" or not (url.database or "").startswith(
        ("tf_acceptance_", "tf_sync_test_")
    ):
        pytest.fail("Live sync tests require a disposable tf_acceptance_ or tf_sync_test_ MySQL database")
    from train_factory.config.settings import get_settings

    if make_url(get_settings().mysql_url) != url:
        pytest.fail("Loaded application settings do not match the explicit sync test database")
    return base


class _SyncSession(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", 60)
        return super().request(method, url, **kwargs)


def sync_artifact_path(storage_path):
    """Translate an explicitly declared API bind mount for content assertions."""
    local = Path(storage_path)
    if local.is_file():
        return local
    server_root = os.getenv("TEST_SYNC_SERVER_DATA_DIR")
    host_root = os.getenv("SYNC_DATA_DIR")
    if not server_root or not host_root:
        pytest.fail("Set TEST_SYNC_SERVER_DATA_DIR and SYNC_DATA_DIR to the API's shared sync bind mount")
    try:
        relative = PurePosixPath(storage_path).relative_to(PurePosixPath(server_root))
        base = Path(host_root).resolve()
        local = base.joinpath(*relative.parts).resolve()
        local.relative_to(base)
    except ValueError:
        pytest.fail("Sync artifact is outside the explicitly configured shared bind mount")
    return local


@pytest.fixture(scope="session")
def sync_integration_session(sync_integration_settings):
    username = os.getenv("TEST_SYNC_USERNAME")
    password_file = os.getenv("TEST_SYNC_PASSWORD_FILE")
    if not username or not password_file:
        pytest.fail("Set TEST_SYNC_USERNAME and TEST_SYNC_PASSWORD_FILE for live API authentication")
    try:
        password = Path(password_file).read_text(encoding="utf-8").strip()
    except OSError:
        pytest.fail("The sync test account password file could not be read", pytrace=False)
    session = _SyncSession()
    try:
        login = session.post(
            f"{sync_integration_settings}/auth/login",
            json={"username": username, "password": password},
        )
        if login.status_code != 200:
            pytest.fail(f"Sync test account login failed with HTTP {login.status_code}")
        token = login.json().get("access_token")
        if not token:
            pytest.fail("Sync test account login did not return an access token")
        session.headers["Authorization"] = f"Bearer {token}"
        profile = session.get(f"{sync_integration_settings}/auth/me")
        if profile.status_code != 200 or profile.json().get("user_id") != os.getenv("TEST_SYNC_USER_ID"):
            pytest.fail("Authenticated sync account does not match TEST_SYNC_USER_ID")
        yield session
    finally:
        session.close()
