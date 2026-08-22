"""Security regressions for user-triggered remote repository downloads."""

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from train_factory.api.routes import dataset_routes, registry_routes
from train_factory.storage.services import (
    dataset_download_service as dataset_download_module,
)
from train_factory.storage.services import (
    model_download_service as model_download_module,
)
from train_factory.core import remote_download_security
from train_factory.config.settings import Settings


INVALID_REMOTE_REPOSITORIES = (
    "",
    "repo-only",
    "/etc/passwd",
    "../private/repo",
    "./owner/repo",
    r"C:\\private\\repo",
    r"owner\\repo",
    "file:///etc/passwd",
    "https://huggingface.co/owner/repo",
    "owner/../repo",
    "owner/./repo",
    "owner//repo",
    "owner/repo/extra",
    "owner/%2e%2e",
    "owner/repo?revision=main",
    "owner/repo#main",
)


def _unexpected_side_effect(*_args, **_kwargs):
    pytest.fail("invalid repository input reached a side effect")


@pytest.mark.parametrize("remote_repo", INVALID_REMOTE_REPOSITORIES)
def test_model_download_service_rejects_noncanonical_repo_ids_before_side_effects(
    monkeypatch,
    remote_repo,
):
    service = model_download_module.ModelDownloadService()
    monkeypatch.setattr(model_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr(model_download_module.os, "makedirs", _unexpected_side_effect)

    with pytest.raises(ValueError, match="repository"):
        service.start_download(
            download_source="huggingface",
            remote_repo=remote_repo,
            user_id="user-1",
        )


@pytest.mark.parametrize("remote_repo", INVALID_REMOTE_REPOSITORIES)
def test_dataset_download_service_rejects_noncanonical_repo_ids_before_side_effects(
    monkeypatch,
    remote_repo,
):
    service = dataset_download_module.DatasetDownloadService()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr(dataset_download_module.os, "makedirs", _unexpected_side_effect)

    with pytest.raises(ValueError, match="repository"):
        service.start_download(
            source_type="huggingface",
            remote_repo=remote_repo,
            dataset_name="unsafe",
            user_id="user-1",
        )


@pytest.mark.parametrize(
    ("route_module", "payload", "endpoint"),
    (
        (
            registry_routes,
            registry_routes.DownloadModelRequest(
                download_source="huggingface",
                remote_repo="../private/model",
            ),
            registry_routes.download_model,
        ),
        (
            dataset_routes,
            dataset_routes.DownloadDatasetRequest(
                dataset_name="unsafe",
                source_type="huggingface",
                remote_repo="../private/dataset",
            ),
            dataset_routes.download_dataset,
        ),
    ),
)
def test_download_routes_return_400_for_noncanonical_repo_ids(
    monkeypatch,
    route_module,
    payload,
    endpoint,
):
    monkeypatch.setattr(route_module, "requires_tenant_provenance", lambda _user: True)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(payload, {"user_id": "user-1"}))

    assert exc_info.value.status_code == 400
    assert "repository" in str(exc_info.value.detail).lower()


@pytest.mark.parametrize("auth_enabled", (True, False))
def test_huggingface_model_download_uses_anonymous_token_only_in_auth_mode(
    monkeypatch,
    tmp_path,
    auth_enabled,
):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    service = model_download_module.ModelDownloadService()
    monkeypatch.setattr(model_download_module.settings, "auth_enabled", auth_enabled)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    service._download_from_huggingface(
        "registry-1",
        "owner/model",
        str(tmp_path),
    )

    if auth_enabled:
        assert captured["token"] is False
    else:
        assert "token" not in captured


class _FakeDatasetSplit:
    def __iter__(self):
        yield {"text": "example"}

    def __len__(self):
        return 1

    def to_json(self, *_args, **_kwargs):
        return None


@pytest.mark.parametrize("auth_enabled", (True, False))
def test_huggingface_dataset_download_uses_anonymous_token_only_in_auth_mode(
    monkeypatch,
    tmp_path,
    auth_enabled,
):
    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"train": _FakeDatasetSplit()}

    service = dataset_download_module.DatasetDownloadService()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", auth_enabled)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)
    monkeypatch.setattr(service, "_calculate_size", lambda _path: 1)
    monkeypatch.setattr(service, "_clean_hf_cache", lambda *_args: None)

    service._download_from_huggingface(
        "dataset-1",
        "owner/dataset",
        str(tmp_path),
        clean_cache=False,
    )

    if auth_enabled:
        assert captured["kwargs"]["token"] is False
        assert captured["kwargs"]["trust_remote_code"] is False
    else:
        # legacy 模式：不传匿名 token，但仍强制禁止远端代码执行
        assert "token" not in captured["kwargs"]
        assert captured["kwargs"].get("trust_remote_code") is False


@pytest.mark.parametrize(
    ("repo_type", "info_method"),
    (("model", "model_info"), ("dataset", "dataset_info")),
)
def test_huggingface_metadata_preflight_is_anonymous_and_sums_file_sizes(
    monkeypatch,
    repo_type,
    info_method,
):
    calls = []

    class FakeHfApi:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def model_info(self, repo_id, **kwargs):
            calls.append(("model_info", repo_id, kwargs))
            return SimpleNamespace(
                siblings=[SimpleNamespace(size=12), SimpleNamespace(size=30)]
            )

        def dataset_info(self, repo_id, **kwargs):
            calls.append(("dataset_info", repo_id, kwargs))
            return SimpleNamespace(
                siblings=[SimpleNamespace(size=12), SimpleNamespace(size=30)]
            )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    size = remote_download_security.preflight_remote_repo_size(
        download_source="huggingface",
        repo_type=repo_type,
        remote_repo="owner/repo",
        max_bytes=100,
        anonymous=True,
        timeout_seconds=7,
    )

    assert size == 42
    assert calls[0] == ("init", {"token": False})
    assert calls[1] == (
        info_method,
        "owner/repo",
        {"files_metadata": True, "timeout": 7, "token": False},
    )


def test_metadata_preflight_rejects_repository_over_size_limit(monkeypatch):
    class FakeHfApi:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, _repo_id, **_kwargs):
            return SimpleNamespace(siblings=[SimpleNamespace(size=101)])

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    with pytest.raises(ValueError, match="exceeds"):
        remote_download_security.preflight_remote_repo_size(
            download_source="huggingface",
            repo_type="model",
            remote_repo="owner/repo",
            max_bytes=100,
            anonymous=True,
            timeout_seconds=7,
        )


@pytest.mark.parametrize(
    "siblings",
    (
        [],
        [SimpleNamespace(size=None, lfs=None)],
        [SimpleNamespace(size="not-a-size", lfs=None)],
        [SimpleNamespace(size=-1, lfs=None)],
    ),
)
def test_metadata_preflight_fails_closed_when_size_cannot_be_verified(
    monkeypatch,
    siblings,
):
    class FakeHfApi:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, _repo_id, **_kwargs):
            return SimpleNamespace(siblings=siblings)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    with pytest.raises(
        remote_download_security.DownloadMetadataUnavailable,
        match="Unable to verify",
    ):
        remote_download_security.preflight_remote_repo_size(
            download_source="huggingface",
            repo_type="model",
            remote_repo="owner/repo",
            max_bytes=100,
            anonymous=True,
            timeout_seconds=7,
        )


def test_metadata_preflight_sanitizes_upstream_errors(monkeypatch):
    class FakeHfApi:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, _repo_id, **_kwargs):
            raise RuntimeError("Authorization: Bearer host-secret-token")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    with pytest.raises(
        remote_download_security.DownloadMetadataUnavailable,
        match="Unable to verify remote repository size",
    ) as exc_info:
        remote_download_security.preflight_remote_repo_size(
            download_source="huggingface",
            repo_type="model",
            remote_repo="owner/repo",
            max_bytes=100,
            anonymous=True,
            timeout_seconds=7,
        )

    assert "host-secret-token" not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload_factory",
    (
        lambda: registry_routes.DownloadModelRequest(
            download_source="huggingface",
            remote_repo=f"owner/{'x' * 91}",
        ),
        lambda: registry_routes.DownloadModelRequest(
            download_source="huggingface",
            remote_repo="owner/model",
            display_name="x" * 257,
        ),
        lambda: registry_routes.DownloadModelRequest(
            download_source="huggingface",
            remote_repo="owner/model",
            description="x" * 4001,
        ),
        lambda: dataset_routes.DownloadDatasetRequest(
            dataset_name="x" * 256,
            source_type="huggingface",
            remote_repo="owner/dataset",
        ),
        lambda: dataset_routes.DownloadDatasetRequest(
            dataset_name="dataset",
            source_type="huggingface",
            remote_repo="owner/dataset",
            model_type=["embedding"] * 9,
        ),
        lambda: dataset_routes.DownloadDatasetRequest(
            dataset_name="dataset",
            source_type="huggingface",
            remote_repo="owner/dataset",
            description="x" * 4001,
        ),
    ),
)
def test_download_request_models_bound_user_controlled_payloads(payload_factory):
    with pytest.raises(ValidationError):
        payload_factory()


def test_download_limiter_enforces_per_user_and_global_limits():
    limiter = remote_download_security.DownloadConcurrencyLimiter()
    first = limiter.acquire("user-1", global_limit=2, per_user_limit=1)

    with pytest.raises(ValueError, match="per-user"):
        limiter.acquire("user-1", global_limit=2, per_user_limit=1)

    second = limiter.acquire("user-2", global_limit=2, per_user_limit=1)
    with pytest.raises(ValueError, match="global"):
        limiter.acquire("user-3", global_limit=2, per_user_limit=1)

    first.release()
    replacement = limiter.acquire("user-3", global_limit=2, per_user_limit=1)

    first.release()
    second.release()
    replacement.release()
    assert limiter.active_total == 0


def _install_fake_modelscope_snapshot(monkeypatch, callback):
    modelscope_module = types.ModuleType("modelscope")
    hub_module = types.ModuleType("modelscope.hub")
    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")
    snapshot_module.snapshot_download = callback
    modelscope_module.hub = hub_module
    hub_module.snapshot_download = snapshot_module
    monkeypatch.setitem(sys.modules, "modelscope", modelscope_module)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub_module)
    monkeypatch.setitem(
        sys.modules,
        "modelscope.hub.snapshot_download",
        snapshot_module,
    )


@pytest.mark.parametrize("auth_enabled", (True, False))
def test_modelscope_model_download_uses_empty_token_only_in_auth_mode(
    monkeypatch,
    tmp_path,
    auth_enabled,
):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    _install_fake_modelscope_snapshot(monkeypatch, fake_snapshot_download)
    service = model_download_module.ModelDownloadService()
    monkeypatch.setattr(model_download_module.settings, "auth_enabled", auth_enabled)
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    service._download_from_modelscope(
        "registry-1",
        "owner/model",
        str(tmp_path),
    )

    if auth_enabled:
        assert captured["token"] == ""
    else:
        assert "token" not in captured


def _install_fake_modelscope_dataset(monkeypatch, load_callback):
    modelscope_module = types.ModuleType("modelscope")
    msdatasets_module = types.ModuleType("modelscope.msdatasets")

    class FakeMsDataset:
        load = staticmethod(load_callback)

    msdatasets_module.MsDataset = FakeMsDataset
    modelscope_module.msdatasets = msdatasets_module
    monkeypatch.setitem(sys.modules, "modelscope", modelscope_module)
    monkeypatch.setitem(sys.modules, "modelscope.msdatasets", msdatasets_module)


@pytest.mark.parametrize("auth_enabled", (True, False))
def test_modelscope_dataset_download_uses_empty_token_only_in_auth_mode(
    monkeypatch,
    tmp_path,
    auth_enabled,
):
    captured = {}

    class FakeDownloadedDataset:
        def to_hf_dataset(self):
            return {"train": _FakeDatasetSplit()}

    def fake_load(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeDownloadedDataset()

    _install_fake_modelscope_dataset(monkeypatch, fake_load)
    service = dataset_download_module.DatasetDownloadService()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", auth_enabled)
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)
    monkeypatch.setattr(service, "_calculate_size", lambda _path: 1)

    service._download_from_modelscope(
        "dataset-1",
        "owner/dataset",
        str(tmp_path),
    )

    if auth_enabled:
        assert captured["kwargs"]["token"] == ""
        assert captured["kwargs"]["trust_remote_code"] is False
    else:
        assert "token" not in captured["kwargs"]
        assert "trust_remote_code" not in captured["kwargs"]


@pytest.mark.parametrize("repo_type", ("model", "dataset"))
def test_modelscope_metadata_preflight_is_anonymous(
    monkeypatch,
    repo_type,
):
    calls = []
    api_module = types.ModuleType("modelscope_hub.api")
    package_module = types.ModuleType("modelscope_hub")

    class FakeHubApi:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def list_repo_files(self, repo_id, requested_type, **kwargs):
            calls.append(("list", repo_id, requested_type, kwargs))
            return [SimpleNamespace(size=40), SimpleNamespace(size=2)]

    api_module.HubApi = FakeHubApi
    package_module.api = api_module
    monkeypatch.setitem(sys.modules, "modelscope_hub", package_module)
    monkeypatch.setitem(sys.modules, "modelscope_hub.api", api_module)

    size = remote_download_security.preflight_remote_repo_size(
        download_source="modelscope",
        repo_type=repo_type,
        remote_repo="owner/repo",
        max_bytes=100,
        anonymous=True,
        timeout_seconds=7,
    )

    assert size == 42
    assert calls == [
        ("init", {"token": ""}),
        (
            "list",
            "owner/repo",
            repo_type,
            {"revision": "master", "recursive": True},
        ),
    ]


@pytest.mark.parametrize("repo_type", ("model", "dataset"))
def test_modelscope_path_only_metadata_reports_unknown_size(
    monkeypatch,
    repo_type,
):
    api_module = types.ModuleType("modelscope_hub.api")
    package_module = types.ModuleType("modelscope_hub")

    class FakeHubApi:
        def __init__(self, **_kwargs):
            pass

        def list_repo_files(self, _repo_id, _requested_type, **_kwargs):
            return ["config.json", "weights/model.safetensors"]

    api_module.HubApi = FakeHubApi
    package_module.api = api_module
    monkeypatch.setitem(sys.modules, "modelscope_hub", package_module)
    monkeypatch.setitem(sys.modules, "modelscope_hub.api", api_module)

    size = remote_download_security.preflight_remote_repo_size(
        download_source="modelscope",
        repo_type=repo_type,
        remote_repo="owner/repo",
        max_bytes=100,
        anonymous=True,
        timeout_seconds=7,
    )

    assert size is None


def test_remote_download_settings_have_bounded_defaults():
    fields = Settings.model_fields

    assert fields["download_max_concurrent_global"].default == 4
    assert fields["download_max_concurrent_per_user"].default == 2
    assert fields["model_download_max_bytes"].default == 30 * 1024**3
    assert fields["dataset_download_max_bytes"].default == 10 * 1024**3
    assert fields["download_storage_max_bytes_global"].default == 200 * 1024**3
    assert fields["download_storage_max_bytes_per_user"].default == 60 * 1024**3
    assert fields["download_metadata_timeout_seconds"].default == 15
    assert fields["download_max_workers"].default == 4


@pytest.mark.parametrize(
    ("module", "service_factory", "start_kwargs"),
    (
        (
            model_download_module,
            model_download_module.ModelDownloadService,
            {
                "download_source": "huggingface",
                "remote_repo": "owner/model",
                "user_id": "user-1",
            },
        ),
        (
            dataset_download_module,
            dataset_download_module.DatasetDownloadService,
            {
                "source_type": "huggingface",
                "remote_repo": "owner/dataset",
                "dataset_name": "dataset",
                "user_id": "user-1",
            },
        ),
    ),
)
def test_services_enforce_concurrency_before_filesystem_side_effects(
    monkeypatch,
    module,
    service_factory,
    start_kwargs,
):
    class RejectingLimiter:
        def acquire(self, *_args, **_kwargs):
            raise remote_download_security.DownloadConcurrencyExceeded(
                "global limit exceeded"
            )

    monkeypatch.setattr(module.settings, "auth_enabled", True)
    monkeypatch.setattr(
        module,
        "download_concurrency_limiter",
        RejectingLimiter(),
        raising=False,
    )
    monkeypatch.setattr(module.os, "makedirs", _unexpected_side_effect)

    with pytest.raises(ValueError, match="global limit"):
        service_factory().start_download(**start_kwargs)


class _RecordingLease:
    def __init__(self):
        self.release_calls = 0

    def release(self):
        self.release_calls += 1


def test_model_background_rechecks_size_cleans_files_and_releases_slot(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"x" * 11)
    service = model_download_module.ModelDownloadService()
    lease = _RecordingLease()
    record_updates = []

    monkeypatch.setattr(
        service,
        "_download_from_huggingface",
        lambda *_args: None,
    )
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "_update_model_record",
        lambda *args, **kwargs: record_updates.append((args, kwargs)),
    )

    service._download_model_background(
        "registry-1",
        "huggingface",
        "owner/model",
        str(model_path),
        lease=lease,
        max_bytes=10,
    )

    assert not model_path.exists()
    assert lease.release_calls == 1
    assert record_updates[-1][1]["download_status"] == "failed"
    assert "exceeds" in record_updates[-1][1]["download_error"]


def test_dataset_background_rechecks_size_cleans_files_and_releases_slot(
    monkeypatch,
    tmp_path,
):
    storage_path = tmp_path / "dataset"
    storage_path.mkdir()
    (storage_path / "train.jsonl").write_bytes(b"x" * 11)
    service = dataset_download_module.DatasetDownloadService()
    lease = _RecordingLease()
    record_updates = []

    monkeypatch.setattr(
        service,
        "_download_from_huggingface",
        lambda *_args, **_kwargs: (1, 11),
    )
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)
    monkeypatch.setattr(
        service,
        "_update_dataset_record",
        lambda *args, **kwargs: record_updates.append((args, kwargs)),
    )

    service._download_dataset_background(
        "dataset-1",
        "huggingface",
        "owner/dataset",
        str(storage_path),
        lease=lease,
        max_bytes=10,
    )

    assert not storage_path.exists()
    assert lease.release_calls == 1
    assert record_updates[-1][1]["status"] == "error"
    assert "exceeds" in record_updates[-1][1]["error_message"]


def test_authenticated_hf_dataset_cache_counts_toward_post_download_limit(
    monkeypatch,
    tmp_path,
):
    storage_path = tmp_path / "dataset"
    storage_path.mkdir()

    def fake_load_dataset(*_args, **kwargs):
        cache_dir = kwargs["cache_dir"]
        cache_file = dataset_download_module.os.path.join(cache_dir, "payload.bin")
        dataset_download_module.os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as handle:
            handle.write(b"x" * 11)
        return {"train": _FakeDatasetSplit()}

    service = dataset_download_module.DatasetDownloadService()
    monkeypatch.setattr(dataset_download_module.settings, "auth_enabled", True)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(service, "_update_download_status", lambda *_args: None)

    with pytest.raises(ValueError, match="exceeds"):
        service._download_from_huggingface(
            "dataset-1",
            "owner/dataset",
            str(storage_path),
            clean_cache=False,
            max_bytes=10,
        )

    assert not (storage_path / ".download-cache").exists()


@pytest.mark.parametrize(
    ("route_module", "payload", "endpoint", "service", "error", "status_code"),
    (
        (
            registry_routes,
            registry_routes.DownloadModelRequest(
                download_source="huggingface",
                remote_repo="owner/model",
            ),
            registry_routes.download_model,
            model_download_module.model_download_service,
            remote_download_security.DownloadConcurrencyExceeded("global limit"),
            429,
        ),
        (
            registry_routes,
            registry_routes.DownloadModelRequest(
                download_source="huggingface",
                remote_repo="owner/model",
            ),
            registry_routes.download_model,
            model_download_module.model_download_service,
            remote_download_security.DownloadSizeExceeded("size exceeds limit"),
            413,
        ),
        (
            registry_routes,
            registry_routes.DownloadModelRequest(
                download_source="huggingface",
                remote_repo="owner/model",
            ),
            registry_routes.download_model,
            model_download_module.model_download_service,
            remote_download_security.DownloadMetadataUnavailable(
                "Unable to verify remote repository size"
            ),
            502,
        ),
        (
            dataset_routes,
            dataset_routes.DownloadDatasetRequest(
                dataset_name="dataset",
                source_type="huggingface",
                remote_repo="owner/dataset",
            ),
            dataset_routes.download_dataset,
            dataset_download_module.dataset_download_service,
            remote_download_security.DownloadConcurrencyExceeded("global limit"),
            429,
        ),
        (
            dataset_routes,
            dataset_routes.DownloadDatasetRequest(
                dataset_name="dataset",
                source_type="huggingface",
                remote_repo="owner/dataset",
            ),
            dataset_routes.download_dataset,
            dataset_download_module.dataset_download_service,
            remote_download_security.DownloadSizeExceeded("size exceeds limit"),
            413,
        ),
        (
            dataset_routes,
            dataset_routes.DownloadDatasetRequest(
                dataset_name="dataset",
                source_type="huggingface",
                remote_repo="owner/dataset",
            ),
            dataset_routes.download_dataset,
            dataset_download_module.dataset_download_service,
            remote_download_security.DownloadMetadataUnavailable(
                "Unable to verify remote repository size"
            ),
            502,
        ),
    ),
)
def test_download_routes_map_resource_limits_to_specific_status_codes(
    monkeypatch,
    route_module,
    payload,
    endpoint,
    service,
    error,
    status_code,
):
    monkeypatch.setattr(route_module, "requires_tenant_provenance", lambda _user: True)

    def reject_download(**_kwargs):
        raise error

    monkeypatch.setattr(service, "start_download", reject_download)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(payload, {"user_id": "user-1"}))

    assert exc_info.value.status_code == status_code


@pytest.mark.parametrize(
    ("route_module", "payload", "endpoint", "service"),
    (
        (
            registry_routes,
            registry_routes.DownloadModelRequest(
                download_source="huggingface",
                remote_repo="owner/model",
            ),
            registry_routes.download_model,
            model_download_module.model_download_service,
        ),
        (
            dataset_routes,
            dataset_routes.DownloadDatasetRequest(
                dataset_name="dataset",
                source_type="huggingface",
                remote_repo="owner/dataset",
            ),
            dataset_routes.download_dataset,
            dataset_download_module.dataset_download_service,
        ),
    ),
)
def test_download_routes_do_not_expose_unexpected_upstream_errors(
    monkeypatch,
    route_module,
    payload,
    endpoint,
    service,
):
    monkeypatch.setattr(route_module, "requires_tenant_provenance", lambda _user: True)

    def reject_download(**_kwargs):
        raise RuntimeError("Authorization: Bearer host-secret-token")

    monkeypatch.setattr(service, "start_download", reject_download)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoint(payload, {"user_id": "user-1"}))

    assert exc_info.value.status_code == 500
    assert "host-secret-token" not in exc_info.value.detail


def test_per_user_download_limit_cannot_exceed_global_limit(tmp_path):
    with pytest.raises(ValidationError, match="per-user"):
        Settings(
            _env_file=None,
            debug=False,
            jwt_secret_key="test-only-secret-that-is-not-published",
            download_max_concurrent_global=1,
            download_max_concurrent_per_user=2,
            training_cache=tmp_path / "cache",
            models_dir=tmp_path / "models",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            local_cache_dir=tmp_path / "local-cache",
        )


def test_remote_download_limits_are_exposed_in_compose_and_env_example():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load(
        (root / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    env_example = (root / ".env.example").read_text(encoding="utf-8")
    expected = {
        "DOWNLOAD_MAX_CONCURRENT_GLOBAL": "4",
        "DOWNLOAD_MAX_CONCURRENT_PER_USER": "2",
        "MODEL_DOWNLOAD_MAX_BYTES": str(30 * 1024**3),
        "DATASET_DOWNLOAD_MAX_BYTES": str(10 * 1024**3),
        "DOWNLOAD_STORAGE_MAX_BYTES_GLOBAL": str(200 * 1024**3),
        "DOWNLOAD_STORAGE_MAX_BYTES_PER_USER": str(60 * 1024**3),
        "DOWNLOAD_METADATA_TIMEOUT_SECONDS": "15",
        "DOWNLOAD_MAX_WORKERS": "4",
    }

    for name, default in expected.items():
        assert f"{name}={default}" in env_example
        expected_expression = f"${{{name}:-{default}}}"
        assert (
            compose["services"]["train-factory-api"]["environment"][name]
            == expected_expression
        )
        assert (
            compose["services"]["train-factory-api-dev"]["environment"][name]
            == expected_expression
        )


def test_modelscope_metadata_client_is_an_explicit_runtime_dependency():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"modelscope-hub>=0.2.0"' in pyproject


@pytest.mark.parametrize(
    "split_name",
    (
        "../escape",
        "owner/escape",
        "/absolute",
        r"C:\\escape",
        r"owner\\escape",
        ".",
        "..",
        ".hidden",
    ),
)
def test_dataset_download_rejects_unsafe_remote_split_names(
    tmp_path,
    split_name,
):
    class RejectWrites:
        def to_json(self, *_args, **_kwargs):
            pytest.fail("unsafe split name reached a file write")

    with pytest.raises(ValueError, match="split"):
        dataset_download_module.DatasetDownloadService._save_split(
            RejectWrites(),
            str(tmp_path),
            split_name,
            "jsonl",
        )


@pytest.mark.parametrize(
    ("route_module", "payload", "endpoint", "service", "service_result"),
    (
        (
            registry_routes,
            registry_routes.DownloadModelRequest(
                download_source="huggingface",
                remote_repo="owner/model",
            ),
            registry_routes.download_model,
            model_download_module.model_download_service,
            {
                "registry_id": "model-1",
                "model_name": "model",
                "display_name": "Model",
                "status": "downloading",
                "model_path": "/models/model-1",
            },
        ),
        (
            dataset_routes,
            dataset_routes.DownloadDatasetRequest(
                dataset_name="dataset",
                source_type="huggingface",
                remote_repo="owner/dataset",
            ),
            dataset_routes.download_dataset,
            dataset_download_module.dataset_download_service,
            {
                "dataset_id": "dataset-1",
                "dataset_name": "dataset",
                "status": "downloading",
                "storage_path": "/datasets/dataset-1",
            },
        ),
    ),
)
def test_download_routes_run_blocking_preflight_off_the_event_loop(
    monkeypatch,
    route_module,
    payload,
    endpoint,
    service,
    service_result,
):
    calls = []

    async def fake_to_thread(function, *args, **kwargs):
        calls.append((function, args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(route_module, "requires_tenant_provenance", lambda _user: True)
    monkeypatch.setattr(service, "start_download", lambda **_kwargs: service_result)
    monkeypatch.setattr(
        route_module,
        "asyncio",
        SimpleNamespace(to_thread=fake_to_thread),
        raising=False,
    )

    asyncio.run(endpoint(payload, {"user_id": "user-1"}))

    assert len(calls) == 1
    assert calls[0][0] == service.start_download
