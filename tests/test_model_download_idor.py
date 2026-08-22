"""Tests for model download task user-isolation (IDOR fix).

Verifies ModelDownloadService.list_downloads filters by owner so one tenant
cannot enumerate another tenant's download tasks via GET /api/models/downloads.
"""
from train_factory.storage.services.model_download_service import ModelDownloadService


def test_list_downloads_filters_by_owner():
    svc = ModelDownloadService()
    svc._download_progress = {
        "a": {"registry_id": "a", "user_id": "alice", "model_name": "m1"},
        "b": {"registry_id": "b", "user_id": "bob", "model_name": "m2"},
    }
    alice = [d["registry_id"] for d in svc.list_downloads(user_id="alice")]
    assert alice == ["a"]
    bob = [d["registry_id"] for d in svc.list_downloads(user_id="bob")]
    assert bob == ["b"]


def test_list_downloads_no_filter_returns_all():
    """Auth disabled / anonymous (user_id falsy) -> all tasks visible."""
    svc = ModelDownloadService()
    svc._download_progress = {
        "a": {"registry_id": "a", "user_id": "alice"},
        "b": {"registry_id": "b", "user_id": "bob"},
    }
    assert len(svc.list_downloads()) == 2
    assert len(svc.list_downloads(user_id=None)) == 2
