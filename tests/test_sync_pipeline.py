"""
Sync pipeline functional tests.

Tests the business logic of the two-level threshold pipeline:
- Counter lifecycle (increment, accumulate, reset)
- Level 2 handler: on_generation_completed callback
- Training threshold trigger logic
- Manual trigger APIs (sync-now, trigger-generation, trigger-training)
- Runtime API config resolution

Run: pytest tests/test_sync_pipeline.py -v
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from train_factory.core.time_utils import now_naive
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_SYNC_INTEGRATION") != "1",
    reason="set RUN_SYNC_INTEGRATION=1 and use an isolated API/database",
)

BASE = os.getenv("TEST_API_BASE", "http://localhost:18000/api").rstrip("/")
EXTERNAL_API_HOST = os.getenv("TEST_EXTERNAL_API_HOST", "localhost")
MOCK_PORT = 19877
TEST_USER_ID = os.getenv("TEST_SYNC_USER_ID", "pytest-user")
OTHER_USER_ID = os.getenv("TEST_SYNC_OTHER_USER_ID", "pytest-other-user")


# ── Mock External API Server ─────────────────────────────


def _make_mock_data(count=30):
    data = []
    now = now_naive()
    for i in range(count):
        ts = (now - timedelta(hours=count - i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        data.append(
            {
                "id": f"pipe-{i:04d}",
                "text": f"Pipeline test content #{i}.",
                "source": "pipeline-test",
                "session_id": f"sess-{i % 3:03d}",
                "doc_id": f"doc-{i % 2:03d}",
                "created_at": ts,
            }
        )
    return data


MOCK_DATA = _make_mock_data(30)


class PipelineMockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": "unauthorized"}')
            return

        offset = int(params.get("offset", ["0"])[0])
        limit = int(params.get("limit", ["1000"])[0])
        since = params.get("since", [None])[0]

        items = MOCK_DATA
        if since:
            since_dt = datetime.strptime(since, "%Y-%m-%dT%H:%M:%S")
            items = [
                it
                for it in items
                if datetime.strptime(it["created_at"], "%Y-%m-%dT%H:%M:%SZ") > since_dt
            ]

        total = len(items)
        page = items[offset : offset + limit]
        has_more = (offset + len(page)) < total

        resp = json.dumps({"total": total, "items": page, "has_more": has_more})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp.encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def pipeline_mock_server():
    server = HTTPServer(("0.0.0.0", MOCK_PORT), PipelineMockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    try:
        yield f"http://{EXTERNAL_API_HOST}:{MOCK_PORT}/api/pipeline"
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=5)


# ── Helpers ──────────────────────────────────────────────


def _create_api_config(api_url, token="pipeline-token"):
    r = requests.post(
        f"{BASE}/sync/api-configs",
        json={
            "config_name": "pytest-pipeline-api",
            "api_url": api_url,
            "auth_config": {"token": token},
        },
    )
    assert r.status_code == 201
    return r.json()["config"]


def _create_sync_config(
    api_config_id, generation_threshold=99999, training_threshold=99999, is_active=False,
):
    r = requests.post(
        f"{BASE}/sync/tasks",
        json={
            "task_name": f"pytest-pipeline-sync-{uuid.uuid4().hex[:12]}",
            "external_api_config_id": api_config_id,
            "generation_threshold": generation_threshold,
            "training_threshold": training_threshold,
            "sync_interval_seconds": 600,
            "is_active": is_active,
            "generation_config": {},
        },
    )
    assert r.status_code == 201
    return r.json()["task"]


def _cleanup(sync_id=None, api_id=None):
    if sync_id:
        stopped = requests.post(f"{BASE}/sync/tasks/{sync_id}/stop")
        assert stopped.status_code == 200, stopped.text
        deleted = requests.delete(f"{BASE}/sync/tasks/{sync_id}")
        assert deleted.status_code == 200, deleted.text
    if api_id:
        deleted = requests.delete(f"{BASE}/sync/api-configs/{api_id}")
        assert deleted.status_code == 200, deleted.text


def _create_owned_input_batch(task, tmp_path, record_count=50):
    """Create real input rows for callback tests without launching a model."""
    from train_factory.storage.services.external_sync_service import external_sync_service

    path = tmp_path / f"{task['task_id']}.jsonl"
    path.write_text(
        "".join(json.dumps({"content": f"input-{index}"}) + "\n" for index in range(record_count)),
        encoding="utf-8",
    )
    batch, created = external_sync_service.create_batch(
        task_id=task["task_id"],
        user_id=task["user_id"],
        record_count=record_count,
        storage_path=str(path),
        since_time=None,
    )
    assert created is True
    return batch


def _claim_generation_batches(task_id, generation_task_id, batches):
    """Use the production atomic claim, including its actual batch side effects."""
    from train_factory.storage.services.external_sync_service import external_sync_service

    task = external_sync_service.get_task_raw(task_id)
    assert task is not None and batches
    external_sync_service.update_task(task_id, status="generating")
    generation = external_sync_service.create_generation_and_claim_batches(
        task_id=task_id,
        generation_task_id=generation_task_id,
        user_id=task["user_id"],
        input_batch_ids=[batch["batch_id"] for batch in batches],
        input_record_count=sum(batch["record_count"] for batch in batches),
    )
    stored, _ = external_sync_service.list_batches(task_id)
    expected_batch_ids = {batch["batch_id"] for batch in batches}
    assert set(generation["input_batch_ids"]) == expected_batch_ids
    assert {
        batch["batch_id"] for batch in stored
        if batch["generation_task_id"] == generation_task_id
    } == expected_batch_ids
    for batch in stored:
        if batch["batch_id"] in generation["input_batch_ids"]:
            assert batch["status"] == "generation_queued"
            assert batch["generation_task_id"] == generation_task_id
            assert batch["user_id"] == task["user_id"]
    return generation


# ── Counter Lifecycle Tests ──────────────────────────────


class TestCounterLifecycle:
    """Verify counter increment, accumulation, and reset behavior."""

    def test_pending_records_increment(self, pipeline_mock_server):
        """sync-now should increment pending_record_count by the number of fetched records."""
        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            # Initial state: counters at 0
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status")
            assert r.json()["pending_record_count"] == 0

            # Sync once
            r = requests.post(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/sync-now")
            assert r.status_code == 200

            # After sync: pending should equal fetched count
            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status")
            status = r.json()
            assert status["pending_record_count"] == 30
            assert status["total_record_count"] == 30
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_total_records_never_reset(self, pipeline_mock_server):
        """total_record_count accumulates and is never reset (unlike pending)."""
        from train_factory.storage.services.external_sync_service import (
            external_sync_service,
        )

        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            # Sync to get records
            requests.post(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/sync-now")

            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status")
            assert r.json()["total_record_count"] == 30

            # Reset pending (simulates what _trigger_generation does)
            external_sync_service.reset_pending_records(sync_cfg["task_id"])

            r = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status")
            status = r.json()
            assert status["pending_record_count"] == 0, "pending should be reset"
            assert status["total_record_count"] == 30, "total should NOT be reset"
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_training_sample_counter_increment_and_reset(self):
        """pending_training_samples increments on generation complete, resets on training trigger."""
        from train_factory.storage.services.external_sync_service import (
            external_sync_service,
        )

        cfg = external_sync_service.create_task(
            task_name="pytest-counter-test",
            user_id=TEST_USER_ID,
            external_api_url="http://unused",
            external_auth_config={"token": "x"},
            training_threshold=100,
        )
        try:
            config_id = cfg["task_id"]

            # Increment training samples (simulates generation completion)
            external_sync_service.increment_pending_training_samples(config_id, 50)
            c1 = external_sync_service.get_task_raw(config_id)
            assert c1["pending_training_samples"] == 50
            assert c1["total_training_samples"] == 50

            # Increment again (second generation)
            external_sync_service.increment_pending_training_samples(config_id, 60)
            c2 = external_sync_service.get_task_raw(config_id)
            assert c2["pending_training_samples"] == 110
            assert c2["total_training_samples"] == 110

            # Reset pending (simulates training trigger)
            external_sync_service.reset_pending_training_samples(config_id)
            c3 = external_sync_service.get_task_raw(config_id)
            assert c3["pending_training_samples"] == 0, "pending should be reset"
            assert c3["total_training_samples"] == 110, "total should NOT be reset"
        finally:
            external_sync_service.delete_task(cfg["task_id"])


# ── Level 2 Handler Tests ────────────────────────────────


class TestLevel2Handler:
    """Verify on_generation_completed callback and training threshold logic."""

    def test_non_sync_generation_ignored(self):
        """on_generation_completed should silently ignore generations not triggered by sync."""
        from train_factory.sync.level2_handler import on_generation_completed

        # Call with a non-existent generation task ID — should not raise
        on_generation_completed(
            generation_task_id="non-existent-task",
            output_dataset_id="ds-001",
            output_sample_count=100,
        )

    def test_generation_complete_updates_counters(self, pipeline_mock_server):
        """on_generation_completed should increment pending_training_samples."""
        from train_factory.storage.services.external_sync_service import (
            external_sync_service,
        )
        from train_factory.sync.level2_handler import on_generation_completed

        gen_task_id = f"test-gen-{uuid.uuid4().hex[:12]}"

        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(
            api_cfg["config_id"],
            training_threshold=99999,  # won't trigger training
        )
        try:
            config_id = sync_cfg["task_id"]

            # Sync to create batches
            synced = requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")
            assert synced.status_code == 200, synced.text

            # Get the batches
            batches_resp = requests.get(f"{BASE}/sync/tasks/{config_id}/batches")
            batches = batches_resp.json()["batches"]
            batch_ids = [b["batch_id"] for b in batches]
            assert batches and sum(batch["record_count"] for batch in batches) == 30

            _claim_generation_batches(config_id, gen_task_id, batches)

            # Before callback: training samples = 0
            before = external_sync_service.get_task_raw(config_id)
            assert before["pending_training_samples"] == 0

            # Call the Level 2 callback
            completion = on_generation_completed(
                generation_task_id=gen_task_id,
                output_dataset_id="fake-output-ds",
                output_sample_count=150,
            )
            assert completion["completed"] is True
            assert completion["credited_sample_count"] == 150
            assert completion["completed_batch_count"] == len(batch_ids)

            # After callback: training samples should be incremented
            after = external_sync_service.get_task_raw(config_id)
            assert after["pending_training_samples"] == 150
            assert after["total_training_samples"] == 150

            # Generation record should be updated
            gen_record = external_sync_service.get_generation_by_task_id(gen_task_id)
            assert gen_record["status"] == "completed"
            assert gen_record["output_dataset_id"] == "fake-output-ds"
            assert gen_record["output_sample_count"] == 150

            # Batch statuses should be updated to generation_done
            batches_after = requests.get(f"{BASE}/sync/tasks/{config_id}/batches")
            for b in batches_after.json()["batches"]:
                if b["batch_id"] in batch_ids:
                    assert b["status"] == "generation_done"
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_training_threshold_check(self, tmp_path):
        """on_generation_completed should trigger training when threshold reached."""
        from train_factory.storage.services.external_sync_service import (
            external_sync_service,
        )
        from unittest.mock import patch
        from train_factory.sync.level2_handler import on_generation_completed

        gen_task_id = f"test-gen-{uuid.uuid4().hex[:12]}"

        cfg = external_sync_service.create_task(
            task_name="pytest-l2-threshold",
            user_id=TEST_USER_ID,
            external_api_url="http://unused",
            external_auth_config={"token": "x"},
            generation_threshold=99999,
            training_threshold=100,
        )
        try:
            config_id = cfg["task_id"]

            batch = _create_owned_input_batch(cfg, tmp_path)
            _claim_generation_batches(config_id, gen_task_id, [batch])

            # Mock _trigger_training to verify it's called
            with patch(
                "train_factory.sync.level2_handler._trigger_training"
            ) as mock_train:
                # 150 samples >= 100 threshold → should trigger
                completion = on_generation_completed(
                    generation_task_id=gen_task_id,
                    output_dataset_id="ds-output",
                    output_sample_count=150,
                )
                assert completion["completed"] is True
                assert completion["credited_sample_count"] == 150
                mock_train.assert_called_once()

            after = external_sync_service.get_task_raw(config_id)
            assert after["pending_training_samples"] == 150
        finally:
            external_sync_service.delete_task(config_id)

    def test_training_not_triggered_below_threshold(self, tmp_path):
        """on_generation_completed should NOT trigger training when below threshold."""
        from train_factory.storage.services.external_sync_service import (
            external_sync_service,
        )
        from unittest.mock import patch
        from train_factory.sync.level2_handler import on_generation_completed

        gen_task_id = f"test-gen-{uuid.uuid4().hex[:12]}"

        cfg = external_sync_service.create_task(
            task_name="pytest-l2-below",
            user_id=TEST_USER_ID,
            external_api_url="http://unused",
            external_auth_config={"token": "x"},
            generation_threshold=99999,
            training_threshold=1000,
        )
        try:
            config_id = cfg["task_id"]

            batch = _create_owned_input_batch(cfg, tmp_path)
            _claim_generation_batches(config_id, gen_task_id, [batch])

            with patch(
                "train_factory.sync.level2_handler._trigger_training"
            ) as mock_train:
                # 50 samples < 1000 threshold → should NOT trigger
                completion = on_generation_completed(
                    generation_task_id=gen_task_id,
                    output_dataset_id="ds-below",
                    output_sample_count=50,
                )
                assert completion["completed"] is True
                assert completion["credited_sample_count"] == 50
                mock_train.assert_not_called()
            assert external_sync_service.get_task_raw(config_id)["pending_training_samples"] == 50
        finally:
            external_sync_service.delete_task(config_id)

    def test_training_skipped_when_already_in_progress(self, tmp_path):
        """on_generation_completed should skip training if status is 'training'."""
        from train_factory.storage.services.external_sync_service import (
            external_sync_service,
        )
        from unittest.mock import patch
        from train_factory.sync.level2_handler import on_generation_completed

        gen_task_id = f"test-gen-{uuid.uuid4().hex[:12]}"

        cfg = external_sync_service.create_task(
            task_name="pytest-l2-busy",
            user_id=TEST_USER_ID,
            external_api_url="http://unused",
            external_auth_config={"token": "x"},
            generation_threshold=99999,
            training_threshold=10,
        )
        try:
            config_id = cfg["task_id"]

            batch = _create_owned_input_batch(cfg, tmp_path)
            _claim_generation_batches(config_id, gen_task_id, [batch])
            # Another training is active by the time this claimed generation completes.
            external_sync_service.update_task(config_id, status="training")

            with patch(
                "train_factory.sync.level2_handler._trigger_training"
            ) as mock_train:
                # 100 samples >= 10 threshold, but status="training" → skip
                completion = on_generation_completed(
                    generation_task_id=gen_task_id,
                    output_dataset_id="ds-busy",
                    output_sample_count=100,
                )
                assert completion["completed"] is True
                assert completion["credited_sample_count"] == 100
                mock_train.assert_not_called()
            after = external_sync_service.get_task_raw(config_id)
            assert after["pending_training_samples"] == 100
            assert after["status"] == "training"
        finally:
            external_sync_service.delete_task(config_id)

    def test_generation_claim_rejects_wrong_owner_without_consuming_batch(self, tmp_path):
        from train_factory.storage.services.external_sync_service import external_sync_service

        task = external_sync_service.create_task(
            task_name=f"pytest-owner-fence-{uuid.uuid4().hex[:12]}",
            user_id=TEST_USER_ID,
            external_api_url="http://unused",
            external_auth_config={"token": "x"},
        )
        generation_id = f"test-gen-{uuid.uuid4().hex[:12]}"
        try:
            batch = _create_owned_input_batch(task, tmp_path)
            external_sync_service.update_task(task["task_id"], status="generating")
            with pytest.raises(ValueError, match="batches are no longer available"):
                external_sync_service.create_generation_and_claim_batches(
                    task_id=task["task_id"],
                    generation_task_id=generation_id,
                    user_id=OTHER_USER_ID,
                    input_batch_ids=[batch["batch_id"]],
                    input_record_count=50,
                )
            assert external_sync_service.get_generation_by_task_id(generation_id) is None
            batches, _ = external_sync_service.list_batches(task["task_id"])
            assert len(batches) == 1
            assert batches[0]["status"] == "fetched"
            assert batches[0]["generation_task_id"] is None
            assert external_sync_service.get_task_raw(task["task_id"])["pending_training_samples"] == 0
        finally:
            external_sync_service.delete_task(task["task_id"])


# ── Manual Trigger API Tests ─────────────────────────────


class TestManualTriggerAPI:
    """Verify sync-now, trigger-generation, trigger-training endpoints do real work."""

    def test_sync_now_rejects_future_records_without_advancing_checkpoint(
        self, pipeline_mock_server, monkeypatch,
    ):
        data = _make_mock_data(2)
        data[-1]["created_at"] = (now_naive() + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        monkeypatch.setitem(globals(), "MOCK_DATA", data)
        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            response = requests.post(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/sync-now")
            assert response.status_code == 200
            status = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/status").json()
            assert status["status"] == "error"
            detail = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}")
            assert detail.status_code == 200
            assert "future skew limit" in detail.json()["task"]["error_message"]
            assert status["pending_record_count"] == 0
            assert status["total_record_count"] == 0
            assert status["last_sync_at"] is None
            batches = requests.get(f"{BASE}/sync/tasks/{sync_cfg['task_id']}/batches").json()
            assert batches["total"] == 0
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_sync_now_creates_batch(self, pipeline_mock_server):
        """POST sync-now should fetch data from external API and create a batch record."""
        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            config_id = sync_cfg["task_id"]

            # Before: no batches
            r = requests.get(f"{BASE}/sync/tasks/{config_id}/batches")
            assert r.json()["total"] == 0

            # Trigger sync
            r = requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")
            assert r.status_code == 200

            # After: batch created with real data
            r = requests.get(f"{BASE}/sync/tasks/{config_id}/batches")
            data = r.json()
            assert data["total"] >= 1
            assert data["batches"][0]["record_count"] == 30
            assert data["batches"][0]["status"] == "fetched"

            # Verify the JSONL file exists and has correct content
            batch_path = data["batches"][0]["storage_path"]
            assert os.path.exists(batch_path)
            with open(batch_path) as f:
                lines = f.readlines()
            assert len(lines) == 30
            first = json.loads(lines[0])
            assert first["metadata"]["source"] == "pipeline-test"
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_sync_now_deduplicates_on_repeat(self, pipeline_mock_server):
        """Calling sync-now twice with same data should not create duplicate records."""
        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            config_id = sync_cfg["task_id"]

            # First sync
            requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")
            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            count_after_first = r.json()["pending_record_count"]
            assert count_after_first == 30

            # Second sync (same data source, no new records)
            requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")
            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            count_after_second = r.json()["pending_record_count"]

            assert count_after_second == count_after_first, (
                f"Duplicate records detected: {count_after_first} → {count_after_second}"
            )
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_sync_now_updates_last_sync_at(self, pipeline_mock_server):
        """sync-now should update last_sync_at to the max timestamp of fetched data."""
        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            config_id = sync_cfg["task_id"]

            # Before: no last_sync_at
            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            assert r.json().get("last_sync_at") is None

            requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")

            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            assert r.json()["last_sync_at"] is not None
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_sync_now_with_invalid_api_sets_error(self):
        """sync-now with unreachable API should set status to 'error'."""
        api_cfg = _create_api_config(
            f"http://{EXTERNAL_API_HOST}:19999/nonexistent",
            token="bad",
        )
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            config_id = sync_cfg["task_id"]

            requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")

            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            status = r.json()
            assert status["status"] == "error"
            assert status["pending_record_count"] == 0
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])


# ── Runtime Resolution Tests ─────────────────────────────


class TestRuntimeResolution:
    """Verify that sync resolves API URL and token from external API config at runtime."""

    def test_sync_uses_resolved_url(self, pipeline_mock_server):
        """When referencing an API config, sync-now should use the URL from that config."""
        api_cfg = _create_api_config(pipeline_mock_server)
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            config_id = sync_cfg["task_id"]

            # sync-now should succeed because it resolves the URL from api_config
            r = requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")
            assert r.status_code == 200

            # Verify data was actually fetched
            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            assert r.json()["pending_record_count"] == 30
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])

    def test_token_rotation_takes_effect(self, pipeline_mock_server):
        """Updating the token in API config should take effect on next sync."""
        api_cfg = _create_api_config(pipeline_mock_server, token="valid-token")
        sync_cfg = _create_sync_config(api_cfg["config_id"])
        try:
            config_id = sync_cfg["task_id"]

            # First sync succeeds
            r = requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")
            assert r.status_code == 200

            # Update token to invalid one
            requests.patch(
                f"{BASE}/sync/api-configs/{api_cfg['config_id']}",
                json={"auth_config": {"token": ""}},
            )

            # Second sync should fail (empty token → 401 from mock server)
            requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")

            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            assert r.json()["status"] == "error"

            # Restore valid token
            requests.patch(
                f"{BASE}/sync/api-configs/{api_cfg['config_id']}",
                json={"auth_config": {"token": "restored-token"}},
            )

            # Third sync should succeed again
            r = requests.post(f"{BASE}/sync/tasks/{config_id}/sync-now")
            assert r.status_code == 200

            r = requests.get(f"{BASE}/sync/tasks/{config_id}/status")
            # Status should recover to idle after successful sync
            assert r.json()["status"] in ("idle", "syncing")
        finally:
            _cleanup(sync_cfg["task_id"], api_cfg["config_id"])
