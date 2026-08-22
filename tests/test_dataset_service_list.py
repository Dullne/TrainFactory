from datetime import datetime

from train_factory.storage.services.dataset_service import DatasetService


class _FakeRow:
    def __init__(self, **kwargs):
        self._mapping = kwargs


def test_dataset_list_row_is_response_compatible_and_lightweight():
    row = _FakeRow(
        dataset_id="ds-1",
        dataset_name="unit-dataset",
        display_name="Unit Dataset",
        description="desc",
        dataset_type="embedding_universal",
        usage="train",
        model_type=["embedding"],
        source_type="generated",
        source_path="/tmp/data.jsonl",
        remote_repo=None,
        hf_subset=None,
        storage_path="/tmp/data.jsonl",
        storage_backend="local",
        storage_uri=None,
        version=1,
        file_format="jsonl",
        num_rows=12,
        num_train=12,
        num_eval=None,
        num_test=None,
        file_size=1024,
        source_task_type="generation",
        source_task_id="task-1",
        tags=["auto-generated"],
        status="ready",
        error_message=None,
        user_id="user-1",
        created_at=datetime(2026, 1, 1, 0, 0, 0),
        updated_at=datetime(2026, 1, 2, 0, 0, 0),
    )

    payload = DatasetService._row_to_list_dict(row)

    assert payload["dataset_id"] == "ds-1"
    assert payload["usage"] == "train"
    assert payload["num_rows"] == 12
    assert payload["tags"] == ["auto-generated"]
    assert payload["columns"] is None
    assert payload["sample_data"] is None
    assert payload["content_schema"] is None
    assert payload["extra_metadata"] is None
