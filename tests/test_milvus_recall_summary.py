from train_factory.api.routes.milvus_routes import _compute_recall_summary


def test_compute_recall_summary_basic():
    results = [
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
        {"chunk_id": "c3"},
    ]
    summary = _compute_recall_summary(results, ["c2", "c9"], top_k=3)
    assert summary["gold_count"] == 2
    assert summary["hit_count"] == 1
    assert summary["recall_at_k"] == 0.5
    assert summary["hit_chunk_ids"] == ["c2"]
    assert summary["miss_chunk_ids"] == ["c9"]


def test_compute_recall_summary_with_duplicates_and_blanks():
    results = [
        {"chunk_id": "c1"},
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
    ]
    summary = _compute_recall_summary(results, ["", "c1", "c1", "  c2  "], top_k=3)
    assert summary["gold_count"] == 2
    assert summary["hit_count"] == 2
    assert summary["recall_at_k"] == 1.0
    assert summary["hit_chunk_ids"] == ["c1", "c2"]
    assert summary["miss_chunk_ids"] == []


def test_compute_recall_summary_top_k_cutoff():
    results = [
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
        {"chunk_id": "c3"},
    ]
    summary = _compute_recall_summary(results, ["c3"], top_k=2)
    assert summary["gold_count"] == 1
    assert summary["hit_count"] == 0
    assert summary["recall_at_k"] == 0.0
    assert summary["hit_chunk_ids"] == []
    assert summary["miss_chunk_ids"] == ["c3"]
