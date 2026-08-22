import json
from pathlib import Path

from train_factory.generation.pipeline import DatasetGenerationPipeline, PipelineConfig


def test_jsonl_prefers_metadata_chunk_id(tmp_path: Path):
    input_path = tmp_path / "sync_batch.jsonl"
    records = [
        {
            "content": "hello",
            "metadata": {
                "external_id": "ext-1",
                "session_id": "sess-1",
                "doc_id": "doc-1",
                "chunk_id": "ext-1:sess-1:doc-1",
            },
        },
        {
            "content": "world",
            "metadata": {
                "external_id": "ext-2",
            },
        },
    ]
    with open(input_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pipeline = DatasetGenerationPipeline(PipelineConfig(input_path=str(input_path)))
    docs = pipeline._load_file(input_path, "jsonl")

    assert len(docs) == 2
    assert docs[0].doc_id == "ext-1:sess-1:doc-1"
    # Fallback to available metadata id when chunk_id is missing.
    assert docs[1].doc_id == "ext-2"
