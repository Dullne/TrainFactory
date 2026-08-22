"""
Dataset type → schema mapping for TrainFactory.

Provides automatic schema inference based on dataset_type, used for:
1. Auto-populating content_schema on dataset creation
2. Detecting primary/secondary text fields for preview and search
3. Extracting searchable fields for dataset_records (hot-record mode)
"""

from typing import Any, Dict, Optional, Tuple


# Maps dataset_type → field definitions + primary/secondary text fields.
DATASET_TYPE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "embedding_universal": {
        "fields": {
            "query": "string",
            "answer": "string",
            "positives": "list[string]",
            "negatives": "list[string]",
        },
        "primary_text_field": "query",
        "secondary_text_field": "answer",
    },
    "embedding_pair": {
        "fields": {"query": "string", "positive": "string"},
        "primary_text_field": "query",
        "secondary_text_field": "positive",
    },
    "embedding_triplet": {
        "fields": {"query": "string", "positive": "string", "negative": "string"},
        "primary_text_field": "query",
        "secondary_text_field": "positive",
    },
    "embedding_cosine": {
        "fields": {"text1": "string", "text2": "string", "label": "float"},
        "primary_text_field": "text1",
        "secondary_text_field": "text2",
        "label_field": "label",
    },
    "rerank_pair": {
        "fields": {"query": "string", "document": "string", "label": "float"},
        "primary_text_field": "query",
        "secondary_text_field": "document",
        "label_field": "label",
    },
    "rerank_triplet": {
        "fields": {"query": "string", "positive": "string", "negative": "string"},
        "primary_text_field": "query",
        "secondary_text_field": "positive",
    },
    "rerank_listwise": {
        "fields": {"query": "string", "documents": "list[string]", "labels": "list[float]"},
        "primary_text_field": "query",
    },
    "sft_instruct": {
        "fields": {"instruction": "string", "response": "string"},
        "primary_text_field": "instruction",
        "secondary_text_field": "response",
    },
    "dpo_preference": {
        "fields": {"instruction": "string", "chosen": "string", "rejected": "string"},
        "primary_text_field": "instruction",
    },
    "rl_reward": {
        "fields": {"instruction": "string", "response": "string", "reward": "float"},
        "primary_text_field": "instruction",
        "label_field": "reward",
    },
    "qa_pair": {
        "fields": {"query": "string", "answer": "string"},
        "primary_text_field": "query",
        "secondary_text_field": "answer",
    },
    "custom": {
        "fields": {},
        "primary_text_field": None,
        "secondary_text_field": None,
    },
}


def infer_schema(dataset_type: str) -> Dict[str, Any]:
    """Return the content_schema dict for *dataset_type*.

    Falls back to ``"custom"`` when the type is unknown.
    """
    return DATASET_TYPE_SCHEMA.get(dataset_type, DATASET_TYPE_SCHEMA["custom"])


def detect_preview_fields(dataset_type: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(primary_text_field, secondary_text_field)`` for preview/search."""
    schema = infer_schema(dataset_type)
    return schema.get("primary_text_field"), schema.get("secondary_text_field")


def extract_searchable_fields(data: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
    """Extract searchable columns from a JSON record.

    Used when writing rows into ``dataset_records`` (hot-record mode).
    Returns a dict with optional keys: ``text_query``, ``text_content``, ``label``.
    """
    schema = infer_schema(dataset_type)
    result: Dict[str, Any] = {}

    primary = schema.get("primary_text_field")
    if primary and primary in data:
        val = data[primary]
        if isinstance(val, str):
            result["text_query"] = val

    secondary = schema.get("secondary_text_field")
    if secondary and secondary in data:
        val = data[secondary]
        if isinstance(val, str):
            result["text_content"] = val

    label_field = schema.get("label_field")
    if label_field and label_field in data:
        try:
            result["label"] = float(data[label_field])
        except (TypeError, ValueError):
            pass

    return result
