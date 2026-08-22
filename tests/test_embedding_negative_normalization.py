"""Embedding negatives use one shared scalar/list normalization rule."""

import pytest
from datasets import Dataset

from train_factory.data.formats.base import DataFormatType
from train_factory.data.formats.embedding import (
    QPNFormat,
    detect_embedding_format,
    normalize_embedding_negatives,
)
from train_factory.trainers.encoder.embedding_trainer import EmbeddingTrainer


def test_scalar_negative_is_one_item_not_a_character_sequence():
    sample = {"query": "q", "positive": "p", "negative": "hard negative"}
    data_format = QPNFormat()

    assert data_format.validate(sample) is True
    assert data_format.parse(sample).negatives == ["hard negative"]
    assert normalize_embedding_negatives("hard negative") == ["hard negative"]
    assert detect_embedding_format(sample) == DataFormatType.QUERY_POSITIVE_NEGATIVES


@pytest.mark.parametrize(
    "value",
    ["", "   ", [], [""], ["valid", "  "], None, 123],
)
def test_empty_or_invalid_negatives_are_rejected(value):
    sample = {"query": "q", "positive": "p", "negatives": value}
    data_format = QPNFormat()

    assert data_format.validate(sample) is False
    with pytest.raises(ValueError, match="negative"):
        data_format.parse(sample)


def test_list_negatives_preserve_item_boundaries():
    negatives = ["first", "second"]
    sample = {"query": "q", "positive": "p", "negatives": negatives}

    assert QPNFormat().parse(sample).negatives == negatives


def test_trainer_uses_shared_normalizer_before_counting_columns():
    dataset = Dataset.from_dict(
        {
            "query": ["q"],
            "positives": [["p"]],
            "negatives": ["single hard negative"],
        }
    )
    trainer = object.__new__(EmbeddingTrainer)

    prepared = trainer._prepare_datasets({"train": dataset})["train"]

    assert prepared.column_names == ["anchor", "positive", "negative_0"]
    assert prepared[0] == {
        "anchor": "q",
        "positive": "p",
        "negative_0": "single hard negative",
    }
