from datasets import Dataset

from train_factory.data.formats.dpo import (
    PreferenceRankedFormat,
    infer_rankings_higher_is_better,
    is_preferred_ranking,
    resolve_rankings_higher_is_better,
    sort_ranked_preference_pairs,
)
from train_factory.data.preprocessors.dpo_preprocessor import DPOPreprocessor
from train_factory.trainers.decoder.llm_trainer import LLMTrainer


class DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {"input_ids": [1], "attention_mask": [1]}

    def encode(self, text, add_special_tokens=False):
        return [1] if text else []


def build_preprocessor():
    return DPOPreprocessor(tokenizer=DummyTokenizer())


def build_preprocessor_with_config(config):
    return DPOPreprocessor(tokenizer=DummyTokenizer(), config=config)


def test_preprocess_standard_preference_format():
    dataset = Dataset.from_dict({
        "prompt": ["What is DPO?"],
        "chosen": ["A preference-optimization method."],
        "rejected": ["I do not know."],
        "chosen_score": [0.9],
        "rejected_score": [0.1],
    })

    processed = build_preprocessor().preprocess(dataset)
    sample = processed[0]

    assert processed.column_names == [
        "prompt", "chosen", "rejected", "chosen_score", "rejected_score"
    ]
    assert sample["prompt"] == "What is DPO?"
    assert sample["chosen"] == "A preference-optimization method."
    assert sample["rejected"] == "I do not know."
    assert sample["chosen_score"] == 0.9
    assert sample["rejected_score"] == 0.1


def test_preprocess_instruction_and_dict_response_format():
    dataset = Dataset.from_dict({
        "instruction": ["Summarize this text"],
        "chosen": [{"content": "Good summary"}],
        "rejected": [{"response": "Bad summary"}],
    })

    processed = build_preprocessor().preprocess(dataset)
    sample = processed[0]

    assert sample == {
        "prompt": "Summarize this text",
        "chosen": "Good summary",
        "rejected": "Bad summary",
    }


def test_preprocess_query_positive_negative_list_format():
    dataset = Dataset.from_dict({
        "query": ["search query"],
        "positive": [["best answer", "backup answer"]],
        "negatives": [["", "wrong answer"]],
    })

    processed = build_preprocessor().preprocess(dataset)
    sample = processed[0]

    assert sample == {
        "prompt": "search query",
        "chosen": "best answer",
        "rejected": "wrong answer",
    }


def test_preprocess_ranked_responses_uses_best_and_worst():
    dataset = Dataset.from_dict({
        "prompt": ["Rank these"],
        "responses": [["best", "middle", "worst"]],
        "rankings": [[0.95, 0.6, 0.1]],
    })

    processed = build_preprocessor().preprocess(dataset)
    sample = processed[0]

    assert sample == {
        "prompt": "Rank these",
        "chosen": "best",
        "rejected": "worst",
        "chosen_score": 0.95,
        "rejected_score": 0.1,
    }


def test_preprocess_ranked_positions_use_lowest_rank_as_best():
    dataset = Dataset.from_dict({
        "prompt": ["Rank these"],
        "responses": [["best", "middle", "worst"]],
        "rankings": [[1, 2, 3]],
    })

    processed = build_preprocessor().preprocess(dataset)
    sample = processed[0]

    assert sample == {
        "prompt": "Rank these",
        "chosen": "best",
        "rejected": "worst",
        "chosen_score": 1.0,
        "rejected_score": 3.0,
    }


def test_preprocess_ranked_positions_respects_explicit_higher_is_better():
    dataset = Dataset.from_dict({
        "prompt": ["Rank these"],
        "responses": [["lowest", "middle", "highest"]],
        "rankings": [[1, 2, 3]],
    })

    processed = build_preprocessor_with_config({
        "rl_config": {"rankings_direction": "higher_is_better"},
    }).preprocess(dataset)
    sample = processed[0]

    assert sample == {
        "prompt": "Rank these",
        "chosen": "highest",
        "rejected": "lowest",
        "chosen_score": 3.0,
        "rejected_score": 1.0,
    }


def test_preprocess_filters_invalid_samples():
    dataset = Dataset.from_dict({
        "instruction": ["valid", ""],
        "chosen": ["good", "missing prompt"],
        "rejected": ["bad", "still invalid"],
    })

    processed = build_preprocessor().preprocess(dataset)

    assert len(processed) == 1
    assert processed[0] == {
        "prompt": "valid",
        "chosen": "good",
        "rejected": "bad",
    }


def test_ranked_samples_require_two_valid_response_score_pairs():
    dataset = Dataset.from_dict({
        "prompt": ["keep", "drop"],
        "responses": [["best", "worst"], ["usable", "", "fallback"]],
        "rankings": [[0.9, 0.1], [None, 1.0, 2.0]],
    })

    processed = build_preprocessor().preprocess(dataset)

    assert len(processed) == 1
    assert processed[0] == {
        "prompt": "keep",
        "chosen": "best",
        "rejected": "worst",
        "chosen_score": 0.9,
        "rejected_score": 0.1,
    }


def test_ranked_samples_with_equal_edge_scores_are_kept_when_min_max_differ():
    dataset = Dataset.from_dict({
        "prompt": ["keep"],
        "responses": [["best-a", "worst", "best-b"]],
        "rankings": [[1.0, 0.0, 1.0]],
    })

    processed = build_preprocessor().preprocess(dataset)

    assert len(processed) == 1
    assert processed[0] == {
        "prompt": "keep",
        "chosen": "best-a",
        "rejected": "worst",
        "chosen_score": 1.0,
        "rejected_score": 0.0,
    }


def test_llm_trainer_preference_pipeline_uses_dpo_preprocessor(monkeypatch):
    dataset = Dataset.from_dict({
        "prompt": ["Rank these"],
        "responses": [["best", "middle", "worst"]],
        "rankings": [[0.95, 0.6, 0.1]],
    })

    trainer = LLMTrainer({
        "training_method": "dpo",
        "dataset_configs": [{"path": "unused", "split": "train"}],
        "max_length": 1024,
    })

    def fake_load_datasets(dataset_configs, split_filter=None):
        return dataset if split_filter == "train" else None

    monkeypatch.setattr(trainer, "_load_datasets", fake_load_datasets)

    train_dataset, eval_dataset = trainer._load_and_normalize_datasets(
        "preference",
        tokenizer=DummyTokenizer(),
    )

    assert eval_dataset is None
    assert len(train_dataset) == 1
    assert train_dataset[0] == {
        "prompt": "Rank these",
        "chosen": "best",
        "rejected": "worst",
        "chosen_score": 0.95,
        "rejected_score": 0.1,
    }


def test_preference_ranked_format_uses_lowest_rank_as_best():
    sample = {
        "prompt": "Rank these",
        "responses": ["best", "middle", "worst"],
        "rankings": [1, 2, 3],
    }

    parsed = PreferenceRankedFormat().parse(sample)

    assert parsed.prompt == "Rank these"
    assert parsed.chosen == "best"
    assert parsed.rejected == "worst"
    assert parsed.chosen_score == 1
    assert parsed.rejected_score == 3


def test_preference_ranked_format_builds_pairs_from_rank_positions():
    sample = {
        "prompt": "Rank these",
        "responses": ["best", "middle", "worst"],
        "rankings": [1, 2, 3],
    }

    pairs = PreferenceRankedFormat().parse_all_pairs(sample)

    assert {(pair.chosen, pair.rejected) for pair in pairs} == {
        ("best", "middle"),
        ("best", "worst"),
        ("middle", "worst"),
    }


def test_preference_ranked_format_respects_explicit_higher_is_better():
    sample = {
        "prompt": "Rank these",
        "responses": ["lowest", "middle", "highest"],
        "rankings": [1, 2, 3],
    }

    parsed = PreferenceRankedFormat(rankings_direction="higher_is_better").parse(sample)

    assert parsed.chosen == "highest"
    assert parsed.rejected == "lowest"
    assert parsed.chosen_score == 3
    assert parsed.rejected_score == 1


def test_infer_rankings_higher_is_better_distinguishes_ordinal_and_score_inputs():
    assert infer_rankings_higher_is_better([1, 2, 3]) is False
    assert infer_rankings_higher_is_better([0, 1, 2]) is False
    assert infer_rankings_higher_is_better([0.9, 0.6, 0.3]) is True
    # Permutations of consecutive integers are still treated as ordinal ranks
    # because the implementation deduplicates and sorts before comparison.
    assert infer_rankings_higher_is_better([1, 3, 2]) is False
    # Integers whose sorted values are not [0..n) or [1..n] are scores.
    assert infer_rankings_higher_is_better([1, 3, 4]) is True
    assert infer_rankings_higher_is_better([5, 4, 3]) is True
    assert infer_rankings_higher_is_better([10, 5, 1]) is True


def test_infer_rankings_with_ties_is_ordinal():
    # 1-based ordinal ranks with ties must be treated as ordinal (lower better),
    # not fall through to score semantics — otherwise chosen/rejected invert.
    assert infer_rankings_higher_is_better([1, 1, 2]) is False
    assert infer_rankings_higher_is_better([1, 2, 2, 3]) is False
    # Float scores with ties stay score semantics (higher better).
    assert infer_rankings_higher_is_better([0.8, 0.8, 0.9]) is True
    # 0-containing tied rankings are ambiguous -> score semantics, matching the
    # {0=bad, 1=good} convention (keeps chosen=high-score behavior).
    assert infer_rankings_higher_is_better([1, 0, 1]) is True
    assert infer_rankings_higher_is_better([0, 0, 1]) is True


def test_resolve_rankings_higher_is_better_honors_explicit_and_auto_modes():
    assert resolve_rankings_higher_is_better([1, 2, 3], "higher_is_better") is True
    assert resolve_rankings_higher_is_better([1, 2, 3], "lower_is_better") is False
    assert resolve_rankings_higher_is_better([1, 2, 3], "auto") is False
    assert resolve_rankings_higher_is_better([0.9, 0.6], "auto") is True


def test_is_preferred_ranking_handles_explicit_and_inferred_directions():
    assert is_preferred_ranking(3, 1, higher_is_better=True) is True
    assert is_preferred_ranking(1, 3, higher_is_better=True) is False
    assert is_preferred_ranking(1, 3, higher_is_better=False) is True
    assert is_preferred_ranking(3, 1, higher_is_better=None) is True


def test_sort_ranked_preference_pairs_sorts_for_higher_lower_and_auto_directions():
    pairs = [("middle", 2), ("best", 3), ("worst", 1)]

    assert sort_ranked_preference_pairs(pairs, higher_is_better=True) == [
        ("best", 3),
        ("middle", 2),
        ("worst", 1),
    ]
    assert sort_ranked_preference_pairs(pairs, higher_is_better=False) == [
        ("worst", 1),
        ("middle", 2),
        ("best", 3),
    ]
    assert sort_ranked_preference_pairs(pairs, rankings_direction="auto") == [
        ("worst", 1),
        ("middle", 2),
        ("best", 3),
    ]
