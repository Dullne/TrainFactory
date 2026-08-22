"""DPO (Direct Preference Optimization) data formats."""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .base import (
    BaseDataFormat,
    DataFormatType,
    PreferenceSample,
)


class PreferenceFormat(BaseDataFormat):
    """
    Basic preference format for DPO.

    Expected columns:
        - prompt: The input prompt
        - chosen: The preferred response
        - rejected: The rejected response
        - chosen_score: Optional score for chosen (optional)
        - rejected_score: Optional score for rejected (optional)
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.PREFERENCE

    @property
    def required_columns(self) -> List[str]:
        return ["prompt", "chosen", "rejected"]

    @property
    def optional_columns(self) -> List[str]:
        return ["chosen_score", "rejected_score"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False
        return (
            bool(sample.get("prompt"))
            and bool(sample.get("chosen"))
            and bool(sample.get("rejected"))
        )

    def parse(self, sample: Dict[str, Any]) -> PreferenceSample:
        return PreferenceSample(
            prompt=sample["prompt"],
            chosen=sample["chosen"],
            rejected=sample["rejected"],
            chosen_score=sample.get("chosen_score"),
            rejected_score=sample.get("rejected_score"),
        )


RankingsDirection = Literal["auto", "higher_is_better", "lower_is_better"]


class PreferenceRankedFormat(BaseDataFormat):
    """
    Ranked preference format for DPO with multiple responses.

    Expected columns:
        - prompt: The input prompt
        - responses: List of responses
        - rankings: List of rankings (1 = best, higher = worse)
            or scores (higher = better)
    """

    def __init__(self, rankings_direction: RankingsDirection = "auto"):
        self.rankings_direction = rankings_direction

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.PREFERENCE_RANKED

    @property
    def required_columns(self) -> List[str]:
        return ["prompt", "responses", "rankings"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False

        responses = sample.get("responses", [])
        rankings = sample.get("rankings", [])

        if not isinstance(responses, list) or len(responses) < 2:
            return False
        if not isinstance(rankings, list) or len(rankings) != len(responses):
            return False

        return bool(sample.get("prompt"))

    def parse(self, sample: Dict[str, Any]) -> PreferenceSample:
        """Parse into a preference sample using best and worst responses."""
        responses = sample["responses"]
        rankings = sample["rankings"]

        paired = sort_ranked_preference_pairs(
            list(zip(responses, rankings)),
            rankings_direction=self.rankings_direction,
        )

        return PreferenceSample(
            prompt=sample["prompt"],
            chosen=paired[0][0],
            rejected=paired[-1][0],
            chosen_score=paired[0][1],
            rejected_score=paired[-1][1],
        )

    def parse_all_pairs(self, sample: Dict[str, Any]) -> List[PreferenceSample]:
        """
        Parse into multiple preference pairs from rankings.

        Returns all preference pairs using the inferred ranking direction.
        """
        responses = sample["responses"]
        rankings = sample["rankings"]
        prompt = sample["prompt"]
        higher_is_better = resolve_rankings_higher_is_better(
            rankings,
            rankings_direction=self.rankings_direction,
        )

        pairs = []
        for i, (resp_i, score_i) in enumerate(zip(responses, rankings)):
            for j, (resp_j, score_j) in enumerate(zip(responses, rankings)):
                if i != j and is_preferred_ranking(score_i, score_j, higher_is_better):
                    pairs.append(PreferenceSample(
                        prompt=prompt,
                        chosen=resp_i,
                        rejected=resp_j,
                        chosen_score=score_i,
                        rejected_score=score_j,
                    ))

        return pairs


def infer_rankings_higher_is_better(rankings: Sequence[Any]) -> bool:
    """
    Infer whether larger ranking values are better.

    TrainFactory documents two ranked-preference conventions:
    - ordinal ranks such as ``1 = best, 2 = worse``
    - numeric scores such as ``0.95 > 0.60 > 0.10``

    To support both without extra user configuration, treat consecutive integer
    sequences like ``[1, 2, 3]`` or ``[0, 1, 2]`` as ordinal ranks; otherwise
    default to score semantics where larger values are preferred.
    """
    normalized = [float(score) for score in rankings]

    if all(score.is_integer() for score in normalized):
        integers = [int(score) for score in normalized]
        unique_values = sorted(set(integers))
        # 1-based ordinal ranks — with or without ties — form [1..k] over the
        # distinct values (e.g. [1,1,2], [1,2,3], [1,2,2,3]). Lower is better.
        if (
            len(unique_values) >= 2
            and unique_values[0] == 1
            and unique_values == list(range(1, len(unique_values) + 1))
        ):
            return False
        # 0-based ordinal ranks WITHOUT ties (legacy [0,1,...,n-1]). Tied
        # 0-containing values (e.g. [1,0,1], [0,0,1]) are ambiguous and fall
        # through to score semantics, matching the common {0=bad, 1=good}
        # score convention so chosen stays the high-score response.
        if (
            len(unique_values) == len(integers)
            and len(unique_values) >= 2
            and unique_values[0] == 0
            and unique_values == list(range(len(unique_values)))
        ):
            return False

    return True


def resolve_rankings_higher_is_better(
    rankings: Sequence[Any],
    rankings_direction: RankingsDirection = "auto",
) -> bool:
    """Resolve ranking direction from explicit config or input-value inference."""
    if rankings_direction == "higher_is_better":
        return True
    if rankings_direction == "lower_is_better":
        return False
    return infer_rankings_higher_is_better(rankings)


def is_preferred_ranking(
    left: Any,
    right: Any,
    higher_is_better: Optional[bool] = None,
) -> bool:
    """Return whether ``left`` is preferred over ``right`` for ranked data."""
    if higher_is_better is None:
        higher_is_better = infer_rankings_higher_is_better([left, right])
    return left > right if higher_is_better else left < right


def sort_ranked_preference_pairs(
    pairs: Sequence[Tuple[Any, Any]],
    higher_is_better: Optional[bool] = None,
    rankings_direction: RankingsDirection = "auto",
) -> List[Tuple[Any, Any]]:
    """Sort ranked preference pairs from best to worst."""
    if higher_is_better is None:
        higher_is_better = resolve_rankings_higher_is_better(
            [score for _, score in pairs],
            rankings_direction=rankings_direction,
        )

    return sorted(pairs, key=lambda item: item[1], reverse=higher_is_better)


class UltraFeedbackFormat(BaseDataFormat):
    """
    UltraFeedback format for DPO.

    Expected columns:
        - instruction: The instruction/prompt
        - chosen: Dict with {"content": ..., "score": ...}
        - rejected: Dict with {"content": ..., "score": ...}
    """

    @property
    def format_type(self) -> DataFormatType:
        return DataFormatType.PREFERENCE

    @property
    def required_columns(self) -> List[str]:
        return ["instruction", "chosen", "rejected"]

    def validate(self, sample: Dict[str, Any]) -> bool:
        if not self.validate_columns(sample):
            return False

        chosen = sample.get("chosen", {})
        rejected = sample.get("rejected", {})

        if not isinstance(chosen, dict) or not isinstance(rejected, dict):
            return False

        return bool(sample.get("instruction"))

    def parse(self, sample: Dict[str, Any]) -> PreferenceSample:
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Handle both dict format and string format
        if isinstance(chosen, dict):
            chosen_content = chosen.get("content", chosen.get("response", ""))
            chosen_score = chosen.get("score")
        else:
            chosen_content = chosen
            chosen_score = None

        if isinstance(rejected, dict):
            rejected_content = rejected.get("content", rejected.get("response", ""))
            rejected_score = rejected.get("score")
        else:
            rejected_content = rejected
            rejected_score = None

        return PreferenceSample(
            prompt=sample["instruction"],
            chosen=chosen_content,
            rejected=rejected_content,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )


# Format registry
DPO_FORMATS = {
    DataFormatType.PREFERENCE: PreferenceFormat,
    DataFormatType.PREFERENCE_RANKED: PreferenceRankedFormat,
}


def get_dpo_format(
    format_type: DataFormatType,
    rankings_direction: RankingsDirection = "auto",
) -> BaseDataFormat:
    """Get DPO format by type."""
    if format_type not in DPO_FORMATS:
        raise ValueError(f"Unknown DPO format: {format_type}")
    if format_type == DataFormatType.PREFERENCE_RANKED:
        return DPO_FORMATS[format_type](rankings_direction=rankings_direction)
    return DPO_FORMATS[format_type]()
