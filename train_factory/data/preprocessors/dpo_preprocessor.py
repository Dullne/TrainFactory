"""DPO preference-data preprocessor."""

import logging
from typing import Any, Dict, Optional

from datasets import Dataset

from ..formats.dpo import RankingsDirection, sort_ranked_preference_pairs
from .base import TokenizingPreprocessor

logger = logging.getLogger(__name__)


class DPOPreprocessor(TokenizingPreprocessor):
    """
    Normalize preference datasets to the raw text schema expected by TRL DPO/ORPO.

    Supported input formats:
    - prompt + chosen + rejected
    - instruction + chosen + rejected
    - query + positive/negative (or pos/neg/negatives)
    - prompt + responses + rankings
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(tokenizer, max_length, config)
        rl_config = (config or {}).get("rl_config") or {}
        direction = rl_config.get("rankings_direction", "auto")
        if direction not in {"auto", "higher_is_better", "lower_is_better"}:
            logger.warning(
                "Unknown ranked preference direction %r, falling back to auto",
                direction,
            )
            direction = "auto"
        self.rankings_direction: RankingsDirection = direction

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Normalize preference data to prompt/chosen/rejected text columns.

        Unlike SFT preprocessing, DPO/ORPO training in this project consumes raw
        preference text and lets TRL tokenize internally.
        """
        dataset = self.filter_invalid(dataset)

        return dataset.map(
            self._preprocess_function,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Preprocessing DPO data",
        )

    def _preprocess_function(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single preference sample."""
        if self._is_ranked_format(sample):
            return self._preprocess_ranked(sample)

        prompt = sample.get("prompt")
        if not prompt:
            prompt = sample.get("instruction")
        if not prompt:
            prompt = sample.get("query")

        chosen = self._extract_chosen(sample)
        rejected = self._extract_rejected(sample)

        processed = {
            "prompt": self._normalize_text(prompt),
            "chosen": self._normalize_text(chosen),
            "rejected": self._normalize_text(rejected),
        }

        chosen_score = self._extract_score(sample.get("chosen_score"))
        rejected_score = self._extract_score(sample.get("rejected_score"))
        if chosen_score is not None:
            processed["chosen_score"] = chosen_score
        if rejected_score is not None:
            processed["rejected_score"] = rejected_score

        return processed

    def _preprocess_ranked(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ranked responses to a single best-vs-worst preference pair."""
        responses = sample["responses"]
        rankings = sample["rankings"]

        paired = [
            (self._normalize_text(response), self._extract_score(score))
            for response, score in zip(responses, rankings)
        ]

        valid_pairs = [
            (response, score)
            for response, score in paired
            if response and score is not None
        ]
        if len(valid_pairs) < 2:
            raise ValueError("Ranked preference sample does not contain two valid responses")

        valid_pairs = sort_ranked_preference_pairs(
            valid_pairs,
            rankings_direction=self.rankings_direction,
        )
        best_response, best_score = valid_pairs[0]
        worst_response, worst_score = valid_pairs[-1]

        return {
            "prompt": self._normalize_text(sample["prompt"]),
            "chosen": best_response,
            "rejected": worst_response,
            "chosen_score": best_score,
            "rejected_score": worst_score,
        }

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate whether a sample can be normalized into a preference pair."""
        try:
            if self._is_ranked_format(sample):
                prompt = self._normalize_text(sample.get("prompt"))
                responses = sample.get("responses", [])
                rankings = sample.get("rankings", [])
                if not prompt:
                    return False
                if not isinstance(responses, list) or not isinstance(rankings, list):
                    return False
                if len(responses) < 2 or len(rankings) != len(responses):
                    return False

                valid_pairs = [
                    (self._normalize_text(response), self._extract_score(score))
                    for response, score in zip(responses, rankings)
                ]
                valid_pairs = [
                    (response, score)
                    for response, score in valid_pairs
                    if response and score is not None
                ]

                if len(valid_pairs) < 2:
                    return False

                scores = [score for _, score in valid_pairs]
                return min(scores) < max(scores)

            prompt = self._normalize_text(
                sample.get("prompt")
                or sample.get("instruction")
                or sample.get("query")
            )
            chosen = self._normalize_text(self._extract_chosen(sample))
            rejected = self._normalize_text(self._extract_rejected(sample))

            return bool(prompt and chosen and rejected)
        except Exception as exc:
            logger.debug("Invalid DPO sample %s: %s", sample, exc)
            return False

    def _extract_chosen(self, sample: Dict[str, Any]) -> Any:
        if "chosen" in sample:
            return sample["chosen"]
        if "positive" in sample:
            return sample["positive"]
        if "pos" in sample:
            return sample["pos"]
        return None

    def _extract_rejected(self, sample: Dict[str, Any]) -> Any:
        if "rejected" in sample:
            return sample["rejected"]
        if "negative" in sample:
            return sample["negative"]
        if "neg" in sample:
            return sample["neg"]
        if "negatives" in sample:
            return sample["negatives"]
        return None

    def _normalize_text(self, value: Any) -> str:
        """Convert various text payload shapes into a trimmed string."""
        if value is None:
            return ""

        if isinstance(value, dict):
            for key in ("content", "response", "text", "value", "output", "answer"):
                if value.get(key):
                    return self._normalize_text(value[key])
            return ""

        if isinstance(value, list):
            for item in value:
                normalized = self._normalize_text(item)
                if normalized:
                    return normalized
            return ""

        return str(value).strip()

    @staticmethod
    def _extract_score(value: Any) -> Optional[float]:
        """Convert a ranking/score field into float when possible."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_ranked_format(sample: Dict[str, Any]) -> bool:
        return (
            "prompt" in sample
            and "responses" in sample
            and "rankings" in sample
        )

    @property
    def preprocessor_type(self) -> str:
        return "dpo"
