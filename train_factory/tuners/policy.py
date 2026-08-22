"""Canonical tuner names and honest feature availability policy."""

from __future__ import annotations

from typing import Any, Dict, Optional


KNOWN_TUNER_TYPES = frozenset({"lora", "qlora", "full", "freeze"})
SUPPORTED_TUNER_TYPES = frozenset({"lora", "full"})


class UnsupportedTunerError(NotImplementedError, ValueError):
    """A recognized tuner cannot be executed truthfully by this build."""


def normalize_tuner_type(value: object) -> Optional[str]:
    """Normalize a tuner name while rejecting unknown or empty values."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("tuner_type must be a string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("tuner_type must not be empty")
    if normalized not in KNOWN_TUNER_TYPES:
        raise ValueError(
            "Unknown tuner_type. Expected one of: "
            + ", ".join(sorted(KNOWN_TUNER_TYPES))
        )
    return normalized


def ensure_tuner_supported(value: object) -> Optional[str]:
    """Return a canonical supported tuner name or fail explicitly."""
    normalized = normalize_tuner_type(value)
    if normalized == "qlora":
        raise UnsupportedTunerError(
            "QLoRA 4-bit quantization is not implemented end-to-end. "
            "Use 'lora' or 'full' instead."
        )
    if normalized == "freeze":
        raise UnsupportedTunerError(
            "Selective layer freezing is not wired through every training path. "
            "Use 'lora' or 'full' instead."
        )
    return normalized


def resolve_runtime_tuner_type(value: object) -> Optional[str]:
    """Resolve built-ins plus custom tuners registered in this process."""
    try:
        return ensure_tuner_supported(value)
    except UnsupportedTunerError:
        raise
    except ValueError:
        if not isinstance(value, str):
            raise
        normalized = value.strip().lower()
        if not normalized:
            raise
        from .registry import TunerRegistry

        if TunerRegistry.is_registered(normalized):
            return normalized
        raise


def canonicalize_tuner_config(
    config: Dict[str, Any],
    *,
    allow_registered_custom: bool = False,
) -> Dict[str, Any]:
    """Return a config with one authoritative tuner and LoRA switch.

    Explicit ``tuner_type`` wins. Legacy configurations without it derive the
    tuner from either top-level or nested ``use_lora``. A full-parameter
    request that simultaneously enables LoRA is rejected instead of silently
    changing the selected training method.
    """
    normalized = dict(config)
    raw_lora_config = normalized.get("lora_config")
    if raw_lora_config is None:
        lora_config: Dict[str, Any] = {}
    elif isinstance(raw_lora_config, dict):
        lora_config = dict(raw_lora_config)
    else:
        raise ValueError("lora_config must be an object")

    legacy_lora_fields = {
        "lora_r": "r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_target_modules": "target_modules",
    }
    for legacy_name, nested_name in legacy_lora_fields.items():
        if nested_name not in lora_config and legacy_name in normalized:
            lora_config[nested_name] = normalized[legacy_name]

    legacy_lora_enabled = bool(normalized.get("use_lora")) or bool(
        lora_config.get("use_lora")
    )
    raw_tuner_type = normalized.get("tuner_type")
    if raw_tuner_type is None:
        tuner_type = "lora" if legacy_lora_enabled else "full"
    elif allow_registered_custom:
        tuner_type = resolve_runtime_tuner_type(raw_tuner_type)
    else:
        tuner_type = ensure_tuner_supported(raw_tuner_type)

    if tuner_type is None:
        tuner_type = "lora" if legacy_lora_enabled else "full"
    model_type = str(normalized.get("model_type") or "").strip().lower()
    if (
        tuner_type not in {"lora", "full"}
        and model_type in {"llm", "decoder_reranker"}
    ):
        raise ValueError(
            f"Registered custom tuner {tuner_type!r} is not supported for "
            f"decoder trainer {model_type!r}"
        )
    if tuner_type == "full" and legacy_lora_enabled:
        raise ValueError("tuner_type='full' conflicts with LoRA being enabled")
    if tuner_type not in {"lora", "full"} and legacy_lora_enabled:
        raise ValueError(
            f"tuner_type={tuner_type!r} conflicts with LoRA being enabled"
        )

    use_lora = tuner_type == "lora"
    normalized["tuner_type"] = tuner_type
    normalized["use_lora"] = use_lora
    lora_config["use_lora"] = use_lora
    normalized["lora_config"] = lora_config
    return normalized
