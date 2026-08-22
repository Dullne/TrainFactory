"""Helpers for keeping persisted and returned JSON RFC-compliant."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any


def _sanitize_json_key(key: Any) -> str:
    """Return the exact string a strict JSON object key should use."""
    if isinstance(key, str):
        return key
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, Integral):
        return str(int(key))
    if isinstance(key, Real):
        number = float(key)
        if math.isfinite(number):
            return json.dumps(number, allow_nan=False)
        if math.isnan(number):
            return "NaN"
        return "Infinity" if number > 0 else "-Infinity"
    raise TypeError(f"Unsupported JSON object key: {key!r}")


def sanitize_json_value(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Python's JSON encoder accepts NaN and infinities by default even though
    they are not valid JSON. Metrics can contain these values when training
    diverges, including inside custom nested metrics. Preserve the input shape
    while making those values safe for strict encoders and API clients.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            sanitized_key = _sanitize_json_key(key)
            if sanitized_key in sanitized:
                raise ValueError(f"JSON object key collision: {sanitized_key!r}")
            sanitized[sanitized_key] = sanitize_json_value(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    return value
