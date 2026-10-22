from __future__ import annotations

import json
from typing import Any

try:
    import toon_format
except ImportError:  # pragma: no cover - exercised only in minimal environments
    toon_format = None

TOON_AVAILABLE = toon_format is not None


def encode_toon(value: Any) -> str:
    """Encode JSON-compatible data as TOON, falling back to compact JSON."""
    if toon_format is None:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    try:
        return toon_format.dumps(value, indent_size=2, delimiter=",", default=str)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
