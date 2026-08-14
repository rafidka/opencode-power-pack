"""Small helpers shared by database record readers."""

import json
from typing import cast


def json_object(value: object) -> dict[str, object]:
    """Return a JSON object from a database value, or an empty object."""
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except TypeError, ValueError:
        return {}
    return cast(dict[str, object], parsed) if isinstance(parsed, dict) else {}


def string(value: object) -> str | None:
    """Return ``value`` when it is a string."""
    return value if isinstance(value, str) else None


def content(value: object) -> str | None:
    """Convert structured tool content to stable text."""
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
