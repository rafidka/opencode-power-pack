"""Parsing helpers for timestamp options."""

import re
from datetime import UTC, datetime, timedelta

from .errors import TimeParseError

_RELATIVE_TIME = re.compile(r"^(?P<amount>\d+)(?P<unit>[mhd])$")
_UNIT_DELTA = {"m": "minutes", "h": "hours", "d": "days"}


def parse_time(value: str, *, now: datetime | None = None) -> int:
    """Parse an ISO 8601 instant or ``30m``/``24h``/``7d`` into UTC epoch ms.

    Relative values mean that duration before ``now``. ISO 8601 values must include
    a UTC offset (``Z`` is accepted); naive local timestamps are rejected.
    """
    match = _RELATIVE_TIME.fullmatch(value)
    if match is not None:
        current = now or datetime.now(UTC)
        if current.tzinfo is None or current.utcoffset() is None:
            msg = "the supplied 'now' value must be timezone-aware"
            raise TimeParseError(msg)
        amount = int(match["amount"])
        delta = timedelta(**{_UNIT_DELTA[match["unit"]]: amount})
        return _to_epoch_ms(current.astimezone(UTC) - delta)

    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        msg = f"invalid time {value!r}; use ISO 8601 with an offset or 30m/24h/7d"
        raise TimeParseError(msg) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        msg = f"ambiguous naive time {value!r}; include a UTC offset such as Z"
        raise TimeParseError(msg)
    return _to_epoch_ms(timestamp.astimezone(UTC))


def _to_epoch_ms(timestamp: datetime) -> int:
    return int(timestamp.timestamp() * 1_000)
