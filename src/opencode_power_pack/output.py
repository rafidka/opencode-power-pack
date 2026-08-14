"""Human and machine-readable output helpers."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.table import Table

type JSONValue = (
    bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None
)

error_console = Console(stderr=True)


def to_jsonable(value: object) -> JSONValue:
    """Convert shared models and standard scalar values to JSON-safe values."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if hasattr(value, "as_dict"):
        return to_jsonable(value.as_dict())  # type: ignore[union-attr]
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): to_jsonable(item) for key, item in mapping.items()}
    if isinstance(value, Iterable):
        iterable = cast(Iterable[object], value)
        return [to_jsonable(item) for item in iterable]
    msg = f"cannot serialize {type(value).__name__} as JSON"
    raise TypeError(msg)


def emit_json(value: object) -> None:
    """Write one JSON document to stdout without Rich styling."""
    print(json.dumps(to_jsonable(value), ensure_ascii=False, separators=(",", ":")))


def emit_json_array(values: Iterable[object]) -> None:
    """Write values as one JSON array, suitable for ``sessions list --format json``."""
    emit_json(list(values))


def emit_jsonl(values: Iterable[object]) -> None:
    """Write one object per stdout line for ``sessions list --format jsonl``."""
    for value in values:
        emit_json(value)


def emit_table(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    title: str | None = None,
) -> None:
    """Render columns and rows as a Rich table for interactive use."""
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(cell) for cell in row))
    Console().print(table)
