"""Conversion of OpenCode V2 ``session_message`` rows into transcript messages.

V2 stores one row per message, ordered by ``seq``, with the role as the row
``type``. User text is ``data.text``; assistant output is the ordered
``data.content`` array of text, reasoning, and tool blocks. Bookkeeping rows
(``synthetic`` reminders, ``system`` notices, ``idle`` and ``agent-switched``
markers) carry no conversation content and are skipped.
"""

import sqlite3
from collections.abc import Iterator
from typing import cast

from ..models import (
    PartKind,
    TokenBreakdown,
    TranscriptMessage,
    TranscriptPart,
    TranscriptTodo,
)
from ._utils import content, json_object, string

USER = "user"
ASSISTANT = "assistant"
COMPACTION = "compaction"

_TOOL_ALIASES = {"bash": "shell", "task": "subagent", "apply_patch": "patch"}
"""Legacy tool names recorded by V1, mapped to the names V2 uses for them."""

_TITLE_INPUT_KEYS = (
    "description",
    "command",
    "path",
    "filePath",
    "pattern",
    "url",
    "id",
    "name",
)
"""Tool input fields that make a readable title, in order of preference.

V2 records no tool title, and its V1 migration drops the one V1 stored, so a
title is derived from the call's own input instead.
"""

_TITLE_LIMIT = 120

TODO_TOOL = "todowrite"


def message_rows(
    connection: sqlite3.Connection, session_id: str
) -> Iterator[sqlite3.Row]:
    """Yield a session's message rows in conversation order."""
    yield from connection.execute(
        "SELECT id, type, time_created, data FROM session_message "
        "WHERE session_id = ? ORDER BY seq",
        (session_id,),
    )


def session_messages(
    connection: sqlite3.Connection, session_id: str
) -> tuple[TranscriptMessage, ...]:
    """Return the conversation messages of one session."""
    return tuple(
        message
        for row in message_rows(connection, session_id)
        if (message := message_from_row(row)) is not None
    )


def message_from_row(row: sqlite3.Row) -> TranscriptMessage | None:
    """Convert one ``session_message`` row, or ``None`` when it is bookkeeping."""
    kind = row["type"]
    data = json_object(row["data"])
    created = _int(_object(data.get("time")).get("created"))
    if created is None:
        created = _int(row["time_created"])
    if kind == USER:
        return TranscriptMessage(
            id=row["id"], role=USER, created=created, parts=_user_parts(data)
        )
    if kind == ASSISTANT:
        model = _object(data.get("model"))
        error = _object(data.get("error"))
        return TranscriptMessage(
            id=row["id"],
            role=ASSISTANT,
            created=created,
            agent=string(data.get("agent")),
            model=string(model.get("id")),
            provider=string(model.get("providerID")),
            cost=_float(data.get("cost")),
            tokens=_tokens(data.get("tokens")),
            error=string(error.get("type")) or string(error.get("name")),
            parts=tuple(
                part
                for block in content_blocks(data)
                if (part := _assistant_part(block)) is not None
            ),
        )
    if kind == COMPACTION:
        reason = string(data.get("reason"))
        return TranscriptMessage(
            id=row["id"],
            role=COMPACTION,
            created=created,
            parts=(
                TranscriptPart(
                    kind=PartKind.COMPACTION,
                    compaction_auto=None if reason is None else reason == "auto",
                ),
            ),
        )
    return None


def session_todos(
    connection: sqlite3.Connection, session_id: str
) -> tuple[TranscriptTodo, ...]:
    """Return a session's current todo list.

    V2 keeps no todo table; the list is the one recorded by the most recent
    ``todowrite`` call, which always carries the complete list.
    """
    rows = connection.execute(
        "SELECT data FROM session_message "
        "WHERE session_id = ? AND type = ? AND instr(data, ?) > 0 "
        "ORDER BY seq DESC",
        (session_id, ASSISTANT, f'"{TODO_TOOL}"'),
    )
    for row in rows:
        for block in reversed(content_blocks(json_object(row["data"]))):
            if string(block.get("name")) != TODO_TOOL:
                continue
            metadata = _object(_object(block.get("state")).get("metadata"))
            todos = metadata.get("todos")
            if not isinstance(todos, list):
                continue
            return tuple(
                TranscriptTodo(content=text, status=string(item.get("status")))
                for item in _objects(cast("list[object]", todos))
                if (text := string(item.get("content"))) is not None
            )
    return ()


def content_blocks(data: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Return the JSON objects of an assistant message's ``content`` array."""
    blocks = data.get("content")
    if not isinstance(blocks, list):
        return ()
    return tuple(_object(block) for block in cast("list[object]", blocks))


def tool_name(block: dict[str, object]) -> str | None:
    """Return a tool block's name, using V2 names for legacy V1 tools."""
    name = string(block.get("name"))
    return _TOOL_ALIASES.get(name, name) if name is not None else None


def _user_parts(data: dict[str, object]) -> tuple[TranscriptPart, ...]:
    parts: list[TranscriptPart] = []
    text = string(data.get("text"))
    if text:
        parts.append(TranscriptPart(kind=PartKind.TEXT, text=text))
    parts.extend(
        TranscriptPart(kind=PartKind.FILE, filename=name)
        for item in _objects(data.get("files"))
        if (name := string(item.get("name"))) is not None
    )
    parts.extend(
        TranscriptPart(kind=PartKind.AGENT, agent_name=name)
        for item in _objects(data.get("agents"))
        if (name := string(item.get("name"))) is not None
    )
    return tuple(parts)


def _assistant_part(block: dict[str, object]) -> TranscriptPart | None:
    block_type = string(block.get("type"))
    if block_type in {"text", "reasoning"}:
        text = string(block.get("text"))
        if not text:
            return None
        kind = PartKind.TEXT if block_type == "text" else PartKind.REASONING
        return TranscriptPart(kind=kind, text=text)
    if block_type == "tool":
        return _tool_part(block)
    return None


def _tool_part(block: dict[str, object]) -> TranscriptPart:
    state = _object(block.get("state"))
    raw_input = state.get("input")
    metadata = _object(state.get("metadata"))
    name = tool_name(block)
    return TranscriptPart(
        kind=PartKind.TOOL,
        tool_name=name,
        tool_title=_tool_title(raw_input),
        tool_status=string(state.get("status")),
        tool_input=content(raw_input),
        tool_output=_tool_output(state),
        child_session_id=(
            string(metadata.get("sessionID")) or string(metadata.get("sessionId"))
            if name == "subagent"
            else None
        ),
    )


def _tool_title(raw_input: object) -> str | None:
    values = _object(raw_input)
    for key in _TITLE_INPUT_KEYS:
        value = string(values.get(key))
        if value is None:
            continue
        lines = value.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        if len(lines) > 1 or len(title) > _TITLE_LIMIT:
            title = title[: _TITLE_LIMIT - 1].rstrip() + "…"
        return title
    return None


def _tool_output(state: dict[str, object]) -> str | None:
    texts = [
        text
        for item in _objects(state.get("content"))
        if string(item.get("type")) == "text"
        and (text := string(item.get("text"))) is not None
    ]
    if texts:
        return "\n".join(texts)
    return string(_object(state.get("error")).get("message"))


def _tokens(value: object) -> TokenBreakdown | None:
    tokens = _object(value)
    if not tokens:
        return None
    cache = _object(tokens.get("cache"))
    return TokenBreakdown(
        input=_int(tokens.get("input")),
        output=_int(tokens.get("output")),
        reasoning=_int(tokens.get("reasoning")),
        cache_read=_int(cache.get("read")),
        cache_write=_int(cache.get("write")),
    )


def _object(value: object) -> dict[str, object]:
    """Return a nested JSON object already parsed from a ``data`` column."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _objects(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        cast("dict[str, object]", item)
        for item in cast("list[object]", value)
        if isinstance(item, dict)
    )


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
