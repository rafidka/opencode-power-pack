"""Message, part, and todo conversion into full conversation transcripts."""

import sqlite3
from collections.abc import Mapping
from typing import cast

from ..models import (
    PartKind,
    Session,
    TokenBreakdown,
    Transcript,
    TranscriptMessage,
    TranscriptPart,
    TranscriptQuery,
    TranscriptTodo,
)
from ._utils import content, json_object, string


def session_transcripts(
    connection: sqlite3.Connection,
    query: TranscriptQuery,
    sessions_by_id: Mapping[str, Session],
    columns: Mapping[str, frozenset[str]],
) -> tuple[Transcript, ...]:
    """Build one transcript per requested session, preserving query order.

    Unknown session IDs are skipped. Sub-agent transcripts are inlined into
    ``children`` when requested, following ``task`` tool calls recursively and
    guarding against cycles and repeats within a branch.
    """
    return tuple(
        _build_transcript(
            connection, query, sessions_by_id, columns, session, frozenset()
        )
        for session_id in query.session_ids
        if (session := sessions_by_id.get(session_id)) is not None
    )


def _build_transcript(
    connection: sqlite3.Connection,
    query: TranscriptQuery,
    sessions_by_id: Mapping[str, Session],
    columns: Mapping[str, frozenset[str]],
    session: Session,
    ancestors: frozenset[str],
) -> Transcript:
    messages = _messages(connection, columns, session.id)
    todos = _todos(connection, columns, session.id) if query.include_todos else ()
    children: tuple[Transcript, ...] = ()
    if query.include_children:
        next_ancestors = ancestors | {session.id}
        child_ids = dict.fromkeys(
            part.child_session_id
            for message in messages
            for part in message.parts
            if part.child_session_id is not None
        )
        children = tuple(
            _build_transcript(
                connection,
                query,
                sessions_by_id,
                columns,
                child_session,
                next_ancestors,
            )
            for child_id in child_ids
            if child_id not in next_ancestors
            and (child_session := sessions_by_id.get(child_id)) is not None
        )
    return Transcript(
        session=session, messages=messages, todos=todos, children=children
    )


def _messages(
    connection: sqlite3.Connection,
    columns: Mapping[str, frozenset[str]],
    session_id: str,
) -> tuple[TranscriptMessage, ...]:
    has_time_created = "time_created" in columns.get("message", frozenset())
    order = "time_created, id" if has_time_created else "id"
    rows = connection.execute(
        f"SELECT * FROM message WHERE session_id = ? ORDER BY {order}",
        (session_id,),
    ).fetchall()
    parts_by_message = _parts_by_message(connection, columns, session_id)
    return tuple(
        _message_from_row(
            row["id"],
            json_object(row["data"]),
            row["time_created"] if has_time_created else None,
            tuple(parts_by_message.get(row["id"], ())),
        )
        for row in rows
    )


def _object(value: object) -> dict[str, object]:
    """Return a nested JSON object already parsed from a ``data`` column."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _message_from_row(
    message_id: str,
    data: dict[str, object],
    time_created_column: object,
    parts: tuple[TranscriptPart, ...],
) -> TranscriptMessage:
    role = string(data.get("role")) or ""
    time_data = _object(data.get("time"))
    created = time_data.get("created")
    if not isinstance(created, int):
        created = time_created_column if isinstance(time_created_column, int) else None
    model_data = _object(data.get("model"))
    model = (
        string(data.get("modelID"))
        if role == "assistant"
        else string(model_data.get("modelID"))
    )
    provider = string(data.get("providerID")) or string(model_data.get("providerID"))
    error_data = _object(data.get("error"))
    return TranscriptMessage(
        id=message_id,
        role=role,
        created=created,
        agent=string(data.get("agent")),
        model=model,
        provider=provider,
        cost=_float(data.get("cost")),
        tokens=_tokens(data.get("tokens")),
        error=string(error_data.get("name")),
        parts=parts,
    )


def _tokens(value: object) -> TokenBreakdown | None:
    tokens_data = _object(value)
    if not tokens_data:
        return None
    cache_data = _object(tokens_data.get("cache"))
    return TokenBreakdown(
        input=_int(tokens_data.get("input")),
        output=_int(tokens_data.get("output")),
        reasoning=_int(tokens_data.get("reasoning")),
        cache_read=_int(cache_data.get("read")),
        cache_write=_int(cache_data.get("write")),
    )


def _int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items = cast("list[object]", value)
    return tuple(item for item in items if isinstance(item, str))


def _parts_by_message(
    connection: sqlite3.Connection,
    columns: Mapping[str, frozenset[str]],
    session_id: str,
) -> dict[str, list[TranscriptPart]]:
    part_columns = columns.get("part", frozenset())
    order_bits: list[str] = []
    if "time_created" in part_columns:
        order_bits.append("time_created")
    order_bits.append("id" if "id" in part_columns else "rowid")
    rows = connection.execute(
        f"SELECT * FROM part WHERE session_id = ? ORDER BY {', '.join(order_bits)}",
        (session_id,),
    ).fetchall()
    grouped: dict[str, list[TranscriptPart]] = {}
    for row in rows:
        part = _part_from_row(json_object(row["data"]))
        if part is not None:
            grouped.setdefault(row["message_id"], []).append(part)
    return grouped


def _part_from_row(data: dict[str, object]) -> TranscriptPart | None:
    part_type = string(data.get("type"))
    if part_type == "text":
        # Synthetic text parts are tool-call replays injected for models
        # without native tool calling; they duplicate the tool parts and
        # would double the output, so they are excluded.
        if data.get("synthetic"):
            return None
        text = string(data.get("text"))
        return TranscriptPart(kind=PartKind.TEXT, text=text) if text else None
    if part_type == "reasoning":
        text = string(data.get("text"))
        return TranscriptPart(kind=PartKind.REASONING, text=text) if text else None
    if part_type == "tool":
        return _tool_part(data)
    if part_type == "patch":
        return TranscriptPart(
            kind=PartKind.PATCH, files=_string_tuple(data.get("files"))
        )
    if part_type == "file":
        return TranscriptPart(kind=PartKind.FILE, filename=string(data.get("filename")))
    if part_type == "compaction":
        auto = data.get("auto")
        return TranscriptPart(
            kind=PartKind.COMPACTION,
            compaction_auto=auto if isinstance(auto, bool) else None,
        )
    if part_type == "agent":
        return TranscriptPart(kind=PartKind.AGENT, agent_name=string(data.get("name")))
    return None


def _tool_part(data: dict[str, object]) -> TranscriptPart:
    state = _object(data.get("state"))
    metadata = _object(state.get("metadata"))
    return TranscriptPart(
        kind=PartKind.TOOL,
        tool_name=string(data.get("tool")),
        tool_title=string(state.get("title")),
        tool_status=string(state.get("status")),
        tool_input=content(state.get("input")),
        tool_output=content(state.get("output")),
        child_session_id=string(metadata.get("sessionId")),
    )


def _todos(
    connection: sqlite3.Connection,
    columns: Mapping[str, frozenset[str]],
    session_id: str,
) -> tuple[TranscriptTodo, ...]:
    todo_columns = columns.get("todo", frozenset())
    has_status = "status" in todo_columns
    order = "position" if "position" in todo_columns else "rowid"
    select_columns = "content, status" if has_status else "content"
    rows = connection.execute(
        f"SELECT {select_columns} FROM todo WHERE session_id = ? ORDER BY {order}",
        (session_id,),
    ).fetchall()
    return tuple(
        TranscriptTodo(
            content=row["content"], status=row["status"] if has_status else None
        )
        for row in rows
    )
