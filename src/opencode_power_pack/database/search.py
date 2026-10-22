"""Session content extraction, literal matching, and search ordering."""

import re
import sqlite3

from ..models import (
    PartKind,
    SearchSessionMatch,
    SearchSessionsQuery,
    SearchSource,
    Session,
)
from .messages import session_messages, session_todos

_TEXT_SOURCES = {
    PartKind.TEXT: SearchSource.TEXT,
    PartKind.REASONING: SearchSource.REASONING,
}


def search_session(
    connection: sqlite3.Connection, session: Session, query: SearchSessionsQuery
) -> SearchSessionMatch | None:
    """Return the best current-content match for one session."""
    records: list[tuple[SearchSource, str | None, str]] = []
    if SearchSource.TITLE in query.sources and session.title is not None:
        records.append((SearchSource.TITLE, None, session.title))
    if query.sources & {
        SearchSource.TEXT,
        SearchSource.REASONING,
        SearchSource.TOOL_INPUT,
        SearchSource.TOOL_OUTPUT,
    }:
        for message in session_messages(connection, session.id):
            role = message.role
            if query.roles and role not in query.roles:
                continue
            for part in message.parts:
                if part.text is not None:
                    source = _TEXT_SOURCES.get(part.kind)
                    if source is not None and source in query.sources:
                        records.append((source, role, part.text))
                for source, tool_content in (
                    (SearchSource.TOOL_INPUT, part.tool_input),
                    (SearchSource.TOOL_OUTPUT, part.tool_output),
                ):
                    if source in query.sources and tool_content is not None:
                        records.append((source, role, tool_content))
    if SearchSource.TODO in query.sources:
        records.extend(
            (SearchSource.TODO, None, todo.content)
            for todo in session_todos(connection, session.id)
        )
    found = [
        (
            source,
            role,
            record_content,
            match_index(record_content, query.text, query.case_sensitive),
        )
        for source, role, record_content in records
    ]
    found = [record for record in found if record[3] is not None]
    if not found:
        return None
    source, role, record_content, index = min(
        found, key=lambda record: (source_rank(record[0]), record[3], record[2])
    )
    assert index is not None
    return SearchSessionMatch(
        session=session,
        source=source,
        role=role,
        snippet=snippet(record_content, index, query.snippet_length),
        match_count=len(found),
    )


def sort_and_limit_matches(
    matches: list[SearchSessionMatch], query: SearchSessionsQuery
) -> tuple[SearchSessionMatch, ...]:
    """Apply deterministic search ordering and an optional limit."""
    matches.sort(key=lambda match: match.session.id, reverse=True)
    matches.sort(
        key=lambda match: (
            match.session.updated if match.session.updated is not None else -1
        ),
        reverse=True,
    )
    matches.sort(key=lambda match: source_rank(match.source))
    if query.limit is not None:
        matches = matches[: query.limit]
    return tuple(matches)


def source_rank(source: SearchSource) -> int:
    """Return the stable priority for a matching content source."""
    return (
        SearchSource.TITLE,
        SearchSource.TEXT,
        SearchSource.TODO,
        SearchSource.REASONING,
        SearchSource.TOOL_INPUT,
        SearchSource.TOOL_OUTPUT,
    ).index(source)


def match_index(content: str, needle: str, case_sensitive: bool) -> int | None:
    """Find a literal match, mapping Unicode casefold offsets to source offsets."""
    if case_sensitive:
        index = content.find(needle)
        return index if index >= 0 else None
    folded = "".join(char.casefold() for char in content)
    index = folded.find(needle.casefold())
    if index < 0:
        return None
    position = 0
    for original_index, char in enumerate(content):
        next_position = position + len(char.casefold())
        if index < next_position:
            return original_index
        position = next_position
    return len(content)


def snippet(content: str, index: int, length: int) -> str:
    """Return a linearly normalized contextual snippet around a match."""
    if length <= 0:
        return ""
    normalized, normalized_index = normalize_for_snippet(content, index)
    start = max(0, normalized_index - max(0, length // 3))
    end = start + length
    result = normalized[start:end]
    return ("…" if start else "") + result + ("…" if end < len(normalized) else "")


def normalize_for_snippet(content: str, index: int) -> tuple[str, int]:
    """Normalize whitespace while retaining the match's corresponding offset."""
    pieces: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    output_length = 0
    for match in re.finditer(r"\S+|\s+", content):
        text = " " if match.group().isspace() else match.group()
        if not pieces and text == " ":
            continue
        output_start = output_length
        pieces.append(text)
        output_length += len(text)
        offsets.append((match.start(), match.end(), output_start))
    normalized = "".join(pieces).strip()
    if not normalized:
        return normalized, 0
    for source_start, source_end, output_start in offsets:
        if source_start <= index < source_end:
            return normalized, output_start
    return normalized, len(normalized)
