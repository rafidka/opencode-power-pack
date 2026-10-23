"""Detection of throwaway sessions, such as tests and greetings.

A session is trivial when it is an idle root session with no sub-agent
sessions, no tool calls, at most one user message, and little text. Titles are
deliberately ignored: OpenCode titles a session from its first message, so a
session titled "Test message" can go on to do real work.
"""

import sqlite3
import time
from collections.abc import Generator, Iterable

from ..models import Session
from ._utils import json_object, string
from .messages import ASSISTANT, USER, content_blocks

TRIVIAL_IDLE_MS = 24 * 60 * 60 * 1_000
"""How long a session must be untouched, so one still in use is never trivial."""

TRIVIAL_MAX_USER_MESSAGES = 1

TRIVIAL_MAX_TEXT_CHARS = 300
"""Most user and assistant text, in characters, that a trivial session can hold."""

_BIND_BATCH_SIZE = 900


def now_ms() -> int:
    """Return the current time in milliseconds since the epoch."""
    return time.time_ns() // 1_000_000


def trivial_session_ids(
    connection: sqlite3.Connection, sessions: Iterable[Session], now: int
) -> frozenset[str]:
    """Return the IDs of the given sessions that are trivial at ``now``."""
    idle = [
        session.id
        for session in sessions
        if session.parent_id is None
        and session.updated is not None
        and now - session.updated >= TRIVIAL_IDLE_MS
    ]
    if not idle:
        return frozenset()
    parents = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT parent_id FROM session_v2 WHERE parent_id IS NOT NULL"
        )
    }
    candidates = [session_id for session_id in idle if session_id not in parents]
    user_counts: dict[str, int] = {}
    for group in _groups(candidates):
        placeholders = ", ".join("?" for _ in group)
        for row in connection.execute(
            "SELECT session_id, COUNT(*) FROM session_message "
            f"WHERE session_id IN ({placeholders}) AND type = ? "
            "GROUP BY session_id",
            (*group, USER),
        ):
            user_counts[row[0]] = row[1]
    candidates = [
        session_id
        for session_id in candidates
        if user_counts.get(session_id, 0) <= TRIVIAL_MAX_USER_MESSAGES
    ]
    text_chars = dict.fromkeys(candidates, 0)
    for group in _groups(candidates):
        placeholders = ", ".join("?" for _ in group)
        for row in connection.execute(
            "SELECT session_id, type, data FROM session_message "
            f"WHERE session_id IN ({placeholders}) AND type IN (?, ?)",
            (*group, USER, ASSISTANT),
        ):
            session_id = row["session_id"]
            if session_id not in text_chars:
                continue
            data = json_object(row["data"])
            if row["type"] == USER:
                text_chars[session_id] += len(string(data.get("text")) or "")
                continue
            for block in content_blocks(data):
                block_type = string(block.get("type"))
                if block_type == "tool":
                    del text_chars[session_id]
                    break
                if block_type == "text":
                    text_chars[session_id] += len(string(block.get("text")) or "")
    return frozenset(
        session_id
        for session_id, chars in text_chars.items()
        if chars < TRIVIAL_MAX_TEXT_CHARS
    )


def _groups(values: list[str]) -> Generator[list[str]]:
    for start in range(0, len(values), _BIND_BATCH_SIZE):
        yield values[start : start + _BIND_BATCH_SIZE]
