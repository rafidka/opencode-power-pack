"""Builders for synthetic OpenCode V2 databases.

Only the tables and columns that ``opencode-power-pack`` reads are created, with
the same names and JSON shapes OpenCode V2 writes.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE project (
    id TEXT PRIMARY KEY,
    worktree TEXT NOT NULL,
    name TEXT,
    time_created INTEGER,
    time_updated INTEGER
);
CREATE TABLE project_directory (
    project_id TEXT NOT NULL,
    directory TEXT NOT NULL,
    PRIMARY KEY (project_id, directory)
);
CREATE TABLE session_v2 (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    parent_id TEXT,
    title TEXT,
    directory TEXT NOT NULL,
    cost REAL DEFAULT 0 NOT NULL,
    tokens_input INTEGER DEFAULT 0 NOT NULL,
    tokens_output INTEGER DEFAULT 0 NOT NULL,
    tokens_reasoning INTEGER DEFAULT 0 NOT NULL,
    tokens_cache_read INTEGER DEFAULT 0 NOT NULL,
    tokens_cache_write INTEGER DEFAULT 0 NOT NULL,
    agent TEXT,
    model TEXT,
    time_created INTEGER NOT NULL,
    time_updated INTEGER NOT NULL,
    time_archived INTEGER
);
CREATE TABLE session_message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    type TEXT NOT NULL,
    seq INTEGER NOT NULL,
    time_created INTEGER NOT NULL,
    time_updated INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE UNIQUE INDEX session_message_session_seq_idx
    ON session_message (session_id, seq);
"""


def create_database(path: Path) -> sqlite3.Connection:
    """Create an empty V2-shaped database at ``path`` and return a connection."""
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def insert_project(
    connection: sqlite3.Connection,
    project_id: str = "p",
    worktree: str = "/work/repo",
    name: str | None = "repo",
) -> None:
    connection.execute(
        "INSERT INTO project (id, worktree, name, time_created, time_updated) "
        "VALUES (?, ?, ?, 100, 900)",
        (project_id, worktree, name),
    )


def insert_session(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    project_id: str = "p",
    parent_id: str | None = None,
    title: str | None = None,
    directory: str = "/work/repo",
    created: int = 1,
    updated: int = 1,
    archived: int | None = None,
    agent: str | None = "build",
    provider: str | None = "openai",
    model: str | None = "gpt",
    cost: float = 0.0,
    tokens: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
) -> None:
    model_json = (
        json.dumps({"id": model, "providerID": provider}) if model is not None else None
    )
    connection.execute(
        "INSERT INTO session_v2 (id, project_id, parent_id, title, directory, cost, "
        "tokens_input, tokens_output, tokens_reasoning, tokens_cache_read, "
        "tokens_cache_write, agent, model, time_created, time_updated, "
        "time_archived) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            project_id,
            parent_id,
            session_id if title is None else title,
            directory,
            cost,
            *tokens,
            agent,
            model_json,
            created,
            updated,
            archived,
        ),
    )


def insert_message(
    connection: sqlite3.Connection,
    session_id: str,
    message_id: str,
    message_type: str,
    data: dict[str, object],
    *,
    created: int = 1,
    seq: int | None = None,
) -> None:
    """Insert a message, appending it to the session unless ``seq`` is given."""
    if seq is None:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq) + 1, 0) FROM session_message "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = int(row[0])
    connection.execute(
        "INSERT INTO session_message "
        "(id, session_id, type, seq, time_created, time_updated, data) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            message_id,
            session_id,
            message_type,
            seq,
            created,
            created,
            json.dumps({"time": {"created": created}, **data}),
        ),
    )


def user(text: str, **extra: object) -> dict[str, object]:
    """Return the data of a user message."""
    return {"text": text, **extra}


def assistant(*content: dict[str, object], **extra: object) -> dict[str, object]:
    """Return the data of an assistant message with the given content blocks."""
    return {
        "agent": "build",
        "model": {"id": "gpt", "providerID": "openai"},
        "content": list(content),
        **extra,
    }


def text(value: str) -> dict[str, object]:
    return {"type": "text", "text": value}


def reasoning(value: str) -> dict[str, object]:
    return {"type": "reasoning", "text": value}


def tool(
    name: str,
    *,
    input: object = None,
    output: str | None = None,
    status: str = "completed",
    metadata: dict[str, object] | None = None,
    call_id: str = "call",
) -> dict[str, object]:
    """Return a V2 tool block; ``output`` becomes its text content."""
    state: dict[str, object] = {
        "status": status,
        "input": {} if input is None else input,
    }
    if output is not None:
        state["content"] = [{"type": "text", "text": output}]
    if metadata is not None:
        state["metadata"] = metadata
    return {"type": "tool", "id": call_id, "name": name, "state": state}


def todowrite(*todos: tuple[str, str]) -> dict[str, object]:
    """Return a ``todowrite`` call recording ``(content, status)`` todos."""
    return tool(
        "todowrite",
        metadata={
            "todos": [
                {"content": content, "status": status, "priority": "medium"}
                for content, status in todos
            ]
        },
        call_id="todo",
    )
