"""Shared SQLite data for command-level session tests."""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def opencode_database(tmp_path: Path) -> Iterator[Path]:
    """Create a small, synthetic OpenCode-shaped database."""
    database = tmp_path / "opencode.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
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
        CREATE INDEX project_directory_project_id_idx
            ON project_directory(project_id);
        CREATE TABLE session (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            parent_id TEXT,
            title TEXT,
            directory TEXT,
            time_created INTEGER,
            time_updated INTEGER,
            time_archived INTEGER,
            agent TEXT,
            provider TEXT,
            provider_id TEXT,
            model TEXT,
            model_id TEXT,
            cost REAL,
            tokens_input INTEGER,
            tokens_output INTEGER,
            tokens_reasoning INTEGER,
            tokens_cache_read INTEGER,
            tokens_cache_write INTEGER
        );
        CREATE INDEX session_project_updated_idx
            ON session(project_id, time_updated DESC);
        CREATE INDEX session_parent_id_idx ON session(parent_id);
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            time_created INTEGER,
            data TEXT NOT NULL
        );
        CREATE INDEX message_session_id_idx ON message(session_id);
        CREATE TABLE part (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            type TEXT NOT NULL,
            text TEXT,
            data TEXT,
            time_created INTEGER,
            time_updated INTEGER
        );
        CREATE INDEX part_session_id_idx ON part(session_id);
        CREATE INDEX part_message_id_idx ON part(message_id);
        CREATE TABLE todo (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT,
            time_created INTEGER,
            time_updated INTEGER
        );
        CREATE INDEX todo_session_id_idx ON todo(session_id);
        """
    )
    connection.executemany(
        "INSERT INTO project VALUES (?, ?, ?, ?, ?)",
        [
            ("project-root", "/workspace/acme/widget", "widget", 100, 900),
            (
                "project-child",
                "/workspace/acme/widget/packages/child",
                "widget-child",
                100,
                900,
            ),
            ("project-other", "/workspace/other/widget", "widget", 100, 900),
        ],
    )
    connection.executemany(
        "INSERT INTO project_directory VALUES (?, ?)",
        [
            ("project-root", "/workspace/acme/widget"),
            ("project-root", "/worktrees/widget-feature"),
            ("project-child", "/workspace/acme/widget/packages/child"),
            ("project-other", "/workspace/other/widget"),
        ],
    )
    connection.executemany(
        "INSERT INTO session VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "session-root",
                "project-root",
                None,
                "Fix naïve parser 100%_ready",
                "/workspace/acme/widget",
                1_000,
                5_000,
                None,
                "build",
                "openai",
                "openai",
                json.dumps({"providerID": "openai", "id": "gpt-test"}),
                "gpt-test",
                1.25,
                10,
                20,
                3,
                4,
                5,
            ),
            (
                "session-root-second",
                "project-root",
                "session-root",
                "A second root conversation",
                "/worktrees/widget-feature",
                1_100,
                5_000,
                None,
                "plan",
                "anthropic",
                "anthropic",
                json.dumps({"providerID": "anthropic", "id": "claude-test"}),
                "claude-test",
                0.5,
                1,
                2,
                0,
                0,
                0,
            ),
            (
                "session-child",
                "project-child",
                None,
                "Child workspace needle",
                "/workspace/acme/widget/packages/child",
                1_200,
                4_000,
                None,
                "build",
                "openai",
                "openai",
                json.dumps({"providerID": "openai", "id": "gpt-test"}),
                json.dumps({"providerID": "openai", "id": "gpt-test"}),
                0.0,
                0,
                0,
                0,
                0,
                0,
            ),
            (
                "session-archived",
                "project-root",
                None,
                "Archived needle",
                "/workspace/acme/widget",
                1_300,
                3_000,
                6_000,
                "build",
                "openai",
                "openai",
                "gpt-test",
                "gpt-test",
                0.0,
                0,
                0,
                0,
                0,
                0,
            ),
            (
                "session-other",
                "project-other",
                None,
                "Other widget needle",
                "/workspace/other/widget",
                1_400,
                2_000,
                None,
                "review",
                "cohere",
                "cohere",
                json.dumps({"providerID": "cohere", "id": "command-test"}),
                "command-test",
                0.0,
                0,
                0,
                0,
                0,
                0,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
        [
            (
                "message-user",
                "session-root",
                "user",
                1_010,
                json.dumps({"role": "user"}),
            ),
            (
                "message-assistant",
                "session-root",
                "assistant",
                1_020,
                json.dumps({"role": "assistant"}),
            ),
            (
                "message-child",
                "session-child",
                "user",
                1_210,
                json.dumps({"role": "user"}),
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "part-user",
                "session-root",
                "message-user",
                "text",
                "Needle needle in user text; literal 100%_ready and 東京.",
                json.dumps(
                    {
                        "type": "text",
                        "text": (
                            "Needle needle in user text; literal 100%_ready and 東京."
                        ),
                    }
                ),
                1_011,
                1_011,
            ),
            (
                "part-assistant",
                "session-root",
                "message-assistant",
                "text",
                "The assistant also mentions needle.",
                json.dumps(
                    {"type": "text", "text": "The assistant also mentions needle."}
                ),
                1_021,
                1_021,
            ),
            (
                "part-reasoning",
                "session-root",
                "message-assistant",
                "reasoning",
                "private reasoning needle",
                json.dumps({"type": "reasoning", "text": "private reasoning needle"}),
                1_022,
                1_022,
            ),
            (
                "part-tool",
                "session-root",
                "message-assistant",
                "tool",
                None,
                json.dumps(
                    {
                        "type": "tool",
                        "state": {
                            "input": "tool input needle",
                            "output": "tool output needle",
                        },
                    }
                ),
                1_023,
                1_023,
            ),
            (
                "part-child",
                "session-child",
                "message-child",
                "text",
                "child body needle",
                json.dumps({"type": "text", "text": "child body needle"}),
                1_211,
                1_211,
            ),
        ],
    )
    connection.executemany(
        "INSERT INTO todo VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "todo-root",
                "session-root",
                "Investigate needle",
                "pending",
                1_030,
                1_030,
            ),
            ("todo-child", "session-child", "Child todo", "completed", 1_220, 1_220),
        ],
    )
    connection.commit()
    connection.close()
    yield database
