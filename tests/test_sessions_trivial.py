"""Tests for trivial session detection and the options that filter on it."""

import json
from pathlib import Path

import pytest
from opencode_v2 import (
    assistant,
    create_database,
    insert_message,
    insert_project,
    insert_session,
    text,
    tool,
    user,
)
from typer.testing import CliRunner

from opencode_power_pack.cli import app
from opencode_power_pack.database import OpenCodeDatabase
from opencode_power_pack.database.trivial import (
    TRIVIAL_IDLE_MS,
    TRIVIAL_MAX_TEXT_CHARS,
)
from opencode_power_pack.models import (
    ListSessionsQuery,
    SessionFilter,
    TrivialMode,
)

_UPDATED = 1_000
_NOW = _UPDATED + TRIVIAL_IDLE_MS


@pytest.fixture
def database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from opencode_power_pack.database import sessions

    monkeypatch.setattr(sessions, "now_ms", lambda: _NOW)
    path = tmp_path / "opencode.db"
    connection = create_database(path)
    insert_project(connection)

    def session(session_id: str, **changes: object) -> None:
        values: dict[str, object] = {"updated": _UPDATED}
        values.update(changes)
        insert_session(connection, session_id, **values)  # pyright: ignore[reportArgumentType]

    def exchange(session_id: str, prompt: str, *reply: dict[str, object]) -> None:
        count = connection.execute(
            "SELECT COUNT(*) FROM session_message WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        insert_message(
            connection, session_id, f"{session_id}-u{count}", "user", user(prompt)
        )
        insert_message(
            connection,
            session_id,
            f"{session_id}-a{count}",
            "assistant",
            assistant(*reply),
        )

    session("empty")
    session("greeting")
    exchange("greeting", "Hi", text("Hello! What can I help with?"))
    session("recent", updated=_NOW - 1)
    exchange("recent", "Hi", text("Hello!"))
    session("tool-call")
    exchange("tool-call", "Test", tool("shell", input={"command": "ls"}, output="x"))
    session("two-turns")
    exchange("two-turns", "Ping", text("Pong"))
    exchange("two-turns", "Ping", text("Pong"))
    session("long-answer")
    exchange("long-answer", "Explain", text("x" * TRIVIAL_MAX_TEXT_CHARS))
    session("parent")
    session("child", parent_id="parent")
    connection.commit()
    connection.close()
    return path


def _ids(database: Path, mode: TrivialMode, *, children: bool = False) -> set[str]:
    query = ListSessionsQuery(
        filters=SessionFilter(trivial_mode=mode, include_children=children)
    )
    with OpenCodeDatabase(database) as opened:
        return {session.id for session in opened.list_sessions(query)}


def test_trivial_sessions_are_idle_childless_toolless_and_short(
    database: Path,
) -> None:
    assert _ids(database, TrivialMode.TRIVIAL) == {"empty", "greeting"}


def test_substantive_is_the_default_and_all_includes_both(database: Path) -> None:
    assert _ids(database, SessionFilter().trivial_mode) == {
        "recent",
        "tool-call",
        "two-turns",
        "long-answer",
        "parent",
    }
    assert _ids(database, TrivialMode.ALL) == {
        "empty",
        "greeting",
        "recent",
        "tool-call",
        "two-turns",
        "long-answer",
        "parent",
    }


def test_child_sessions_are_never_trivial(database: Path) -> None:
    assert "child" in _ids(database, TrivialMode.SUBSTANTIVE, children=True)
    assert "child" not in _ids(database, TrivialMode.TRIVIAL, children=True)


def test_selectors_resolve_trivial_sessions(database: Path) -> None:
    with OpenCodeDatabase(database) as opened:
        assert "greeting" in {session.id for session in opened.all_sessions()}


@pytest.mark.parametrize("command", ["list", "grep"])
def test_cli_trivial_options(database: Path, command: str) -> None:
    runner = CliRunner()
    base = ["--database", str(database), "sessions", command]
    if command == "grep":
        base.append("h")

    def ids(*options: str) -> set[str]:
        result = runner.invoke(app, [*base, *options, "--format", "jsonl"])
        assert result.exit_code == 0, result.output
        records = [json.loads(line) for line in result.stdout.splitlines()]
        return {record.get("session", record)["id"] for record in records}

    assert "greeting" not in ids()
    assert "greeting" in ids("--include-trivial")
    assert ids("--trivial-only") <= {"empty", "greeting"}
    assert "greeting" in ids("--trivial-only")

    conflict = runner.invoke(app, [*base, "--include-trivial", "--trivial-only"])
    assert conflict.exit_code != 0
    assert "--trivial-only" in conflict.output
