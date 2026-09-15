"""Tests for the mutating session commands and their selector handling."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from opencode_power_pack.cli import app
from opencode_power_pack.errors import OpencodeServerError, UsageError
from opencode_power_pack.server import SessionNotFound, data_home_for_database

runner = CliRunner()


class _FakeServer:
    """Records the session mutations a command asked OpenCode to perform."""

    def __init__(self, missing: frozenset[str] = frozenset()) -> None:
        self.deleted: list[str] = []
        self.archived: list[tuple[str, int]] = []
        self.missing = missing

    def delete_session(self, session_id: str) -> None:
        if session_id in self.missing:
            raise SessionNotFound(f"session {session_id} does not exist")
        self.deleted.append(session_id)

    def archive_session(self, session_id: str, archived_at: int) -> None:
        if session_id in self.missing:
            raise SessionNotFound(f"session {session_id} does not exist")
        self.archived.append((session_id, archived_at))


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _FakeServer:
    """Replace the real ``opencode serve`` lifecycle with an in-process double."""
    return _install_server(monkeypatch, _FakeServer())


def _install_server(monkeypatch: pytest.MonkeyPatch, fake: _FakeServer) -> _FakeServer:
    from opencode_power_pack.commands.sessions import _mutations

    @contextmanager
    def factory(database: Path) -> Iterator[_FakeServer]:
        fake.database = database  # type: ignore[attr-defined]
        yield fake

    monkeypatch.setattr(_mutations, "opencode_server", factory)
    return fake


def invoke(database: Path, *arguments: str, stdin: str | None = None) -> Result:
    return runner.invoke(
        app,
        ["--database", str(database), *arguments],
        input=stdin,
        env={"NO_COLOR": "1"},
    )


def test_deletes_by_full_id_and_unique_prefix(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(
        opencode_database, "sessions", "delete", "session-other", "session-c", "--yes"
    )
    assert result.exit_code == 0, result.output
    assert server.deleted == ["session-other", "session-child"]
    assert "Deleted 2 sessions." in result.stderr


def test_deletes_archived_and_child_sessions(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(
        opencode_database,
        "sessions",
        "delete",
        "session-archived",
        "session-root-second",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    assert server.deleted == ["session-archived", "session-root-second"]


def test_rejects_ambiguous_and_unknown_selectors(
    opencode_database: Path, server: _FakeServer
) -> None:
    ambiguous = invoke(opencode_database, "sessions", "delete", "session-r", "--yes")
    assert ambiguous.exit_code == 2
    assert "ambiguous" in ambiguous.stderr
    assert "session-root" in ambiguous.stderr

    unknown = invoke(opencode_database, "sessions", "delete", "nope", "--yes")
    assert unknown.exit_code == 2
    assert "no session matches" in unknown.stderr
    assert server.deleted == []


def test_deduplicates_selectors(opencode_database: Path, server: _FakeServer) -> None:
    result = invoke(
        opencode_database,
        "sessions",
        "delete",
        "session-other",
        "session-o",
        "--yes",
    )
    assert result.exit_code == 0, result.output
    assert server.deleted == ["session-other"]


def test_requires_confirmation_when_not_interactive(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(opencode_database, "sessions", "delete", "session-other")
    assert result.exit_code == 2
    assert "--yes" in result.stderr
    assert server.deleted == []


def test_archive_needs_no_confirmation_and_sends_a_timestamp(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(opencode_database, "sessions", "archive", "session-other")
    assert result.exit_code == 0, result.output
    assert [session_id for session_id, _ in server.archived] == ["session-other"]
    assert server.archived[0][1] > 1_700_000_000_000
    assert "Archived 1 session." in result.stderr


def test_accepts_piped_tab_separated_list_output(
    opencode_database: Path, server: _FakeServer
) -> None:
    listed = invoke(opencode_database, "sessions", "list")
    assert listed.exit_code == 0, listed.output
    assert "\t" in listed.stdout

    result = invoke(opencode_database, "sessions", "archive", stdin=listed.stdout)
    assert result.exit_code == 0, result.output
    assert [session_id for session_id, _ in server.archived] == [
        "session-root",
        "session-child",
        "session-other",
    ]


def test_accepts_piped_json_jsonl_and_bare_ids(
    opencode_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    as_json = invoke(opencode_database, "sessions", "list", "--format", "json")
    from_json = _install_server(monkeypatch, _FakeServer())
    result = invoke(
        opencode_database, "sessions", "delete", "--yes", stdin=as_json.stdout
    )
    assert result.exit_code == 0, result.output
    assert from_json.deleted == ["session-root", "session-child", "session-other"]

    as_jsonl = invoke(opencode_database, "sessions", "list", "--format", "jsonl")
    from_jsonl = _install_server(monkeypatch, _FakeServer())
    result = invoke(
        opencode_database, "sessions", "delete", "--yes", stdin=as_jsonl.stdout
    )
    assert result.exit_code == 0, result.output
    assert from_jsonl.deleted == ["session-root", "session-child", "session-other"]

    bare = _install_server(monkeypatch, _FakeServer())
    result = invoke(
        opencode_database,
        "sessions",
        "delete",
        "--yes",
        stdin="session-child\nsession-o\n\n",
    )
    assert result.exit_code == 0, result.output
    assert bare.deleted == ["session-child", "session-other"]


def test_accepts_piped_search_matches(
    opencode_database: Path, server: _FakeServer
) -> None:
    matches = invoke(
        opencode_database, "sessions", "search", "needle", "--format", "json"
    )
    assert matches.exit_code == 0, matches.output
    result = invoke(opencode_database, "sessions", "archive", stdin=matches.stdout)
    assert result.exit_code == 0, result.output
    assert {session_id for session_id, _ in server.archived} == {
        "session-root",
        "session-child",
        "session-other",
    }


def test_arguments_take_precedence_over_stdin(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(
        opencode_database,
        "sessions",
        "archive",
        "session-other",
        stdin="session-child\n",
    )
    assert result.exit_code == 0, result.output
    assert [session_id for session_id, _ in server.archived] == ["session-other"]


def test_empty_pipe_is_a_no_op(opencode_database: Path, server: _FakeServer) -> None:
    result = invoke(opencode_database, "sessions", "delete", "--yes", stdin="")
    assert result.exit_code == 0, result.output
    assert server.deleted == []
    assert "No sessions to delete." in result.stderr


def test_unparsable_piped_line_is_reported(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(
        opencode_database,
        "sessions",
        "delete",
        "--yes",
        stdin="2026-01-01\tsome repo\tsome title\n",
    )
    assert result.exit_code == 2
    assert "stdin line 1" in result.stderr
    assert server.deleted == []


def test_reports_per_session_failures(
    opencode_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _install_server(
        monkeypatch, _FakeServer(missing=frozenset({"session-child"}))
    )
    result = invoke(
        opencode_database,
        "sessions",
        "delete",
        "session-other",
        "session-child",
        "--yes",
        "--format",
        "json",
    )
    assert result.exit_code == 1
    assert fake.deleted == ["session-other"]
    outcomes = json.loads(result.stdout)
    assert [outcome["status"] for outcome in outcomes] == ["applied", "failed"]
    assert "does not exist" in outcomes[1]["error"]


def test_machine_formats_describe_outcomes(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(
        opencode_database,
        "sessions",
        "archive",
        "session-other",
        "--format",
        "jsonl",
    )
    assert result.exit_code == 0, result.output
    outcome = json.loads(result.stdout.splitlines()[0])
    assert outcome == {
        "id": "session-other",
        "title": "Other widget needle",
        "repository": "widget",
        "status": "applied",
        "error": None,
    }


def test_piped_stdout_lists_applied_ids(
    opencode_database: Path, server: _FakeServer
) -> None:
    result = invoke(
        opencode_database, "sessions", "archive", "session-other", "session-child"
    )
    assert result.exit_code == 0, result.output
    assert result.stdout.split() == ["session-other", "session-child"]


def test_server_requires_the_standard_database_layout(tmp_path: Path) -> None:
    standard = tmp_path / "share" / "opencode" / "opencode.db"
    assert data_home_for_database(standard) == tmp_path / "share"

    with pytest.raises(OpencodeServerError, match="data home"):
        data_home_for_database(tmp_path / "elsewhere.db")


def test_selectors_resolve_without_the_cli() -> None:
    from opencode_power_pack.models import Session
    from opencode_power_pack.session_selectors import (
        SessionResolver,
        SessionSelector,
        selectors_from_arguments,
    )

    def session(identifier: str) -> Session:
        return Session(
            id=identifier,
            title=None,
            created=None,
            updated=None,
            project_id=None,
            directory=None,
        )

    def selector(text: str) -> SessionSelector:
        return SessionSelector(text=text, origin="argument")

    resolver = SessionResolver([session("ses_abc"), session("ses_abd"), session("ses")])
    assert resolver.resolve(selector("ses_abc")).id == "ses_abc"
    assert resolver.resolve(selector("ses")).id == "ses"
    assert resolver.resolve(selector("ses_abd")).id == "ses_abd"
    with pytest.raises(UsageError, match="ambiguous"):
        resolver.resolve(selector("ses_ab"))
    assert [
        selector.text for selector in selectors_from_arguments([" a ", "", "b"])
    ] == [
        "a",
        "b",
    ]
