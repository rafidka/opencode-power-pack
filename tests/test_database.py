import json
import sqlite3
from pathlib import Path

import pytest

from opencode_power_pack.database import OpenCodeDatabase, resolve_database_path
from opencode_power_pack.errors import DatabaseError
from opencode_power_pack.models import (
    ListSessionsQuery,
    SearchSessionsQuery,
    SearchSource,
    SessionFilter,
    SortDirection,
)


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE project (id TEXT, worktree TEXT, name TEXT);
        CREATE TABLE project_directory (project_id TEXT, directory TEXT);
        CREATE TABLE session (
            id TEXT, project_id TEXT, parent_id TEXT, title TEXT, directory TEXT,
            cost REAL, agent TEXT, model TEXT, time_created INTEGER,
            time_updated INTEGER, time_archived INTEGER, tokens_input INTEGER,
            tokens_output INTEGER, tokens_reasoning INTEGER, tokens_cache_read INTEGER,
            tokens_cache_write INTEGER
        );
        CREATE TABLE message (id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE part (message_id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE todo (session_id TEXT, content TEXT);
    """)
    connection.execute("INSERT INTO project VALUES ('p', '/work/repo', 'repo')")
    connection.execute(
        "INSERT INTO session VALUES (?, 'p', NULL, ?, '/work/repo', 1, "
        "'agent', ?, 1, ?, NULL, 1, 2, 3, 4, 5)",
        ("s", "Straße title", json.dumps({"providerID": "openai", "id": "gpt"}), 2),
    )
    connection.execute(
        "INSERT INTO message VALUES ('m', 's', ?)", (json.dumps({"role": "user"}),)
    )
    connection.execute(
        "INSERT INTO part VALUES ('m', 's', ?)",
        (json.dumps({"type": "text", "text": "a 100%_ literal Straße"}),),
    )
    connection.commit()
    connection.close()


def test_database_path_read_only_listing_and_literal_unicode_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "opencode.db"
    _database(path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert resolve_database_path() == tmp_path / "data" / "opencode" / "opencode.db"
    assert resolve_database_path(path) == path

    with OpenCodeDatabase(path) as database:
        assert database.list_sessions(ListSessionsQuery())[0].repository == "repo"
        match = database.search_sessions(SearchSessionsQuery("STRASSE"))[0]
        assert match.source is SearchSource.TITLE
        assert (
            database.search_sessions(SearchSessionsQuery("100%_"))[0].source
            is SearchSource.TEXT
        )
        assert database._connection is not None
        with pytest.raises(sqlite3.OperationalError):
            database._connection.execute("DELETE FROM session")


def test_incompatible_schema_is_a_friendly_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.db"
    sqlite3.connect(path).close()

    with pytest.raises(DatabaseError, match="incompatible OpenCode database schema"):
        OpenCodeDatabase(path).list_repositories()


def test_search_snippet_preserves_casefolded_match_location() -> None:
    from opencode_power_pack.database.search import match_index, snippet

    content = "before" + ("\n\t " * 80) + "Straße after unrelated tail"
    index = match_index(content, "STRASSE", case_sensitive=False)

    assert index is not None
    result = snippet(content, index, 16)
    assert "Straße" in result
    assert "unrelated" not in result


def test_since_is_inclusive_and_until_is_exclusive(opencode_database: Path) -> None:
    with OpenCodeDatabase(opencode_database) as database:
        inclusive = database.list_sessions(
            ListSessionsQuery(
                filters=SessionFilter(since=5_000, until=5_001),
            )
        )
        exclusive = database.list_sessions(
            ListSessionsQuery(
                filters=SessionFilter(until=5_000),
            )
        )

    assert [session.id for session in inclusive] == ["session-root"]
    assert [session.id for session in exclusive] == ["session-child", "session-other"]


def test_list_and_search_id_tiebreaking_follows_their_contracts(
    opencode_database: Path,
) -> None:
    connection = sqlite3.connect(opencode_database)
    connection.executemany(
        "INSERT INTO session "
        "(id, project_id, parent_id, title, directory, time_created, time_updated) "
        "VALUES (?, 'project-root', NULL, 'needle tie', "
        "'/workspace/acme/widget', 1, 7000)",
        [("tie-a",), ("tie-b",)],
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(opencode_database) as database:
        descending = database.list_sessions(ListSessionsQuery())
        ascending = database.list_sessions(
            ListSessionsQuery(direction=SortDirection.ASC)
        )
        matches = database.search_sessions(SearchSessionsQuery("needle"))

    assert [session.id for session in descending[:2]] == ["tie-b", "tie-a"]
    assert [session.id for session in ascending[-2:]] == ["tie-a", "tie-b"]
    assert [match.session.id for match in matches[:2]] == ["tie-b", "tie-a"]
    assert all(match.source is SearchSource.TITLE for match in matches[:4])
    assert matches[4].source is SearchSource.TEXT
