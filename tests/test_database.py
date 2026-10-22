import sqlite3
from pathlib import Path

import pytest
from opencode_v2 import (
    assistant,
    create_database,
    insert_message,
    insert_project,
    insert_session,
    text,
    user,
)

from opencode_power_pack.database import OpenCodeDatabase, resolve_database_path
from opencode_power_pack.database.client import _SQLITE_SESSION_ID_BIND_BATCH_SIZE
from opencode_power_pack.errors import DatabaseError
from opencode_power_pack.models import (
    ListSessionsQuery,
    SearchSessionsQuery,
    SearchSource,
    SessionFilter,
    SortDirection,
)


def _database(path: Path) -> None:
    connection = create_database(path)
    insert_project(connection)
    insert_session(
        connection,
        "s",
        title="Straße title",
        cost=1,
        updated=2,
        agent="agent",
        tokens=(1, 2, 3, 4, 5),
    )
    insert_message(connection, "s", "m", "user", user("a 100%_ literal Straße"))
    insert_message(
        connection, "s", "a", "assistant", assistant(text("first"), text("second"))
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
        match = database.search_sessions(SearchSessionsQuery(text="STRASSE"))[0]
        assert match.source is SearchSource.TITLE
        assert (
            database.search_sessions(SearchSessionsQuery(text="100%_"))[0].source
            is SearchSource.TEXT
        )
        assert database._connection is not None
        with pytest.raises(sqlite3.OperationalError):
            database._connection.execute("DELETE FROM session_v2")


def test_semantic_record_ids_identify_messages_and_assistant_blocks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    _database(path)

    with OpenCodeDatabase(path) as database:
        documents = database.fetch_semantic_documents({"s"})[0].documents

    extracted = [
        (document.source_record_id, document.content) for document in documents
    ]
    assert extracted == [
        ("title", "Straße title"),
        ("m", "a 100%_ literal Straße"),
        ("a:0", "first"),
        ("a:1", "second"),
    ]


def test_semantic_fetch_splits_selected_ids_at_sqlite_bind_batch_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    _database(path)
    session_ids = tuple(
        f"bulk-{index:04}" for index in range(_SQLITE_SESSION_ID_BIND_BATCH_SIZE + 1)
    )
    connection = sqlite3.connect(path)
    for session_id in session_ids:
        insert_session(connection, session_id)
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        assert database._connection is not None
        statements: list[str] = []
        database._connection.set_trace_callback(statements.append)
        batches = database.fetch_semantic_documents(session_ids)

    session_queries = [
        statement
        for statement in statements
        if statement.startswith("SELECT id, title, time_updated FROM session_v2")
    ]
    assert len(batches) == _SQLITE_SESSION_ID_BIND_BATCH_SIZE + 1
    assert len(session_queries) == 2


def test_incompatible_schema_is_a_friendly_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.db"
    sqlite3.connect(path).close()

    with pytest.raises(DatabaseError, match="incompatible OpenCode database schema"):
        OpenCodeDatabase(path).list_repositories()


def test_v1_database_is_rejected_with_a_migration_hint(tmp_path: Path) -> None:
    path = tmp_path / "v1.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE project (id TEXT, worktree TEXT, name TEXT);
        CREATE TABLE project_directory (project_id TEXT, directory TEXT);
        CREATE TABLE session (id TEXT);
        CREATE TABLE message (id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE part (message_id TEXT, session_id TEXT, data TEXT);
        """
    )
    connection.close()

    with pytest.raises(DatabaseError, match="OpenCode V1 database"):
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
    for session_id in ("tie-a", "tie-b"):
        insert_session(
            connection,
            session_id,
            project_id="project-root",
            title="needle tie",
            directory="/workspace/acme/widget",
            updated=7000,
        )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(opencode_database) as database:
        descending = database.list_sessions(ListSessionsQuery())
        ascending = database.list_sessions(
            ListSessionsQuery(direction=SortDirection.ASC)
        )
        matches = database.search_sessions(SearchSessionsQuery(text="needle"))

    assert [session.id for session in descending[:2]] == ["tie-b", "tie-a"]
    assert [session.id for session in ascending[-2:]] == ["tie-a", "tie-b"]
    assert [match.session.id for match in matches[:2]] == ["tie-b", "tie-a"]
    assert all(match.source is SearchSource.TITLE for match in matches[:4])
    assert matches[4].source is SearchSource.TEXT
