import json
import sqlite3
from pathlib import Path

from opencode_power_pack.database import OpenCodeDatabase
from opencode_power_pack.models import PartKind, TranscriptQuery


def _base_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE project (id TEXT, worktree TEXT, name TEXT);
        CREATE TABLE project_directory (project_id TEXT, directory TEXT);
        CREATE TABLE session (
            id TEXT, project_id TEXT, parent_id TEXT, title TEXT, directory TEXT,
            cost REAL, agent TEXT, model TEXT, time_created INTEGER,
            time_updated INTEGER, time_archived INTEGER, tokens_input INTEGER,
            tokens_output INTEGER, tokens_reasoning INTEGER, tokens_cache_read INTEGER,
            tokens_cache_write INTEGER
        );
        """
    )
    connection.execute("INSERT INTO project VALUES ('p', '/work/repo', 'repo')")


def _insert_session(connection: sqlite3.Connection, session_id: str) -> None:
    connection.execute(
        "INSERT INTO session VALUES "
        "(?, 'p', NULL, ?, '/work/repo', 0, 'build', 'gpt', 1, 1, NULL, 0, 0, 0, 0, 0)",
        (session_id, session_id),
    )


def _full_database(path: Path) -> None:
    """Build a database with every optional transcript column present."""
    connection = sqlite3.connect(path)
    _base_schema(connection)
    connection.executescript(
        """
        CREATE TABLE message (
            id TEXT, session_id TEXT, time_created INTEGER, time_updated INTEGER,
            data TEXT
        );
        CREATE TABLE part (
            id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER,
            time_updated INTEGER, data TEXT
        );
        CREATE TABLE todo (
            session_id TEXT, content TEXT, status TEXT, priority TEXT,
            position INTEGER, time_created INTEGER, time_updated INTEGER
        );
        """
    )
    connection.commit()
    connection.close()


def _minimal_database(path: Path) -> None:
    """Build a database mirroring the minimal shape tolerated in production."""
    connection = sqlite3.connect(path)
    _base_schema(connection)
    connection.executescript(
        """
        CREATE TABLE message (id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE part (message_id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE todo (session_id TEXT, content TEXT);
        """
    )
    connection.commit()
    connection.close()


def _insert_message(
    connection: sqlite3.Connection,
    message_id: str,
    session_id: str,
    data: dict[str, object],
    time_created: int | None = None,
) -> None:
    has_time_created = any(
        row["name"] == "time_created"
        for row in connection.execute("PRAGMA table_info(message)")
    )
    if has_time_created:
        connection.execute(
            "INSERT INTO message (id, session_id, time_created, data) "
            "VALUES (?, ?, ?, ?)",
            (message_id, session_id, time_created, json.dumps(data)),
        )
    else:
        connection.execute(
            "INSERT INTO message (id, session_id, data) VALUES (?, ?, ?)",
            (message_id, session_id, json.dumps(data)),
        )


def _insert_part(
    connection: sqlite3.Connection,
    message_id: str,
    session_id: str,
    data: dict[str, object],
    part_id: str | None = None,
    time_created: int | None = None,
) -> None:
    has_id = any(
        row["name"] == "id" for row in connection.execute("PRAGMA table_info(part)")
    )
    has_time_created = any(
        row["name"] == "time_created"
        for row in connection.execute("PRAGMA table_info(part)")
    )
    columns = ["message_id", "session_id"]
    values: list[object] = [message_id, session_id]
    if has_id:
        columns.insert(0, "id")
        values.insert(0, part_id)
    if has_time_created:
        columns.append("time_created")
        values.append(time_created)
    columns.append("data")
    values.append(json.dumps(data))
    placeholders = ", ".join("?" for _ in values)
    connection.execute(
        f"INSERT INTO part ({', '.join(columns)}) VALUES ({placeholders})", values
    )


def test_messages_are_ordered_chronologically_despite_reversed_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-root")
    _insert_message(
        connection, "zzz-first", "session-root", {"role": "user"}, time_created=1_000
    )
    _insert_message(
        connection,
        "aaa-second",
        "session-root",
        {"role": "assistant"},
        time_created=2_000,
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-root",))
        )

    assert [message.id for message in transcripts[0].messages] == [
        "zzz-first",
        "aaa-second",
    ]


def test_parts_are_ordered_chronologically_within_a_message(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-root")
    _insert_message(connection, "message-1", "session-root", {"role": "user"}, 1_000)
    _insert_part(
        connection,
        "message-1",
        "session-root",
        {"type": "text", "text": "second"},
        part_id="part-b",
        time_created=20,
    )
    _insert_part(
        connection,
        "message-1",
        "session-root",
        {"type": "text", "text": "first"},
        part_id="part-a",
        time_created=10,
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-root",))
        )

    texts = [part.text for part in transcripts[0].messages[0].parts]
    assert texts == ["first", "second"]


def test_part_kinds_map_correctly_and_excluded_parts_are_dropped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-kinds")
    _insert_message(
        connection, "message-1", "session-kinds", {"role": "assistant"}, 1_000
    )
    parts: list[tuple[str, dict[str, object]]] = [
        ("part-01", {"type": "text", "text": "hello"}),
        ("part-02", {"type": "text", "text": "replayed", "synthetic": True}),
        ("part-03", {"type": "text", "text": ""}),
        ("part-04", {"type": "reasoning", "text": "thinking"}),
        ("part-05", {"type": "reasoning", "text": ""}),
        (
            "part-06",
            {
                "type": "tool",
                "tool": "task",
                "callID": "call-1",
                "state": {
                    "status": "completed",
                    "input": {"prompt": "do it"},
                    "output": "done",
                    "title": "Spawn helper",
                    "metadata": {"sessionId": "session-sub"},
                },
            },
        ),
        ("part-07", {"type": "patch", "hash": "abc", "files": ["a.py", 1, "b.py"]}),
        ("part-08", {"type": "file", "mime": "text/plain", "filename": "notes.txt"}),
        ("part-09", {"type": "compaction", "auto": True, "overflow": False}),
        ("part-10", {"type": "agent", "name": "reviewer"}),
        ("part-11", {"type": "step-start"}),
        ("part-12", {"type": "step-finish"}),
        ("part-13", {"type": "unknown-future-type", "text": "should be dropped"}),
    ]
    for index, (part_id, data) in enumerate(parts):
        _insert_part(
            connection,
            "message-1",
            "session-kinds",
            data,
            part_id=part_id,
            time_created=index,
        )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-kinds",))
        )

    result_parts = transcripts[0].messages[0].parts
    kinds = [part.kind for part in result_parts]
    assert kinds == [
        PartKind.TEXT,
        PartKind.REASONING,
        PartKind.TOOL,
        PartKind.PATCH,
        PartKind.FILE,
        PartKind.COMPACTION,
        PartKind.AGENT,
    ]

    text_part = result_parts[0]
    assert text_part.text == "hello"

    reasoning_part = result_parts[1]
    assert reasoning_part.text == "thinking"

    tool_part = result_parts[2]
    assert tool_part.tool_name == "task"
    assert tool_part.tool_title == "Spawn helper"
    assert tool_part.tool_status == "completed"
    assert tool_part.tool_input == json.dumps(
        {"prompt": "do it"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert tool_part.tool_output == "done"
    assert tool_part.child_session_id == "session-sub"

    patch_part = result_parts[3]
    assert patch_part.files == ("a.py", "b.py")

    file_part = result_parts[4]
    assert file_part.filename == "notes.txt"

    compaction_part = result_parts[5]
    assert compaction_part.compaction_auto is True

    agent_part = result_parts[6]
    assert agent_part.agent_name == "reviewer"


def test_assistant_tokens_cost_and_error_are_extracted(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-root")
    _insert_message(
        connection,
        "message-assistant",
        "session-root",
        {
            "role": "assistant",
            "agent": "build",
            "cost": 1.5,
            "tokens": {
                "total": 100,
                "input": 40,
                "output": 30,
                "reasoning": 10,
                "cache": {"read": 15, "write": 5},
            },
            "modelID": "gpt-test",
            "providerID": "openai",
            "finish": "stop",
            "error": {"name": "AbortError", "data": {}},
        },
        1_000,
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-root",))
        )

    message = transcripts[0].messages[0]
    assert message.role == "assistant"
    assert message.agent == "build"
    assert message.model == "gpt-test"
    assert message.provider == "openai"
    assert message.cost == 1.5
    assert message.error == "AbortError"
    assert message.tokens is not None
    assert message.tokens.input == 40
    assert message.tokens.output == 30
    assert message.tokens.reasoning == 10
    assert message.tokens.cache_read == 15
    assert message.tokens.cache_write == 5


def test_user_model_and_provider_come_from_nested_model_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-root")
    _insert_message(
        connection,
        "message-user",
        "session-root",
        {
            "role": "user",
            "time": {"created": 1_234},
            "agent": "build",
            "model": {"providerID": "anthropic", "modelID": "claude-test"},
        },
        1_000,
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-root",))
        )

    message = transcripts[0].messages[0]
    assert message.role == "user"
    assert message.model == "claude-test"
    assert message.provider == "anthropic"
    assert message.created == 1_234


def test_todos_respect_include_flag_and_position_order(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-root")
    connection.execute(
        "INSERT INTO todo (session_id, content, status, position) "
        "VALUES ('session-root', 'second', 'pending', 2)"
    )
    connection.execute(
        "INSERT INTO todo (session_id, content, status, position) "
        "VALUES ('session-root', 'first', 'completed', 1)"
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        without_todos = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-root",), include_todos=False)
        )
        with_todos = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-root",), include_todos=True)
        )

    assert without_todos[0].todos == ()
    assert [todo.content for todo in with_todos[0].todos] == ["first", "second"]
    assert [todo.status for todo in with_todos[0].todos] == ["completed", "pending"]


def test_children_are_inlined_only_when_requested(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-parent")
    _insert_session(connection, "session-sub")
    _insert_message(
        connection, "message-parent", "session-parent", {"role": "assistant"}, 1_000
    )
    _insert_part(
        connection,
        "message-parent",
        "session-parent",
        {
            "type": "tool",
            "tool": "task",
            "state": {
                "status": "completed",
                "metadata": {"sessionId": "session-sub"},
            },
        },
        part_id="part-task",
        time_created=1,
    )
    _insert_message(
        connection, "message-sub", "session-sub", {"role": "assistant"}, 1_100
    )
    _insert_part(
        connection,
        "message-sub",
        "session-sub",
        {"type": "text", "text": "sub-agent reply"},
        part_id="part-sub",
        time_created=1,
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        without_children = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-parent",), include_children=False)
        )
        with_children = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-parent",), include_children=True)
        )

    assert without_children[0].children == ()
    assert len(with_children[0].children) == 1
    child = with_children[0].children[0]
    assert child.session.id == "session-sub"
    assert child.messages[0].parts[0].text == "sub-agent reply"


def test_cyclic_parent_child_sessions_terminate(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-a")
    _insert_session(connection, "session-b")
    _insert_message(connection, "message-a", "session-a", {"role": "assistant"}, 1_000)
    _insert_part(
        connection,
        "message-a",
        "session-a",
        {
            "type": "tool",
            "tool": "task",
            "state": {"metadata": {"sessionId": "session-b"}},
        },
        part_id="part-a",
        time_created=1,
    )
    _insert_message(connection, "message-b", "session-b", {"role": "assistant"}, 1_000)
    _insert_part(
        connection,
        "message-b",
        "session-b",
        {
            "type": "tool",
            "tool": "task",
            "state": {"metadata": {"sessionId": "session-a"}},
        },
        part_id="part-b",
        time_created=1,
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-a",), include_children=True)
        )

    assert transcripts[0].session.id == "session-a"
    child = transcripts[0].children[0]
    assert child.session.id == "session-b"
    assert child.children == ()


def test_minimal_schema_without_optional_columns_still_works(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    _minimal_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-root")
    _insert_message(connection, "message-b", "session-root", {"role": "assistant"})
    _insert_message(connection, "message-a", "session-root", {"role": "user"})
    _insert_part(
        connection,
        "message-a",
        "session-root",
        {"type": "text", "text": "user text"},
    )
    _insert_part(
        connection,
        "message-b",
        "session-root",
        {"type": "text", "text": "assistant text"},
    )
    connection.execute(
        "INSERT INTO todo (session_id, content) VALUES ('session-root', 'a todo')"
    )
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-root",), include_todos=True)
        )

    assert [message.id for message in transcripts[0].messages] == [
        "message-a",
        "message-b",
    ]
    assert transcripts[0].todos[0].content == "a todo"
    assert transcripts[0].todos[0].status is None


def test_unknown_session_ids_are_skipped_and_order_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    _full_database(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _insert_session(connection, "session-a")
    _insert_session(connection, "session-b")
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        transcripts = database.fetch_transcripts(
            TranscriptQuery(session_ids=("session-b", "does-not-exist", "session-a"))
        )

    assert [transcript.session.id for transcript in transcripts] == [
        "session-b",
        "session-a",
    ]
