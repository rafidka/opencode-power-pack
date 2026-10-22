import json
import sqlite3
from pathlib import Path

from opencode_v2 import (
    assistant,
    create_database,
    insert_message,
    insert_project,
    insert_session,
    reasoning,
    text,
    todowrite,
    tool,
    user,
)

from opencode_power_pack.database import OpenCodeDatabase
from opencode_power_pack.models import PartKind, Transcript, TranscriptQuery


def _database(path: Path, *session_ids: str) -> sqlite3.Connection:
    connection = create_database(path)
    insert_project(connection)
    for session_id in session_ids:
        insert_session(connection, session_id)
    return connection


def _transcripts(
    path: Path, connection: sqlite3.Connection, query: TranscriptQuery
) -> tuple[Transcript, ...]:
    connection.commit()
    connection.close()
    with OpenCodeDatabase(path) as database:
        return database.fetch_transcripts(query)


def test_messages_follow_seq_rather_than_ids_or_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "s")
    insert_message(connection, "s", "zzz-first", "user", user("one"), created=9, seq=0)
    insert_message(
        connection, "s", "aaa-second", "assistant", assistant(), created=1, seq=1
    )

    transcripts = _transcripts(path, connection, TranscriptQuery(session_ids=("s",)))

    assert [message.id for message in transcripts[0].messages] == [
        "zzz-first",
        "aaa-second",
    ]


def test_user_messages_carry_text_files_and_agent_mentions(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "s")
    insert_message(
        connection,
        "s",
        "m",
        "user",
        user(
            "look at this",
            files=[{"mime": "image/png", "name": "image.png", "data": "..."}],
            agents=[{"name": "pr-reviewer", "mention": {"text": "@pr-reviewer"}}],
        ),
        created=1_234,
    )

    message = _transcripts(path, connection, TranscriptQuery(session_ids=("s",)))[
        0
    ].messages[0]

    assert message.role == "user"
    assert message.created == 1_234
    assert [
        (part.kind, part.text, part.filename, part.agent_name) for part in message.parts
    ] == [
        (PartKind.TEXT, "look at this", None, None),
        (PartKind.FILE, None, "image.png", None),
        (PartKind.AGENT, None, None, "pr-reviewer"),
    ]


def test_assistant_blocks_map_to_parts_and_empty_or_unknown_blocks_are_dropped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "s")
    insert_message(
        connection,
        "s",
        "m",
        "assistant",
        assistant(
            text("hello"),
            text(""),
            reasoning("thinking"),
            reasoning(""),
            tool(
                "task",
                input={"description": "Spawn helper", "prompt": "do it"},
                output="done",
                metadata={"sessionId": "session-sub"},
            ),
            {"type": "unknown-future-type", "text": "should be dropped"},
        ),
    )

    parts = (
        _transcripts(path, connection, TranscriptQuery(session_ids=("s",)))[0]
        .messages[0]
        .parts
    )

    assert [part.kind for part in parts] == [
        PartKind.TEXT,
        PartKind.REASONING,
        PartKind.TOOL,
    ]
    assert parts[0].text == "hello"
    assert parts[1].text == "thinking"
    call = parts[2]
    assert call.tool_name == "subagent"
    assert call.tool_title == "Spawn helper"
    assert call.tool_status == "completed"
    assert call.tool_input == json.dumps(
        {"description": "Spawn helper", "prompt": "do it"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert call.tool_output == "done"
    assert call.child_session_id == "session-sub"


def test_tool_names_titles_and_error_output(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "s")
    failed = tool("shell", input={"command": "false"}, status="error")
    failed_state = failed["state"]
    assert isinstance(failed_state, dict)
    failed_state["error"] = {"type": "tool.execution", "message": "exit code 1"}
    insert_message(
        connection,
        "s",
        "m",
        "assistant",
        assistant(
            tool("bash", input={"command": "git status\ngit log", "workdir": "/x"}),
            tool("apply_patch", input={"patchText": "*** Begin Patch"}),
            tool("read", input={"path": "src/app.py"}),
            tool("subagent", metadata={"sessionID": "session-v2-child"}),
            failed,
        ),
    )

    parts = (
        _transcripts(path, connection, TranscriptQuery(session_ids=("s",)))[0]
        .messages[0]
        .parts
    )

    assert [(part.tool_name, part.tool_title) for part in parts] == [
        ("shell", "git status…"),
        ("patch", None),
        ("read", "src/app.py"),
        ("subagent", None),
        ("shell", "false"),
    ]
    assert parts[3].child_session_id == "session-v2-child"
    assert parts[4].tool_status == "error"
    assert parts[4].tool_output == "exit code 1"


def test_assistant_model_tokens_cost_and_error_are_extracted(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "s")
    insert_message(
        connection,
        "s",
        "m",
        "assistant",
        assistant(
            agent="plan",
            model={"id": "gpt-test", "providerID": "openai", "variant": "high"},
            cost=1.5,
            tokens={
                "input": 40,
                "output": 30,
                "reasoning": 10,
                "cache": {"read": 15, "write": 5},
            },
            error={"type": "aborted", "message": "Aborted"},
        ),
    )

    message = _transcripts(path, connection, TranscriptQuery(session_ids=("s",)))[
        0
    ].messages[0]

    assert message.role == "assistant"
    assert message.agent == "plan"
    assert message.model == "gpt-test"
    assert message.provider == "openai"
    assert message.cost == 1.5
    assert message.error == "aborted"
    assert message.tokens is not None
    assert message.tokens.total == 100


def test_bookkeeping_messages_are_skipped_and_compactions_are_kept(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "s")
    insert_message(connection, "s", "u", "user", user("hi"))
    insert_message(connection, "s", "r", "synthetic", {"text": "<system-reminder>"})
    insert_message(connection, "s", "n", "system", {"text": "instructions changed"})
    insert_message(connection, "s", "i", "idle", {"outcome": "succeeded"})
    insert_message(connection, "s", "w", "agent-switched", {"agent": "build"})
    insert_message(
        connection, "s", "c", "compaction", {"status": "completed", "reason": "auto"}
    )

    messages = _transcripts(path, connection, TranscriptQuery(session_ids=("s",)))[
        0
    ].messages

    assert [(message.id, message.role) for message in messages] == [
        ("u", "user"),
        ("c", "compaction"),
    ]
    assert messages[1].parts[0].kind is PartKind.COMPACTION
    assert messages[1].parts[0].compaction_auto is True


def test_todos_come_from_the_latest_todowrite_and_respect_the_flag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "s")
    insert_message(
        connection, "s", "m1", "assistant", assistant(todowrite(("stale", "pending")))
    )
    insert_message(
        connection,
        "s",
        "m2",
        "assistant",
        assistant(todowrite(("first", "completed"), ("second", "pending"))),
    )
    insert_message(connection, "s", "m3", "assistant", assistant(text("done")))
    connection.commit()
    connection.close()

    with OpenCodeDatabase(path) as database:
        without_todos = database.fetch_transcripts(
            TranscriptQuery(session_ids=("s",), include_todos=False)
        )
        with_todos = database.fetch_transcripts(
            TranscriptQuery(session_ids=("s",), include_todos=True)
        )

    assert without_todos[0].todos == ()
    assert [(todo.content, todo.status) for todo in with_todos[0].todos] == [
        ("first", "completed"),
        ("second", "pending"),
    ]


def test_children_are_inlined_only_when_requested(tmp_path: Path) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "session-parent", "session-sub")
    insert_message(
        connection,
        "session-parent",
        "message-parent",
        "assistant",
        assistant(tool("task", metadata={"sessionId": "session-sub"})),
    )
    insert_message(
        connection,
        "session-sub",
        "message-sub",
        "assistant",
        assistant(text("sub-agent reply")),
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
    connection = _database(path, "session-a", "session-b")
    insert_message(
        connection,
        "session-a",
        "message-a",
        "assistant",
        assistant(tool("subagent", metadata={"sessionID": "session-b"})),
    )
    insert_message(
        connection,
        "session-b",
        "message-b",
        "assistant",
        assistant(tool("subagent", metadata={"sessionID": "session-a"})),
    )

    transcripts = _transcripts(
        path,
        connection,
        TranscriptQuery(session_ids=("session-a",), include_children=True),
    )

    assert transcripts[0].session.id == "session-a"
    child = transcripts[0].children[0]
    assert child.session.id == "session-b"
    assert child.children == ()


def test_unknown_session_ids_are_skipped_and_order_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.db"
    connection = _database(path, "session-a", "session-b")

    transcripts = _transcripts(
        path,
        connection,
        TranscriptQuery(session_ids=("session-b", "does-not-exist", "session-a")),
    )

    assert [transcript.session.id for transcript in transcripts] == [
        "session-b",
        "session-a",
    ]
