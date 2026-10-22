"""Shared SQLite data for command-level session tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
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


@pytest.fixture
def opencode_database(tmp_path: Path) -> Iterator[Path]:
    """Create a small, synthetic OpenCode V2-shaped database."""
    database = tmp_path / "opencode.db"
    connection = create_database(database)
    insert_project(connection, "project-root", "/workspace/acme/widget", "widget")
    insert_project(
        connection,
        "project-child",
        "/workspace/acme/widget/packages/child",
        "widget-child",
    )
    insert_project(connection, "project-other", "/workspace/other/widget", "widget")
    connection.executemany(
        "INSERT INTO project_directory VALUES (?, ?)",
        [
            ("project-root", "/workspace/acme/widget"),
            ("project-root", "/worktrees/widget-feature"),
            ("project-child", "/workspace/acme/widget/packages/child"),
            ("project-other", "/workspace/other/widget"),
        ],
    )
    insert_session(
        connection,
        "session-root",
        project_id="project-root",
        title="Fix naïve parser 100%_ready",
        directory="/workspace/acme/widget",
        created=1_000,
        updated=5_000,
        agent="build",
        provider="openai",
        model="gpt-test",
        cost=1.25,
        tokens=(10, 20, 3, 4, 5),
    )
    insert_session(
        connection,
        "session-root-second",
        project_id="project-root",
        parent_id="session-root",
        title="A second root conversation",
        directory="/worktrees/widget-feature",
        created=1_100,
        updated=5_000,
        agent="plan",
        provider="anthropic",
        model="claude-test",
        cost=0.5,
        tokens=(1, 2, 0, 0, 0),
    )
    insert_session(
        connection,
        "session-child",
        project_id="project-child",
        title="Child workspace needle",
        directory="/workspace/acme/widget/packages/child",
        created=1_200,
        updated=4_000,
        model="gpt-test",
    )
    insert_session(
        connection,
        "session-archived",
        project_id="project-root",
        title="Archived needle",
        directory="/workspace/acme/widget",
        created=1_300,
        updated=3_000,
        archived=6_000,
    )
    connection.execute(
        "UPDATE session_v2 SET model = 'gpt-test' WHERE id = 'session-archived'"
    )
    insert_session(
        connection,
        "session-other",
        project_id="project-other",
        title="Other widget needle",
        directory="/workspace/other/widget",
        created=1_400,
        updated=2_000,
        agent="review",
        provider="cohere",
        model="command-test",
    )
    insert_message(
        connection,
        "session-root",
        "message-user",
        "user",
        user("Needle needle in user text; literal 100%_ready and 東京."),
        created=1_010,
    )
    insert_message(
        connection,
        "session-root",
        "message-assistant",
        "assistant",
        assistant(
            text("The assistant also mentions needle."),
            reasoning("private reasoning needle"),
            tool("grep", input="tool input needle", output="tool output needle"),
            todowrite(("Investigate needle", "pending")),
        ),
        created=1_020,
    )
    insert_message(
        connection,
        "session-root",
        "message-reminder",
        "synthetic",
        {"text": "<system-reminder>synthetic needle</system-reminder>"},
        created=1_030,
    )
    insert_message(
        connection,
        "session-child",
        "message-child",
        "user",
        user("child body needle"),
        created=1_210,
    )
    insert_message(
        connection,
        "session-child",
        "message-child-assistant",
        "assistant",
        assistant(todowrite(("Child todo", "completed"))),
        created=1_220,
    )
    connection.commit()
    connection.close()
    yield database
