"""Tests for the sessions show command."""

import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import pytest
import typer
from typer.testing import CliRunner, Result

from opencode_power_pack.cli import app
from opencode_power_pack.commands.sessions.show_sessions import _run_show_sessions
from opencode_power_pack.errors import UsageError
from opencode_power_pack.models import AppContext

runner = CliRunner()

_BODY_RE = re.compile(
    r'<oc-(text|reasoning|input|output)(?: truncated="\d+")?>\n(.*?)\n</oc-\1>',
    re.DOTALL,
)


def _bodies(output: str, tag: str) -> list[str]:
    """Return the verbatim content of every ``<oc-{tag}>`` element in ``output``."""
    return [body for name, body in _BODY_RE.findall(output) if name == tag]


def _create_minimal_schema(connection: sqlite3.Connection) -> None:
    """Create the smallest schema the command's database layer tolerates."""
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
        CREATE TABLE message (id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE part (message_id TEXT, session_id TEXT, data TEXT);
        CREATE TABLE todo (session_id TEXT, content TEXT);
        """
    )
    connection.execute("INSERT INTO project VALUES ('project-x', '/work/x', 'x')")


def _insert_session(
    connection: sqlite3.Connection, session_id: str, title: str
) -> None:
    connection.execute(
        "INSERT INTO session VALUES "
        "(?, 'project-x', NULL, ?, '/work/x', 0, 'build', 'gpt', 1, 1, NULL, "
        "0, 0, 0, 0, 0)",
        (session_id, title),
    )


def _insert_message_and_part(
    connection: sqlite3.Connection,
    message_id: str,
    session_id: str,
    role: str,
    part_data: dict[str, object],
) -> None:
    connection.execute(
        "INSERT INTO message VALUES (?, ?, ?)",
        (message_id, session_id, json.dumps({"role": role})),
    )
    connection.execute(
        "INSERT INTO part VALUES (?, ?, ?)",
        (message_id, session_id, json.dumps(part_data)),
    )


def invoke(database: Path, *arguments: str, stdin: str | None = None) -> Result:
    return runner.invoke(
        app,
        ["--database", str(database), *arguments],
        input=stdin,
        env={"NO_COLOR": "1"},
    )


class _FakeContext:
    """A minimal stand-in for ``typer.Context`` that satisfies ``get_app_context``."""

    def __init__(self, database: Path) -> None:
        self._app_context = AppContext(database=database)

    def find_root(self) -> _FakeContext:
        return self

    @property
    def obj(self) -> AppContext:
        return self._app_context


class _FakeTTYStdin:
    """A stream that looks like an interactive terminal and must not be read."""

    def isatty(self) -> bool:
        return True

    def read(self) -> str:
        raise AssertionError("stdin should not be read when isatty() is True")


def test_shows_a_single_session_with_title_and_text(opencode_database: Path) -> None:
    result = invoke(opencode_database, "sessions", "show", "session-root")
    assert result.exit_code == 0, result.output
    assert "<oc-session" in result.stdout
    assert 'title="Fix naïve parser 100%_ready"' in result.stdout
    texts = _bodies(result.stdout, "text")
    assert "Needle needle in user text; literal 100%_ready and 東京." in texts
    assert "The assistant also mentions needle." in texts


def test_output_parses_as_xml(opencode_database: Path) -> None:
    # Verbatim message bodies are deliberately unescaped (see transcript.py),
    # so this only holds for XML-safe fixture content, such as the shared
    # `opencode_database` fixture; it is not a universal property of the
    # format, and it is not asserted for content containing `&`, `<`, or `>`.
    result = invoke(opencode_database, "sessions", "show", "session-root")
    assert result.exit_code == 0, result.output
    ET.fromstring(f"<root>\n{result.stdout}</root>")


def test_verbatim_fidelity_of_markdown_shaped_content(tmp_path: Path) -> None:
    """The property the XML-tagged format exists for: a message body that
    itself contains Markdown structure survives byte-for-byte, and none of
    its lines are mistaken for the transcript's own structure. The shared
    `opencode_database` fixture has no such content, so this builds its own.
    """
    database = tmp_path / "opencode.db"
    connection = sqlite3.connect(database)
    _create_minimal_schema(connection)
    _insert_session(connection, "session-md", "Markdown body")
    body = (
        "## Heading in the conversation\n"
        "\n"
        "A rule follows.\n"
        "\n"
        "---\n"
        "\n"
        "```python\n"
        "def needle():\n"
        "    return 1\n"
        "```"
    )
    _insert_message_and_part(
        connection, "message-md", "session-md", "user", {"type": "text", "text": body}
    )
    connection.commit()
    connection.close()

    result = invoke(database, "sessions", "show", "session-md")
    assert result.exit_code == 0, result.output
    assert f"<oc-text>\n{body}\n</oc-text>" in result.stdout
    for line in body.splitlines():
        assert not line.startswith("<oc-"), result.stdout


def test_reasoning_is_opt_in(opencode_database: Path) -> None:
    default = invoke(opencode_database, "sessions", "show", "session-root")
    assert default.exit_code == 0, default.output
    assert "<oc-reasoning" not in default.stdout
    assert "private reasoning needle" not in default.stdout

    with_reasoning = invoke(
        opencode_database, "sessions", "show", "session-root", "--include-reasoning"
    )
    assert with_reasoning.exit_code == 0, with_reasoning.output
    assert "private reasoning needle" in _bodies(with_reasoning.stdout, "reasoning")


def test_tool_inputs_and_outputs_are_opt_in(opencode_database: Path) -> None:
    default = invoke(opencode_database, "sessions", "show", "session-root")
    assert default.exit_code == 0, default.output
    assert "<oc-input" not in default.stdout
    assert "<oc-output" not in default.stdout
    assert "tool input needle" not in default.stdout
    assert "tool output needle" not in default.stdout

    with_inputs = invoke(
        opencode_database, "sessions", "show", "session-root", "--include-tool-inputs"
    )
    assert with_inputs.exit_code == 0, with_inputs.output
    assert "tool input needle" in _bodies(with_inputs.stdout, "input")
    assert "<oc-output" not in with_inputs.stdout

    with_outputs = invoke(
        opencode_database,
        "sessions",
        "show",
        "session-root",
        "--include-tool-outputs",
    )
    assert with_outputs.exit_code == 0, with_outputs.output
    assert "tool output needle" in _bodies(with_outputs.stdout, "output")
    assert "<oc-input" not in with_outputs.stdout


def test_max_tool_chars_truncates_tool_output(opencode_database: Path) -> None:
    result = invoke(
        opencode_database,
        "sessions",
        "show",
        "session-root",
        "--include-tool-outputs",
        "--max-tool-chars",
        "10",
    )
    assert result.exit_code == 0, result.output
    assert "tool output needle" not in result.stdout
    match = re.search(r'<oc-output truncated="(\d+)">', result.stdout)
    assert match is not None, result.stdout
    assert int(match.group(1)) == len("tool output needle") - 10


def test_include_todos_appends_the_todo_list(opencode_database: Path) -> None:
    default = invoke(opencode_database, "sessions", "show", "session-root")
    assert default.exit_code == 0, default.output
    assert "<oc-todos>" not in default.stdout
    assert "Investigate needle" not in default.stdout

    with_todos = invoke(
        opencode_database, "sessions", "show", "session-root", "--include-todos"
    )
    assert with_todos.exit_code == 0, with_todos.output
    assert "<oc-todos>" in with_todos.stdout
    assert "Investigate needle" in with_todos.stdout


def test_include_children_nests_child_session_in_the_spawning_tool(
    tmp_path: Path,
) -> None:
    database = tmp_path / "opencode.db"
    connection = sqlite3.connect(database)
    _create_minimal_schema(connection)
    _insert_session(connection, "session-parent", "Parent")
    _insert_session(connection, "session-sub", "Sub-agent")
    _insert_message_and_part(
        connection,
        "message-parent",
        "session-parent",
        "assistant",
        {
            "type": "tool",
            "tool": "task",
            "state": {
                "status": "completed",
                "metadata": {"sessionId": "session-sub"},
            },
        },
    )
    _insert_message_and_part(
        connection,
        "message-sub",
        "session-sub",
        "assistant",
        {"type": "text", "text": "Sub-agent reply needle"},
    )
    connection.commit()
    connection.close()

    without_children = invoke(database, "sessions", "show", "session-parent")
    assert without_children.exit_code == 0, without_children.output
    assert "<oc-tool" in without_children.stdout
    assert "Sub-agent reply needle" not in without_children.stdout

    with_children = invoke(
        database, "sessions", "show", "session-parent", "--include-children"
    )
    assert with_children.exit_code == 0, with_children.output
    match = re.search(
        r"<oc-tool[^>]*>\n(<oc-session[\s\S]*?</oc-session>)\n</oc-tool>",
        with_children.stdout,
    )
    assert match is not None, with_children.stdout
    assert 'id="session-sub"' in match.group(1)
    assert "Sub-agent reply needle" in match.group(1)


def test_unique_prefix_resolves(opencode_database: Path) -> None:
    result = invoke(opencode_database, "sessions", "show", "session-c")
    assert result.exit_code == 0, result.output
    assert "Child workspace needle" in result.stdout


def test_ambiguous_and_unknown_selectors_are_rejected(
    opencode_database: Path,
) -> None:
    ambiguous = invoke(opencode_database, "sessions", "show", "session-r")
    assert ambiguous.exit_code == 2
    assert "ambiguous" in ambiguous.stderr
    assert ambiguous.stdout == ""

    unknown = invoke(opencode_database, "sessions", "show", "nope")
    assert unknown.exit_code == 2
    assert "no session matches" in unknown.stderr
    assert unknown.stdout == ""


def test_accepts_piped_bare_ids(opencode_database: Path) -> None:
    result = invoke(
        opencode_database, "sessions", "show", stdin="session-root\nsession-child\n"
    )
    assert result.exit_code == 0, result.output
    assert "Fix naïve parser" in result.stdout
    assert "Child workspace needle" in result.stdout


def test_accepts_piped_list_table_output(opencode_database: Path) -> None:
    listed = invoke(opencode_database, "sessions", "list")
    assert listed.exit_code == 0, listed.output
    assert "\t" in listed.stdout

    result = invoke(opencode_database, "sessions", "show", stdin=listed.stdout)
    assert result.exit_code == 0, result.output
    assert "Fix naïve parser" in result.stdout
    assert "Child workspace needle" in result.stdout
    assert "Other widget needle" in result.stdout


def test_accepts_piped_json_output(opencode_database: Path) -> None:
    as_json = invoke(opencode_database, "sessions", "list", "--format", "json")
    assert as_json.exit_code == 0, as_json.output

    result = invoke(opencode_database, "sessions", "show", stdin=as_json.stdout)
    assert result.exit_code == 0, result.output
    assert "Fix naïve parser" in result.stdout
    assert "Child workspace needle" in result.stdout
    assert "Other widget needle" in result.stdout


def test_empty_pipe_reports_no_sessions_and_succeeds(opencode_database: Path) -> None:
    result = invoke(opencode_database, "sessions", "show", stdin="")
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "No sessions to show." in result.stderr


def test_no_arguments_with_a_tty_stdin_is_a_usage_error(
    opencode_database: Path,
) -> None:
    context = cast(typer.Context, _FakeContext(opencode_database))
    with pytest.raises(UsageError) as error:
        _run_show_sessions(
            context,
            sessions=(),
            output=None,
            include_reasoning=False,
            include_tool_inputs=False,
            include_tool_outputs=False,
            include_todos=False,
            include_children=False,
            max_tool_chars=2000,
            stdin=_FakeTTYStdin(),
        )
    assert error.value.exit_code == 2


def test_multiple_sessions_are_shown_in_order_with_no_separator(
    opencode_database: Path,
) -> None:
    result = invoke(
        opencode_database, "sessions", "show", "session-child", "session-root"
    )
    assert result.exit_code == 0, result.output
    child_index = result.stdout.index("Child workspace needle")
    root_index = result.stdout.index("Fix naïve parser")
    assert child_index < root_index
    assert result.stdout.count("<oc-session") == 2
    assert "</oc-session>\n<oc-session" in result.stdout


def test_output_option_writes_a_file_and_leaves_stdout_empty(
    opencode_database: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "conversation.md"
    to_stdout = invoke(opencode_database, "sessions", "show", "session-root")
    assert to_stdout.exit_code == 0, to_stdout.output

    result = invoke(
        opencode_database,
        "sessions",
        "show",
        "session-root",
        "--output",
        str(destination),
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert "conversation.md" in result.stderr
    assert destination.read_text(encoding="utf-8") == to_stdout.stdout


def test_archived_session_can_still_be_shown(opencode_database: Path) -> None:
    result = invoke(opencode_database, "sessions", "show", "session-archived")
    assert result.exit_code == 0, result.output
    assert "Archived needle" in result.stdout
