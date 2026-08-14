"""End-to-end coverage for the public Typer command line interface."""

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner, Result

from opencode_power_pack import __version__
from opencode_power_pack.cli import app

runner = CliRunner()


def invoke(database: Path, *arguments: str, env: dict[str, str] | None = None):
    command_env = {"NO_COLOR": "1"}
    if env is not None:
        command_env.update(env)
    return runner.invoke(
        app, ["--database", str(database), *arguments], env=command_env
    )


def json_output(result: Result) -> object:
    output = result.stdout
    assert "\x1b" not in output
    return json.loads(output)


def test_root_sessions_help_and_version() -> None:
    assert runner.invoke(app, ["--help"]).exit_code == 0
    sessions_help = runner.invoke(app, ["sessions", "--help"])
    assert sessions_help.exit_code == 0
    assert "list" in sessions_help.stdout
    assert "search" in sessions_help.stdout
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert version.stdout.strip() == __version__
    from opencode_power_pack.commands.sessions import app as sessions_app
    from opencode_power_pack.commands.sessions import register_available_commands

    register_available_commands()
    command_names = [command.name for command in sessions_app.registered_commands]
    assert command_names.count("list") == 1
    assert command_names.count("search") == 1


def test_list_defaults_database_overrides_and_machine_formats(
    opencode_database: Path,
) -> None:
    result = invoke(opencode_database, "sessions", "list", "--format", "json")
    assert result.exit_code == 0, result.output
    sessions = json_output(result)
    assert isinstance(sessions, list)
    assert [session["id"] for session in sessions] == [
        "session-root",
        "session-child",
        "session-other",
    ]
    assert all(
        set(session) >= {"id", "title", "projectId", "directory"}
        for session in sessions
    )

    jsonl = invoke(opencode_database, "sessions", "list", "--format", "jsonl")
    assert jsonl.exit_code == 0, jsonl.output
    assert [json.loads(line)["id"] for line in jsonl.stdout.splitlines()] == [
        "session-root",
        "session-child",
        "session-other",
    ]
    assert "\x1b" not in jsonl.stdout

    environment = runner.invoke(
        app,
        ["sessions", "list", "--format", "json"],
        env={"OPENCODE_DATABASE": str(opencode_database), "NO_COLOR": "1"},
    )
    assert environment.exit_code == 0, environment.output
    assert [item["id"] for item in json_output(environment)] == [
        "session-root",
        "session-child",
        "session-other",
    ]

    zero = invoke(
        opencode_database,
        "sessions",
        "list",
        "--max-count",
        "0",
        "--format",
        "json",
    )
    assert zero.exit_code == 0, zero.output
    assert json_output(zero) == []


def test_list_repository_children_and_archive_controls(opencode_database: Path) -> None:
    root_only = invoke(
        opencode_database,
        "sessions",
        "list",
        "--repo",
        "/workspace/acme/widget",
        "--format",
        "json",
    )
    assert root_only.exit_code == 0, root_only.output
    assert {item["id"] for item in json_output(root_only)} == {
        "session-root",
    }

    with_children = invoke(
        opencode_database,
        "sessions",
        "list",
        "--repo",
        "/workspace/acme/widget",
        "--include-children",
        "--format",
        "json",
    )
    assert with_children.exit_code == 0, with_children.output
    assert {item["id"] for item in json_output(with_children)} == {
        "session-root",
        "session-root-second",
    }

    archived = invoke(
        opencode_database,
        "sessions",
        "list",
        "--archived-only",
        "--format",
        "json",
    )
    assert archived.exit_code == 0, archived.output
    assert [item["id"] for item in json_output(archived)] == ["session-archived"]

    provider = invoke(
        opencode_database,
        "sessions",
        "list",
        "--provider",
        "cohere",
        "--format",
        "json",
    )
    assert provider.exit_code == 0, provider.output
    assert [item["id"] for item in json_output(provider)] == ["session-other"]

    model = invoke(
        opencode_database,
        "sessions",
        "list",
        "--agent",
        "build",
        "--model",
        "gpt-test",
        "--format",
        "json",
    )
    assert model.exit_code == 0, model.output
    assert {item["id"] for item in json_output(model)} == {
        "session-root",
        "session-child",
    }


def test_list_ambiguous_and_unique_repository_filters(opencode_database: Path) -> None:
    ambiguous = invoke(opencode_database, "sessions", "list", "--repo", "widget")
    assert ambiguous.exit_code == 2
    assert "ambiguous" in ambiguous.output.lower()

    unique = invoke(
        opencode_database,
        "sessions",
        "list",
        "--repo",
        "/worktrees/widget-feature",
        "--include-children",
        "--format",
        "json",
    )
    assert unique.exit_code == 0, unique.output
    assert {item["id"] for item in json_output(unique)} == {
        "session-root",
        "session-root-second",
    }


def test_search_defaults_roles_sources_and_literal_unicode_matching(
    opencode_database: Path,
) -> None:
    default = invoke(
        opencode_database, "sessions", "search", "needle", "--format", "json"
    )
    assert default.exit_code == 0, default.output
    matches = json_output(default)
    assert isinstance(matches, list)
    assert {match["session"]["id"] for match in matches} == {
        "session-root",
        "session-child",
        "session-other",
    }
    root_match = next(
        match for match in matches if match["session"]["id"] == "session-root"
    )
    assert root_match["matchCount"] == 2
    assert all(match["source"] in {"title", "text"} for match in matches)

    user = invoke(
        opencode_database,
        "sessions",
        "search",
        "assistant",
        "--role",
        "assistant",
        "--format",
        "json",
    )
    assert user.exit_code == 0, user.output
    assert [match["role"] for match in json_output(user)] == ["assistant"]

    reasoning = invoke(
        opencode_database,
        "sessions",
        "search",
        "private reasoning",
        "--include-reasoning",
        "--format",
        "json",
    )
    assert reasoning.exit_code == 0, reasoning.output
    assert [match["source"] for match in json_output(reasoning)] == ["reasoning"]

    tool = invoke(
        opencode_database,
        "sessions",
        "search",
        "tool output",
        "--include-tool-outputs",
        "--format",
        "json",
    )
    assert tool.exit_code == 0, tool.output
    assert [match["source"] for match in json_output(tool)] == ["tool-output"]

    todo = invoke(
        opencode_database,
        "sessions",
        "search",
        "Investigate",
        "--include-todos",
        "--format",
        "json",
    )
    assert todo.exit_code == 0, todo.output
    assert [match["source"] for match in json_output(todo)] == ["todo"]

    literal = invoke(
        opencode_database,
        "sessions",
        "search",
        "100%_ready",
        "--format",
        "json",
    )
    assert literal.exit_code == 0, literal.output
    assert [match["session"]["id"] for match in json_output(literal)] == [
        "session-root"
    ]

    unicode = invoke(
        opencode_database, "sessions", "search", "東京", "--format", "json"
    )
    assert unicode.exit_code == 0, unicode.output
    assert [match["session"]["id"] for match in json_output(unicode)] == [
        "session-root"
    ]


def test_invalid_options_and_database_errors(
    tmp_path: Path, opencode_database: Path
) -> None:
    invalid = invoke(opencode_database, "sessions", "list", "--format", "nope")
    assert invalid.exit_code == 2

    invalid_search = invoke(
        opencode_database, "sessions", "search", "needle", "--in", "nope"
    )
    assert invalid_search.exit_code == 2

    missing = invoke(tmp_path / "missing.db", "sessions", "list")
    assert missing.exit_code == 1
    assert "database" in missing.output.lower()

    incompatible = tmp_path / "incompatible.db"
    sqlite3.connect(incompatible).close()
    schema = invoke(incompatible, "sessions", "list")
    assert schema.exit_code == 1
    assert "schema" in schema.output.lower() or "database" in schema.output.lower()
