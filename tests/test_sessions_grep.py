import json
from pathlib import Path

import pytest
from sessions_contract import EXPECTED_SESSION
from typer.testing import CliRunner

from opencode_power_pack import cli
from opencode_power_pack.commands.sessions import _search_common, grep_sessions
from opencode_power_pack.models import SearchSource

runner = CliRunner()


def test_defaults_grep_active_title_and_text(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    result = runner.invoke(cli.app, ["sessions", "grep", "needle", "--format", "json"])

    assert result.exit_code == 0
    query = search_api["query"]
    assert query.sources == frozenset({SearchSource.TITLE, SearchSource.TEXT})
    assert query.filters.archive_mode.value == "active"
    assert query.filters.repository_project_ids == frozenset()
    assert query.roles == frozenset({"user", "assistant"})


def test_grep_collects_repeated_filters_and_all_sources(
    search_api: dict[str, object],
) -> None:
    result = runner.invoke(
        cli.app,
        [
            "sessions",
            "grep",
            "needle",
            "-r",
            "one",
            "--repo",
            "two",
            "--role",
            "user",
            "--role",
            "assistant",
            "--agent",
            "build",
            "--provider",
            "openai",
            "--model",
            "model-x",
            "--in",
            "title",
            "--include-reasoning",
            "--include-tool-inputs",
            "--include-tool-outputs",
            "--include-children",
            "--include-archived",
            "-n",
            "0",
            "--snippet-length",
            "10",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    query = search_api["query"]
    assert search_api["repo_names"] == ["one", "two"]
    assert query.filters.repository_project_ids == frozenset({"id:one", "id:two"})
    assert query.roles == frozenset({"user", "assistant"})
    assert query.sources == frozenset(
        {
            SearchSource.TITLE,
            SearchSource.REASONING,
            SearchSource.TOOL_INPUT,
            SearchSource.TOOL_OUTPUT,
        }
    )
    assert query.filters.include_children is True
    assert query.filters.archive_mode.value == "all"
    assert query.limit == 0
    assert query.snippet_length == 10


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["", "--format", "json"], "must not be empty"),
        (["needle", "--repo", "one", "--cwd"], "cannot be used"),
        (["needle", "--include-archived", "--archived-only"], "cannot be used"),
        (["needle", "-n", "-1"], "Invalid value"),
        (["needle", "--snippet-length", "0"], "Invalid value"),
        (
            [
                "needle",
                "--since",
                "2025-01-02T00:00:00Z",
                "--until",
                "2025-01-01T00:00:00Z",
            ],
            "must be earlier",
        ),
    ],
)
def test_grep_validation_is_clean(arguments: list[str], message: str) -> None:
    result = runner.invoke(cli.app, ["sessions", "grep", *arguments])

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output


def test_table_output_and_no_match(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    table = runner.invoke(
        cli.app, ["sessions", "grep", "needle"], env={"COLUMNS": "240"}
    )

    assert table.exit_code == 0
    assert "Updated" in table.output
    assert "Repository" in table.output
    assert "example-repo" in table.output
    assert "/work/example" in table.output
    assert "A title" in table.output
    assert "matching" in table.output

    class EmptyDatabase:
        def __init__(self, _: Path) -> None:
            pass

        def __enter__(self) -> EmptyDatabase:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def list_repositories(self) -> tuple[str, ...]:
            return ()

        def search_sessions(self, _: object) -> tuple[()]:
            return ()

    monkeypatch.setattr(grep_sessions, "OpenCodeDatabase", EmptyDatabase)
    empty = runner.invoke(cli.app, ["sessions", "grep", "needle"])
    assert empty.exit_code == 0
    assert empty.output == "No matching sessions.\n"


def test_table_abbreviates_home_directory(
    search_api: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_search_common.Path, "home", lambda: Path("/work"))

    table = runner.invoke(
        cli.app, ["sessions", "grep", "needle"], env={"COLUMNS": "240"}
    )

    assert table.exit_code == 0
    assert "~/example" in table.output


def test_json_and_jsonl_are_exact_and_use_context_database(
    search_api: dict[str, object],
) -> None:
    arguments = ["--database", "/override.db", "sessions", "grep", "needle"]
    json_result = runner.invoke(cli.app, [*arguments, "--format", "json"])
    jsonl_result = runner.invoke(cli.app, [*arguments, "--format", "jsonl"])
    expected = {
        "session": EXPECTED_SESSION,
        "source": "text",
        "role": "assistant",
        "snippet": "A matching\nsnippet",
        "matchCount": 2,
    }

    assert json_result.exit_code == 0
    assert json.loads(json_result.output) == [expected]
    assert jsonl_result.exit_code == 0
    assert json.loads(jsonl_result.output) == expected
    assert search_api["explicit_path"] == Path("/override.db")


@pytest.mark.parametrize("option", ["--no-rerank", "--semantic"])
def test_grep_has_no_semantic_options(option: str) -> None:
    result = runner.invoke(cli.app, ["sessions", "grep", "needle", option])

    assert result.exit_code == 2
    assert "No such option" in result.output


@pytest.mark.parametrize("excluded", ["session-root", "session-root-second"])
def test_exclude_leaves_out_the_whole_session_tree(
    opencode_database: Path, excluded: str
) -> None:
    def ids(*options: str) -> set[str]:
        result = runner.invoke(
            cli.app,
            [
                "--database",
                str(opencode_database),
                "sessions",
                "grep",
                "needle",
                "--include-children",
                "--format",
                "jsonl",
                *options,
            ],
        )
        assert result.exit_code == 0, result.output
        return {
            json.loads(line)["session"]["id"] for line in result.stdout.splitlines()
        }

    assert "session-root" in ids()
    remaining = ids("--exclude", excluded)
    assert "session-root" not in remaining
    assert "session-root-second" not in remaining
    assert "session-child" in remaining


def test_exclude_rejects_unknown_sessions(opencode_database: Path) -> None:
    result = runner.invoke(
        cli.app,
        [
            "--database",
            str(opencode_database),
            "sessions",
            "grep",
            "needle",
            "--exclude",
            "missing",
        ],
    )
    assert result.exit_code != 0
    assert "no session matches" in result.output
